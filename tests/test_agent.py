import base64
from types import SimpleNamespace
import uuid

import pytest

from mgm_hackathon.agent import Ctx, parse_pdfs
from mgm_hackathon.schemas import Confidence, InvoiceEntry, Schema


class DummyMessage:
  def __init__(self, content: str):
    self.content = content


class DummyStructuredLLM:
  def __init__(self, schema):
    self.schema = schema
    self.last_messages = None

  async def ainvoke(self, messages):
    self.last_messages = messages
    return {
      "raw": DummyMessage("dummy raw response"),
      "parsed": self.schema(
        kind="ambiguous",
        inner_kind="other",
        entries=[
          InvoiceEntry(
            description="Latte",
            date=None,
            amount=9.99,
            confidence=Confidence(level=70),
          )
        ],
        total_price=9.99,
        currency="EUR",
        language="en",
        confidence=Confidence(level=95),
        free_text="Structured summary",
      ),
    }


class DummyLLM:
  def __init__(self):
    self.structured = None

  def __repr_name__(self):
    return "DummyLLM"

  def with_structured_output(self, schema, include_raw=True, strict=True):
    self.structured = DummyStructuredLLM(schema)
    return self.structured


@pytest.mark.asyncio
async def test_parse_pdfs_builds_file_messages(tmp_path):
  pdf_path = tmp_path / "sample.pdf"
  pdf_payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f\n"
  pdf_path.write_bytes(pdf_payload)

  file_uuid = str(uuid.uuid4())
  dummy_llm = DummyLLM()

  schema_uuid = Ctx.default_schema_uuid()
  fake_context = SimpleNamespace(
    files={file_uuid: str(pdf_path)},
    parsing_schemas={schema_uuid: Schema},
  )
  fake_context.get_llm = lambda purpose="default": dummy_llm
  fake_context.get_parsing_schema = lambda uuid=None: Schema

  runtime = SimpleNamespace(context=fake_context)

  raw_content, parsed = await parse_pdfs(
    [file_uuid], schema_uuid=None, user_prompt="Please parse", runtime=runtime
  )

  assert raw_content == "dummy raw response"
  assert isinstance(parsed, Schema)
  assert parsed.total_price == pytest.approx(9.99)

  structured_llm = dummy_llm.structured
  assert structured_llm is not None
  sent_messages = structured_llm.last_messages
  assert sent_messages is not None
  assert sent_messages[0]["role"] == "user"
  content = sent_messages[0]["content"]
  assert content[0] == {"type": "text", "text": "Please parse"}
  file_part = content[1]
  expected_base64 = base64.b64encode(pdf_payload).decode("utf-8")
  assert file_part["type"] == "file"
  assert file_part["mime_type"] == "application/pdf"
  assert file_part["base64"] == expected_base64
