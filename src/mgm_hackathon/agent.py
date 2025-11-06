import uuid
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pathlib import Path
import dotenv
import base64
from langgraph.runtime import Runtime
import langsmith
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, create_model
from typing import List, Literal, Optional, Dict, Type

from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field
import langchain.tools as lang_tools
from langchain.tools import ToolRuntime

import asyncio
import aiofiles

from .util import _merge_dicts
from .schemas import _generate_default_schemas, ReprBuilder, schema_union
from .chat_models import MultiChatModel

dotenv.load_dotenv()


multi_llm = MultiChatModel()

@dataclass
class Ctx:
  files: Dict[str, str] = field(default_factory=dict)
  parsing_schemas: Dict[str, Type[BaseModel]] = field(default_factory=_generate_default_schemas)
  # llm: BaseChatModel = field(default_factory=lambda: ChatOpenAI(model='gpt-5-mini', reasoning_effort='minimal'))
  # openai_kwargs: dict = field(default_factory=lambda: dict(model='gpt-5-mini', reasoning_effort='minimal'))
  llm_choice: dict = field(default_factory=lambda: dict(
    # default='router-gemini',
    default='cerebras-gpt-oss-120b',
    ocr='gpt-4.1-nano'
  ))

  # @property
  # def llm(self):
  #   return ChatOpenAI(**self.openai_kwargs)

  def get_llm(self, purpose: Literal['default', 'ocr'] = 'default'):
    choice = self.llm_choice.get(purpose, 'default')
    return multi_llm[choice]

  def get_parsing_schema(self, uuid: None|str|List[str] = None):
    # if uuid is None or uuid not in self.parsing_schemas:
    #   uuid = Ctx.default_schema_uuid()
    # return self.parsing_schemas[uuid]
    if isinstance(uuid, str):
      uuids = [uuid]
    elif uuid is None:
      uuids = []
    else:
      uuids = list(uuid)
    uuids = [u for u in uuids if u in self.parsing_schemas]
    if not uuids:
      # uuids = list((u for u in self.parsing_schemas if UUID(u).int))
      uuids = list(self.parsing_schemas)
    if len(uuids) == 1:
      return self.parsing_schemas[uuids[0]]
    else:
      return schema_union(*(self.parsing_schemas[u] for u in uuids))

  @staticmethod
  def default_schema_uuid():
    return str(uuid.UUID(int=0))

  @staticmethod
  def ctx_tools():
    @lang_tools.tool(description='Returns a lookup dictionary for all available files')
    async def get_files(runtime: ToolRuntime[Ctx]):
      return runtime.context.files
    @lang_tools.tool(description='Returns a lookup dictionary for all available parsing schemas. Invoke fallback-schema by passing `None`')
    async def get_parsing_schemas(runtime: ToolRuntime[Ctx]):
      return runtime.context.parsing_schemas
    
    return [get_files, get_parsing_schemas]



@lang_tools.tool(
  description='Analyze the given files from `file_uuids`. If set the Output is returned in a structure specified by `schema_uuid`. If `schema_uuid` is specified as `None`, a fallback schema is used. The returned content will be in the preferred markdown+extension format.',
  response_format='content_and_artifact',
)
async def parse_pdfs(file_uuids: List[str], schema_uuid: Optional[str], user_prompt: str, runtime: ToolRuntime[Ctx]):
  llm = runtime.context.get_llm('ocr')

  content = []
  if user_prompt:
    content.append(dict(type='text', text=user_prompt))
  for file_uuid in file_uuids:
    if not (path := runtime.context.files.get(file_uuid)):
      continue
    async with aiofiles.open(path, "rb") as f:
      f_content = await f.read()
      f_base64 = base64.b64encode(f_content).decode("utf-8")
    file_fields = (
        dict(mime_type='application/pdf', base64=f_base64)
      if llm.__repr_name__() != 'ChatOpenAI' else
        dict(file=dict(filename=Path(path).name, file_data=f'data:application/pdf;base64,{f_base64}'))
    )

    content.append(
      _merge_dicts(file_fields, dict(type='file'))
    )

  msg = dict(role='user', content=content)

  # llm = runtime.context.llm
  schema = runtime.context.get_parsing_schema(schema_uuid) # pyright: ignore[reportCallIssue]
  llm_structured = llm.with_structured_output(schema, include_raw=True, strict=True)

  # response = llm_structured.invoke([msg])
  response: dict = await llm_structured.ainvoke([msg])
  raw_msg = response['raw']
  parsed_msg = response['parsed']
  if parsed_msg.__repr_name__() == 'SchemaUnion':
    parsed_msg = parsed_msg.value
  res = (raw_msg.content, parsed_msg)
  # _repr = ReprBuilder(parsed_msg)._repr_root()
  # res = (_repr, parsed_msg)
  return res

# @lang_tools.tool(description='Parses the ')
# async def parse_schema(artifact_uuid: str, runtime: ToolRuntime[Ctx]):
#   schema = runtime.context.parsing_schemas[artifact_uuid]
#   _repr = ReprBuilder(schema)._repr_root()
#   return _repr

tools = [parse_pdfs] + Ctx.ctx_tools()


agent = create_agent(
  # model=ChatOpenAI(model='gpt-5-mini', reasoning_effort='minimal'),
  model=Ctx().get_llm(),
  tools=tools,
  context_schema=Ctx
)

if __name__ == '__main__':
  import nest_asyncio
  nest_asyncio.apply() # For jupyter notebook compability

  pdfs = {
    str(uuid.uuid4()): p for p in [
      'data/archive/2018/de/hotel/20180915_THE MADISON HAMBURG.pdf',
      'data/archive/2018/de/cafe/20181024_018.pdf',
      'data/archive/2020/de/miscellaneous/unterwegs-20200314_002.pdf',
      # 'data/use cases.pdf',
    ]
  }

  responses = []

  responses.append(agent.invoke(
    dict(messages=[dict(role='user', content='Do something')]),
    context=Ctx(pdfs)
  ))

  responses.append(asyncio.run(
    agent.ainvoke( # pylint: disable=await-outside-async
      dict(messages=[dict(role='user', content=f'Tell me something interesting about the file with id {list(pdfs)[0]}')]),
      context=Ctx(pdfs)
    )
  ))

  responses.append(asyncio.run(
    agent.ainvoke(
      dict(messages=[dict(role='user', content='Make an analysis of the content and tell me something interesting about the invoice from the madison hotel')]),
      context=Ctx(pdfs)
    )
  ))




  # llm_structured = llm.with_structured_output(Schema, include_raw=True, strict=True)

  # response = llm_structured.invoke([msg])
  # raw_msg = response['raw']
  # parsed_msg = response['parsed']
  # print(parsed_msg.model_dump())
  # # A hotel invoice (Rechnung Nr. 474081) from THE MADISON Hamburg to APImeister Consulting GmbH for a stay from 09.09.2018 to 14.09.2018 (five nights at €110 each), with a total charge of €550.00 (net €514.02 + €35.98 VAT). The page also shows payment by Mastercard (card ending 5621), tax/bank details and contact information for the hotel.


