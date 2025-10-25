import gradio as gr
import dotenv
import uuid
import json
from pathlib import Path

from dataclasses import dataclass
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import List, Optional

dotenv.load_dotenv()

from .agent import agent, Ctx
# from .util import multistrip


@dataclass
class FullCtx:
  agent_ctx: Ctx
  chat_hist: List[AnyMessage]

  def __post_init__(self):
    self.update_chat_hist([])

  def update_chat_hist(self, new_mgs: List[AnyMessage], overwrite=True):
    all_msgs = add_messages(self.chat_hist, new_mgs)
    if overwrite:
      self.chat_hist = all_msgs
    return all_msgs

leading_user_input_mark = '+++ Prompt with metadata +++'.upper()

def gen_user_message(base_msg: Optional[str], uploaded_pdfs: List[str]):
  if not base_msg:
    base_msg = 'Please analyze the uploaded files.'
  tagged_content = []
  if uploaded_pdfs:
    count_uploads = len(uploaded_pdfs)
    tagged_content.append((
      f'{'a new pdf' if count_uploads==1 else f'{count_uploads} new pdfs'} associated with the user prompt',
      '\n'.join((f'{i:>2}) {uuid}' for i, uuid in enumerate(uploaded_pdfs)))
    ))
  if base_msg:
    tagged_content.append(('original user prompt', base_msg))
  else:
    tagged_content.append(('no user prompt', ''))
  res = leading_user_input_mark
  res += '\n\n\n\n'.join((f'[{head.upper()}]\n{body}' for head, body in tagged_content))
  return res

async def process_multimodal_message(in_msg, _history, ctx: FullCtx):
  msg = in_msg.get('text')
  files = in_msg.get('files')
  if not msg and not files:
    return 'Please provide a message or upload a file.', ctx
  
  new_file_uuids = []
  # Add uploaded files to persistent context
  for file in files:
    file_path = Path(file)
    file_uuid = str(uuid.uuid4())
    new_file_uuids.append(file_uuid)
    ctx.agent_ctx.files[file_uuid] = str(file_path)
  
  # Build user message for the agent
  # user_message = msg if msg else 'Please analyze the uploaded files.'
  user_message = gen_user_message(msg, new_file_uuids)
  
  # Use the agent with persistent context
  response = await agent.ainvoke(
    # {'messages': [{'role': 'user', 'content': user_message}]},
    dict(messages=ctx.update_chat_hist([('human', user_message)])),
    context=ctx.agent_ctx
  )
  agent_dialog = response['messages']
  print(json.dumps([m.model_dump() for m in agent_dialog], indent=2))
  final_response = next((m for m in agent_dialog[::-1] if isinstance(m, AIMessage)), '')
  artifacts = [x for x in (getattr(m, 'artifact', None) for m in agent_dialog) if x is not None]


  # TODO: what about the artifacts?

  return final_response.content, ctx
    
  # except Exception as e:
  #   return f'Error: {str(e)}', ctx

# Create persistent context state
system_prompt = f'''
You are a helpful AI agent who's main task it is to analyze invoices.

By default you answer in the language used by the user.


The user interacts with you via a chat-ui - the total source code can be found at "https://github.com/nextdorf/mgm-hackathon". The user will upload one or more pdfs and expect you to analyze them and extract relevant structured data. Since the input pdfs might be hard to read, your reported confidence level plays a significant role.

Furthermore, you aim to help the user with any follow-up question they might have or report the extracted data in the ideal/specified format.

Whenever possible, aim to give short but informative and relevant answers (1-2 sentences), unless you are requested to explain something in detail.


The User input which directly comes from the chat will be marked by the first line being "{leading_user_input_mark}". These types of messages usually contain tags of the form "[...]" followed by the body associated with the tag. It may be used to inject metadata for the respective user prompt.
'''.strip()
ctx_state = gr.State(FullCtx(Ctx(), [('system', system_prompt)]))

# Create ChatInterface with multimodal support and session state
demo = gr.ChatInterface(
  fn=process_multimodal_message,
  multimodal=True,
  title='🤖 Multimodal Chat',
  description='Chat with AI using text and files (PDF, images)',
  type='messages',
  additional_inputs=[ctx_state],
  additional_outputs=[ctx_state],
)


