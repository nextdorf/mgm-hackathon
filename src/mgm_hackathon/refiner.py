from typing import Annotated, List, Literal, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from langgraph.runtime import Runtime

from .agent import agent as core_agent, Ctx
from .util import chain_pairs, multistrip


@dataclass
class Refining:
  messages: Annotated[List[AnyMessage], add_messages] # = field(default_factory=list)
  id_msg_to_refine: Optional[str] = field(default=None)
  refined_msg: Optional[str] = field(default=None)
  reasoning_effort: Optional[Literal['minimal', 'low', 'medium', 'high']] = field(default='medium')

class RefineSchema(BaseModel):
  refined_version: str = Field(description='A refined version of the input message (the input message = the last message in the history). The refined_version is considered to be the output message. It is intended to be a drop in replacement for the input message.')
  freetext: str = Field(description='Use this field for any relevant information which should be present in the answer but would be out of place in any other schema-field. In the case that any of the checks below fail, append an explanation here.')
  avoid_uuids: bool = Field(description='The backend uses uuids to index several kinds of data. The user will not have any use for them, unless explicitly stated by one of the human chat messages. Thus, any reference to retrieved data should refer to them by human-friendly labels/names/descriptions/etc')
  ensure_natural_conversation_flow: bool = Field(description='Due to the nature of how LLMs work, some of the replies by the AI may have a self-prompting structure (e.g. preface the prompt with a short summary to enrich the context with relevant information). However, this form structured reply often leads to a chat with an unnatural flow. The refined version should ensure a natural conversation flow')

async def refine_msg(state: Refining, runtime: Runtime[Ctx]):
  if state.id_msg_to_refine is None:
    state.id_msg_to_refine = state.messages[-1].id
  msg = next((x for x in state.messages[::-1] if x.id == state.id_msg_to_refine), None)
  if msg is None:
    return {}
  
  if state.reasoning_effort:
    base_llm_kwargs = runtime.context.openai_kwargs
    base_llm_kwargs.update(reasoning_effort=state.reasoning_effort)
    base_llm = ChatOpenAI(**base_llm_kwargs)
  else:
    base_llm = state.llm
  llm = base_llm.with_structured_output(RefineSchema, strict=True, include_raw=True)

  prompt = multistrip('''
    Determine a refined version of the input message.

    The overall goal of the refinement process is to ensure that the user will not be bothered with "behind-the-scene" data and a stream of thought of irrelevant information. Even if the content may be important to be present in the chat history for proper context enrichment, it shall be removed if it is not immedeatly informative for the human-user of the chatbot. 

    Use the free-text field of the schema which may be relevant for the developer of the chatbot and for typing out the checks and requirement-validations

    Furthermore, perform the checks and ensure that the requirements which are specified by the schema-definition of the structured output.
  ''')
  response = await llm.ainvoke(prompt)
  _raw_msg = response['raw']
  parsed_msg = response['parsed']




def _build_refiner(and_compile=True):
  refiner = StateGraph(Refining)
  refiner.add_node('core', core_agent)
  refiner.add_node('refine_msg', refine_msg)
  for e1, e2 in chain_pairs(START, 'core refine_msg'.split(), END):
    refiner.add_edge(e1, e2)
  if and_compile:
    res = refiner.compile(name='Refiner')
    return res
  else:
    return refiner

refiner = _build_refiner()



