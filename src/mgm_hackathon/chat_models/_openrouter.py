from typing import Literal
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from openai import responses
from pydantic import BaseModel, Field
import base64

from ..util import _merge_dicts


# See https://openrouter.ai/request-builder

def _merge_dicts(src: dict, dst: dict):
  for k, v in src.items():
    dst[k] = _merge_dicts(v, dst.get(k, {})) if isinstance(v, dict) else v
  return dst

class ChatOpenRouter(ChatOpenAI):
  def __init__(self, *, model='google/gemini-2.5-flash-lite', **kwargs):
    default_kwargs = dict(
      api_key=secret_from_env("OPENROUTER_API_KEY", default=None),
      base_url='https://openrouter.ai/api/v1',
      model=model,
      default_headers = {'HTTP-Referer': 'https://github.com/nextdorf/mgm-hackathon', 'X-Title': 'Nextdorf'},
      extra_body=dict(provider=dict(
        sort='throughput',
        allow_fallbacks=True,
        order='cerebras baseten/fp4 groq google-vertex'.split()
      )),
    )
    _merge_dicts(default_kwargs, kwargs)
    if hasattr(kwargs['api_key'], '__call__'):
      kwargs['api_key'] = kwargs['api_key']()

    super().__init__(**kwargs)


if __name__ == '__main__':

  llm0 = ChatOpenAI(model='gpt-5-nano', reasoning_effort='minimal')


  llm_gem = ChatOpenRouter(model='google/gemini-2.5-flash-lite')
  llm_gpt = ChatOpenRouter(model='openai/gpt-oss-120b')
  llm = llm_gem

  responses = []

  responses.append(response := llm.invoke('Who are you?'))

  class MathQuestion(BaseModel):
    x: int = Field()
    y: int = Field()
    op: Literal['+', '-', '*'] = Field(description='The operation of the math question')
    result: int = Field(description='The result of `x` `op` `y`')
    answer: str = Field('Very short written answer, Examples:\n  "Calculate the sum of 6 and 7" -> "The sum of 6 and 7 is 13"\n  "I was so bored that I started counting the passing cars. I saw 7 black cars and all of them had 4 passengers. How many people in black cars is that in total?" -> "You saw in total 28 people in black cars."')

  responses.append(response := llm.with_structured_output(MathQuestion).invoke('What is 7 times 9?'))

  with open('../../data/archive/2018/de/hotel/20181117_THE MADISON HAMBURG.pdf', 'rb') as f:
    f_base64 = base64.b64encode(f.read()).decode()
  responses.append(response := llm.invoke([dict(role='user', content=[
    dict(type='text', text='Summarize the content of the file'),
    dict(type='file', mime_type='application/pdf', base64=f_base64) if llm.__repr_name__() != 'ChatOpenAI' else dict(type='file', file=dict(filename='Dummyfile.pdf', file_data=f'data:application/pdf;base64,{f_base64}')),
  ])]))

  print(response.content)


