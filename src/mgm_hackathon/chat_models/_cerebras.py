from typing import Literal
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from openai import responses
from pydantic import BaseModel, Field

from ..util import _merge_dicts

# See https://openrouter.ai/request-builder

class ChatCerebras(ChatOpenAI):
  def __init__(self, *, model: Literal['llama3.1-8b', 'llama-3.3-70b', 'gpt-oss-120b', 'qwen-3-32b'] = 'gpt-oss-120b', **kwargs):
    default_kwargs = dict(
      api_key=secret_from_env("CEREBRAS_API_KEY", default=None),
      base_url='https://api.cerebras.ai/v1',
      model=model,
      default_headers = {'HTTP-Referer': 'https://github.com/nextdorf/mgm-hackathon', 'X-Title': 'Nextdorf'},
    )
    _merge_dicts(default_kwargs, kwargs)
    if hasattr(kwargs['api_key'], '__call__'):
      kwargs['api_key'] = kwargs['api_key']()

    super().__init__(**kwargs)


if __name__ == '__main__':

  llm_gpt = ChatCerebras(model='gpt-oss-120b', extra_body=dict(reasoning_effort='low'))
  llm_llama = ChatCerebras(model='llama3.1-8b')
  llm_qwen = ChatCerebras(model='qwen-3-32b')
  llm = llm_qwen

  responses = []

  responses.append(response := llm.invoke('Who are you?'))

  class MathQuestion(BaseModel):
    x: int = Field()
    y: int = Field()
    op: Literal['+', '-', '*'] = Field(description='The operation of the math question')
    result: int = Field(description='The result of `x` `op` `y`')
    answer: str = Field('Very short written answer, Examples:\n  "Calculate the sum of 6 and 7" -> "The sum of 6 and 7 is 13"\n  "I was so bored that I started counting the passing cars. I saw 7 black cars and all of them had 4 passengers. How many people in black cars is that in total?" -> "You saw in total 28 people in black cars."')

  responses.append(response := llm.with_structured_output(MathQuestion).invoke('What is 7 times 9?'))

