from typing import Dict, Literal
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


from ..util import LazyVal
from ._openrouter import ChatOpenRouter
from ._cerebras import ChatCerebras



class MultiChatModel:
  llm_ids = (
    'gpt', 'gpt-extra',
    'gemini', 'gemini-extra', 'gemini-exp',
    # 'claude', 'claude-extra', 'claude-opus',
    'router-gemini', 'router-gpt',
    'cerebras-gpt', 'cerebras-llama', 'cerebras-qwen',
  )
  def __init__(self, llm_id = 'gemini'):
    self._llms: Dict[str, LazyVal[BaseChatModel]] = {
      'gpt': LazyVal(lambda: ChatOpenAI(model='gpt-5-mini', reasoning_effort='low')),
      'gpt-extra': LazyVal(lambda: ChatOpenAI(model='gpt-5', reasoning_effort='medium')),
      'gemini': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')),
      'gemini-extra': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.5-flash')),
      'gemini-exp': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')),
      # 'claude': LazyVal(lambda: ChatAnthropic(model='claude-haiku-4-5')), # pyright: ignore[reportCallIssue]
      # 'claude-extra': LazyVal(lambda: ChatAnthropic(model='claude-sonnet-4-5')), # pyright: ignore[reportCallIssue]
      # 'claude-opus': LazyVal(lambda: ChatAnthropic(model='claude-opus-4-5')), # pyright: ignore[reportCallIssue]
      'router-gemini': LazyVal(lambda: ChatOpenRouter(model='google/gemini-2.5-flash-lite')),
      'router-gpt': LazyVal(lambda: ChatOpenRouter(model='openai/gpt-oss-120b')),

      'cerebras-gpt': LazyVal(lambda: ChatCerebras(model='gpt-oss-120b', extra_body=dict(reasoning_effort='low'))),
      'cerebras-llama': LazyVal(lambda: ChatCerebras(model='llama3.1-8b')),
      'cerebras-qwen': LazyVal(lambda: ChatCerebras(model='qwen-3-32b')),

    }
    assert MultiChatModel.llm_ids == tuple(self._llms)
    self.llm_id = llm_id

  @property
  def llm(self):
    return self._llms[self.llm_id].val

  @property
  def model_provider(self) -> Literal['gpt', 'gemini', 'claude', 'router', 'cerebras']:
    return self.llm_id.split('-')[0] # pyright: ignore[reportReturnType]

  @property
  def model_name(self):
    provider = self.model_provider
    if provider == 'gpt':
      return self.llm.model_name
    elif provider == 'gemini':
      return self.llm.model.split('/', 1)[-1]
    else:
      try:
        return self.llm.model
      except:
        return 'UNKNOWN'

  def with_id(self, llm_id):
    res = MultiChatModel(llm_id)
    res._llms = { k: v.copy() for k,v in self._llms.items() } # pylint: disable=protected-access
    return res

  def __setitem__(self, llm_id, llm):
    self._llms[llm_id].val = llm
  def __getitem__(self, llm_id):
    return self._llms[llm_id].val
  def __delitem__(self, llm_id):
    del self._llms[llm_id].val
  def __iter__(self):
    return (x.val for x in self._llms.values())

  def items(self):
    return ((k, v.val) for k,v in self._llms.items())

  def __repr__(self):
    return f'{MultiChatModel.__qualname__}(id={self.llm_id}, model={self.model_name})'


if __name__ == '__main__':
  import time
  import numpy as np
  from langgraph.graph import add_messages


  multi_llm = MultiChatModel()
  responses = {}
  hist = add_messages([], [
    ('system', 'You are being benchmarked. Follow the instructions strictly and answer with 1000 words.'),
    ('user', 'Cite the first 1000 words of the bible. Don\'t talk back and just do what I am asking immediatly'),
  ])
  hist

  for k, llm in multi_llm.items():
    if k.split('-')[0] in 'gpt gemini'.split():
      continue
    # print(k)
    ts, rs = [], []
    ts.append(time.time_ns())
    for ch in llm.stream(hist):
      rs.append(ch.content)
      ts.append(time.time_ns())
    ts = (np.asarray(ts[1:]) - np.asarray(ts[:-1])) / 1e9
    responses[k] = dict(
      chunks = rs,
      full = ''.join(rs),
      ts = ts,
      tft = ts[0],
      t_tot = np.sum(ts),
      dt_mean = np.mean(ts[1:]),
      dt_std = np.std(ts[1:], ddof=1),
    )

    print('{k}: ({tft:.3f} + {dt_mean:.3f} ± {dt_std:.3f})s -> {t_tot:.3f}s'.format(k=k, **responses[k]))

    # s_full = responses[k]['full']
    # s = s_full if len(s_full) <= 150 else f'{s_full[:72]} ... {s_full[-72:]}'
    words = responses[k]['full'].split(' ')
    s = ' '.join(words) if len(words) <= 150 else f'{' '.join(words[:74])} ... {' '.join(words[-74:])}'
    print('\n  | '.join(f'  | {s}'.splitlines()))




