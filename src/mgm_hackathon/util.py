from typing import Any, Callable, Generic, List, Optional, Type, TypeVar
import json
from pathlib import Path
import anyio

T = TypeVar('T')


def multistrip(s: str):
  lines = s.splitlines()
  line_ident = [(len(l) - len(l.lstrip())) if l.strip() else None for l in lines]
  ident_est = min(ident for ident in line_ident if ident is not None)
  unidented = '\n'.join(l[ident_est:] for l in lines)
  leading_ident = next(ident for ident in line_ident if ident is not None) - ident_est
  res = ' '*leading_ident + unidented.strip()
  return res

def chain_pairs(*its: Any|List[Any], sep:Any=None, res_factory:Type=list):
  def chain(*its: Any|List[Any]):
    for it in its:
      if not isinstance(it, str) and hasattr(it, '__iter__'):
        yield from it
      else:
        yield it
  def gen(*its: Any|List[Any]):
    a, b = sep, sep
    for it in chain(*its):
      a = b
      b = it
      if a != sep and b != sep:
        yield a, b
  res = gen(*its)
  return res if res_factory is None else res_factory(res)

async def temp_async_json_dump(data: dict) -> Path:
  async with anyio.NamedTemporaryFile(
    mode="w+",
    suffix=".json",
    prefix="invoice_",
    delete=False,  # keep for Gradio downloads
  ) as f:
    # async write
    await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    await f.flush()
    return Path(f.name)



class LazyVal(Generic[T]):
  def __init__(self, f: Callable[[], T]):
    self.__f = f
    self.__calculated: bool = False
    self.__val: Optional[T] = None
  @property
  def val(self) -> T:
    if not self.__calculated:
      self.__val = self.__f()
      self.__calculated = True
    return self.__val # pyright: ignore[reportReturnType]
  @val.setter
  def val(self, val: T):
    self.__val = val
    self.__calculated = True
  @val.deleter
  def val(self):
    self.__val = None
    self.__calculated = False

  def copy(self, recalculate=False):
    res = LazyVal(self.__f)
    if recalculate:
      del res.val
    return res


def _merge_dicts(src: dict, dst: dict):
  for k, v in src.items():
    dst[k] = _merge_dicts(v, dst.get(k, {})) if isinstance(v, dict) else v
  return dst




