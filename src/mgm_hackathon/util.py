from typing import Any, List, Type


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

