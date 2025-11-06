from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Type
import numpy as np
import re

from pydantic import BaseModel
from ._types import Confidence, Date, Datetime, Time


@dataclass
class ReprBuilder:
  schema: Type[BaseModel]
  theme: Literal['light', 'dark']=field(default='dark')
  ignore:List[str] = field(default_factory=lambda:'free_text currency'.split())

  def col_map(self, x: float):
    t = x*np.pi/2
    if self.theme=='light':
      r, g = map(round, (160*np.cos(t), 160*np.sin(t)))
    elif self.theme=='dark':
      r, g = map(round, 72+183*np.array((np.cos(t), np.sin(t))))
    else:
      raise ValueError(self.theme)
    return f'#{r:02x}{g:02x}{0:02x}'
  def gen_col_f(self, c: Optional[Confidence]):
    if c is None:
      return repr
    else:
      col = self.col_map(c.level/100)
      fmt = f"<span style='color:{col}'>{{s}}</span>"
      return lambda s: fmt.format(s=s)


  @property
  def currency_fields(self):
    for name, f in self.schema.__class__.model_fields.items():
      if name.lower().strip() == 'currency':
        matches_in_quotes = re.findall(r'`[A-Za-z0-9_-]+`', f.description) # pyright: ignore[reportCallIssue]
        return [m[1:-1] for m in matches_in_quotes]
    return []
  @property
  def fields(self):
    return [f for f in self.schema.__class__.model_fields if f not in self.ignore]
  @property
  def confidence(self):
    for name, f in self.schema.__class__.model_fields.items():
      if issubclass(f.annotation, Confidence):
        # return name
        return getattr(self.schema, name)
    return None

  def nest(self, inner: Type[BaseModel]):
    return ReprBuilder(inner, self.theme, self.ignore)

  def _repr_nested(self, fields: Optional[List[str]] = None):
    if fields is None:
      fields = self.fields

    entries = [self._repr_val(f) for f in fields]
    res_nested = f'<tr><td>{'</td><td>'.join(entries)}</td></tr>'
    return res_nested
  def _repr_list(self, vals: List[Type[BaseModel]]):
    # reprs = [ReprBuilder(val, self.theme, self.ignore) for val in vals]
    reprs = [self.nest(val) for val in vals]
    fields = reprs[0].fields if reprs else []
    for _fields in (r.fields for r in reprs[1:]):
      assert set(fields) == set(_fields), vals

    res_lines = [f'<tr><th>{'</th><th>'.join(fields)}</th></tr>']
    res_lines.extend(r._repr_nested(fields) for r in reprs)
    res = f'<table>\n  {'\n  '.join(res_lines)}\n</table>'
    return res
    
  def _repr_root(self):
    res_lines = []
    for f in self.fields:
      val_repr = self._repr_val(f)
      res_lines.append(f'<tr><td>{f}</td><td>{val_repr}</td></tr>')

    res = f'<table>\n  {'\n  '.join(res_lines)}\n</table>'
    return res

  def _repr_val(self, field_name: str, col_f: Optional[Callable] = None):
    def is_nested_type(x):
      if isinstance(val, BaseModel):
        not_nested_schemas = (Confidence, Date, Datetime, Time)
        return not any(isinstance(val, s) for s in not_nested_schemas)
      else:
        return False

    if not col_f:
      col_f = self.gen_col_f(self.confidence)

    val = getattr(self.schema, field_name)
    val_repr = (
        f'\n{self._repr_list(val)}\n'
      if isinstance(val, list) else
        f'\n{self.nest(val)._repr_root()}\n'
      # if isinstance(val, BaseModel) and not isinstance(val, Confidence) else
      if is_nested_type(val) else
        col_f(f'{val:.02f} {getattr(self.schema, 'currency')}')
      if field_name in self.currency_fields else
        col_f(val)
      )
    return val_repr

