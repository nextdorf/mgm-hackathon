from typing import Optional
from pydantic import BaseModel, Field, model_validator
from datetime import date, time, datetime

class Confidence(BaseModel):
  level: int = Field(ge=0, le=100, description='Confidence level from 0 to 100 which estimates the correctness of the sibling properties <fields> of the owning object for down-stream. `level`=0 means that the object definitely contains faulty data, while `level`=100 means that the data perfectly reflect the source. `level`=50 may be used if further checks are necessary. If any of the siblings <fields> contain a `Confidence` object than this `level` should be equal or lower than any of the siblings\' `level`s')
  def __repr__(self):
    return f'{self.level/100:.0%}'
  def __str__(self):
    return repr(self)

class Date(BaseModel):
  day: int = Field(ge=1, le=31)
  month: int = Field(ge=1, le=12)
  year: int = Field()

  @model_validator(mode='after')
  def valid_date(self):
    _ = self.asdate
    return self

  @property
  def asdate(self):
    return date(self.year, self.month, self.day)
  def __repr__(self):
    return self.asdate.strftime('%d.%m.%Y')
  def __str__(self):
    return repr(self)

class Time(BaseModel):
  hour: int = Field(ge=0, lt=24)
  minute: int = Field(ge=0, lt=60)
  second: Optional[int] = Field(None, ge=0, lt=60)

  @property
  def astime(self):
    return time(self.hour, self.minute, self.second if self.second else 0)
  def __repr__(self):
    return self.astime.strftime(f'%H:%M{':%S' if self.second is not None else ''}')
  def __str__(self):
    return repr(self)

class Datetime(BaseModel):
  date: Date
  time: Optional[Time] = Field(None)

  @property
  def asdatetime(self):
    kw = dict(year=self.date.year, month=self.date.month, day=self.date.day)
    if (time := self.time) is not None:
      kw.update(hour=time.hour, minute=time.minute) # pylint: disable=no-member
      if (second:=time.second) is not None: # pylint: disable=no-member
        kw.update(second=second)
    return datetime(**kw)
  def __repr__(self):
    return f'{self.date}{f', {self.time}' if self.time else ''}'
  def __str__(self):
    return repr(self)

