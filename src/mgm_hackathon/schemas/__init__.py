from pydantic import BaseModel, Field, conint, create_model
from typing import Optional, Literal, List, Type, Union
import uuid
import numpy as np

from ..util import multistrip
from . flights_models import FlightModel
from . hotel_models import HotelModel
from . public_transport_models import PublicTransportModel
from . restaurant_models import RestaurantModel
from . shop_models import ShopModel
from ._types import Confidence

from .repr_builder import ReprBuilder



class InvoiceEntry(BaseModel):
  description: Optional[str] = Field(description='description of the entry')
  date: Optional[str] = Field(description='date of the entry')
  amount: float = Field(description='Amount')
  currency: str = Field(description='Currency of `amount`.')
  confidence: Confidence
class Schema(BaseModel):
  kind: Literal['ambiguous']
  inner_kind: Literal['cafe', 'flights', 'hotel', 'public-transport', 'restaurant', 'shop', 'other'] = Field(description='The kind of the input PDF')
  entries: List[InvoiceEntry] = Field(description='List of invoice entries. Leave empty if none are found. Make sure to have one entry per invoice item')
  total_price: float = Field(description='The total price to pay')
  currency: str = Field(description='Currency of `total_price`.')
  language: Literal['de', 'en', 'other']
  confidence: Confidence
  free_text: str = Field(description='A free text comment. Add any other relevant information not part of the schema but important for the user here.')

def _generate_default_schemas():
  # res = { str(uuid.UUID(int=0)): Schema }
  res = dict()
  res.update({str(uuid.uuid4()) : schema for schema in [FlightModel, HotelModel, PublicTransportModel, RestaurantModel, ShopModel]})
  # Schema(kind='cafe', entries=[], total_price=0, currency='', confidence_level=0, language='de', free_text='').model_dump_json()
  return res

def schema_union(*schemas: Type[BaseModel]):
  schema_dict = {s.__name__: (Optional[s], Field(None)) for s in schemas}
  res = create_model( # pyright: ignore[reportCallIssue]
    'SchemaUnion',
    choice=(Literal[tuple(schema_dict)], Field(description='The name of the non-None field. All other fields should be None')),
    **schema_dict
  )
  setattr(res, 'value', property(lambda this: getattr(this, this.choice)))
  return res



