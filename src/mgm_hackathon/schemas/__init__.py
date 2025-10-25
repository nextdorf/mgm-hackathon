from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import uuid

from . flights_models import FlightModel
from . hotel_models import HotelModel
from . public_transport_models import PublicTransportModel
from . restaurant_models import RestaurantModel
from . shop_models import ShopModel




class InvoiceEntry(BaseModel):
  description: Optional[str] = Field(description='description of the entry')
  date: Optional[str] = Field(description='date of the entry')
  amount: Optional[float] = Field(description='Amount')
  confidence_level: float = Field(description='Confidence level from 0 to 1 ')
class Schema(BaseModel):
  kind: Literal['cafe', 'flights', 'hotel', 'misc', 'public transport', 'restaurant', 'retail', 'unknown'] = Field(description='The kind of the input PDF')
  entries: List[InvoiceEntry] = Field(description='List of invoice entries. Leave empty if none are found. Make sure to have one entry per invoice item')
  total_price: float = Field(description='The total price to pay')
  currency: str = Field(description='Currency of total_prices.')
  confidence_level: float = Field(description='Confidence level from 0 to 1 ')
  language: Literal['de', 'en', 'other']
  free_text: str = Field(description='A free text comment. Add any other relevant information not part of the schema but important for the user here.')


def _generate_default_schemas():
  res = { str(uuid.UUID(int=0)): Schema }
  res.update({str(uuid.uuid4()) : schema for schema in [FlightModel, HotelModel, PublicTransportModel, RestaurantModel, ShopModel]})
  # Schema(kind='cafe', entries=[], total_price=0, currency='', confidence_level=0, language='de', free_text='').model_dump_json()
  return res

