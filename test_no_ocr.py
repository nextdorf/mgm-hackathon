from langchain_openai import ChatOpenAI
from pathlib import Path
import dotenv
import base64
import langsmith

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


dotenv.load_dotenv()


llm = ChatOpenAI(model='gpt-5-mini')
# pdf_path = Path('data/archive/2018/de/hotel/20180915_THE MADISON HAMBURG.pdf')
# pdf_path = Path('data/archive/2018/de/cafe/20181024_018.pdf')

# pdf_path = Path('data/archive/2020/de/miscellaneous/unterwegs-20200314_002.pdf')

pdf_path = Path('data/use cases.pdf')


with open(pdf_path, "rb") as f:
  file_base64 = base64.b64encode(f.read()).decode("utf-8")

msg = dict(
  role='user',
  content=[
    dict(type='text', text='Descirbe in 1-3 sentences the content. The output language should match the language of the pdf'),
    dict(
      type='file',
      file=dict(
        filename=pdf_path.name,
        file_data=f'data:application/pdf;base64,{file_base64}',
      )
    ),
  ]
)

class Date(BaseModel):
  day: int
  month: int
  year: int

  def __repr__(self) -> str:
    return f'{self.day:02}.{self.month:02}.{self.year:04}'

class InvoiceEntry(BaseModel):
  description: Optional[str] = Field(description='description of the entry')
  date: Optional[Date] = Field(description='date of the entry')
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



llm_structured = llm.with_structured_output(Schema, include_raw=True, strict=True)

response = llm_structured.invoke([msg])
raw_msg = response['raw']
parsed_msg = response['parsed']
print(parsed_msg.model_dump())
# A hotel invoice (Rechnung Nr. 474081) from THE MADISON Hamburg to APImeister Consulting GmbH for a stay from 09.09.2018 to 14.09.2018 (five nights at €110 each), with a total charge of €550.00 (net €514.02 + €35.98 VAT). The page also shows payment by Mastercard (card ending 5621), tax/bank details and contact information for the hotel.


