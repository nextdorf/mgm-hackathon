from langchain_openai import ChatOpenAI
from pathlib import Path
import dotenv
import base64

dotenv.load_dotenv()


llm = ChatOpenAI(model='gpt-5-mini')
pdf_path = Path('data/archive/2018/de/hotel/20180915_THE MADISON HAMBURG.pdf')


with open(pdf_path, "rb") as f:
  file_base64 = base64.b64encode(f.read()).decode("utf-8")

msg = dict(
  role='user',
  content=[
    dict(type='text', text='Descirbe in 1-3 sentences the content'),
    dict(
      type='file',
      file=dict(
        filename=pdf_path.name,
        file_data=f'data:application/pdf;base64,{file_base64}',
      )
    ),
  ]
)

response = llm.invoke([
  msg
])
print(response.content)
# A hotel invoice (Rechnung Nr. 474081) from THE MADISON Hamburg to APImeister Consulting GmbH for a stay from 09.09.2018 to 14.09.2018 (five nights at €110 each), with a total charge of €550.00 (net €514.02 + €35.98 VAT). The page also shows payment by Mastercard (card ending 5621), tax/bank details and contact information for the hotel.


