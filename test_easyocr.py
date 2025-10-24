import easyocr

# reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
# result = reader.readtext('chinese.jpg')

reader = easyocr.Reader(['en', 'de'])

import fitz  # PyMuPDF
from pathlib import Path

pdf_path = Path('data/archive/2018/de/hotel/20180915_THE MADISON HAMBURG.pdf')
img_path = Path('test_img.png')

with fitz.open(pdf_path) as doc:
  page = doc.load_page(0)
  pix = page.get_pixmap()
  pix.save(img_path)

res = reader.readtext(str(img_path))
