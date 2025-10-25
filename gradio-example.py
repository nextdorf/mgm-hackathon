import gradio as gr
import base64
import dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from agent import agent


def process_multimodal_message(in_msg, history):
  msg = in_msg.get('text')
  files = in_msg.get('files')
  if not msg and not files:
    return 'Please provide a message or upload a file.'
  
  try:
    llm = ChatOpenAI(model='gpt-5-mini')
    
    # Build content array
    content = []
    
    # Add text if provided
    if msg:
      content.append({'type': 'text', 'text': msg})
    
    # Add files if provided
    if files:
      for file in files:
        file_path = Path(file)
        with open(file_path, 'rb') as f:
          file_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine file type
        if file_path.suffix.lower() == '.pdf':
          file_data = f'data:application/pdf;base64,{file_base64}'
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
          mime_type = f'image/{file_path.suffix.lower().replace(".", "")}'
          if file_path.suffix.lower() == '.jpg':
            mime_type = 'image/jpeg'
          file_data = f'data:{mime_type};base64,{file_base64}'
        else:
          continue  # Skip unsupported files
        
        content.append({
          'type': 'file',
          'file': {
            'filename': file_path.name,
            'file_data': file_data
          }
        })
    
    # Create message
    msg = {'role': 'user', 'content': content}
    
    # Get response
    response = llm.invoke([msg])
    return response.content
    
  except Exception as e:
    return f'Error: {str(e)}'

# Create ChatInterface with multimodal support
demo = gr.ChatInterface(
  fn=process_multimodal_message,
  multimodal=True,
  title='🤖 Multimodal Chat',
  description='Chat with AI using text and files (PDF, images)'
)

demo.launch(share=False)