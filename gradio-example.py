import gradio as gr


def fn(msg, hist):
  return 'Yes'

demo = gr.ChatInterface(fn)

demo.launch()


