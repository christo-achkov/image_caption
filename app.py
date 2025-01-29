import gradio as gr

from ImageCaptioning import caption_image

iface = gr.Interface(fn=caption_image, inputs=gr.Image(), outputs="text", title='image captioning')
iface.launch()