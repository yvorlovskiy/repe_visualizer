import requests as re
import gradio as gr
import os

#Set API Key
baseten_api_key = os.environ["BASETEN_API_KEY"]


def format_prompt(prompt, honesty_coefficient):
    return {"prompt": prompt, "honesty_coefficient": honesty_coefficient}

def call_baseten_model(prompt, honesty_coefficient):
    url = "https://app.baseten.co/models/rwn44d23/predict" 
    headers = {"Authorization": f"Api-Key {baseten_api_key}"} 
    formatted_data = format_prompt(prompt, honesty_coefficient)

    response = re.post(url, json=formatted_data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Parse and return as needed
    else:
        return "Error: " + response.text

def gradio_interface(prompt, honesty_coefficient):
    return call_baseten_model(prompt, honesty_coefficient)

iface = gr.Interface(
    fn=gradio_interface, 
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Slider(minimum=-2, maximum=2, step=0.1, value=0, label="Honesty Coefficient")
    ], 
    outputs="text"
)
iface.launch()