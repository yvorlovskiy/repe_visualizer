import requests as re
import gradio as gr
import os

# Set API Key
baseten_api_key = os.environ["BASETEN_API_KEY"]

def format_prompt(prompt, **kwargs):
    # Build data dictionary dynamically from kwargs and always include the prompt
    data = {"prompt": prompt}
    data.update({k: v for k, v in kwargs.items() if v is not None})
    return data

def call_baseten_model(prompt, **kwargs):
    url = "https://app.baseten.co/models/rwn44d23/predict" 
    headers = {"Authorization": f"Api-Key {baseten_api_key}"}
    formatted_data = format_prompt(prompt, **kwargs)

    response = re.post(url, json=formatted_data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Parse and return as needed
    else:
        return "Error: " + response.text

def gradio_interface(prompt, control, honesty_coefficient, emotion):
    if control == "honesty":
        return call_baseten_model(prompt, control=control, honesty_coefficient=honesty_coefficient)
    elif control == "emotion":
        return call_baseten_model(prompt, control=control, emotion=emotion)
    else:
        return "Invalid control selection."

iface = gr.Interface(
    fn=gradio_interface, 
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Dropdown(
            choices=["honesty", "emotion"], 
            label="Rep Control Selector", 
            info="Select which representation control you want to use"
        ),
        gr.Slider(
            minimum=-2, maximum=2, step=0.1, value=0, 
            label="Honesty Coefficient"
        ),
        gr.Dropdown(
            choices=["happiness", "sadness", "anger", "fear", "disgust", "surprise"], 
            label="Emotion", 
            info="Emotion selector for emotion controls"
        )
    ], 
    outputs="text",
    live=False
)

iface.launch()
