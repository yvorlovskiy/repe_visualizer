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
    print(formatted_data)
    response = re.post(url, json=formatted_data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Parse and return as needed
    else:
        return "Error: " + response.text

def gradio_interface(prompt, control_type, honesty_coefficient, emotion, emotion_coefficient):
    return call_baseten_model(prompt, control_type=control_type, honesty_coefficient=honesty_coefficient, emotion=emotion, emotion_coefficient=emotion_coefficient)

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
            label="Honesty Coefficient",
            visible=lambda inputs: inputs["Rep Control Selector"] == "honesty"
        ),
        gr.Dropdown(
            choices=["happiness", "sadness", "anger", "fear", "disgust", "surprise"], 
            label="Emotion",
            info="Emotion selector for emotion controls",
            visible=lambda inputs: inputs["Rep Control Selector"] == "emotion"
        ),
        gr.Slider(
            minimum=-2, maximum=2, step=0.1, value=0, 
            label="Emotion Coefficient",
            visible=lambda inputs: inputs["Rep Control Selector"] == "emotion"
        )
    ],
    outputs="text",
    live=False
)

iface.launch()
