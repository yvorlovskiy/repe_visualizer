'''
example use:
truss predict -d '{"prompt": "What is a large language model?"}'
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import os

from representation_engineering.examples.honesty.utils import honesty_function_dataset, plot_lat_scans, plot_detection_results
from representation_engineering.repe import repe_pipeline_registry
from utils import honesty_utils, emotion_utils 

CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.1"

class Model:
    def __init__(self, **kwargs) -> None:
        # enables representation reading and control in the pipeline 
        repe_pipeline_registry()
        self._data_dir = kwargs["data_dir"]
        self.tokenizer = None
        self.model = None
        self.rep_control_pipeline = None
        
        self.representation_controls = {
            "honesty": {
                "dataset_path": os.path.join(self._data_dir, 'facts', 'facts_true_false.csv'),
                "load_dataset": honesty_utils.get_dataset,
                "get_rep_reader": honesty_utils.get_rep_reader,
                "get_activations": honesty_utils.get_activations,
            }
            # "emotion": {
            #     "dataset_path": os.path.join(self._data_dir, 'emotions'),
            #     "load_dataset": emotion_utils.get_dataset,
            #     "get_rep_reader": emotion_utils.get_rep_readers,
            #     "get_activations": emotion_utils.get_activations,
            # }
            
        }

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float16, device_map="auto"
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, 
                                                use_fast=use_fast_tokenizer, 
                                                padding_side="left", legacy=False)
        
        for rep_name, control in self.representation_controls.items():
            if "dataset" not in control:
                control["dataset"] = control["load_dataset"](control["dataset_path"], self.tokenizer)
            if "rep_reader" not in control:
                control["rep_reader"] = control["get_rep_reader"](self.model, self.tokenizer, control["dataset"])
                print(f'{rep_name} reader loaded successfully')
                
        # self.dataset = honesty_utils.get_honesty_dataset(os.path.join(self._data_dir, 'facts', 'facts_true_false.csv'))
        # print("Honesty dataset loaded successfully!")
        # self.honesty_rep_reader = honesty_utils.get_honesty_rep_reader(self.model, self.tokenizer, self.dataset)
        # print("Honesty rep reader loaded successfully!")
        self.rep_control_pipeline = self.get_rep_control_pipeline()
        print("Representation control pipeline loaded successfully!")

       
    def predict(self, request: dict):
        prompt = request.get("prompt", "")
        control_type = request.get("control_type", "honesty")
        honesty_coeff = request.get("honesty_coefficient", 2)
        max_new_tokens = request.get("max_new_tokens", 128)
        
        get_activations = self.representation_controls[control_type]["get_activations"]
        print(type(self.representation_controls["honesty"]["rep_reader"]))

        print(self.representation_controls["honesty"]["rep_reader"].directions)

        # Apply activations with basic arguments
        control_outputs = self.rep_control_pipeline(
            text_inputs=prompt, 
            activations=get_activations(self.model, self.representation_controls["honesty"]["rep_reader"], honesty_coeff), 
            max_new_tokens=max_new_tokens,
            repetition_penalty=1,
            no_repeat_ngram_size=3
            
        )

        # Assuming control_outputs contains the generated text
        return control_outputs[0]['generated_text']
    
    def get_rep_control_pipeline(self):
        layer_id = list(range(-5, -18, -1))
        block_name="decoder_block"
        control_method="reading_vec"

        rep_control_pipeline = pipeline(
            "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=layer_id,
            control_method=control_method)
        return rep_control_pipeline
    
    def get_honesty_scores(self, texts, rep_token=-1):
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        honesty_scores_dict = {}
        
        for text in texts:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            scores = []
            for layer in hidden_layers:
                layer_hidden_states = hidden_states[layer][:, rep_token, :]
                score = torch.dot(layer_hidden_states.squeeze(), 
                                self.honesty_rep_reader.directions[layer] * 
                                self.honesty_rep_reader.direction_signs[layer])
                scores.append(score.item())
            
            honesty_score = sum(scores) / len(scores)  # Taking the mean score as the honesty score
            tokens = self.tokenizer.tokenize(text)
            
            # Assuming that each token gets the same honesty score for simplicity
            for token in tokens:
                honesty_scores_dict[token] = honesty_score
    
        return honesty_scores_dict
        
