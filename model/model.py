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

CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.1"

class Model:
    def __init__(self, **kwargs) -> None:
        # enables representation reading and control in the pipeline 
        repe_pipeline_registry()
        self._data_dir = kwargs["data_dir"]
        self.tokenizer = None
        self.model = None
        self.honesty_rep_reader = None
        self.rep_control_pipeline = None
        self.dataset = None 
        self.honesty_activations = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float16, device_map="auto"
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, 
                                                use_fast=use_fast_tokenizer, 
                                                padding_side="left", legacy=False)
        self.dataset = self.get_dataset()
        print("Honesty dataset loaded successfully!")
        self.honesty_rep_reader = self.get_honesty_rep_reader()
        print("Honesty rep reader loaded successfully!")
        self.rep_control_pipeline = self.get_rep_control_pipeline()
        print("Representation control pipeline loaded successfully!")

       
    def predict(self, request: dict):
        prompt = request.get("prompt", "")
        honesty_coeff = request.get("honesty_coefficient", 2)
        max_new_tokens = request.get("max_new_tokens", 128)

        # Apply activations with basic arguments
        control_outputs = self.rep_control_pipeline(
            text_inputs=prompt, 
            activations=self.get_activations(honesty_coeff), 
            max_new_tokens=max_new_tokens,
            repetition_penalty=1,
            no_repeat_ngram_size=3
            
        )

        # Assuming control_outputs contains the generated text
        return control_outputs[0]['generated_text']
    
    def get_honesty_rep_reader(self):
        
        rep_token = -1
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        self.tokenizer.pad_token_id = 0

        rep_reading_pipeline =  pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        dataset = self.dataset
        
        honesty_rep_reader = rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            n_difference=n_difference,
            train_labels=dataset['train']['labels'],
            direction_method=direction_method,
            batch_size=32,
        )
        return honesty_rep_reader
    
    
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
    
    def get_activations(self, coeff = 2):
        layer_id = list(range(-5, -18, -1))
        activations = {}
        for layer in layer_id:
            activations[layer] = torch.tensor(coeff * self.honesty_rep_reader.directions[layer] * self.honesty_rep_reader.direction_signs[layer]).to(self.model.device).half()
        return activations
    
    def get_dataset(self):
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
        data_path = os.path.join(self._data_dir, 'facts', 'facts_true_false.csv')
        dataset = honesty_function_dataset(data_path, self.tokenizer, user_tag, assistant_tag)
        return dataset
    
