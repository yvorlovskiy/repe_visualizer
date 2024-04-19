from transformers import pipeline
import torch
import numpy as np
from datasets import load_dataset

def get_rep_reader(model, tokenizer, dataset):
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    direction_finder_kwargs={"n_components": 1}

    train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

    train_data, train_labels = train_dataset['sentence'], train_dataset['label']
    test_data = test_dataset['sentence']

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "


    train_data = [template.format(instruction=s) for s in train_data]
    test_data = [template.format(instruction=s) for s in test_data]

    rep_reader = rep_reading_pipeline.get_directions(
        train_data, 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=train_labels, 
        direction_method=direction_method,
        direction_finder_kwargs=direction_finder_kwargs
    )

    return rep_reader

def get_activations(model, rep_reader, harmless_coefficient=.5):
    layer_id = list(range(-25, -33, -1))
    activations = {}
    component_index = 0
    for layer in layer_id:
        activations[layer] = torch.tensor(harmless_coefficient * rep_reader.directions[layer][component_index] * rep_reader.direction_signs[layer][component_index]).to(model.device).half()
    return activations

def get_dataset(data_path, tokenizer):
    dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
    return dataset