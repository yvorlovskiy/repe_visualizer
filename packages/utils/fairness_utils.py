from transformers import pipeline
import torch

from representation_engineering.examples.fairness.utils import bias_dataset

def get_rep_reader(model, tokenizer, dataset):
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
    )

    return rep_reader

def get_activations(model, rep_reader, fairness_coefficient=0.5):
    layer_id = list(range(-11, -30, -1))

    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(fairness_coefficient * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()
    return activations

def get_dataset(data_path, tokenizer):
    user_tag =  "[INST]"
    assistant_tag =  "[/INST]"
    dataset = bias_dataset(user_tag=user_tag, assistant_tag=assistant_tag)
    return dataset

