
import torch
from transformers import pipeline

from representation_engineering.examples.honesty.utils import honesty_function_dataset

def get_rep_reader(model, tokenizer, dataset):
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    tokenizer.pad_token_id = 0

    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=dataset['train']['labels'],
        direction_method=direction_method,
        batch_size=32,
    )
    return rep_reader


def get_activations(model, rep_reader, honesty_coefficient = .5):
    layer_id = list(range(-5, -18, -1))
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(honesty_coefficient * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()
    return activations



def get_dataset(data_path, tokenizer):
    user_tag = "[INST]"
    assistant_tag = "[/INST]"
    dataset = honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag)
    return dataset