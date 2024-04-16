

from sklearn import pipeline
import torch
import tqdm
from representation_engineering.examples.primary_emotions.utils import primary_emotions_concept_dataset


def get_emotion_rep_readers(model, tokenizer, data, emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]):
    
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    emotion_rep_readers = {}
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    for emotion in tqdm(emotions):
        train_data = data[emotion]['train']
                
        rep_reader = rep_reading_pipeline.get_directions(
            train_data['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=train_data['labels'], 
            direction_method=direction_method,
        )
    
    emotion_rep_readers[emotion] = rep_reader
    return emotion_rep_readers


def get_emotion_activations(model, rep_reader, coeff = .5):
    layer_id = list(range(-11, -30, -1))
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()

    
def get_emotions_dataset(data_path):
    user_tag =  "[INST]"
    assistant_tag =  "[/INST]"
    data = primary_emotions_concept_dataset(data_path, user_tag=user_tag, assistant_tag=assistant_tag)
    
    return data

