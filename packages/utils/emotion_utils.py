

from representation_engineering.examples.primary_emotions.utils import primary_emotions_concept_dataset


def get_emotions_dataset(data_path):
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    data_dir = "../../data/emotions"
    user_tag =  "[INST]"
    assistant_tag =  "[/INST]"
    data = primary_emotions_concept_dataset(data_path, user_tag=user_tag, assistant_tag=assistant_tag)
    
    return data


def get_emotion_rep_readers():