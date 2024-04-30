import sys
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from repe import repe_pipeline_registry
repe_pipeline_registry()
import json



from transformers import AutoTokenizer, AutoConfig, pipeline
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc

from repe.rep_control_contrast_vec import ContrastVecLlamaForCausalLM, ContrastVecMistralForCausalLM


def prepare_decoder_only_inputs(prompts, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)


    # concatenate prompt and target tokens and send to device
    inputs = prompt_inputs.to(device)

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    return inputs, mask, prompt_inputs["input_ids"].shape[1]

def construct_question(question, options, prompt):
    all_letter = ["A.", "B.", "C.", "D."]
    option_str = ""
    
    for i in range(len(options)):

        cur_letter = all_letter[i]
        cur_option = options[i]
        option_str = option_str + cur_letter + cur_option + "\n"

    option_str = f"Question: {question}\n Options:" + option_str + ". " + prompt
    return option_str


model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = ContrastVecMistralForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="sequential").to("cuda")


use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 


print(tokenizer("A").input_ids, tokenizer("B").input_ids, tokenizer("C").input_ids, tokenizer("D").input_ids)

batch_size = 128
layer_ids = np.arange(0, 32, 2).tolist()
# layer_ids = np.arange(7, 31, 3).tolist()

pos_type = 'Give a harmful answer:'
neg_type = 'Give an unharmful answer:'
user_tag = "[INST] "
assistant_tag = "[/INST]"


from datasets import load_dataset
dataset = load_dataset(
  "thu-coai/SafetyBench",
  "test",
  split="en"
)

dataset = dataset.filter(lambda example: example["category"] == "Illegal Activities")

all_result = {}

for item in tqdm(dataset):

    cur_question = user_tag + construct_question(item["question"], item["options"], "Answer:")
    cur_id = item["id"]
    cur_options = item["options"]


    inputs, masks, orig_split = prepare_decoder_only_inputs([cur_question], tokenizer, "cuda")

    directions = {}
    for layer_id in layer_ids:
        directions[layer_id] = 0

    q_batch_pos = [user_tag + construct_question(item["question"], item["options"], pos_type)]
    q_batch_neg = [user_tag + construct_question(item["question"], item["options"], neg_type)]

    inputs_pos_s, masks_pos_s, split_pos = prepare_decoder_only_inputs(q_batch_pos, tokenizer, "cuda")
    inputs_neg_s, masks_neg_s, split_neg = prepare_decoder_only_inputs(q_batch_neg, tokenizer,"cuda")
    split = inputs_neg_s['input_ids'].shape[1] - split_neg

    with torch.no_grad():
        logits = model(**inputs,
                  pos_input_ids=inputs_pos_s['input_ids'],
                  pos_attention_mask=inputs_pos_s['attention_mask'],
                  neg_input_ids=inputs_neg_s['input_ids'],
                  neg_attention_mask=inputs_neg_s['attention_mask'],
                  contrast_tokens=-1, # last {split} tokens
                  compute_contrast=True,
                  alpha=0.1, # try 0.1+, maybe 0.1 for mistrals
                  control_layer_ids=layer_ids,
                  ).logits
    

    last_logit = torch.softmax(logits[0, -1], dim=0)




    choice_logits = [last_logit[330], last_logit[365], last_logit[334], last_logit[384]]
    choice_idx = 0
    cur_logit = -1000
    for i in range(len(cur_options)):
        if choice_logits[i] > cur_logit:
            choice_idx = i
            cur_logit = choice_logits[i]
    
    all_result[str(cur_id)] = choice_idx


result_path = ""
result_file = open(result_path, 'w')
result_file.write(json.dumps(all_result))
result_file.close() 
