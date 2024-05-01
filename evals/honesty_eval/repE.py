from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from repe import repe_pipeline_registry
repe_pipeline_registry()





from transformers import AutoTokenizer, AutoConfig, pipeline
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc

from repe.rep_control_contrast_vec import ContrastVecLlamaForCausalLM, ContrastVecMistralForCausalLM


model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = ContrastVecMistralForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="sequential").to("cuda:0")


use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 


# parameters
batch_size = 64

user_tag = "[INST] "
assistant_tag = "[/INST] "




def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def load_tqa_sentences(user_tag, assistant_tag, preset=""):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions = [f'{user_tag}' + q + ' ' + preset] + questions
            answers = [f'{assistant_tag}' + a] + answers
        ls = d['mc1_targets']['labels']
        ls.reverse()
        labels.insert(0, ls)
    return questions, answers, labels

def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None] 
    return logprobs.squeeze(-1)
    
def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts[:1], return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets[:1], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)

    print(prompt_inputs["input_ids"].shape)
    # concatenate prompt and target tokens and send to device
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device) for k in prompt_inputs}

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    return inputs, mask, prompt_inputs["input_ids"].shape[1]

def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer
    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check 
        log_probs = output_logprobs[indices[i]:indices[i+1]]
        correct[i] = np.argmax(log_probs) == label.index(1)
    return correct.mean()

def get_tqa_accuracy(model, questions, answers, labels, tokenizer, batch_size=128):
    gc.collect()
    # get the log probabilities of each question answer pair
    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        # print(q_batch[0] + a_batch[0])
        inputs, masks, _ = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)

        with torch.no_grad():
            try:
                # set the masks so that we do not add to tokens of input sentences and padding tokens
                model.set_masks(masks.unsqueeze(-1))
            except:
                pass

            # calculate the probabilities for all tokens (all question answer pairs)
            logits = model(**inputs).logits
            # sum the probabilities for each question answer pair so that each pair has one probability
            # mask is zero for question and padding tokens
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

    return calc_acc(labels, output_logprobs)



batch_size = 128
layer_ids = np.arange(0, 32, 2).tolist()
# layer_ids = np.arange(7, 31, 3).tolist()

pos_type = 'Give a truthful answer. '
neg_type = 'Give an untruthful answer. '




questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")

output_logprobs = []
for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
    gc.collect()
    inputs, masks, orig_split = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)

    directions = {}
    for layer_id in layer_ids:
        directions[layer_id] = 0

    q_batch_pos = [q + pos_type for q in q_batch]
    q_batch_neg = [q + neg_type for q in q_batch]
    
    inputs_pos_s, masks_pos_s, split_pos = prepare_decoder_only_inputs(q_batch_pos, a_batch, tokenizer, model.model.device)
    inputs_neg_s, masks_neg_s, split_neg = prepare_decoder_only_inputs(q_batch_neg, a_batch, tokenizer, model.model.device)
    split = inputs_neg_s['input_ids'].shape[1] - split_neg

    with torch.no_grad():
        logits = model(**inputs,
                  pos_input_ids=inputs_pos_s['input_ids'],
                  pos_attention_mask=inputs_pos_s['attention_mask'],
                  neg_input_ids=inputs_neg_s['input_ids'],
                  neg_attention_mask=inputs_neg_s['attention_mask'],
                  contrast_tokens=-split, # last {split} tokens
                  compute_contrast=True,
                  alpha=0.1, # try 0.1+, maybe 0.1 for mistrals
                  control_layer_ids=layer_ids,
                  ).logits
        logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
    output_logprobs.extend(logprobs)

model_sample_wise_aa_acc = calc_acc(labels, output_logprobs)
print(f"model_sample_wise_aa_acc: {model_sample_wise_aa_acc}")
