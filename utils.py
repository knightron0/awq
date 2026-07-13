from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import math

def load_test_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test_dataset = dataset["test"]["text"]
    return "\n\n".join(test_dataset)

def load_train_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]["text"]
    return "\n\n".join(train_dataset)

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model

def eval_perplexity(model, text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids

    num_tokens = tokens.shape[1]
    eval_seqlen = 512
    prev_end = 0

    total_loss = 0
    total_tokens = 0

    for begin in tqdm(range(0, num_tokens, eval_seqlen), desc="eval ppl"):
        input_ids = tokens[:, begin:(begin + eval_seqlen)]
        target_ids = input_ids.clone()

        # already accounted for tokens
        num_new_tokens = (begin + eval_seqlen) - prev_end
        target_ids[:, :-num_new_tokens] = -100

        outputs = model(input_ids=input_ids, labels=target_ids)
        
        # outputs returns averaged loss, we gotta unaverage it
        unaveraged_loss = outputs.loss.item() * (num_new_tokens - 1)
        total_loss += unaveraged_loss
        total_tokens += (num_new_tokens - 1)

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)