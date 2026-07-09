from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math

def load_test_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test_dataset = dataset["test"]["text"]
    print(len(test_dataset))
    return "\n\n".join(test_dataset)

model_name = "EleutherAI/pythia-31m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = load_test_data()

tokens = tokenizer(text, return_tensors="pt").input_ids

device = "cpu" #broke
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

num_tokens = tokens.shape[1]
max_length = 2048
stride = 512
prev_end = 0

total_loss = 0
total_tokens = 0

for begin in range(0, num_tokens, stride):
    input_ids = tokens[:, begin:(begin + stride)]
    target_ids = input_ids.clone()

    # already accounted for tokens
    num_new_tokens = (begin + stride) - prev_end
    target_ids[:, :-num_new_tokens] = -100

    outputs = model(input_ids=input_ids, labels=target_ids)
    
    # outputs returns averaged loss, we gotta unaverage it
    unaveraged_loss = outputs.loss.item() * (num_new_tokens - 1)
    total_loss += unaveraged_loss
    total_tokens += (num_new_tokens - 1)

avg_nll = total_loss / total_tokens
print("perplexity", math.exp(avg_nll))
