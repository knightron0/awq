from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import math

def load_test_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test_dataset = dataset["test"]["text"]
    print(len(test_dataset))
    return "\n\n".join(test_dataset)

def rtn_quantize(w):
    # w: (m x n)
    # w_max: (m x n/128 x 128)
    # w_min: (m x n/128 x 128)
    w_max = torch.reshape(w, (w.shape[0], w.shape[1] // 128, 128)).amax(dim=2)
    w_max = w_max.unsqueeze(-1).expand(-1, -1, 128).flatten(1)
    w_min = torch.reshape(w, (w.shape[0], w.shape[1] // 128, 128)).amin(dim=2)
    w_min = w_min.unsqueeze(-1).expand(-1, -1, 128).flatten(1)
    
    # scale: (m x n/128)
    scale = (w_max - w_min) / 15
    z = torch.clamp(torch.round(-w_min / scale), min=0, max=15)

    quant_w = torch.round(w / scale) + z
    quant_w = quant_w.clamp(0, 15)

    dequant_w = (quant_w - z) * scale
    assert dequant_w.shape == w.shape
    return dequant_w

def run_rtn_quantize(model):
    with torch.no_grad():
        modules = list(model.named_modules())
        for name, module in tqdm(modules, desc="rtn quantize"):
            if isinstance(module, torch.nn.Linear) and "embed" not in name:
                dequant_weight = rtn_quantize(module.weight)
                module.weight.copy_(dequant_weight)
    return model


model_name = "EleutherAI/pythia-14m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = load_test_data()

tokens = tokenizer(text, return_tensors="pt").input_ids

device = "cpu" #broke
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

model = run_rtn_quantize(model)

num_tokens = tokens.shape[1]
max_length = 2048
stride = 512
prev_end = 0

total_loss = 0
total_tokens = 0

for begin in tqdm(range(0, num_tokens, stride), desc="eval ppl"):
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

