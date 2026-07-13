from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import math
from utils import load_model, load_test_data, eval_perplexity
from awq import collect_weights_activations, search_awq_scales, apply_awq_scale
from rtn import run_rtn_quantize

model_name = "EleutherAI/pythia-14m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
text = load_test_data()

# baseline FP16
fp16_model = load_model(model_name, device)
print("FP16: ")
print(eval_perplexity(fp16_model, text, tokenizer))

# awq
calibration_model = load_model(model_name, device)
cache = collect_weights_activations(calibration_model, tokenizer, max_sequences=128, model_name=model_name)
awq_result = search_awq_scales(cache)
best_scales = awq_result["scales"]

awq_model = load_model(model_name, device)
awq_model = apply_awq_scale(awq_model, best_scales)
print("AWQ: ")
print(eval_perplexity(awq_model, text, tokenizer))

# RTN 
rtn_model = load_model(model_name, device)
rtn_model = run_rtn_quantize(rtn_model)
print("RTN: ")
print(eval_perplexity(rtn_model, text, tokenizer))
