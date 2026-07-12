from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import math
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_test_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test_dataset = dataset["test"]["text"]
    return "\n\n".join(test_dataset)

def load_train_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]["text"]
    return "\n\n".join(train_dataset)

def rtn_quantize(w):
    original_dtype = w.dtype
    w = w.float()
    # w: (m x n)
    # w_max: (m x n/128 x 128)
    # w_min: (m x n/128 x 128)
    w_max = torch.reshape(w, (w.shape[0], w.shape[1] // 128, 128)).amax(dim=2)
    w_max = w_max.unsqueeze(-1).expand(-1, -1, 128).flatten(1)
    w_min = torch.reshape(w, (w.shape[0], w.shape[1] // 128, 128)).amin(dim=2)
    w_min = w_min.unsqueeze(-1).expand(-1, -1, 128).flatten(1)
    
    # scale: (m x n/128)
    scale = ((w_max - w_min) / 15).clamp(min=1e-5)
    z = torch.clamp(torch.round(-w_min / scale), min=0, max=15)

    quant_w = torch.round(w / scale) + z
    quant_w = quant_w.clamp(0, 15)

    dequant_w = (quant_w - z) * scale
    assert dequant_w.shape == w.shape
    return dequant_w.to(original_dtype)

def run_rtn_quantize(model):
    with torch.no_grad():
        modules = list(model.named_modules())
        for name, module in tqdm(modules, desc="rtn quantize"):
            if isinstance(module, torch.nn.Linear) and "embed" not in name:
                dequant_weight = rtn_quantize(module.weight)
                module.weight.copy_(dequant_weight)
    return model

def load_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model

def eval_perplexity(model, text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids

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
    return math.exp(avg_nll)

def collect_weights_activations(model, tokenizer, cache_path=CACHE_DIR / "pythia14m_wikitext2_calib.pt"):
    if cache_path.exists():
        return torch.load(cache_path)

    text = load_train_data()
    tokens = tokenizer(text, return_tensors="pt").input_ids

    max_length = 2048 
    stride = 512 
    max_sequences = 256
    
    # trim, we don't need *all* train tokens
    tokens = tokens[:, : (stride * max_sequences) + max_length]
    num_tokens = tokens.shape[1]
    weights_cache = {}
    activation_cache = {}

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0].detach().cpu()
            activation_cache.setdefault(name, []).append(x)
        return hook

    handles = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "embed" not in name:
            handle = module.register_forward_pre_hook(make_hook(name))
            weights_cache[name] = module.weight.detach().cpu().clone()
            handles.append(handle)

    model.eval()
    with torch.no_grad():
        for begin in tqdm(range(0, num_tokens, stride), desc="calibration"):
            input_ids = tokens[:, begin:(begin + stride)].to(model.device)
            model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    cache = {
        "weights": weights_cache,
        "activations": activation_cache,
        "metadata": {
            "model_name": model_name,
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "stride": stride,
            "max_length": max_length,
            "max_sequences": max_sequences,
        },
    }
    torch.save(cache, cache_path)
    return cache


def calculate_layer_loss(name, s, cache):
    # given one scale tensor for one linear layer
    # AWQ formula: Q(Ws)(X/s) - WX
    weights_cache = cache["weights"]
    activation_cache = cache["activations"]

    total_loss = 0
    total_cnt = 0
    W = weights_cache[name].float()
    s = s.float().clamp(min=1e-5)
    Ws = W * s
    QWs = rtn_quantize(Ws)

    for activation in tqdm(activation_cache[name], desc=f"loss {name}", leave=False):
        x = activation.squeeze().float()
        scaled_activation = x / s
        loss = ((scaled_activation @ QWs.float().t() - x @ W.t()) ** 2).mean()
        total_loss += loss
        total_cnt += 1

    return total_loss / total_cnt

def normalize_awq_scale(s):
    s = s.float().clamp(min=1e-5)
    return s / torch.sqrt(s.max() * s.min())

def calculate_base_scale(cache):
    activation_cache = cache["activations"]
    scales = {}
    for name in activation_cache: 
        baseline = activation_cache[name][0]
        sum_channel = None
        for x in activation_cache[name]:
            x = torch.squeeze(x).float()
            if sum_channel is not None:
                sum_channel += x.abs().sum(dim=0)
            else:
                sum_channel = x.abs().sum(dim=0)
        sum_channel /= (torch.squeeze(baseline).shape[0] * len(activation_cache[name]))
        scales[name] = sum_channel.clamp(min=1e-5)
    return scales

def apply_awq_scale(model, scales):
    with torch.no_grad():
        modules = list(model.named_modules())
        for name, module in tqdm(modules, desc="awq quantize"):
            if isinstance(module, torch.nn.Linear) and "embed" not in name:
                s = scales[name].float().clamp(min=1e-5)
                W = module.weight.float()

                # (Q(Ws))/s
                W_new = rtn_quantize(W * s) / s
                module.weight.copy_(W_new)
    return model

def search_awq_scales(cache, cache_path=CACHE_DIR / "pythia14m_wikitext2_awq_scales.pt"):
    if cache_path.exists():
        return torch.load(cache_path)

    base_scales = calculate_base_scale(cache)
    alpha_grid = torch.arange(0, 1, 0.05)
    best_scales = {}
    best_alphas = {}

    for name in base_scales:
        best_loss = float("inf")
        best_scale = None
        best_alpha = None

        for alpha in tqdm(alpha_grid, desc=f"alpha grid ({name})"):
            alpha_scale = normalize_awq_scale(torch.pow(base_scales[name].clone(), alpha))
            loss = calculate_layer_loss(name, alpha_scale, cache)
            if loss < best_loss:
                best_loss = loss
                best_scale = alpha_scale
                best_alpha = alpha

        best_scales[name] = best_scale
        best_alphas[name] = best_alpha

    result = {
        "scales": best_scales,
        "alphas": best_alphas,
    }
    torch.save(result, cache_path)
    return result

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cpu" #broke
model = load_model()

text = load_test_data()

# baseline FP16
fp16_model = load_model()
print("FP16: ")
print(eval_perplexity(fp16_model, text, tokenizer))

# calculate AWQ values 
cache = collect_weights_activations(model, tokenizer)
awq_result = search_awq_scales(cache)
best_scales = awq_result["scales"]
best_alphas = awq_result["alphas"]

awq_model = load_model()
awq_model = apply_awq_scale(awq_model, best_scales)
print("after AWQ: ")
print(eval_perplexity(awq_model, text, tokenizer))

# RTN 
rtn_model = load_model()
rtn_model = run_rtn_quantize(rtn_model)
print("after RTN: ")
print(eval_perplexity(rtn_model, text, tokenizer))
