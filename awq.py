import torch
from tqdm import tqdm
from utils import load_train_data
from rtn import rtn_quantize

def collect_weights_activations(model, tokenizer, max_sequences=256, model_name=""):
    text = load_train_data()
    tokens = tokenizer(text, return_tensors="pt").input_ids

    calib_seqlen = 512
    
    # trim, we don't need *all* train tokens
    tokens = tokens[:, : calib_seqlen * max_sequences]
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
        for begin in tqdm(range(0, num_tokens, calib_seqlen), desc="calibration"):
            input_ids = tokens[:, begin:(begin + calib_seqlen)].to(model.device)
            model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    cache = {
        "weights": weights_cache,
        "activations": activation_cache,
        "metadata": {
            "model_name": model_name,
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "calib_seqlen": calib_seqlen,
            "max_sequences": max_sequences,
        },
    }
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

def search_awq_scales(cache):
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
    return result