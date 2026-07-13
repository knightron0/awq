import torch
from tqdm import tqdm

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