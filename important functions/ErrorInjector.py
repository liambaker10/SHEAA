import transformers
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # Specify the data type you want to retrieve the number of bits for
bitwidth = torch.finfo(dtype).bits

def error_map(injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, scale_factor = .1, p = 1e-10) -> torch.Tensor:
    with torch.no_grad():
        error_map = (2 * torch.ones((*injectee_shape, dtype_bitwidth), dtype=torch.int, device=device)) ** torch.arange(0, dtype_bitwidth, dtype=torch.int, device=device).flip(dims=(-1, )).expand((*injectee_shape, dtype_bitwidth))

        filter = (p * nn.functional.dropout(torch.ones_like(error_map, dtype=torch.float, device=device), 1 - p)).int()

        error_map = (filter * error_map * scale_factor).sum(dim=-1).int()

    return error_map

def error_inject(model, attack, sf, p):
    error_maps = {}

    for param_name, param in model.named_parameters():
        # Options for attacks are here, you can do weights in general bias in general
        # Then you can do specific kinds of weights/biases attn.weights, proj.weights
        if attack in param_name:# or "bias" in param_name:

            injectee_shape = param.shape

            error_maps[param_name] = error_map(injectee_shape, bitwidth, device, sf, p)

            error_fin = error_maps[param_name]

            param.data = (param.data.to(torch.int) ^ error_fin).to(torch.float)