import transformers
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import GPT2Model



def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in the given pre-trained language model.

    Args:
        model (nn.Module): The pre-trained language model.

    Returns:
        int: Total number of parameters in the model.
    """
    total_params = 0

    if isinstance(model, dict):
        for param in model.values():
            total_params += param.numel()
    else:
        for param in model.parameters():
            total_params += param.numel()

    return total_params

model = GPT2Model.from_pretrained('gpt2')

total_params = count_parameters(model)
print("Total number of parameters:", total_params)
