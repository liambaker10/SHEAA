import torch
import torch.nn as nn
import random

def flip_bits(tensor, num_bits):
        binary_tensor = tensor.byte()
        num_bits_per_element = binary_tensor.numel() * 8
        
        bit_indices = random.sample(range(num_bits_per_element), num_bits)
        
        modified_tensor = tensor.clone()
        
        for index in bit_indices:
            element_index = index // 8 
            bit_offset = index % 8      
            modified_tensor.view(-1)[element_index] = modified_tensor.view(-1)[element_index].to(torch.long) ^ (1 << bit_offset)
        
        return modified_tensor

def modify_parameter(model: nn.Module, param_index: int, num_bits: int):
        parameter = list(model.parameters())[param_index]
        modified_parameter = flip_bits(parameter.data, num_bits)
        parameter.data.copy_(modified_parameter)

modify_parameter(model, param_index, num_bits)
