import torch
import torch.nn as nn
import random
from transformers import BartTokenizer, BartForConditionalGeneration

def BartSingleInjector(param_index, num_bits, input_text):
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

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    modify_parameter(model, param_index, num_bits)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)

    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    print("Summary:", summary)

# BartSingleInjector(0,1000,'Liam Baker is a ')
