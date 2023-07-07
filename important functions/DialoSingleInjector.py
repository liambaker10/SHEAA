import torch
import torch.nn as nn
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def DialoSingleInjector(param_index, num_bits, input_text):
    def flip_bits(tensor, num_bits):
        # Convert the tensor to binary
        binary_tensor = tensor.byte()
        
        # Get the number of bits in each element
        num_bits_per_element = binary_tensor.numel() * 8
        
        # Generate random bit indices to flip
        bit_indices = random.sample(range(num_bits_per_element), num_bits)
        
        # Create a copy of the tensor to modify
        modified_tensor = tensor.clone()
        
        # Flip the selected bits in the modified tensor
        for index in bit_indices:
            element_index = index // 8  # Index of the element in the tensor
            bit_offset = index % 8      # Offset of the bit within the element
            modified_tensor.view(-1)[element_index] = modified_tensor.view(-1)[element_index].to(torch.long) ^ (1 << bit_offset)
        
        return modified_tensor

    def modify_parameter(model: nn.Module, param_index: int, num_bits: int):
        parameter = list(model.parameters())[param_index]
        modified_parameter = flip_bits(parameter.data, num_bits)
        parameter.data.copy_(modified_parameter)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    modify_parameter(model, param_index, num_bits)

    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

# Generate a response using the DialoGPT model
    response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id)
    # pretty print last ouput tokens from bot
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("DialoGPT: ", response)

# DialoSingleInjector(1, 85, "How are you?")
