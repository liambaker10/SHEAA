import torch
import torch.nn as nn
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def GPT2SingleInjector(param_index, num_bits, input_text):
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

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # print("Before modifications:")
    # print(list(model.parameters())[99])

    modify_parameter(model, param_index, num_bits)

    # After modification
    # print("After modifications:")
    # print(list(model.parameters())[99])

    # Generate text using the modified model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)

# GPT2SingleInjector(7,9,'Liam Baker')
