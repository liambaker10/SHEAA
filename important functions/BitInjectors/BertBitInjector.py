import torch
import torch.nn as nn
import random
from transformers import BertForMaskedLM, BertTokenizer

def BertSingleInjector(param_index, num_bits, input_text):
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

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    modify_parameter(model, param_index, num_bits)

    input_tokens = tokenizer.tokenize(input_text)
    masked_index = input_tokens.index("[MASK]")


    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

        assert (
            predictions.shape[1] >= masked_index + 1
        ), "Masked index is out of bounds."

        predicted_token_index = torch.argmax(
            predictions[0, masked_index + 1]
        ).item()  
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_index)

    # Print the generated word
    print("Generated word:")
    print(predicted_token)

# BertSingleInjector(9, 10, "Liam Baker is a [MASK].")
