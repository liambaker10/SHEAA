import torch
import torch.nn as nn
import random
from transformers import RobertaTokenizer, RobertaForMaskedLM

def RobertaSingleInjector(param_index, num_bits, input_text):
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

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    modify_parameter(model, param_index, num_bits)

    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_index = token_ids.index(tokenizer.mask_token_id)
    input_tensor = torch.tensor([token_ids]).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    predictions = outputs.logits[0, masked_index]
    predicted_token_ids = torch.argmax(predictions, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_ids.tolist())

    output_text = input_text.replace("<mask>", predicted_token)

    print("Output Text:", output_text)

# RobertaSingleInjector(50, 5, "Liam Baker is a <mask>.")
