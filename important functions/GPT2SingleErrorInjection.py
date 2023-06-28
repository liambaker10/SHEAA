import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def GPT2SingleErrorInjector(param_index, new_val, input_text):
    def modify_parameter(model: nn.Module, param_index: int, new_val):
        parameter = list(model.parameters())[param_index]

        # Modify the parameter with the new value
        parameter.data.fill_(new_val)

    # Create the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    modify_parameter(model, param_index, new_val)

    # Set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Generate text using the modified model
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)


# GPT2SingleErrorInjector(99, 0.5, 'Liam Baker loves')
