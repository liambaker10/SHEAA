import transformers
import torch
import torch.nn as nn
import sys
import pprint
import json
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def count_parameters(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_parameter_value_by_index(model, parameter_index):
    parameter_value = None
    model_parameters = list(model.parameters())
    if parameter_index < len(model_parameters):
        parameter_value = model_parameters[parameter_index].data
    return parameter_value

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
num_params = int(input("Enter the number of parameters to change: "))
new_value = int(input("Enter a new value: "))
x = random.randint(0, count_parameters(gpt2_model) - num_params)

state_dict = gpt2_model.state_dict()  # Get the model's state_dict

for i in range(x, x + num_params):
    parameter_index_gpt2 = i
    parameter_value = get_parameter_value_by_index(gpt2_model, parameter_index_gpt2)
    if parameter_value is not None:
        parameter_value.fill_(new_value)  # Modify the parameter value

# Create an instance of the modified model
modified_model = GPT2LMHeadModel.from_pretrained('gpt2')
modified_model.load_state_dict(state_dict, strict=False)  # Load the modified state_dict

# Create a tokenizer for the model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the device to run the model on (e.g., 'cuda' if you have a GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the specified device
modified_model = modified_model.to(device)

# Set the modified model to evaluation mode
modified_model.eval()

# Generate text using the modified model
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
with torch.no_grad():
    output = modified_model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)
