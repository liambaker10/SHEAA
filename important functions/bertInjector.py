import torch
import torch.nn as nn
import weightdistribution as wd
from transformers import BertForMaskedLM, BertTokenizer

# Create the BERT model and tokenizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def get_parameter_importance(model: nn.Module) -> dict:
    parameter_importance = {}
    for name, parameter in model.named_parameters():
        parameter_importance[name] = torch.std(parameter).item()  # Calculate importance based on standard deviation
    return parameter_importance

def modify_parameters(model: nn.Module, num_params: int, modification_func: callable):
    parameter_importance = get_parameter_importance(model)
    sorted_params = sorted(parameter_importance.items(), key=lambda x: x[1], reverse=True)
    total_params = len(sorted_params)

    if num_params > total_params:
        num_params = total_params

    selected_params = [param[0] for param in sorted_params][:num_params]

    for parameter_name in selected_params:
        parameter = dict(model.named_parameters())[parameter_name]
        modified_parameter = modification_func(parameter)
        dict(model.named_parameters())[parameter_name].data.copy_(modified_parameter)

# Specify the number of distinct parameters to modify and the modification function
num_params = int(input("How many parameters would you like to modify? "))  # Update the value here
new_val = float(input("New value: "))
modification_func = lambda parameter: parameter.clone().fill_(new_val).to(device)  # Example modification: set the parameter values to a new value

# Modify a specific number of distinct parameters of highest importance
modify_parameters(model, num_params, modification_func)

# Specify the input text with the masked token
input_text = "I want to eat a [MASK]."
input_tokens = tokenizer.tokenize(input_text)
masked_index = input_tokens.index("[MASK]")

# Encode the input text
input_ids = tokenizer.encode(input_tokens, return_tensors='pt').to(device)

# Get the predictions for the masked token
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

    # Ensure the predictions tensor has the expected shape
    assert predictions.shape[1] >= masked_index + 1, "Masked index is out of bounds."

    # Get the predicted token for the masked position
    predicted_token_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.decode(predicted_token_index)

# Replace the masked token with the predicted token
completed_text = " ".join(input_tokens[:masked_index] + [predicted_token] + input_tokens[masked_index+1:])

# Print the completed text
print("Completed text:")
print(completed_text)

wd.plot_weight_distribution("bert=base-uncased", model)
