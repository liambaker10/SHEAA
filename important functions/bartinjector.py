import torch
import torch.nn as nn
import weightdistribution as wd
from transformers import BartTokenizer, BartForConditionalGeneration

def BartErrorInjector(num_params, new_val, input_text):
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

        selected_params = [param[0] for param in sorted_params]
        modified_params = set()

        for name, parameter in model.named_parameters():
            if len(modified_params) < num_params and name in selected_params:
                modified_params.add(name)
                modified_parameter = modification_func(parameter)
                parameter.data.copy_(modified_parameter)
            else:
                parameter.requires_grad_(False)

    # Create the BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Specify the number of distinct parameters to modify and the modification function
    modification_func = lambda parameter: parameter.clone().fill_(new_val)  # Example modification: set the parameter values to a new value

    # Set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move the model to the device

    # Modify a specific number of distinct parameters of highest importance
    modify_parameters(model, num_params, modification_func)

    # Tokenize your input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the summary using the BART model
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    # Print the summary
    print("Summary:", summary)

# BartErrorInjector(100, -3, "Kieran Seven")
