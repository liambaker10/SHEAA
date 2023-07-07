import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def DialoErrorInjector(num_params, new_val, user_input):
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

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    modification_func = lambda parameter: parameter.clone().fill_(new_val)  # Example modification: set the parameter values to 0.0

    # Modify a specific number of distinct parameters of highest importance
    modify_parameters(model, num_params, modification_func)

    # Set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
# Let's chat for 5 lines
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

# Generate a response using the DialoGPT model
    response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id)
    # pretty print last ouput tokens from bot
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("DialoGPT: ", response)

# DialoErrorInjector(1, 1, "How are you?")
