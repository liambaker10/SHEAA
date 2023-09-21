import torch
import torch.nn as nn
import weightdistribution as wd  
from transformers import GPT2LMHeadModel, GPT2Tokenizer
def GPT2ErrorInjector(num_params, new_val, input_text):
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

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    modification_func = lambda parameter: parameter.clone().fill_(new_val)  # Example modification: set the parameter values to 0.0

    modify_parameters(model, num_params, modification_func)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)
    
# GPT2ErrorInjector(1, -1, 'Liam Baker')
