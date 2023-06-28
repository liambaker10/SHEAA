import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForMaskedLM

def RobertaErrorInjector(num_params, new_val, input_text):
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
        modified_params = set()

        for name, parameter in model.named_parameters():
            if len(modified_params) < num_params and name in selected_params:
                modified_params.add(name)
                modified_parameter = modification_func(parameter)
                parameter.data.copy_(modified_parameter)
            else:
                parameter.requires_grad_(False)

    # Load the RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load the RoBERTa model
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    # Specify the number of distinct parameters to modify and the modification function
    num_params = num_params  # Number of parameters to modify
    modification_func = lambda parameter: parameter.clone().fill_(new_val)  # Modification function

    # Modify a specific number of distinct parameters of highest importance
    modify_parameters(model, num_params, modification_func)

    # Set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Tokenize the input text
    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Find the position of the masked token
    masked_index = token_ids.index(tokenizer.mask_token_id)

    # Convert token IDs to tensors
    input_tensor = torch.tensor([token_ids]).to(device)

    # Pass the input tensor through the RoBERTa model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the predictions for the masked token
    predictions = outputs.logits[0, masked_index]
    predicted_token_ids = torch.argmax(predictions, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_ids.tolist())

    # Replace the masked token in the input text with the predicted token
    output_text = input_text.replace("<mask>", predicted_token)

    # Print the output text
    print("Output Text:", output_text)

# RobertaErrorInjector(10, -2, "Liam Baker is a <mask>.")
