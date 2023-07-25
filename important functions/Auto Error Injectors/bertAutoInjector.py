import torch
import torch.nn as nn

# import weightdistribution as wd
from transformers import BertForMaskedLM, BertTokenizer


def BertErrorInjection(num_params, new_val, input_text):
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def get_parameter_importance(model: nn.Module) -> dict:
        parameter_importance = {}
        for name, parameter in model.named_parameters():
            parameter_importance[name] = torch.std(
                parameter
            ).item() 
        return parameter_importance

    def modify_parameters(
        model: nn.Module, num_params: int, modification_func: callable
    ):
        parameter_importance = get_parameter_importance(model)
        sorted_params = sorted(
            parameter_importance.items(), key=lambda x: x[1], reverse=True
        )
        total_params = len(sorted_params)

        if num_params > total_params:
            num_params = total_params

        selected_params = [param[0] for param in sorted_params][:num_params]

        for parameter_name in selected_params:
            parameter = dict(model.named_parameters())[parameter_name]
            modified_parameter = modification_func(parameter)
            dict(model.named_parameters())[parameter_name].data.copy_(
                modified_parameter
            )

    modification_func = (
        lambda parameter: parameter.clone().fill_(new_val).to(device)
    ) 

    modify_parameters(model, num_params, modification_func)

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


    print("Generated word:")
    print(predicted_token)


# Ex: BertErrorInjection(10,-2,"I want to eat a [MASK].")
