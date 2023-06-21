import transformers
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import GPT2Model, BertModel



def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in the given pre-trained language model.

    Args:
        model (nn.Module): The pre-trained language model.

    Returns:
        int: Total number of parameters in the model.
    """
    total_params = 0

    if isinstance(model, dict):
        for param in model.values():
            total_params += param.numel()
    else:
        for param in model.parameters():
            total_params += param.numel()

    return total_params

def get_parameter_value_by_index(model, parameter_index):
    """
    Get the value assigned to a parameter based on its index in the model's parameter list.

    Args:
        model: The model instance.
        parameter_index (int): The index of the parameter.

    Returns:
        The value assigned to the parameter.
    """
    parameter_value = None

    if isinstance(model, dict):
        raise ValueError("The model should not be a dictionary when retrieving parameter by index.")
    else:
        model_parameters = list(model.parameters())
        if parameter_index < len(model_parameters):
            parameter_value = model_parameters[parameter_index].data

    return parameter_value


gpt2_model = GPT2Model.from_pretrained('gpt2')
bert_model = BertModel.from_pretrained('bert-base-uncased')
parameter_index_gpt2 = 2
parameter_index_bert = 2


total_params_gpt2 = count_parameters(gpt2_model)
value_gpt2 = get_parameter_value_by_index(gpt2_model, parameter_index_gpt2)

total_params_bert_base = count_parameters(bert_model)
value_bert = get_parameter_value_by_index(bert_model, parameter_index_bert)
print("Total number of parameters in gpt2:", total_params_gpt2)
print("Value of parameter at index {} in gpt2 is:".format(parameter_index_gpt2))
print(value_gpt2)
print("Total number of parameters in bert-base:", total_params_bert_base)
print("Value of parameter at index {} in bert is:".format(parameter_index_gpt2))
print(value_bert)
