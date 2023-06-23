import transformers
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import GPT2Model, BertModel
from transformers.modeling_roberta import RobertaModel




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

def get_parameter_value_by_index_modified(model, parameter_index):
    """
    Get the value assigned to a parameter based on its index in the model's parameter list.

    Args:
        model: The model instance.
        parameter_index (int): The index of the parameter.

    Returns:
        The value assigned to the parameter.
    """
    if isinstance(model, dict):
        raise ValueError("The model should not be a dictionary when retrieving a parameter by index.")
    else:
        model_parameters = list(model.parameters())
        if parameter_index < len(model_parameters):
            parameter_tensor = model_parameters[parameter_index]
            parameter_tensor_shape = parameter_tensor.shape
            parameter_tensor_random = torch.randn(parameter_tensor_shape)
            model_parameters[parameter_index] = parameter_tensor_random
            return model_parameters[parameter_index].data

    return None


def get_parameter_values(model):
    """
    Get the values of all parameters in the model.

    Args:
        model: The model instance.

    Returns:
        A list of tensors, where each tensor contains the values of parameters at a specific index position.
    """
    parameter_values = []

    if isinstance(model, dict):
        raise ValueError("The model should not be a dictionary when retrieving parameter values.")
    else:
        model_parameters = list(model.parameters())
        num_indices = len(model_parameters[0])

        for i in range(num_indices):
            values = []
            for param in model_parameters:
                if param.size(0) > i:
                    values.append(param[i])
            parameter_values.append(values)

    return parameter_values






gpt2_model = GPT2Model.from_pretrained('gpt2')
bert_model = BertModel.from_pretrained('bert-base-uncased')
RoBERTa_model = RobertaModel.from_pretrained('roberta-base')


parameter_index_gpt2 = 5
parameter_index_bert = 5
parameter_index_roberta = 5



total_params_gpt2 = count_parameters(gpt2_model)
value_gpt2 = get_parameter_value_by_index(gpt2_model, parameter_index_gpt2)
# all_param_values_gpt2 = get_parameter_values(gpt2_model)

total_params_bert_base = count_parameters(bert_model)
value_bert = get_parameter_value_by_index(bert_model, parameter_index_bert)
# all_param_values_bert = get_parameter_values(bert_model)

total_params_roberta = count_parameters(RoBERTa_model)
value_roberta = get_parameter_value_by_index(RoBERTa_model, parameter_index_roberta)
# all_param_values_roberta = get_parameter_values(RoBERTa_model)


print("Total number of parameters in gpt2:", total_params_gpt2)
print("Value of parameter at index {} in gpt2 is:".format(parameter_index_gpt2))
print(value_gpt2)
# print("Values of all parameters in gpt2 are:")
# for index, tensor_list in enumerate(all_param_values_gpt2):
#     print(f"Parameter values at index {index}:")
#     for tensor in tensor_list:
#         print(tensor)
#         print("------")
print("Total number of parameters in bert-base:", total_params_bert_base)
print("Value of parameter at index {} in bert is:".format(parameter_index_gpt2))
print(value_bert)
# print("Values of all parameters in bert-base are:")
# for index, tensor_list in enumerate(all_param_values_bert):
#     print(f"Parameter values at index {index}:")
#     for tensor in tensor_list:
#         print(tensor)
#         print("------")
print("Total number of parameters in RoBERTa-base:", total_params_roberta)
print("Value of parameter at index {} in RoBERTa-base is:".format(parameter_index_roberta))
print(value_roberta)
# print("Values of all parameters in roBERTa-base are:")
# for index, tensor_list in enumerate(all_param_values_roberta):
#     print(f"Parameter values at index {index}:")
#     for tensor in tensor_list:
#         print(tensor)
#         print("------")
