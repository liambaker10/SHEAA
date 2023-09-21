<<<<<<< HEAD
from transformers import (
    AutoModel,
    AutoConfig,
    BartTokenizer,
    BartForConditionalGeneration,
)
import matplotlib.pyplot as plt
import torch
from torch import nn


def BartErrorInjector(input_text, num_params, new_val):
    def plot_weight_distribution(model):
        # Create an empty list to store the parameter values
        parameter_values = []
=======
import matplotlib.pyplot as plt

def plot_weight_distribution(model, pull):
>>>>>>> e4ec5f24406d1b5397300d90b1a383ed5a86454a

        # Create an empty tensor to concatenate the parameters
        concatenated_tensor = None

        # Iterate through the named parameters of the model
        for name, param in model.named_parameters():
            if "weight" in name:
                # Reshape the parameter tensor to a 1D tensor
                param_tensor = param.view(-1)

<<<<<<< HEAD
                # Concatenate the parameter tensor to the existing tensor
                if concatenated_tensor is None:
                    concatenated_tensor = param_tensor
                else:
                    concatenated_tensor = torch.cat((concatenated_tensor, param_tensor))
=======
    # Iterate through the named parameters of the model
    for name, param in model.named_parameters():
        if pull in name:
            # Reshape the parameter tensor to a 1D tensor
            param_tensor = param.view(-1)
>>>>>>> e4ec5f24406d1b5397300d90b1a383ed5a86454a

        # Convert the GPU tensor to a CPU tensor (if necessary)
        concatenated_tensor_cpu = concatenated_tensor.cpu()

        # Detach the tensor from the computation graph and convert it to a NumPy array
        parameter_values = concatenated_tensor_cpu.detach().numpy()

        # Plotting the weight distribution
        start = -10.0
        stop = 10.0
        step = 0.05
        bins = [round(start + i * step, 1) for i in range(int((stop - start) / step))]

        plt.hist(parameter_values, bins=bins, edgecolor="black", log=True)

        plt.xlabel("Parameter Values")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution")
        plt.show()

    def get_parameter_importance(model: nn.Module) -> dict:
        parameter_importance = {}
        for name, parameter in model.named_parameters():
            parameter_importance[name] = torch.std(
                parameter
            ).item()  # Calculate importance based on standard deviation
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

        selected_params = [param[0] for param in sorted_params[:num_params]]

        for name, parameter in model.named_parameters():
            if name in selected_params:
                parameter.data = modification_func(
                    parameter.data
                )  # Modify the parameter in-place

        return model

    # Create the BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Specify the number of distinct parameters to modify and the modification function
    modification_func = lambda parameter: parameter.fill_(
        new_val
    )  # Example modification: set the parameter values to a new value

    # Set the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the device

    # Modify a specific number of distinct parameters of highest importance
    modified_model = modify_parameters(model, num_params, modification_func)

    # Tokenize your input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the summary using the BART model
    summary_ids = modified_model.generate(
        input_ids, num_beams=4, max_length=100, early_stopping=True
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    # Print the summary
    print(summary)

    plot_weight_distribution(
        modified_model.cpu()
    )  # Move the model back to the CPU for plotting


<<<<<<< HEAD
BartErrorInjector("When was Villanova established?", 9, -10.0)
=======
>>>>>>> e4ec5f24406d1b5397300d90b1a383ed5a86454a
