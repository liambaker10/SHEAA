from transformers import GPT2Model, GPT2Config
import matplotlib.pyplot as plt
import torch


model_name = "gpt2"  # or specify a different GPT-2 variant
config = GPT2Config.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, config=config)


# Create an empty list to store the parameter values
parameter_values = []

# Create an empty tensor to concatenate the parameters
concatenated_tensor = None

# Iterate through the named parameters of the model
for name, param in model.named_parameters():
    if "weight" in name:
        # Reshape the parameter tensor to a 1D tensor
        param_tensor = param.view(-1)

        # Concatenate the parameter tensor to the existing tensor
        if concatenated_tensor is None:
            concatenated_tensor = param_tensor
        else:
            concatenated_tensor = torch.cat((concatenated_tensor, param_tensor))

# Move the concatenated tensor to the GPU
#concatenated_tensor = concatenated_tensor.cuda()

# Convert the GPU tensor to a CPU tensor (if necessary)
concatenated_tensor_cpu = concatenated_tensor.cpu()

# Detach the tensor from the computation graph and convert it to a NumPy array
parameter_values = concatenated_tensor_cpu.detach().numpy()


# Plotting the weight distribution
start = -3.0
stop = 3.0
step = 0.05
bins = [round(start + i * step, 1) for i in range(int((stop - start) / step))]

plt.hist(parameter_values, bins=bins, edgecolor="black", log=True)

plt.xlabel("Parameter Values")
plt.ylabel("Frequency")
plt.title("Weight Parameter Frequency Distribution")
plt.show()
