# cd into the correct folder

from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.multiprocessing as mp
import torch.cuda 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)

enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)
model.to(device)
model.eval()



# Create an empty list to store the parameter values
parameter_values = []

# Create an empty tensor to concatenate the parameters
concatenated_tensor = None

# Iterate through the named parameters of the model
for name, param in model.named_parameters():
    if 'weight' in name:
    # Reshape the parameter tensor to a 1D tensor
        param_tensor = param.view(-1)

    # Concatenate the parameter tensor to the existing tensor
        if concatenated_tensor is None:
            concatenated_tensor = param_tensor
        else:
         concatenated_tensor = torch.cat((concatenated_tensor, param_tensor))

# Move the concatenated tensor to the GPU
concatenated_tensor = concatenated_tensor.cuda()

# Convert the GPU tensor to a CPU tensor (if necessary)
concatenated_tensor_cpu = concatenated_tensor.cpu()

# Detach the tensor from the computation graph and convert it to a NumPy array
parameter_values = concatenated_tensor_cpu.detach().numpy()


# Plotting the weight distribution
start = -3.0
stop = 3.0
step = 0.05
bins = [round(start + i*step , 1) for i in range (int((stop - start)/ step ))]

plt.hist(parameter_values, bins=bins, edgecolor = 'black', log=True)

plt.xlabel('Parameter Values')
plt.ylabel('Frequency')
plt.title('Weight Parameter Frequency Distribution')
plt.show()