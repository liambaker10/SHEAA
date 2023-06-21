# cd into the correct folder
%cd D:\Documents\VSCODE\Python\gpt-2-Pytorch-master\gpt-2-Pytorch-master

from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)

enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)
model.to(device)
model.eval()

# def count_parameters(models):
#     return sum(p.numel() for p in models.parameters() if p.requires_grad)

# count_parameters(model)


def calculate_parameter_values(model, param_queue):
    parameter_values = []
    for _, param in model.named_parameters():
        param_values = param.data.view(-1).cpu().numpy()
        parameter_values.extend(param_values)
    param_queue.put(parameter_values)

model = MyModel()

# Number of processes to utilize (adjust according to your system)
num_processes = mp.cpu_count()

# Create a queue to store parameter values from different processes
param_queue = mp.Queue()

# Create and start the processes
processes = []
for _ in range(num_processes):
    process = mp.Process(target=calculate_parameter_values, args=(model, param_queue))
    process.start()
    processes.append(process)

# Collect the parameter values from the queue
parameter_values = []
for _ in range(num_processes):
    parameter_values.extend(param_queue.get())

# Join the processes
for process in processes:
    process.join()

# Plotting the weight distribution
plt.hist(parameter_values, bins=50)
plt.xlabel('Parameter Values')
plt.ylabel('Frequency')
plt.title('Weight Distribution of Model Parameters')
plt.show()
