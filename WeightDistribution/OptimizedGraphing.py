from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)

enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)
model.to(device)
model.eval()



def compute_parameter_values(model, param_indices, param_queue):
    parameter_values = []
    for index in param_indices:
        param = model.parameters()[index]
        param_values = param.data.view(-1).cpu().numpy()
        parameter_values.extend(param_values)
    param_queue.put(parameter_values)


# Number of processes to utilize (adjust according to your system)
num_processes = mp.cpu_count()

# Number of parameters to sample and plot
num_sampled_parameters = 1000

# Randomly sample parameter indices
all_param_indices = list(range(len(list(model.parameters()))))
sampled_param_indices = random.sample(all_param_indices, num_sampled_parameters)

# Create a queue to store parameter values from different processes
param_queue = mp.Queue()

# Create and start the processes
processes = []
for _ in range(num_processes):
    process = mp.Process(target=compute_parameter_values, args=(model, sampled_param_indices, param_queue))
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
start = -1.0
stop = 1.2  # We use 1.6 to include the stop value (1.5)
step = 0.1
bins = [round(start + i * step, 1) for i in range(int((stop - start) / step))]

plt.hist(parameter_values, bins=bins, edgecolor='black', log=True)
plt.xlabel('Parameter Values')
plt.ylabel('Frequency')
plt.title('Weight Distribution of Sampled Model Parameters')
plt.show()
