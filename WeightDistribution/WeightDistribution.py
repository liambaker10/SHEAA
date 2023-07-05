# cd into the correct folder
# %cd D:\Documents\VSCODE\Python\gpt-2-Pytorch-master\gpt-2-Pytorch-master

# from GPT2.model import GPT2LMHeadModel
# from GPT2.utils import load_weight
# from GPT2.config import GPT2Config
# from GPT2.sample import sample_sequence
# from GPT2.encoder import get_encoder
from transformers import AutoModel, AutoConfig
import matplotlib.pyplot as plt
import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)

# enc = get_encoder()
# config = GPT2Config()
# model = GPT2LMHeadModel(config)
# model = load_weight(model, state_dict)
# model.to(device)
# model.eval()
model_name = "facebook/bart-large-cnn"

config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)

# Collecting parameter values
parameter_values = []
for _, param in model.named_parameters():
    param_values = param.data.view(-1).cpu().numpy()
    parameter_values.extend(param_values)

# Plotting the weight distribution
start = -0.9
stop = 1.3  # We use 1.6 to include the stop value (1.5)
step = 0.1
bins = [round(start + i * step, 1) for i in range(int((stop - start) / step))]

plt.hist(parameter_values, bins=bins, edgecolor="black", log=True)
plt.xlabel("Parameter Values")
plt.ylabel("Frequency")
plt.title("Weight Distribution of Model Parameters")
plt.show()
