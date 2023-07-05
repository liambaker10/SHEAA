import torch
import os
import sys
import torch
import random
import argparse
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer

# from GPT2.model import (GPT2LMHeadModel)
# from GPT2.utils import load_weight
# from GPT2.config import GPT2Config
# from GPT2.sample import sample_sequence
# from GPT2.encoder import get_encoder


# state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# enc = get_encoder()
# config = GPT2Config()
# model = GPT2LMHeadModel(config)
# model = load_weight(model, state_dict)
# model.to(device)
# model.eval()

# Load BART model and tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# def param_test():
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         output = str(param)[:50] + '...' if len(str(param)) > 50 else str(param)
#     # Print the parameter name, shape, and the truncated output
#         print(f"Parameter Name: {name}")
#         print(f"Parameter Shape: {param.shape}")
#         print(f"Parameter Tensor: {output}")
#         print()
def param_test(model):
    for name, param in model.named_parameters():
        if "bias" or "weight" in name:
            output = str(param)[:50] + "..." if len(str(param)) > 50 else str(param)
            # Print the parameter name, shape, and the truncated output
            print(f"Parameter Name: {name}")
            print(f"Parameter Shape: {param.shape}")
            print(f"Parameter Tensor: {output}")
            print()


# Call param_test() function with the loaded BART model
param_test(model)
