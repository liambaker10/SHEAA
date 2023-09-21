import torch
import os
import sys
import torch
import random
import argparse
import numpy as np

# from GPT2.model import GPT2LMHeadModel
# from GPT2.utils import load_weight
# from GPT2.config import GPT2Config
# from GPT2.sample import sample_sequence
# from GPT2.encoder import get_encoder
<<<<<<< HEAD

# from GPT2.model import GPT2LMHeadModel
# from GPT2.utils import load_weight
# from GPT2.config import GPT2Config
# from GPT2.sample import sample_sequence
# from GPT2.encoder import get_encoder
from transformers import BartForConditionalGeneration, BartTokenizer
=======
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BertLMHeadModel,
    BertTokenizer,
    RobertaForCausalLM,
    RobertaTokenizer,
)

# python -u "D:\Documents\Innovate\SHEAA\ParamTest.py"
>>>>>>> c82cf20dbd4ea16b82a1058069feb8431ad3405b

# state_dict = torch.load(
#     "gpt2-pytorch_model.bin",
#     map_location="cpu" if not torch.cuda.is_available() else None,
# )
<<<<<<< HEAD
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# enc = get_encoder()
# config = GPT2Config()
# model = GPT2LMHeadModel(config)
# model = load_weight(model, state_dict)
# model.to(device)
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
=======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BART model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForCausalLM.from_pretrained(model_name)
# enc = get_encoder()
# config = GPT2Config()
# model = GPT2LMHeadModel(config)
# model = load_weight(model, state_dict)
>>>>>>> c82cf20dbd4ea16b82a1058069feb8431ad3405b
model.to(device)
model.eval()

for name, param in model.named_parameters():
    if "weight" or "bias" in name:
        output = str(param)[:50] + "..." if len(str(param)) > 50 else str(param)
        # Print the parameter name, shape, and the truncated output
        print(f"Parameter Name: {name}")
        print(f"Parameter Shape: {param.shape}")
        print(f"Parameter Tensor: {output}")
        print()
