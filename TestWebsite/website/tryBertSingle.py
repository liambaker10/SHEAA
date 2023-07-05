from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import BertForMaskedLM, BertTokenizer
import torch
from torch import nn
import random

tryBertSingle = Blueprint("tryBertSingle", __name__)


@tryBertSingle.route("/tryBertSingle")
def tryModel():
    return render_template("tryBertSingle.html")


@tryBertSingle.route("/tryBertSingle/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    param_index = int(request.form["param_index"])
    bit_num = int(request.form["bit_num"])
    input = msg
    return BertSingleInjector(input, param_index=param_index, num_bits=bit_num)


def BertSingleInjector(input_text, param_index=1, num_bits=1):
    def flip_bits(tensor, num_bits):
        # Convert the tensor to binary
        binary_tensor = tensor.byte()

        # Get the number of bits in each element
        num_bits_per_element = binary_tensor.numel() * 8

        # Generate random bit indices to flip
        bit_indices = random.sample(range(num_bits_per_element), num_bits)

        # Create a copy of the tensor to modify
        modified_tensor = tensor.clone()

        # Flip the selected bits in the modified tensor
        for index in bit_indices:
            element_index = index // 8  # Index of the element in the tensor
            bit_offset = index % 8  # Offset of the bit within the element
            modified_tensor.view(-1)[element_index] = modified_tensor.view(-1)[
                element_index
            ].to(torch.long) ^ (1 << bit_offset)

        return modified_tensor

    def modify_parameter(model: nn.Module, param_index: int, num_bits: int):
        parameter = list(model.parameters())[param_index]
        modified_parameter = flip_bits(parameter.data, num_bits)
        parameter.data.copy_(modified_parameter)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    modify_parameter(model, param_index, num_bits)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the summary using the BART model
    summary_ids = model.generate(
        input_ids, num_beams=4, max_length=60, early_stopping=True
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary
