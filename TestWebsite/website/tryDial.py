from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

tryDial = Blueprint("tryDial", __name__)


@tryDial.route("/tryDial")
def tryModel():
    return render_template("tryDial.html")


@tryDial.route("/tryDial/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    maxLength = int(request.form["len"])
    temp = float(request.form["temp"])
    return get_Chat_response(input, length=maxLength, temperature=temp)


def get_Chat_response(text, length=50, temperature=1.0):
    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(
            str(text) + tokenizer.eos_token, return_tensors="pt"
        )

        # append the new user input tokens to the chat history
        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if step > 0
            else new_user_input_ids
        )

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
        )

        # pretty print last ouput tokens from bot
        return tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
        )
