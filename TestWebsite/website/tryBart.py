from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration

tryBart = Blueprint("tryBart", __name__)


@tryBart.route("/tryBart")
def tryModel():
    return render_template("tryBart.html")


@tryBart.route("/tryBart/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    new_length = request.form["len"]
    new_temp = request.form["temp"]
    max_len = int(new_length)
    temp = float(new_temp)
    input = msg
    return get_Chat_response(input, max_length=max_len, temperature=temp)


def get_Chat_response(text, max_length=50, temperature=1.0):
    # Load the model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Tokenize the input prompt
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate the response
    output = model.generate(input_ids, max_length=100, temperature=temperature)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response
