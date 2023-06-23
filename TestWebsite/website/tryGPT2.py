from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import pipeline


tryGPT2 = Blueprint("tryGPT2", __name__)


@tryGPT2.route("/tryGPT2")
def tryModel():
    return render_template("tryModelGPT2.html")


@tryGPT2.route("/tryGPT2/get", methods=["GET", "POST"])  # type: ignore
def chat():
    msg = request.form["msg"]
    new_length = request.form["len"]
    max_len = int(new_length)
    input = msg
    return get_Chat_response(input, max_len)


def get_Chat_response(text, max_length=50):
    generator = pipeline("text-generation", model="gpt2")
    output = generator(text, max_length=max_length, num_return_sequences=1)
    # formatted = output[21 : len(output) - 3].replace("/", "")
    formatted = output[0]["generated_text"]
    return formatted
