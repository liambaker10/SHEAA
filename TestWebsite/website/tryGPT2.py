from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import pipeline


tryGPT2 = Blueprint("tryGPT2", __name__)


@tryGPT2.route("/tryGPT2")
def tryModel():
    return render_template("tryModels.html")


@tryGPT2.route("/tryGPT2/get", methods=["GET", "POST"])  # type: ignore
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    generator = pipeline("text-generation", model="gpt2")
    output = str(generator(text, max_length=30, num_return_sequences=1))
    formatted = output[21 : len(output) - 3].replace("/", "")
    return formatted.replace("\n", " ").replace("\n", " ")
