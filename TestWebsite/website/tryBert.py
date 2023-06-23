from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import pipeline

tryBert = Blueprint("tryBert", __name__)


@tryBert.route("/tryBert")
def tryModel():
    return render_template("tryBert.html")


@tryBert.route("/tryBert/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    outputs = unmasker(text)

    responses = [output["sequence"] for output in outputs]
    formatted_response = "<br>".join(responses)
    return formatted_response
