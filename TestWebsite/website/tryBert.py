from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import pipeline

tryBert = Blueprint("tryBert", __name__)


@tryBert.route("/tryBert")
def tryModel():
    return render_template("tryBert.html")


@tryBert.route("/tryBert/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    number_sentences = request.form["num_sentences"]
    sentence_value = int(number_sentences)
    input = msg
    return get_Chat_response(input, num_sentences=sentence_value)


def get_Chat_response(text, num_sentences=5):
    unmasker = pipeline("fill-mask", model="bert-base-uncased", top_k=num_sentences)
    outputs = unmasker(text)

    responses = [output["sequence"] for output in outputs]
    formatted_response = "<br>".join(responses)
    return formatted_response


# def get_Chat_response(text):
#     unmasker = pipeline("fill-mask", model="bert-base-uncased")
#     outputs = unmasker(text)

#     responses = [output["sequence"] for output in outputs]
#     formatted_response = "<br>".join(responses)
#     return formatted_response

# def truncate_sequence(sequence, max_length):
#     # Truncate the sequence to the specified maximum length
#     if len(sequence) > max_length:
#         sequence = sequence[:max_length]
#     return sequence


# def get_Chat_response(text, num_sentences=5):
#     unmasker = pipeline("fill-mask", model="bert-base-uncased", top_k=num_sentences)
#     outputs = unmasker(text)

#     responses = [
#         truncate_sequence(output["sequence"], max_length) for output in outputs
#     ]
#     formatted_response = "<br>".join(responses)
#     return formatted_response
