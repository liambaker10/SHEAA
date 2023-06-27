from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import pipeline

tryRoBERTa = Blueprint("tryRoBERTa", __name__)


@tryRoBERTa.route("/tryRoBERTa")
def tryModel():
    return render_template("tryRoBERTa.html")


@tryRoBERTa.route("/tryRoBERTa/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    number_sentences = request.form["num_sentences"]
    sentence_value = int(number_sentences)
    input = msg
    return get_Chat_response(input, num_sentences=sentence_value)


def get_Chat_response(text, num_sentences=5):
    unmasker = pipeline(
        "fill-mask",
        model="roberta-large",
        tokenizer="roberta-large",
        top_k=num_sentences,
    )
    input_text = text.replace("[MASK]", "<mask>")
    outputs = unmasker(input_text)
    top_k_outputs = outputs[: int(num_sentences)]
    responses = [
        output["sequence"].replace("<s>", "").replace("</s>", "").strip()
        for output in top_k_outputs
    ]
    formatted_response = "<br>".join(responses)
    return formatted_response
    # unmasker = pipeline("fill-mask", model="roberta-large", top_k=num_sentences)
    # outputs = unmasker(text)

    # responses = [output["sequence"] for output in outputs]
    # formatted_response = "<br>".join(responses)
    # return formatted_response
