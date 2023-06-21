from flask import Blueprint, render_template

AboutModels = Blueprint("AboutModels", __name__)


@AboutModels.route("/AboutModels")
def AboutModel():
    return render_template("AboutModels.html")


@AboutModels.route("/AboutModels/GPT-2")
def GPT2():
    return render_template("GPT2.html")


@AboutModels.route("/AboutModels/GPT-3.5")
def GPT3():
    return render_template("GPT3.html")


@AboutModels.route("/AboutModels/Bert-base")
def Bert():
    return render_template("Bert.html")


@AboutModels.route("/AboutModels/Bart")
def Bart():
    return render_template("Bart.html")


@AboutModels.route("/AboutModels/dialGPT")
def dialGPT():
    return render_template("dialGPT.html")
