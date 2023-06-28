from flask import Blueprint, render_template, Flask

AboutModel = Blueprint("AboutModel", __name__)


@AboutModel.route("/AboutModel")
def AboutModels():
    return render_template("AboutModel.html")


@AboutModel.route("/AboutModel/GPT-2")
def GPT2():
    return render_template("GPT2.html")


@AboutModel.route("/AboutModel/GPT-3.5")
def GPT3():
    return render_template("GPT3.html")


@AboutModel.route("/AboutModel/Bert-base")
def Bert():
    return render_template("Bert.html")


@AboutModel.route("/AboutModel/Bart")
def Bart():
    return render_template("Bart.html")


@AboutModel.route("/AboutModel/dialGPT")
def dialGPT():
    return render_template("dialGPT.html")


@AboutModel.route("/AboutModel/RoBERTa")
def RoBERTa():
    return render_template("RoBERTa.html")
