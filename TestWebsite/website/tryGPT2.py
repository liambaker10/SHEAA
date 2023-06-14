from flask import Blueprint, render_template

tryGPT2 = Blueprint("tryGPT2", __name__)


@tryGPT2.route("/tryGPT2")
def tryPage():
    return render_template("tryModels.html")
