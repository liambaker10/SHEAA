from flask import Blueprint, render_template

pipeline = Blueprint('pipeline', __name__)

@pipeline.route('/QA')
def gpt2():
    return render_template("pipeline.html")