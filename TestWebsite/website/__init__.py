from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ahsdgfkjaghdfkj"

    from .views import views
    from .AboutModels import AboutModels
    from .tryGPT2 import tryGPT2
    from .pipeline import pipeline

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(AboutModels, url_prefix="/")
    app.register_blueprint(tryGPT2, url_prefix="/")
    app.register_blueprint(pipeline, url_prefix="/")

    return app
