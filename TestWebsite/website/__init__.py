from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ahsdgfkjaghdfkj"

    from .views import views
    from .AboutModel import AboutModel
    from .tryGPT2 import tryGPT2
    from .tryBert import tryBert
    from .tryRoBERTa import tryRoBERTa
    from .tryDial import tryDial
    from .tryBertError import tryBertError
    from .tryGPT2error import tryGPT2error

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(AboutModel, url_prefix="/")
    app.register_blueprint(tryGPT2, url_prefix="/")
    app.register_blueprint(tryBert, url_prefix="/")
    app.register_blueprint(tryRoBERTa, url_prefix="/")
    app.register_blueprint(tryDial, url_prefix="/")
    app.register_blueprint(tryBertError, url_prefix="/")
    app.register_blueprint(tryGPT2error, url_prefix="/")

    return app
