from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ahsdgfkjaghdfkj"

    from .views import views
    from .about import about
    from .AboutModel import AboutModel
    from .tryGPT2 import tryGPT2
    from .tryBert import tryBert
    from .tryRoBERTa import tryRoBERTa
    from .tryDial import tryDial
    from .tryBertError import tryBertError
    from .tryGPT2error import tryGPT2error
    from .tryBart import tryBart
    from .tryBartError import tryBartError
    from .tryRoBERTaError import tryRoBERTaError
    from .tryBartSingle import tryBartSingle
    from .tryBertSingle import tryBertSingle
    from .tryRobertaSingle import tryRobertaSingle
    from .tryGPT2Single import tryGPT2Single

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(about, url_prefix="/")
    app.register_blueprint(AboutModel, url_prefix="/")
    app.register_blueprint(tryGPT2, url_prefix="/")
    app.register_blueprint(tryBert, url_prefix="/")
    app.register_blueprint(tryRoBERTa, url_prefix="/")
    app.register_blueprint(tryDial, url_prefix="/")
    app.register_blueprint(tryBertError, url_prefix="/")
    app.register_blueprint(tryGPT2error, url_prefix="/")
    app.register_blueprint(tryBart, url_prefix="/")
    app.register_blueprint(tryBartError, url_prefix="/")
    app.register_blueprint(tryRoBERTaError, url_prefix="/")
    app.register_blueprint(tryBartSingle, url_prefix="/")
    app.register_blueprint(tryBertSingle, url_prefix="/")
    app.register_blueprint(tryRobertaSingle, url_prefix="/")
    app.register_blueprint(tryGPT2Single, url_prefix="/")

    return app
