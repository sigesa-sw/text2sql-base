"""
api for the Text2Sql platform
"""

import logging

from typing import List
from flask import Flask  # type: ignore
from nlplatform.base import nlplatformBase  # type: ignore
from api.cache import Cache, MemoryCache  # type: ignore
from flask_sock import Sock  # type: ignore
from websocket import WebSocket  # type: ignore

from api.config import Config
from api.routes.data import create_bp_data
from api.routes.questions import create_bp_questions
from api.routes.sql import create_bp_sql
from api.docs import create_docs_blueprint


class FlaskApp:
    """
    Flask app for the Text2Sql platform
    """

    flask_app = Flask(__name__, static_url_path="")

    def __init__(
        self,
        platform: nlplatformBase,
        cache: Cache = MemoryCache(),
        config: Config = Config(),
    ):
        self.flask_app = Flask(__name__)
        self.config = config or Config()
        self.platform = platform
        self.cache = cache

        self._init_routes()
        self._init_logging()
        self._init_websocket()

    def _init_logging(self):
        """Initialize the logging"""
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR if not self.config.DEBUG else logging.DEBUG)

    def _init_routes(self):
        """Initialize the routes"""
        self.flask_app.register_blueprint(create_docs_blueprint(self))
        self.flask_app.register_blueprint(create_bp_data(self))
        self.flask_app.register_blueprint(create_bp_questions(self))
        self.flask_app.register_blueprint(create_bp_sql(self))

    def _init_websocket(self):
        """Initialize the websocket"""
        self.sock = Sock(self.flask_app)
        self.ws_clients: List[WebSocket] = []

    def run(self):
        """
        Runs the Flask app.
        """
        self.flask_app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            use_reloader=False,
        )
