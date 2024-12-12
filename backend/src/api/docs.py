from flask import Blueprint, redirect
from flasgger import Swagger


def create_docs_blueprint(app_instance):
    # Configure Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
                "definition_filter": lambda definition: True,  # Add this
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/",
    }

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Text2SQL API",
            "version": "1.0.0",
            "description": "API documentation for the Text2SQL service",
        },
        "consumes": [
            "application/json",
        ],
        "produces": [
            "application/json",
        ],
        "definitions": {},  # Add empty definitions
        "parameters": {},   # Add empty parameters
    }
    
    app_instance.swagger = Swagger(app_instance.flask_app, 
                                 config=swagger_config,
                                 template=swagger_template)

    bp = Blueprint("documentation", __name__)

    @bp.route("/")
    def index():
        return redirect("/apidocs")

    return bp
