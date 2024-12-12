from flask import Blueprint, jsonify, request
from api.decorators import create_cache_decorator


def create_bp_data(app_instance):
    bp = Blueprint("data", __name__)
    requires_cache = create_cache_decorator(app_instance)

    @bp.route("/api/v0/get_training_data", methods=["GET"])
    @requires_cache(required_fields=["training_data"])
    def get_training_data():
        """
        Get training data
        ---
        tags:
          - data
        responses:
          200:
            description: Training data retrieved successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [df]
                id:
                  type: string
                  description: Cache ID for the training data
                data:
                  type: string
                  description: JSON string containing training data records
          400:
            description: Error response
            schema:
              properties:
                type:
                  type: string
                  enum: [error]
                error:
                  type: string
                  description: Error message
        """
        df = app_instance.platform.get_training_data()
        return jsonify(
            {
                "type": "df",
                "id": "training_data",
                "data": df.head(10).to_json(orient="records", date_format="iso"),
            }
        )

    @bp.route("/api/v0/remove_training_data", methods=["GET"])
    def remove_training_data():
        """
        Remove training data entry
        ---
        tags:
          - data
        parameters:
          - name: id
            in: body
            required: true
            schema:
              type: object
              properties:
                id:
                  type: string
                  description: ID of training data entry to remove
        responses:
          200:
            description: Training data removed successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [success]
          400:
            description: Error response
            schema:
              properties:
                type:
                  type: string
                  enum: [error]
                error:
                  type: string
                  description: Error message
        """
        cache_id = request.json.get("id")
        if cache_id is None:
            return jsonify({"type": "error", "error": "No id provided"})
        if app_instance.remove_training_data(cache_id=cache_id):
            return jsonify({"type": "success"})
        else:
            return jsonify(
                {
                    "type": "error",
                    "error": "No se ha podido eliminar el dato de entrenamiento",
                }
            )

    @bp.route("/api/v0/train", methods=["POST"])
    def add_training_data():
        """
        Add new training data
        ---
        tags:
          - data
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              properties:
                question:
                  type: string
                  description: Natural language question
                sql:
                  type: string 
                  description: SQL query that answers the question
                ddl:
                  type: string
                  description: DDL statements defining the schema
                documentation:
                  type: string
                  description: Documentation about the schema and data
        responses:
          200:
            description: Training data added successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [success]
                id:
                  type: string
                  description: ID of the new training data entry
          400:
            description: Error response
            schema:
              properties:
                type:
                  type: string
                  enum: [error]
                error:
                  type: string
                  description: Error message
        """
        data = request.get_json()
        question = data.get("question")
        sql = data.get("sql")
        ddl = data.get("ddl")
        documentation = data.get("documentation")
        try:
            new_id = app_instance.train(
                question=question, sql=sql, ddl=ddl, documentation=documentation
            )
            return jsonify({"type": "success", "id": new_id})
        except Exception as e:
            return jsonify({"type": "error", "error": str(e)})

    return bp
