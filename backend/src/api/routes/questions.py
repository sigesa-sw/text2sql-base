from flask import Blueprint, jsonify, request


def create_bp_questions(app_instance):
    bp = Blueprint("questions", __name__)

    @bp.route("/api/v0/generate_questions", methods=["GET"])
    def generate_questions():
        """
        Generate example questions
        ---
        tags:
          - questions
        responses:
          200:
            description: Questions generated successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [question_list]
                questions:
                  type: array
                  items:
                    type: string
                  description: List of example questions
                header:
                  type: string
                  description: Header text to display with questions
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
        training_data = app_instance.platform.get_training_data()
        # If training data is None or empty
        if training_data is None or len(training_data) == 0:
            return jsonify(
                {
                    "type": "error",
                    "error": "No se han encontrado datos de entrenamiento. "
                    "Por favor añade algunas preguntas de ejemplo.",
                }
            )
        # Get the questions from the training data
        try:
            # Filter training data to only include questions where the question is not null
            questions = (
                training_data[training_data["question"].notnull()]
                .sample(5)["question"]
                .tolist()
            )

            # Temporarily this will just return an empty list
            return jsonify(
                {
                    "type": "question_list",
                    "questions": questions,
                    "header": "Prueba con estas preguntas",
                }
            )
        except ValueError as e:
            return jsonify(
                {"type": "error", "error": f"Error al generar preguntas: {str(e)}"}
            )

    @bp.route("/api/v0/generate_sql", methods=["GET"])
    def generate_sql():
        """
        Generate SQL from a natural language question
        ---
        tags:
          - questions
        parameters:
          - name: question
            in: query
            type: string
            required: true
            description: Natural language question to convert to SQL
        responses:
          200:
            description: SQL query generated successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [sql, text]
                id:
                  type: string
                  description: Cache ID for the generated SQL
                sql:
                  type: string
                  description: Generated SQL query
                text:
                  type: string
                  description: Generated text (if SQL is invalid)
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
        question = request.args.get("question")
        if question is None:
            return jsonify(
                {"type": "error", "error": "No se ha proporcionado una pregunta"}
            )
        id_cache = app_instance.cache.generate_id()
        sql = app_instance.platform.generate_sql(
            question=question, allow_llm_to_see_data=app_instance.config.ALLOW_LLM_TO_SEE_DATA
        )
        app_instance.cache.set(cache_id=id_cache, field="question", value=question)
        app_instance.cache.set(cache_id=id_cache, field="sql", value=sql)
        if app_instance.platform.is_sql_valid(sql=sql):
            return jsonify({"type": "sql", "id": id_cache, "sql": sql})
        else:
            return jsonify({"type": "text", "id": id_cache, "text": sql})

    @bp.route("/api/v0/generate_rewritten_question", methods=["GET"])
    def generate_rewritten_question():
        """
        Generate a rewritten question based on the last question and a new question.
        ---
        tags:
          - questions
        parameters:
          - name: last_question
            in: query
            type: string
            required: true
            description: Previous question that was asked
          - name: new_question 
            in: query
            type: string
            required: true
            description: New question to combine with the last question
        responses:
          200:
            description: Question rewritten successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [rewritten_question]
                question:
                  type: string
                  description: The rewritten question
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
        last_question = request.args.get("last_question")
        new_question = request.args.get("new_question")
        rewritten_question = app_instance.platform.generate_rewritten_question(
            last_question=last_question, new_question=new_question
        )
        return jsonify({"type": "rewritten_question", "question": rewritten_question})

    return bp
