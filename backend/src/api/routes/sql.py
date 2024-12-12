from flask import Blueprint, jsonify, request
from api.decorators import create_cache_decorator

def create_bp_sql(app_instance):
    bp = Blueprint("sql", __name__)
    requires_cache = create_cache_decorator(app_instance)

    @bp.route("/api/v0/run_sql", methods=["GET"])
    @requires_cache(required_fields=["sql"])
    def run_sql(cache_id: str, sql: str):
        """
        Run a SQL query
        ---
        tags:
          - sql
        parameters:
          - name: cache_id
            in: path
            type: string
            required: true
            description: Cache ID containing the SQL query
          - name: sql 
            in: path
            type: string
            required: true
            description: SQL query to execute
        responses:
          200:
            description: Query executed successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [df]
                id:
                  type: string
                  description: Cache ID for the results
                df:
                  type: string
                  description: JSON string containing dataframe results
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
        conn = app_instance.db_conn
        try:
            if not app_instance.run_sql_is_set:
                return jsonify(
                    {
                        "type": "error",
                        "error": "La plataforma no soporta la ejecución de consultas SQL",
                    }
                )
            df = app_instance.run_sql(conn=conn, sql=sql)
            print(df)
            if df is None:
                return jsonify(
                    {
                        "type": "error",
                        "error": "No se ha podido ejecutar la consulta SQL",
                    }
                )
            app_instance.cache.set(cache_id=cache_id, field="df", value=df)
            return jsonify(
                {
                    "type": "df",
                    "id": cache_id,
                    "df": df.head(10).to_json(orient="records", date_format="iso")
                }
            )
        except (ValueError, TypeError) as e:
            return jsonify(
                {
                    "type": "error",
                    "error": f"Error al ejecutar la consulta SQL: {str(e)}",
                }
            )
    
    @bp.route("/api/v0/fix_sql", methods=["POST"])
    def fix_sql(cache_id: str, question: str, sql: str):
        """
        Fix SQL query that produced an error
        ---
        tags:
          - sql
        parameters:
          - name: cache_id
            in: path
            type: string
            required: true
            description: Cache ID for the SQL query
          - name: question 
            in: path
            type: string
            required: true
            description: Original natural language question
          - name: sql
            in: path
            type: string
            required: true
            description: SQL query that produced error
          - name: error
            in: body
            schema:
              type: object
              required:
                - error
              properties:
                error:
                  type: string
                  description: Error message from failed SQL query
        responses:
          200:
            description: SQL query fixed successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [sql]
                id:
                  type: string
                  description: Cache ID for the fixed SQL
                text:
                  type: string
                  description: Fixed SQL query
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
        error = request.json.get("error")

        if error is None:
            return jsonify({"type": "error", "error": "No error provided"})

        question = (
            f"I have an error: {error}\n\n"
            + f"Here is the SQL I tried to run: {sql}\n\n"
            + f"This is the question I was trying to answer: {question}\n\n"
            + "Can you rewrite the SQL to fix the error?"
        )

        fixed_sql = app_instance.generate_sql(question=question)

        app_instance.cache.set(cache_id=cache_id, field="sql", value=fixed_sql)

        return jsonify(
            {
                "type": "sql",
                "id": id,
                "text": fixed_sql,
            }
        )

    @bp.route("/api/v0/update_sql", methods=["POST"])
    @requires_cache([])
    def update_sql(cache_id: str):
        """
        Update SQL query in cache
        ---
        tags:
          - sql
        parameters:
          - name: sql
            in: body
            required: true
            schema:
              type: object
              properties:
                sql:
                  type: string
                  description: SQL query to update
        responses:
          200:
            description: SQL query updated successfully
            schema:
              properties:
                type:
                  type: string
                  enum: [sql]
                id:
                  type: string
                  description: Cache ID for the updated SQL
                text:
                  type: string
                  description: Updated SQL query
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
        sql = request.json.get("sql")

        if sql is None:
            return jsonify({"type": "error", "error": "No sql provided"})

        app_instance.cache.set(cache_id=cache_id, field="sql", value=sql)

        return jsonify(
            {
                "type": "sql",
                "id": cache_id,
                "text": sql,
            }
        )

    return bp
