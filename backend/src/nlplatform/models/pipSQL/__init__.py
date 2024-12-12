"""
PipSQL is a Text2SQL platform that uses the PipableAI/pip-sql-1.3b
model to generate SQL queries from natural language questions.

This class inherits from nlplatformBase and provides methods for:
- Initializing the model and tokenizer
- Converting messages between system/user/assistant formats
- Extracting SQL queries from generated text
- Generating SQL queries from natural language questions

Args:
    config (dict, optional): Configuration dictionary. Defaults to None.
        Currently no config options are supported.

Attributes:
    schema (str): The database schema definition used for SQL generation
    model: The loaded pip-sql-1.3b model for text-to-SQL generation
    tokenizer: The tokenizer for the pip-sql-1.3b model
"""

import re
from typing import Any
from nlplatform.base import nlplatformBase
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


class PipSQL(nlplatformBase):
    """
    PipSQL is a Text2SQL platform that uses the PipableAI/pip-sql-1.3b
    model to generate SQL queries from natural language questions.
    """

    def __init__(self, config=None):
        super().__init__(config)
        if config:
            self.schema = config.get("schema", None)
        else:
            self.schema = """
            DATABASE costes;

        CREATE TABLE costes (
            HOSPITAL VARCHAR(100),
            AÑO INTEGER,
            MES INTEGER,
            SERVICIO VARCHAR(100),
            FINANCIADOR VARCHAR(100),
            ALTAS INTEGER,
            COSTE_UNIDAD FLOAT,
            COSTES FLOAT,
            TARIFA_UNIDAD FLOAT,
            INGRESOS FLOAT,
            RENTABILIDAD FLOAT
        );
        """
        self.model = AutoModelForCausalLM.from_pretrained("PipableAI/pip-sql-1.3b")
        self.tokenizer = AutoTokenizer.from_pretrained("PipableAI/pip-sql-1.3b")

    def system_message(self, message: str) -> Any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> Any:
        return {"role": "assistant", "content": message}

    def extract_sql_query(self, text):
        """
        Extracts the SQL query from the generated text.
        """
        # Regular expression to find 'select' (ignoring case)
        # and capture until ';', '```', or end of string
        pattern = re.compile(r"select.*?(?:;|```|$)", re.IGNORECASE | re.DOTALL)

        match = pattern.search(text)
        if match:
            # Remove three backticks from the matched string if they exist
            return match.group(0).replace("```", "")
        else:
            return text

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Generates an SQL query from a natural language question.
        """
        sql = super().generate_sql(question, allow_llm_to_see_data, **kwargs)
        sql = sql.replace("\\_", "_")
        sql = sql.replace("\\", "")
        return self.extract_sql_query(sql)

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        """
        Submits a prompt to the model and returns the generated SQL query.
        """
        prompt = f"""<schema>{self.schema}</schema>
            <question>{prompt}</question>
            <sql>
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = (
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            .split("<sql>")[1]
            .split("</sql>")[0]
        )
        self.log(response)
        return response
