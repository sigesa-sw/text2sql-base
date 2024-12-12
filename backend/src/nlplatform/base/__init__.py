import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any

import pandas as pd  # type: ignore
import sqlparse  # type: ignore

from ..exceptions import ValidationError
from ..types import TrainingPlan, TrainingPlanItem, TableMetadata
from ..connectors.postgres import connect_to_postgres, run_sql_postgres
from ..connectors.oracle import connect_to_oracle, run_sql_oracle


class nlplatformBase(ABC):
    """
    nlplatformBase is the base class for the Text2SQL platform.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)

    def log(self, message: str, title: str = "Info"):
        """Logs a message to the console."""
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        """
        Returns a string that instructs the LLM to respond in the specified language.
        """
        if self.language is None:
            return ""
        return f"Respond in the {self.language} language."

    def generate_sql(self, question: str, allow_llm_to_see_data=False) -> str:
        """
        Example:
        ```python
        platform.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:
        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]
        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]
        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]
        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]
        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data
            (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question)
        ddl_list = self.get_related_ddl(question)
        doc_list = self.get_related_documentation(question)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt)
        self.log(title="LLM Response", message=llm_response)
        if "intermediate_sql" in llm_response:
            if not allow_llm_to_see_data:
                return (
                    "The LLM is not allowed to see the data in your database. "
                    "Your question requires database introspection to generate the necessary SQL. "
                    "Please set allow_llm_to_see_data=True to enable this."
                )
            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)
                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(self.db_conn, intermediate_sql)
                    if df is not None:
                        prompt = self.get_sql_prompt(
                            initial_prompt=initial_prompt,
                            question=question,
                            question_sql_list=question_sql_list,
                            ddl_list=ddl_list,
                            doc_list=doc_list
                            + [
                                f"Pandas DataFrame with the results of the intermediate SQL query {
                                    intermediate_sql
                                }: \n"
                                + df.to_markdown()
                            ],
                        )
                    else:
                        return "El modelo ha devuelto una tabla vacía."
                    self.log(title="Prompt SQL definitivo", message=prompt)
                    llm_response = self.submit_prompt(prompt)
                    self.log(title="Respuesta Modelo", message=llm_response)
                except (ImportError, NameError) as e:
                    return f"Error running intermediate SQL: {e}"
        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response.
        This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If is not markdown formatted, extract by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block,
        # with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    def extract_table_metadata(self, ddl: str) -> TableMetadata:
        """
        Example:
        ```python
        vn.extract_table_metadata("CREATE TABLE hive.bi_ads.customers
        (id INT, name TEXT, sales DECIMAL)")
        ```

        Extracts the table metadata from a DDL statement. This is useful in case the DDL statement
        contains other information besides the table metadata.
        Override this function if your DDL statements need custom extraction logic.

        Args:
            ddl (str): The DDL statement.

        Returns:
            TableMetadata: The extracted table metadata.
        """
        pattern_with_catalog_schema = re.compile(
            r"CREATE TABLE\s+(\w+)\.(\w+)\.(\w+)\s*\(", re.IGNORECASE
        )
        pattern_with_schema = re.compile(
            r"CREATE TABLE\s+(\w+)\.(\w+)\s*\(", re.IGNORECASE
        )
        pattern_with_table = re.compile(r"CREATE TABLE\s+(\w+)\s*\(", re.IGNORECASE)

        match_with_catalog_schema = pattern_with_catalog_schema.search(ddl)
        match_with_schema = pattern_with_schema.search(ddl)
        match_with_table = pattern_with_table.search(ddl)

        if match_with_catalog_schema:
            catalog = match_with_catalog_schema.group(1)
            schema = match_with_catalog_schema.group(2)
            table_name = match_with_catalog_schema.group(3)
            return TableMetadata(catalog, schema, table_name)
        elif match_with_schema:
            schema = match_with_schema.group(1)
            table_name = match_with_schema.group(2)
            return TableMetadata(None, schema, table_name)
        elif match_with_table:
            table_name = match_with_table.group(1)
            return TableMetadata(None, None, table_name)
        else:
            return TableMetadata()

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check
        if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement.
        You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """
        parsed = sqlparse.parse(sql)
        for statement in parsed:
            if statement.get_type() == "SELECT":
                return True
        return False

    def generate_rewritten_question(
        self, last_question: str | None, new_question: str
    ) -> str:
        """
        **Example:**
        ```python
        rewritten_question = vn.generate_rewritten_question(
            "Who are the top 5 customers by sales?",
            "Show me their email addresses"
        )
        ```

        Generate a rewritten question by combining the last question and the new question
        if they are related. If the new question is self-contained and not related
        to the last question, return the new question.

        Args:
            last_question (str): The previous question that was asked.
            new_question (str): The new question to be combined with the last question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The combined question if related, otherwise the new question.
        """
        if last_question is None:
            return new_question
        prompt = [
            self.system_message(
                "Your goal is to combine a sequence of questions into a singular "
                "question if they are related. If the second question does not relate "
                "to the first question and is fully self-contained, return the second "
                "question. Return just the new combined question with no additional "
                "explanations. The question should theoretically be answerable with a "
                "single SQL statement."
            ),
            self.user_message(
                f"First question: {last_question}\nSecond question: {new_question}"
            ),
        ]
        return self.submit_prompt(prompt)

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5
    ) -> list:
        """
        **Example:**
        ```python
        vn.generate_followup_questions("What are the top 10 customers by sales?", sql, df)
        ```

        Generate a list of followup questions that you can ask Vanna.AI.

        Args:
            question (str): The question that was asked.
            sql (str): The LLM-generated SQL query.
            df (pd.DataFrame): The results of the SQL query.
            n_questions (int): Number of follow-up questions to generate.

        Returns:
            list: A list of followup questions that you can ask Vanna.AI.
        """
        message_log = [
            self.system_message(
                "You are a helpful data assistant. The user asked the question: "
                f"'{question}'\n\nThe SQL query for this question was: {sql}\n\n"
                "The following is a pandas DataFrame with the results of the query: \n"
                f"{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Generate a list of {n_questions} followup questions that the user might "
                "ask about this data. Respond with a list of questions, one per line. "
                "Do not answer with any explanations -- just the questions. Remember that "
                "there should be an unambiguous SQL query that can be generated from the "
                "question. Prefer questions that are answerable outside of the context of "
                "this conversation. Prefer questions that are slight modifications of the "
                "SQL query that was generated that allow digging deeper into the data. "
                "Each question will be turned into a button that the user can click to "
                "generate a new SQL query so don't use 'example' type questions. Each "
                "question must have a one-to-one correspondence with an instantiated SQL "
                "query." + self._response_language()
            ),
        ]
        llm_response = self.submit_prompt(message_log)
        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self) -> List[str]:
        """
        **Example:**
        ```python
        vn.generate_questions()
        ```

        Generate a list of questions that you can ask Vanna.AI.
        """
        question_sql = self.get_similar_question_sql(question="")
        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: "
                f"'{question}'\n\nThe following is a pandas DataFrame with the "
                f"results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. "
                "Do not respond with any additional explanation beyond the summary."
                + self._response_language()
            ),
        ]
        summary = self.submit_prompt(message_log)
        return summary

    # ----------------- Use Any Embeddings api ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str) -> List[float]:
        """
        This method is used to generate an
        embedding for a given string of data.
        """

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and
            their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """

    @abstractmethod
    def get_related_ddl(self, question: str) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """

    @abstractmethod
    def get_related_documentation(self, question: str) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """

    @abstractmethod
    def add_question_sql(self, question: str, sql: str) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """

    @abstractmethod
    def add_ddl(self, ddl: str, engine: str | None = None) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.
            engine (str): The database engine.
        Returns:
            str: The ID of the training data that was added.
        """

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """

    @abstractmethod
    def get_training_data(self) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.get_training_data()
        ```

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """

    @abstractmethod
    def remove_training_data(self, data_id: str) -> bool:
        """
        Example:
        ```python
        vn.remove_training_data(id="123-ddl")
        ```

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """

    # ----------------- Use Any Language Model api ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> Any:
        """
        This method is used to create a system message.
        """

    @abstractmethod
    def user_message(self, message: str) -> Any:
        """
        This method is used to create a user message.
        """

    @abstractmethod
    def assistant_message(self, message: str) -> Any:
        """
        This method is used to create an assistant message.
        """

    def str_to_approx_token_count(self, string: str) -> int:
        """
        This method is used to convert a string to an approximate token count.
        """
        return int(len(string) / 4)

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        """
        This method is used to add DDL statements to the prompt.
        """
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        """
        This method is used to add documentation to the prompt.
        """
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[dict], max_tokens: int = 14000
    ) -> str:
        """
        This method is used to add SQL statements to the prompt.
        """
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["question"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"
        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt: str | None,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?",
            "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = (
                f"You are a {self.dialect} expert. Please help to generate a SQL query "
                "to answer the question. Your response should ONLY be based on the given "
                "context and follow the response guidelines and format instructions."
            )

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query "
            "without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a "
            "specific string in a particular column, please generate an intermediate SQL "
            "query to find the distinct strings in that column. Prepend the query with a "
            "comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be "
            "generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the "
            "answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, "
            "and free of syntax errors. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))
        message_log.append(self.user_message(question))
        return message_log

    def get_followup_questions_prompt(
        self, question: str, question_sql_list: list, ddl_list: list, doc_list: list
    ) -> list:
        """
        Example:
        ```python
        vn.get_followup_questions_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{
                "question": "What are the top 10 customers by sales?",
                "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"
            }],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )
        ```

        This method is used to generate a prompt for the LLM to generate followup questions.

        Args:
            question (str): The question to generate followup questions for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            list: The prompt for the LLM to generate followup questions.
        """
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about "
                "this data. Respond with a list of questions, one per line. Do not "
                "answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt) -> str:
        """
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess
                what the business question this query is answering.
                Return just the question without any additional explanation.
                Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str) -> str:
        """
        This method is used to generate a question from a SQL query.
        """
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the "
                    "business question this query is answering. Return just the question "
                    "without any additional explanation. Do not reference the table name "
                    "in the question."
                ),
                self.user_message(sql),
            ]
        )
        return response

    def connect_to_db(self, database: str, db_config: dict):
        if database == "postgres":
            self.db_conn = connect_to_postgres(**db_config)
            self.run_sql_is_set = True
            self.dialect = "postgres"
            self.run_sql = run_sql_postgres
        elif database == "oracle":
            self.db_conn = connect_to_oracle(**db_config)
            self.run_sql_is_set = True
            self.dialect = "oracle"
            self.run_sql = run_sql_oracle
        else:
            raise ValueError(f"Database {database} not supported")

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        allow_llm_to_see_data: bool = False,
    ) -> Union[Tuple[Union[str, None], Union[pd.DataFrame, None]], None]:
        """
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train on the question and SQL query.

        Returns:
            Tuple[str, pd.DataFrame]: The SQL query,
            the results of the SQL query.
        """

        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(
                question=question, allow_llm_to_see_data=allow_llm_to_see_data
            )
        except Exception as e:
            print(e)
            return None, None

        if print_results:
            try:
                display = __import__("IPython.display", fromlist=["display"]).display
                Code = __import__("IPython.display", fromlist=["Code"]).Code
                display(Code(sql))
            except (ImportError, NameError) as e:
                print(sql)

        if not self.run_sql_is_set:
            print("If you want to run the SQL query, connect to a database first.")

            if print_results:
                return None
            else:
                return sql, None
        try:
            df = self.run_sql(self.db_conn, sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromlist=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)
            if df is not None and len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            else:
                return sql, df
        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None
        return sql, df

    def train(
        self,
        question: str | None = None,
        sql: str | None = None,
        ddl: str | None = None,
        engine: str | None = None,
        documentation: str | None = None,
        plan: TrainingPlan | None = None,
    ) -> str:
        """
        **Example:**
        ```python
        platform.train()
        ```

        Train Vanna.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will
        attempt to train on the metadata of that database.

        If you call it with the sql argument, it's equivalent to
        [`vn.add_question_sql()`][vanna.base.base.VannaBase.add_question_sql].

        If you call it with the ddl argument, it's equivalent to
        [`vn.add_ddl()`][vanna.base.base.VannaBase.add_ddl].

        If you call it with the documentation argument, it's equivalent to
        [`vn.add_documentation()`][vanna.base.base.VannaBase.add_documentation].

        Additionally, you can pass a [`TrainingPlan`][vanna.types.TrainingPlan] object.
        Get a training plan with
        [`vn.get_training_plan_generic()`][vanna.base.base.VannaBase.get_training_plan_generic].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            engine (str): The database engine.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        Returns:
            str: The training pl
        """

        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl=ddl, engine=engine)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(ddl=item.item_value, engine=engine)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)
        return "Training plan applied."

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql(
                self.db_conn, "SELECT * FROM INFORMATION_SCHEMA.DATABASES"
            )
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql(self.db_conn, "SHOW DATABASES")
            except Exception as e:
                print(e)
                return []
        if df_databases is None:
            return []
        else:
            return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(
            self.db_conn, f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES"
        )
        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an
        information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS
        into groups of table/column descriptions that can be used
        to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the
        # df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column, schema_column, table_column]
        candidates = ["column_name", "data_type", "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" '
                        f'and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and '
                        f'{schema_column} == "{schema}" and '
                        f'{table_column} == "{table}"'
                    )
                    doc = (
                        f"The following columns are in the {table} table "
                        f"in the {database} database:\n\n"
                    )
                    doc += df_columns_filtered_to_table[columns].to_markdown()
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )
        return plan
