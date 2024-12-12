import os
import time
from typing import Union

import pandas as pd  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore

from nlplatform.exceptions import ImproperlyConfigured, ValidationError


def connect_to_postgres(
    host: str | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    port: int | None = None,
) -> psycopg2.extensions.connection:
    """
    Connect to postgres using the psycopg2 connector.
    **Example:**

    Args:
        host (str): The postgres host.
        dbname (str): The postgres database name.
        user (str): The postgres user.
        password (str): The postgres password.
        port (int): The postgres Port.
    """
    if not host:
        host = os.getenv("HOST")

    if not host:
        raise ImproperlyConfigured("Please set your postgres host")

    if not dbname:
        dbname = os.getenv("DATABASE")

    if not dbname:
        raise ImproperlyConfigured("Please set your postgres database")

    if not user:
        user = os.getenv("PG_USER")

    if not user:
        raise ImproperlyConfigured("Please set your postgres user")

    if not password:
        password = os.getenv("PASSWORD")

    if not password:
        raise ImproperlyConfigured("Please set your postgres password")

    if not port:
        port = int(os.getenv("PORT", 5432))

    if not port:
        raise ImproperlyConfigured("Please set your postgres port")

    conn = None
    # Retry connection a few times if there was an error
    retries = 3
    delay = 1  # seconds
    
    while retries > 0:
        if conn:
            break
            
        time.sleep(delay)
        try:
            conn = psycopg2.connect(
                host=host, dbname=dbname, user=user, password=password, port=port
            )
            break
        except psycopg2.Error:
            retries -= 1
            delay *= 2  # Exponential backoff
            
        if retries == 0:
            raise ValidationError("Failed to connect to postgres after multiple retries")
    return conn


def run_sql_postgres(
    conn: psycopg2.extensions.connection, sql: str
) -> Union[pd.DataFrame, None]:
    """
    Run a sql query on a postgres database.
    """
    try:
        cs = conn.cursor()
        cs.execute(sql)
        results = cs.fetchall()
        # Create a pandas dataframe from the results
        df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
        return df
    except psycopg2.InterfaceError as e:
        # Attempt to reconnect and retry the operation
        if conn:
            conn.close()  # Ensure any existing connection is closed
        conn = connect_to_postgres()
        cs = conn.cursor()
        cs.execute(sql)
        results = cs.fetchall()
        df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
        return df

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
            raise ValidationError(e)

    except Exception as e:
        conn.rollback()
        raise e
    return None
