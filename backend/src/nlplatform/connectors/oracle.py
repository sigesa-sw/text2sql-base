import os
from typing import Union

import pandas as pd  # type: ignore
import oracledb

from nlplatform.exceptions import ImproperlyConfigured, ValidationError


def connect_to_oracle(
    user: str | None = None, password: str | None = None, dsn: str | None = None
):
    """
    Connect to an Oracle db using oracledb package.
    **Example:**
    ```python
    vn.connect_to_oracle(
    user="username",
    password="password",
    dsn="host:port/sid",
    )
    ```
    Args:
        USER (str): Oracle db user name.
        PASSWORD (str): Oracle db user password.
        DSN (str): Oracle db host ip - host:port/sid.
    """
    if not dsn:
        dsn = os.getenv("DSN")

    if not dsn:
        raise ImproperlyConfigured(
            "Please set your Oracle dsn which should include host:port/sid"
        )

    if not user:
        user = os.getenv("USER")

    if not user:
        raise ImproperlyConfigured("Please set your Oracle db user")

    if not password:
        password = os.getenv("PASSWORD")

    if not password:
        raise ImproperlyConfigured("Please set your Oracle db password")

    conn = None

    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
    except oracledb.Error as e:
        raise ValidationError(e)
    return conn


def run_sql_oracle(conn: oracledb.Connection, sql: str) -> Union[pd.DataFrame, None]:
    if conn:
        try:
            sql = sql.rstrip()
            if sql.endswith(
                ";"
            ):  # fix for a known problem with Oracle db where an extra ; will cause an error.
                sql = sql[:-1]
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()
            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
            return df
        except oracledb.Error as e:
            conn.rollback()
            raise ValidationError(e)

        except Exception as e:
            conn.rollback()
            raise e
    return None
