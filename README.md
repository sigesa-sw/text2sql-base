# MLS Querier
Python RAG (Retrieval-Augmented Generation) framework for SQL generation and related functionality.

## How it works

Works in two easy steps - train a RAG "model" on your data, and then ask questions which will return SQL queries that can be set up to automatically run on your database.

1. **Train a RAG "model" on your data**.
2. **Ask questions**.

If you don't know what RAG is, don't worry -- you don't need to know how this works under the hood to use it. You just need to know that you "train" a model, which stores some metadata and then use it to "ask" questions.

## User Interfaces
These are some of the user interfaces that we've built using Vanna. You can use these as-is or as a starting point for your own custom interface.

## User Interfaces
- [vanna-ai/vanna-flask](https://github.com/vanna-ai/vanna-flask)

## Supported LLMs
- [HuggingFace](https://github.com/vanna-ai/vanna/blob/main/src/vanna/hf/hf.py)

## Supported VectorStores
- [Qdrant](https://github.com/vanna-ai/vanna/tree/main/src/vanna/qdrant)

## Supported Databases
- [PostgreSQL](https://www.postgresql.org/)
- [Oracle](https://www.oracle.com/)


## Getting started

### Install
```bash
pip install vanna
```

```python
# The import statement will vary depending on your LLM and vector database. This is an example for OpenAI + ChromaDB

from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': 'sk-...', 'model': 'gpt-4-...'})

# See the documentation for other options

```


## Training
You may or may not need to run these `vn.train` commands depending on your use case. See the [documentation](https://vanna.ai/docs/) for more details.

These statements are shown to give you a feel for how it works.

### Train with DDL Statements
DDL statements contain information about the table names, columns, data types, and relationships in your database.

```python
vn.train(ddl="""
    CREATE TABLE IF NOT EXISTS my-table (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        age INT
    )
""")
```

### Train with Documentation
Sometimes you may want to add documentation about your business terminology or definitions.

```python
vn.train(documentation="Our business defines XYZ as ...")
```

### Train with SQL
You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.

```python
vn.train(sql="SELECT name, age FROM my-table WHERE name = 'John Doe'")
```


## Asking questions
```python
vn.ask("What are the top 10 customers by sales?")
```

You'll get SQL
```sql
SELECT c.c_name as customer_name,
        sum(l.l_extendedprice * (1 - l.l_discount)) as total_sales
FROM   snowflake_sample_data.tpch_sf1.lineitem l join snowflake_sample_data.tpch_sf1.orders o
        ON l.l_orderkey = o.o_orderkey join snowflake_sample_data.tpch_sf1.customer c
        ON o.o_custkey = c.c_custkey
GROUP BY customer_name
ORDER BY total_sales desc limit 10;
```