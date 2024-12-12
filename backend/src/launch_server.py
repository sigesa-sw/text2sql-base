import os
from qdrant_client import QdrantClient  # type: ignore
from nlplatform import QdrantPipSQLPlatform  # type: ignore
from api import FlaskApp  # type: ignore


if __name__ == "__main__":
    platform = QdrantPipSQLPlatform(
        config={
            "client": QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"), 
                port=os.getenv("QDRANT_PORT", 6333)
            )
        }
    )
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "dbname": os.getenv("DB_NAME", "costes"),
    }
    print(db_config)
    platform.connect_to_db(
        database="postgres",
        db_config=db_config
    )
    app = FlaskApp(platform)
    app.run()
