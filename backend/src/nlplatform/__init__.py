"""
QdrantPipSQLPlatform is a Text2SQL platform that uses a
combination of Qdrant_VectorStore and PipSQL.
"""

from nlplatform.connectors.qdrant import QdrantVectorStore  # type: ignore
from nlplatform.models.pipSQL import PipSQL  # type: ignore


class QdrantPipSQLPlatform(QdrantVectorStore, PipSQL):
    """
    nlplatform is a Text2SQL platform that uses a
      combination of Qdrant_VectorStore and PipSQL.
    """

    def __init__(self, config=None):
        QdrantVectorStore.__init__(self, config=config)
        PipSQL.__init__(self)
