"""
Application des migrations de minitoken.
"""

from sqlalchemy import text

from minitoken.database.repository import MinitokenRepository
from minitoken.memory.vector import Embedder

_VECTOR_INDEX_NAME = "ix_memory_embeddings_embedding_cosine"


def apply_migrations(*, repository: MinitokenRepository, embedder: Embedder) -> None:
    engine = repository._engine

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    repository.create_tables()

    dimension = embedder.dimension
    table_name = repository.models.MemoryEmbedding.__tablename__

    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN IF NOT EXISTS embedding vector({dimension})"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_VECTOR_INDEX_NAME} "
                f"ON {table_name} USING ivfflat (embedding vector_cosine_ops) "
                f"WITH (lists = 100)"
            )
        )
