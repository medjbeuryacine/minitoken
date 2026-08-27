"""
Application des migrations de minitoken.
"""

from sqlalchemy import text

from minitoken.database.repository import MinitokenRepository
from minitoken.memory.vector import Embedder

_VECTOR_INDEX_NAME = "ix_memory_embeddings_embedding_cosine"

def _ensure_cascade_on_conversation_memory_fk(engine, repository: MinitokenRepository) -> None:
    """
    Corrige rétroactivement les tables créées avant l'ajout de
    ondelete="CASCADE" sur ConversationMemory.conversation_id (voir
    models.py). Le modèle SQLAlchemy déclare déjà le cascade, mais
    SQLAlchemy ne modifie jamais une contrainte déjà existante en
    base — create_tables() est un no-op sur une table qui existe déjà.
    Idempotent : si la contrainte est déjà en CASCADE, ne fait rien.
    """
    table_name = repository.models.ConversationMemory.__tablename__
    constraint_name = f"{table_name}_conversation_id_fkey"

    with engine.begin() as connection:
        row = connection.execute(
            text(
                "SELECT confdeltype FROM pg_constraint WHERE conname = :name"
            ),
            {"name": constraint_name},
        ).fetchone()

        if row is None:
            # Contrainte introuvable sous ce nom exact — rien à corriger
            # automatiquement, à vérifier manuellement si ça arrive.
            return

        if row[0] == "c":
            # Déjà en CASCADE, rien à faire.
            return

        connection.execute(
            text(f"ALTER TABLE {table_name} DROP CONSTRAINT {constraint_name}")
        )
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD CONSTRAINT {constraint_name} "
                f"FOREIGN KEY (conversation_id) "
                f"REFERENCES {repository.config.conversations_table}({repository.config.conversations_id_column}) "
                f"ON DELETE CASCADE"
            )
        )


def apply_migrations(*, repository: MinitokenRepository, embedder: Embedder) -> None:
    engine = repository._engine

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    repository.create_tables()

    _ensure_cascade_on_conversation_memory_fk(engine, repository)

    dimension = embedder.dimension
    table_name = repository.models.MemoryEmbedding.__tablename__

    # POINT robustesse (limite pgvector) : le type "vector" standard ne
    # peut pas être indexé au-delà de 2000 dimensions (limite dure de
    # pgvector, ivfflat ET hnsw -- pas une limitation de notre choix
    # d'index, une limite de PostgreSQL lui-même : chaque tuple d'index
    # doit tenir dans une page de 8 Ko). Les modèles d'embedding modernes
    # dépassent couramment cette limite (ex: 2048 dimensions pour
    # certains modèles NVIDIA). Le type "halfvec" (flottants 16 bits au
    # lieu de 32) double la limite indexable à 4000 dimensions, au prix
    # d'une précision légèrement réduite -- négligeable pour de la
    # recherche de similarité sémantique. Au-delà de 2000 dimensions, on
    # bascule automatiquement sur halfvec ; en dessous, on garde vector
    # (pleine précision) comme avant, sans changement de comportement
    # pour les développeurs utilisant des modèles plus petits.
    use_halfvec = dimension > 2000
    vector_type = "halfvec" if use_halfvec else "vector"

    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN IF NOT EXISTS embedding {vector_type}({dimension})"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_VECTOR_INDEX_NAME} "
                f"ON {table_name} USING hnsw (embedding {vector_type}_cosine_ops)"
            )
        )
