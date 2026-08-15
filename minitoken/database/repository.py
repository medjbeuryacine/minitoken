"""
Couche d'accès aux données de minitoken.

Connecte la config du développeur (database_url) aux modèles construits par
build_models(), et expose des fonctions simples de lecture/écriture pour
les 3 tables de mémoire. C'est la seule couche de minitoken qui parle
directement à la base de données — memory/ ne fait jamais de SQL, elle
passe toujours par MinitokenRepository.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.orm import Session, sessionmaker

from pgvector.sqlalchemy import Vector

from minitoken.config import MinitokenConfig
from minitoken.database.models import MinitokenModels, build_models

# Enregistre le type "vector" (pgvector) auprès de SQLAlchemy, pour que
# la réflexion automatique (autoload_with) reconnaisse correctement cette
# colonne et expose les méthodes de similarité (cosine_distance, etc.)
# au lieu de la traiter comme un type générique inconnu.
ischema_names["vector"] = Vector


class MinitokenRepository:
    def __init__(self, config: MinitokenConfig):
        self.config = config
        self.models: MinitokenModels = build_models(config)
        self._engine = create_engine(config.database_url)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """
        Crée les 3 tables de minitoken dans la base du projet hôte, si
        elles n'existent pas déjà. Reflète d'abord les tables users/
        conversations existantes dans la même metadata, pour que SQLAlchemy
        puisse résoudre les clés étrangères vers elles.
        """
        from sqlalchemy import Table

        Table(
            self.config.users_table,
            self.models.Base.metadata,
            autoload_with=self._engine,
            extend_existing=True,
        )
        Table(
            self.config.conversations_table,
            self.models.Base.metadata,
            autoload_with=self._engine,
            extend_existing=True,
        )

        self.models.Base.metadata.create_all(self._engine)

    def _session(self) -> Session:
        return self._SessionLocal()

    # ------------------------------------------------------------------
    # conversation_memory
    # ------------------------------------------------------------------

    def get_conversation_memory(self, conversation_id: uuid.UUID):
        with self._session() as session:
            return (
                session.query(self.models.ConversationMemory)
                .filter_by(conversation_id=conversation_id)
                .one_or_none()
            )

    def upsert_conversation_memory(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        summary: str,
        conversation_state: dict,
        message_count_at_last_summary: int,
    ):
        """
        Crée ou met à jour la mémoire d'une conversation, de façon
        ATOMIQUE via INSERT ... ON CONFLICT DO UPDATE (upsert SQL natif).

        Un SELECT-puis-INSERT/UPDATE classique souffre d'une race
        condition sous forte concurrence : deux threads peuvent tous
        les deux voir "aucune ligne existante" avant qu'aucun n'ait
        écrit, et tous les deux tenter un INSERT — le second échoue
        alors avec une UniqueViolation sur la contrainte
        uq_conversation_memory_conversation_id (confirmé par un test de
        charge concurrente). ON CONFLICT DO UPDATE élimine cette classe
        de bug : Postgres gère l'atomicité lui-même, sans fenêtre entre
        lecture et écriture.

        version est incrémenté via l'expression SQL elle-même
        (version + 1), jamais lu puis recalculé côté Python — donc
        toujours correct même sous forte concurrence.
        """
        import json
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self._session() as session:
            table = self.models.ConversationMemory.__table__

            stmt = pg_insert(table).values(
                conversation_id=conversation_id,
                user_id=user_id,
                summary=summary,
                conversation_state=conversation_state,
                message_count_at_last_summary=message_count_at_last_summary,
                version=1,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["conversation_id"],
                set_={
                    "summary": stmt.excluded.summary,
                    "conversation_state": stmt.excluded.conversation_state,
                    "message_count_at_last_summary": stmt.excluded.message_count_at_last_summary,
                    "version": table.c.version + 1,
                },
            ).returning(table)

            result = session.execute(stmt)
            row = result.fetchone()
            session.commit()

            return (
                session.query(self.models.ConversationMemory)
                .filter_by(conversation_id=conversation_id)
                .one()
            )

    # ------------------------------------------------------------------
    # user_memory
    # ------------------------------------------------------------------

    def add_user_fact(
        self,
        *,
        user_id: uuid.UUID,
        fact: str,
        scope: str,
        category: str | None = None,
        source_conversation_id: uuid.UUID | None = None,
        confidence: int | None = None,
    ):
        if scope not in self.config.agent_scopes:
            raise ValueError(
                f"scope '{scope}' invalide. Scopes autorisés par la config : "
                f"{self.config.agent_scopes}"
            )

        with self._session() as session:
            record = self.models.UserMemory(
                user_id=user_id,
                fact=fact,
                scope=scope,
                category=category,
                source_conversation_id=source_conversation_id,
                confidence=confidence,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_user_facts(self, *, user_id: uuid.UUID, scopes: list[str], limit: int):
        """
        Récupère les faits les plus récents d'un utilisateur, filtrés par
        une liste de scopes (typiquement ["global", "<nom_agent>"]).
        Limité à `limit` faits (les plus récents en priorité, via
        updated_at), pour éviter qu'un historique accumulé sur des mois
        d'usage ne fasse exploser le budget de tokens à chaque appel.
        """
        with self._session() as session:
            return (
                session.query(self.models.UserMemory)
                .filter(
                    self.models.UserMemory.user_id == user_id,
                    self.models.UserMemory.scope.in_(scopes),
                )
                .order_by(self.models.UserMemory.updated_at.desc())
                .limit(limit)
                .all()
            )

    # ------------------------------------------------------------------
    # memory_embeddings
    # ------------------------------------------------------------------

    def add_memory_embedding(
        self,
        *,
        user_id: uuid.UUID,
        content: str,
        scope: str,
        embedding: list[float],
        conversation_id: uuid.UUID | None = None,
        importance_score: int | None = None,
    ):
        if scope not in self.config.agent_scopes:
            raise ValueError(
                f"scope '{scope}' invalide. Scopes autorisés par la config : "
                f"{self.config.agent_scopes}"
            )

        with self._session() as session:
            record = self.models.MemoryEmbedding(
                user_id=user_id,
                conversation_id=conversation_id,
                scope=scope,
                content=content,
                importance_score=importance_score,
            )
            session.add(record)
            session.flush()

            # La colonne vector est ajoutée par migration (dimension
            # dépendante du modèle d'embedding) ; on relit la structure
            # réelle de la table (autoload) car le modèle Python ne
            # connaît pas cette colonne ajoutée en SQL brut.
            reflected_table = Table(
                "memory_embeddings", MetaData(), autoload_with=self._engine
            )
            session.execute(
                reflected_table.update()
                .where(reflected_table.c.id == record.id)
                .values(embedding=embedding)
            )
            session.commit()
            session.refresh(record)
            return record

    def search_similar_embeddings(
        self,
        *,
        user_id: uuid.UUID,
        scopes: list[str],
        query_embedding: list[float],
        top_k: int = 5,
    ):
        """
        Recherche de similarité vectorielle, toujours filtrée par user_id
        et scopes AVANT le calcul de similarité (jamais de recherche
        globale non filtrée, pour la sécurité multi-user dont on a
        parlé).
        """
        with self._session() as session:
            table = Table("memory_embeddings", MetaData(), autoload_with=self._engine)
            return session.execute(
                table.select()
                .where(table.c.user_id == user_id, table.c.scope.in_(scopes))
                .order_by(table.c.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            ).fetchall()