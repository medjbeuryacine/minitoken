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

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from minitoken.config import MinitokenConfig
from minitoken.database.models import MinitokenModels, build_models


class MinitokenRepository:
    def __init__(self, config: MinitokenConfig):
        self.config = config
        self.models: MinitokenModels = build_models(config)
        self._engine = create_engine(config.database_url)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """
        Crée les 3 tables de minitoken dans la base du projet hôte, si
        elles n'existent pas déjà. Ne touche jamais aux tables
        users/conversations existantes — seulement à celles définies dans
        models.py (create_all ne crée que les tables rattachées à ce Base).
        """
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
        Crée ou met à jour la mémoire d'une conversation. Incrémente
        `version` à chaque écriture, pour la gestion de concurrence dont on
        a parlé (deux mises à jour parallèles pour la même conversation).
        """
        with self._session() as session:
            existing = (
                session.query(self.models.ConversationMemory)
                .filter_by(conversation_id=conversation_id)
                .one_or_none()
            )

            if existing is None:
                record = self.models.ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    summary=summary,
                    conversation_state=conversation_state,
                    message_count_at_last_summary=message_count_at_last_summary,
                    version=1,
                )
                session.add(record)
            else:
                existing.summary = summary
                existing.conversation_state = conversation_state
                existing.message_count_at_last_summary = message_count_at_last_summary
                existing.version += 1
                record = existing

            session.commit()
            session.refresh(record)
            return record

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

    def get_user_facts(self, *, user_id: uuid.UUID, scopes: list[str]):
        """
        Récupère les faits d'un utilisateur, filtrés par une liste de
        scopes (typiquement ["global", "<nom_agent>"]). C'est ce
        mécanisme qui garantit qu'un agent ne voit jamais les faits
        spécifiques à un autre agent.
        """
        with self._session() as session:
            return (
                session.query(self.models.UserMemory)
                .filter(
                    self.models.UserMemory.user_id == user_id,
                    self.models.UserMemory.scope.in_(scopes),
                )
                .order_by(self.models.UserMemory.updated_at.desc())
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
            # dépendante du modèle d'embedding) ; on l'écrit ici en SQL
            # brut pour ne pas coupler models.py à une dimension fixe.
            session.execute(
                self.models.MemoryEmbedding.__table__.update()
                .where(self.models.MemoryEmbedding.id == record.id)
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
            table = self.models.MemoryEmbedding.__table__
            return session.execute(
                table.select()
                .where(table.c.user_id == user_id, table.c.scope.in_(scopes))
                .order_by(table.c.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            ).fetchall()