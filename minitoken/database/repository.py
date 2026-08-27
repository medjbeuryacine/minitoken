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

    def get_user_memory_keys(self, *, user_id: uuid.UUID) -> list[str]:
        """
        Retourne la liste des memory_key déjà connues pour cet
        utilisateur (tous scopes confondus, valeurs non-NULL
        uniquement) -- à passer à structured.extract_facts() pour que
        le LLM réutilise une clé existante plutôt que d'en inventer une
        nouvelle pour un sujet déjà connu."""
        with self._session() as session:
            rows = (
                session.query(self.models.UserMemory.memory_key)
                .filter(
                    self.models.UserMemory.user_id == user_id,
                    self.models.UserMemory.memory_key.isnot(None),
                )
                .distinct()
                .all()
            )
            return [r[0] for r in rows]

    def add_user_fact(
        self,
        *,
        user_id: uuid.UUID,
        fact: str,
        scope: str,
        type: str = "fact",
        category: str | None = None,
        memory_key: str | None = None,
        source_conversation_id: uuid.UUID | None = None,
        confidence: int | None = None,
        embedding: list[float] | None = None,
    ):
        """
        POINT structuration mémoire : type doit être une des valeurs de
        USER_MEMORY_TYPES -- validé ici, jamais fait confiance
        aveuglément à l'appelant.

        POINT mémoire source de vérité (architecture "user_memory =
        source de vérité, memory_embeddings = index de recherche") :
        déduplication en 2 temps.
        1. Si memory_key est fourni : cherche un fait existant avec la
           MÊME memory_key, pour le même user_id/scope -- si trouvé, le
           fait est MIS À JOUR EN PLACE (même id conservé), jamais
           dupliqué. C'est le mécanisme principal, le plus fiable.
        2. Si memory_key est absent, ou qu'aucun match n'est trouvé par
           memory_key : filet de repli sur l'ancienne logique
           type+category (comparaison texte, moins fiable mais mieux
           que rien).
        Si aucun des deux ne trouve de correspondance, un nouveau fait
        est créé.

        embedding (optionnel) : si fourni, un embedding est
        créé/mis à jour EN PLACE, lié à ce fait via user_memory_id --
        jamais un nouvel embedding séparé créé à côté d'un ancien pour
        le même fait. C'est ce lien qui élimine structurellement le
        risque de deux souvenirs vectoriels contradictoires coexistant
        (ex: "objectif 100kg" et "objectif 140kg" tous les deux
        présents en mémoire vectorielle en même temps) -- voir
        MemoryEmbedding.user_memory_id dans models.py."""
        from minitoken.database.models import USER_MEMORY_TYPES

        if scope not in self.config.agent_scopes:
            raise ValueError(
                f"scope '{scope}' invalide. Scopes autorisés par la config : "
                f"{self.config.agent_scopes}"
            )

        if type not in USER_MEMORY_TYPES:
            raise ValueError(
                f"type '{type}' invalide. Types autorisés : {USER_MEMORY_TYPES}"
            )

        with self._session() as session:
            existing_record = None

            if memory_key:
                existing_record = (
                    session.query(self.models.UserMemory)
                    .filter(
                        self.models.UserMemory.user_id == user_id,
                        self.models.UserMemory.scope == scope,
                        self.models.UserMemory.memory_key == memory_key,
                    )
                    .one_or_none()
                )

            if existing_record is None and category:
                # Filet de repli : ancienne logique type+category, pour
                # les faits sans memory_key (rétrocompatibilité) ou si
                # le LLM a changé de memory_key entre deux appels malgré
                # les instructions du prompt.
                existing_record = (
                    session.query(self.models.UserMemory)
                    .filter(
                        self.models.UserMemory.user_id == user_id,
                        self.models.UserMemory.scope == scope,
                        self.models.UserMemory.type == type,
                        self.models.UserMemory.category == category,
                    )
                    .first()
                )

            if existing_record is not None:
                # MISE À JOUR EN PLACE -- même id conservé, jamais de
                # doublon créé.
                existing_record.fact = fact
                existing_record.type = type
                existing_record.category = category
                if memory_key:
                    existing_record.memory_key = memory_key
                if confidence is not None:
                    existing_record.confidence = confidence
                if source_conversation_id is not None:
                    existing_record.source_conversation_id = source_conversation_id
                session.flush()
                record = existing_record
            else:
                record = self.models.UserMemory(
                    user_id=user_id,
                    fact=fact,
                    scope=scope,
                    type=type,
                    category=category,
                    memory_key=memory_key,
                    source_conversation_id=source_conversation_id,
                    confidence=confidence,
                )
                session.add(record)
                session.flush()

            if embedding is not None:
                linked_embedding = (
                    session.query(self.models.MemoryEmbedding)
                    .filter(self.models.MemoryEmbedding.user_memory_id == record.id)
                    .one_or_none()
                )

                reflected_table = Table(
                    "memory_embeddings", MetaData(), autoload_with=self._engine
                )

                if linked_embedding is not None:
                    # Met à jour le vecteur ET le texte de l'embedding
                    # déjà lié -- jamais un second embedding créé pour
                    # le même fait.
                    session.execute(
                        reflected_table.update()
                        .where(reflected_table.c.id == linked_embedding.id)
                        .values(embedding=embedding, content=fact)
                    )
                else:
                    new_embedding = self.models.MemoryEmbedding(
                        user_id=user_id,
                        conversation_id=source_conversation_id,
                        scope=scope,
                        content=fact,
                        user_memory_id=record.id,
                    )
                    session.add(new_embedding)
                    session.flush()
                    session.execute(
                        reflected_table.update()
                        .where(reflected_table.c.id == new_embedding.id)
                        .values(embedding=embedding)
                    )

            session.commit()
            session.refresh(record)
            return record

    def get_user_facts(self, *, user_id: uuid.UUID, scopes: list[str], limit: int):
        """
        Récupère les faits d'un utilisateur, filtrés par une liste de
        scopes (typiquement ["global", "<nom_agent>"]).

        POINT robustesse mémoire : distingue les faits "global" (prénom,
        préférences durables -- toujours pertinents peu importe le
        sujet de la conversation, censés rester peu nombreux) des faits
        "agent_specific" (objectifs/contraintes propres à ce domaine
        précis). Les faits globaux sont TOUJOURS inclus intégralement --
        jamais tronqués par `limit`, qui ne s'applique qu'aux faits
        agent_specific. Sans cette distinction, un historique
        agent_specific volumineux pourrait évincer un fait global
        essentiel (comme le prénom) simplement parce qu'il est plus
        ancien que `limit` autres faits accumulés depuis.

        `limit` s'applique uniquement aux faits agent_specific, les plus
        récents en priorité (via updated_at) -- évite qu'un historique
        accumulé sur des mois d'usage ne fasse exploser le budget de
        tokens à chaque appel, sans pour autant risquer de perdre les
        faits globaux qui doivent toujours être présents.
        """
        with self._session() as session:
            global_facts = (
                session.query(self.models.UserMemory)
                .filter(
                    self.models.UserMemory.user_id == user_id,
                    self.models.UserMemory.scope == "global",
                )
                .order_by(self.models.UserMemory.updated_at.desc())
                .all()
            )

            agent_specific_scopes = [s for s in scopes if s != "global"]
            agent_specific_facts = []
            if agent_specific_scopes:
                remaining_limit = max(0, limit - len(global_facts))
                agent_specific_facts = (
                    session.query(self.models.UserMemory)
                    .filter(
                        self.models.UserMemory.user_id == user_id,
                        self.models.UserMemory.scope.in_(agent_specific_scopes),
                    )
                    .order_by(self.models.UserMemory.updated_at.desc())
                    .limit(remaining_limit)
                    .all()
                )

            return global_facts + agent_specific_facts

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
        max_per_user_scope: int = 500,
    ):
        """
        POINT robustesse mémoire (point 5) : après insertion, si le
        nombre d'embeddings pour ce user_id/scope dépasse
        max_per_user_scope, les plus anciens en trop sont supprimés --
        nettoyage "au fil de l'eau", pas besoin d'une tâche de fond
        séparée comme pour pending_actions. Cap par utilisateur/scope
        (pas un TTL par âge) : un utilisateur actif après une longue
        pause ne perd jamais ses souvenirs juste parce qu'ils sont
        vieux -- seul le VOLUME accumulé déclenche un nettoyage, jamais
        le temps écoulé. Empêche la table de grossir indéfiniment (coût
        de stockage, lenteur croissante de search_similar_embeddings au
        fil des mois d'usage réel)."""
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

            total_count = (
                session.query(self.models.MemoryEmbedding)
                .filter(
                    self.models.MemoryEmbedding.user_id == user_id,
                    self.models.MemoryEmbedding.scope == scope,
                )
                .count()
            )
            if total_count > max_per_user_scope:
                excess = total_count - max_per_user_scope
                old_ids = [
                    row.id
                    for row in session.query(self.models.MemoryEmbedding.id)
                    .filter(
                        self.models.MemoryEmbedding.user_id == user_id,
                        self.models.MemoryEmbedding.scope == scope,
                    )
                    .order_by(self.models.MemoryEmbedding.created_at.asc())
                    .limit(excess)
                    .all()
                ]
                session.query(self.models.MemoryEmbedding).filter(
                    self.models.MemoryEmbedding.id.in_(old_ids)
                ).delete(synchronize_session=False)

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
        conversation_id: uuid.UUID | None = None,
        max_distance: float = 0.5,
    ):
        """
        Recherche de similarité vectorielle, toujours filtrée par user_id
        et scopes AVANT le calcul de similarité (jamais de recherche
        globale non filtrée, pour la sécurité multi-user).

        conversation_id (optionnel, POINT robustesse mémoire) : si fourni,
        priorise fortement les résultats de CETTE conversation -- résout
        le bug observé où un fragment sémantiquement proche mais tiré
        d'une conversation totalement différente (ex: un ancien Block
        d'un autre fil) polluait le contexte, sans distinction avec les
        résultats réellement pertinents pour la conversation en cours.
        Stratégie : cherche D'ABORD dans la conversation courante ; si
        elle ne fournit pas top_k résultats suffisants, complète avec
        les meilleurs résultats des AUTRES conversations, mais jamais en
        remplacement -- toujours en complément, et seulement s'ils
        passent le seuil de pertinence.

        max_distance (POINT robustesse mémoire) : cosine_distance va de 0
        (identique) à 2 (opposé) -- pgvector, pas une similarité en %.
        0.5 est un seuil raisonnable par défaut : au-delà, le résultat
        n'est généralement plus pertinent, même s'il figure dans le
        top_k techniquement le plus proche disponible. Sans ce seuil,
        top_k=3 retourne TOUJOURS 3 résultats, même si le 3e n'a
        presque aucun rapport avec la requête -- c'est exactement ce
        qui causait la contamination observée (un vieux message éloigné
        remontait simplement parce qu'aucun meilleur candidat n'existait,
        pas parce qu'il était réellement pertinent)."""
        with self._session() as session:
            table = Table("memory_embeddings", MetaData(), autoload_with=self._engine)
            distance_col = table.c.embedding.cosine_distance(query_embedding).label("distance")

            if conversation_id is not None:
                # 1. Cherche d'abord dans la conversation courante uniquement.
                same_conv_rows = session.execute(
                    table.select()
                    .add_columns(distance_col)
                    .where(
                        table.c.user_id == user_id,
                        table.c.scope.in_(scopes),
                        table.c.conversation_id == conversation_id,
                    )
                    .order_by(distance_col)
                    .limit(top_k)
                ).fetchall()
                same_conv_rows = [r for r in same_conv_rows if r.distance <= max_distance]

                if len(same_conv_rows) >= top_k:
                    return same_conv_rows

                # 2. Complète avec d'autres conversations SEULEMENT si la
                # conversation courante n'a pas assez de résultats
                # pertinents -- jamais en remplacement, toujours en plus.
                remaining = top_k - len(same_conv_rows)
                other_conv_rows = session.execute(
                    table.select()
                    .add_columns(distance_col)
                    .where(
                        table.c.user_id == user_id,
                        table.c.scope.in_(scopes),
                        table.c.conversation_id != conversation_id,
                    )
                    .order_by(distance_col)
                    .limit(remaining)
                ).fetchall()
                other_conv_rows = [r for r in other_conv_rows if r.distance <= max_distance]

                return same_conv_rows + other_conv_rows

            # Pas de conversation_id fourni (rétrocompatibilité) --
            # comportement précédent, mais avec le seuil de pertinence
            # appliqué en plus.
            rows = session.execute(
                table.select()
                .add_columns(distance_col)
                .where(table.c.user_id == user_id, table.c.scope.in_(scopes))
                .order_by(distance_col)
                .limit(top_k)
            ).fetchall()
            return [r for r in rows if r.distance <= max_distance]