"""
Modèles SQLAlchemy de minitoken.

IMPORTANT : les tables `users` et `conversations` du projet hôte ont un nom
et une colonne d'ID définis dynamiquement par le développeur (via
MinitokenConfig). On ne peut donc pas écrire les clés étrangères en dur au
niveau du module — les modèles sont construits à l'exécution par
`build_models(config)`, à partir des noms fournis dans la config.

Aucune valeur de table n'est fixée ici : ce fichier ne connaît ni "users"
ni "conversations" tant que build_models() n'a pas reçu une config précise.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base

from minitoken.config import MinitokenConfig


@dataclass
class MinitokenModels:
    """Conteneur des 3 modèles construits pour une config donnée."""

    Base: type
    ConversationMemory: type
    UserMemory: type
    MemoryEmbedding: type


def build_models(config: MinitokenConfig) -> MinitokenModels:
    """
    Construit les 3 tables de minitoken (conversation_memory, user_memory,
    memory_embeddings) avec des clés étrangères pointant vers les tables
    users/conversations réelles du projet hôte, telles que définies dans
    `config`.
    """
    Base = declarative_base()

    users_fk_target = f"{config.users_table}.{config.users_id_column}"
    conversations_fk_target = (
        f"{config.conversations_table}.{config.conversations_id_column}"
    )

    class ConversationMemory(Base):
        __tablename__ = "conversation_memory"
        __table_args__ = (
            UniqueConstraint("conversation_id", name="uq_conversation_memory_conversation_id"),
        )

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

        # Dénormalisé volontairement (voir discussion) pour accélérer les
        # lectures "toute la mémoire de conversation de tel user" sans
        # jointure systématique vers la table conversations du projet hôte.
        user_id = Column(UUID(as_uuid=True), ForeignKey(users_fk_target), nullable=False, index=True)

        conversation_id = Column(
            UUID(as_uuid=True),
            ForeignKey(conversations_fk_target, ondelete="CASCADE"),
            nullable=False,
            index=True,
        )

        summary = Column(Text, nullable=False, default="")
        conversation_state = Column(JSONB, nullable=False, default=dict)

        message_count_at_last_summary = Column(Integer, nullable=False, default=0)
        version = Column(Integer, nullable=False, default=1)

        updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    class UserMemory(Base):
        __tablename__ = "user_memory"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

        user_id = Column(UUID(as_uuid=True), ForeignKey(users_fk_target), nullable=False, index=True)

        fact = Column(Text, nullable=False)

        # Doit correspondre à une valeur de config.agent_scopes
        # ("global" ou un nom d'agent précis).
        scope = Column(String(100), nullable=False, default="global", index=True)

        category = Column(String(100), nullable=True)

        # Traçabilité uniquement — pas une contrainte de filtrage.
        source_conversation_id = Column(
            UUID(as_uuid=True), ForeignKey(conversations_fk_target, ondelete="SET NULL"), nullable=True
        )

        confidence = Column(Integer, nullable=True)  # 0-100, optionnel

        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    class MemoryEmbedding(Base):
        __tablename__ = "memory_embeddings"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

        # Scoping obligatoire : jamais de recherche vectorielle sans filtre
        # user_id.
        user_id = Column(UUID(as_uuid=True), ForeignKey(users_fk_target), nullable=False, index=True)

        # Traçabilité uniquement, nullable.
        conversation_id = Column(
            UUID(as_uuid=True), ForeignKey(conversations_fk_target, ondelete="SET NULL"), nullable=True
        )

        scope = Column(String(100), nullable=False, default="global", index=True)

        content = Column(Text, nullable=False)

        # La colonne vector (pgvector) est ajoutée séparément par la
        # migration Alembic, car sa dimension dépend du modèle d'embedding
        # choisi dans la config (embedding_provider) et n'est donc pas
        # connue de façon statique ici.

        importance_score = Column(Integer, nullable=True)  # 0-100, optionnel

        created_at = Column(DateTime(timezone=True), server_default=func.now())

    return MinitokenModels(
        Base=Base,
        ConversationMemory=ConversationMemory,
        UserMemory=UserMemory,
        MemoryEmbedding=MemoryEmbedding,
    )