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
    RateLimitEvent: type

# Types fixes possibles pour UserMemory.type -- utilisés à la fois côté
# validation (repository.add_user_fact) et côté prompt d'extraction
# (structured.py), pour que les deux restent toujours synchronisés.
USER_MEMORY_TYPES = ["profile", "preference", "goal", "project", "fact"]

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

        # POINT structuration mémoire : type FIXE et fiable (contrairement
        # à category, texte libre imprévisible côté LLM) -- une des 5
        # valeurs de _USER_MEMORY_TYPES ("profile", "preference", "goal",
        # "project", "fact"). Sert à afficher des sections propres et
        # éditables côté frontend (ex: "Profil" / "Préférences" /
        # "Objectifs" / "Projets" / "Faits"), plutôt que de deviner un
        # regroupement à partir de category. Validé côté Python à
        # l'écriture (voir repository.add_user_fact), pas de contrainte
        # SQL CHECK pour rester simple à migrer si de nouveaux types
        # s'ajoutent plus tard.
        type = Column(String(50), nullable=False, default="fact", index=True)

        # Sous-catégorie libre à l'intérieur d'un type -- ex: type="goal",
        # category="force". N'est plus le mécanisme principal de
        # déduplication (voir memory_key ci-dessous), gardé comme filet
        # de repli si memory_key est absent.
        category = Column(String(100), nullable=True)

        # POINT mémoire source de vérité (architecture "user_memory =
        # source de vérité, memory_embeddings = index de recherche") :
        # identifiant logique STABLE d'un fait précis (ex:
        # "bench_press_goal"), fourni par le LLM d'extraction à partir de
        # la liste des clés déjà connues de cet utilisateur (voir
        # structured.py) -- plus fiable qu'une simple correspondance
        # type+category, qui peut légèrement varier d'un appel à l'autre
        # selon la formulation du LLM. Deux faits partageant la même
        # memory_key pour le même user_id/scope représentent LA MÊME
        # information dans le temps -- le second met à jour le premier
        # (même id conservé), jamais un doublon créé. Nullable : un fait
        # sans memory_key (LLM n'en a pas fourni) retombe sur la
        # déduplication par type+category, comme avant.
        memory_key = Column(String(150), nullable=True, index=True)

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

        # POINT mémoire source de vérité : lien optionnel vers le fait
        # user_memory dont cet embedding est la projection vectorielle
        # (voir UserMemory.memory_key). NULL pour les embeddings de
        # conversation brute (échanges complets, jamais liés à un fait
        # précis -- voir client.py record_exchange). Non-NULL pour les
        # embeddings générés à partir d'un fait structuré : dans ce cas,
        # quand le fait est mis à jour (même memory_key), CET embedding
        # est mis à jour en place (même id, nouveau vecteur), jamais
        # dupliqué -- élimine structurellement le risque de deux
        # souvenirs contradictoires coexistant en mémoire vectorielle
        # (ex: "objectif 100kg" et "objectif 140kg" tous les deux
        # présents en même temps).
        user_memory_id = Column(
            UUID(as_uuid=True), ForeignKey("user_memory.id", ondelete="CASCADE"), nullable=True, index=True
        )

        created_at = Column(DateTime(timezone=True), server_default=func.now())

    class RateLimitEvent(Base):
        __tablename__ = "rate_limit_events"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        provider_role = Column(String(100), nullable=False, index=True)
        called_at = Column(DateTime(timezone=True), server_default=func.timezone("UTC", func.now()), index=True)
        tokens_used = Column(Integer, nullable=False, default=0)

    return MinitokenModels(
        Base=Base,
        ConversationMemory=ConversationMemory,
        UserMemory=UserMemory,
        MemoryEmbedding=MemoryEmbedding,
        RateLimitEvent=RateLimitEvent,   # ← ajouté
    )