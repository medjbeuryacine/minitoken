"""
Configuration générique de minitoken.

Ce fichier ne contient AUCUNE valeur spécifique à un projet.
Chaque développeur qui installe minitoken fournit ses propres valeurs
lors de l'instanciation de MinitokenConfig, dans SON projet.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


EmbeddingProviderName = Literal["local", "openai"]

# provider_name n'est plus restreint à une liste fermée : c'est un nom
# libre ("groq", "openai", "nvidia", "mistral", ou n'importe quel autre).
# Seule la valeur "anthropic" a un traitement spécial (API native Claude,
# non OpenAI-compatible). Tout le reste est traité comme OpenAI-compatible
# et nécessite un base_url correspondant (voir response_base_url /
# extraction_base_url ci-dessous).


@dataclass
class MinitokenConfig:
    # --- Connexion à la base de données du projet hôte ---
    database_url: str

    # --- Tables pré-requises côté projet hôte (obligatoires) ---
    users_table: str
    users_id_column: str

    conversations_table: str
    conversations_id_column: str

    # --- Provider utilisé pour générer les réponses principales ---
    # response_provider : nom libre ("groq", "openai", "nvidia", ...) sauf
    # "anthropic" qui a un traitement spécifique. Sert uniquement à
    # étiqueter/logger, et à choisir la branche Anthropic vs
    # OpenAI-compatible — pas à deviner une URL.
    response_provider: str
    response_model: str
    response_api_key: str

    # --- Provider utilisé pour le résumé + l'extraction de faits ---
    # Peut être un provider/modèle différent (souvent plus léger) de celui
    # utilisé pour les réponses.
    # extraction_provider suit exactement la même logique que
    # response_provider ci-dessus : nom libre ("groq", "openai", "nvidia",
    # ...) sauf "anthropic" qui a un traitement spécifique. Sert à
    # étiqueter/logger et à choisir la branche Anthropic vs
    # OpenAI-compatible — pas à deviner une URL (voir extraction_base_url).
    extraction_provider: str
    extraction_model: str
    extraction_api_key: str

    # --- Optionnels / avec valeur par défaut ---

    # Obligatoire (vérifié dans _validate) sauf si le provider correspondant
    # est "anthropic". C'est cette URL qui détermine le vrai fournisseur
    # appelé (ex: "https://api.groq.com/openai/v1",
    # "https://api.openai.com/v1", "https://integrate.api.nvidia.com/v1",
    # ou toute autre API respectant le format OpenAI).
    response_base_url: Optional[str] = None
    extraction_base_url: Optional[str] = None

    # --- Mémoire vectorielle ---
    embedding_provider: EmbeddingProviderName = "local"
    embedding_api_key: Optional[str] = None  # requis seulement si embedding_provider != "local"

    # --- Budget de tokens ---
    token_budget_max: int = 8000

    # --- Scopes d'agents valides pour user_memory / memory_embeddings ---
    # Si non fourni, un seul scope "global" est utilisé (cas mono-agent).
    agent_scopes: list[str] = field(default_factory=lambda: ["global"])

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if not self.database_url:
            raise ValueError("database_url est obligatoire.")

        if not self.users_table or not self.users_id_column:
            raise ValueError(
                "users_table et users_id_column sont obligatoires "
                "(minitoken a besoin de savoir où trouver vos utilisateurs)."
            )

        if not self.conversations_table or not self.conversations_id_column:
            raise ValueError(
                "conversations_table et conversations_id_column sont obligatoires "
                "(minitoken a besoin de savoir où trouver vos conversations)."
            )

        if not self.response_api_key:
            raise ValueError("response_api_key est obligatoire.")

        if self.response_provider != "anthropic" and not self.response_base_url:
            raise ValueError(
                "response_base_url est obligatoire quand response_provider "
                "n'est pas 'anthropic' (indiquez l'URL de base de votre "
                "fournisseur OpenAI-compatible, ex: "
                "'https://api.groq.com/openai/v1')."
            )

        if not self.extraction_api_key:
            raise ValueError("extraction_api_key est obligatoire.")

        if self.extraction_provider != "anthropic" and not self.extraction_base_url:
            raise ValueError(
                "extraction_base_url est obligatoire quand extraction_provider "
                "n'est pas 'anthropic' (indiquez l'URL de base de votre "
                "fournisseur OpenAI-compatible, ex: "
                "'https://api.groq.com/openai/v1')."
            )

        if self.embedding_provider != "local" and not self.embedding_api_key:
            raise ValueError(
                "embedding_api_key est obligatoire quand embedding_provider "
                "n'est pas 'local'."
            )

        if self.token_budget_max <= 0:
            raise ValueError("token_budget_max doit être un entier positif.")

        if not self.agent_scopes:
            raise ValueError("agent_scopes ne peut pas être une liste vide.")

        if "global" not in self.agent_scopes:
            # "global" doit toujours être un scope valide, même si le
            # développeur définit ses propres scopes d'agents.
            self.agent_scopes = ["global", *self.agent_scopes]