"""
Configuration générique de minitoken.

Ce fichier ne contient AUCUNE valeur spécifique à un projet.
Chaque développeur qui installe minitoken fournit ses propres valeurs
lors de l'instanciation de MinitokenConfig, dans SON projet.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


EmbeddingProviderName = Literal["local", "api"]
CompressionProviderName = Literal["local", "extraction_llm", "none"]

# provider_name n'est plus restreint à une liste fermée : c'est un nom
# libre ("groq", "openai", "nvidia", "mistral", ou n'importe quel autre).
# Seule la valeur "anthropic" a un traitement spécial (API native Claude,
# non OpenAI-compatible). Tout le reste est traité comme OpenAI-compatible
# et nécessite un base_url correspondant (voir token_counter_base_url /
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

    # --- Provider utilisé UNIQUEMENT pour compter les tokens (jamais pour
    # générer du texte) — sert à mesurer avec le bon tokenizer si le
    # contexte assemblé (résumé + faits + messages) rentre dans
    # token_budget_max. Doit correspondre au modèle réellement utilisé
    # pour générer les réponses côté application (ex: le même modèle que
    # dans votre graph LangGraph), pour que le comptage soit exact.
    token_counter_provider: str
    token_counter_model: str
    token_counter_api_key: str

    # --- Provider utilisé pour le résumé + l'extraction de faits ---
    # Peut être un provider/modèle différent (souvent plus léger) de celui
    # utilisé pour les réponses.
    # extraction_provider suit exactement la même logique que
    # token_counter_provider ci-dessus : nom libre ("groq", "openai", "nvidia",
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
    token_counter_base_url: Optional[str] = None
    extraction_base_url: Optional[str] = None

    # --- Compression de texte (résumé trop long + question utilisateur) ---
    # "local"  : LLMLingua-2, tourne en local, gratuit, extractif (pas de
    #            reformulation, sélectionne juste les tokens essentiels).
    # "api"    : utilise un LLM génératif via API (config compression_llm_*
    #            ci-dessous), indépendant de extraction_provider.
    # "none"   : désactivé (comportement par défaut, rien ne change).
    compression_mode: Literal["local", "api", "none"] = "none"
    compression_target_ratio: float = 0.5

    # Utilisés uniquement si compression_mode="api". Config totalement
    # indépendante de extraction_provider — un développeur peut choisir un
    # provider différent ici, ce n'est jamais lié automatiquement.
    compression_llm_provider: Optional[str] = None
    compression_llm_model: Optional[str] = None
    compression_llm_api_key: Optional[str] = None
    compression_llm_base_url: Optional[str] = None
    
    # --- Mémoire vectorielle ---
    # "local" : sentence-transformers, tourne en local, gratuit, aucune
    #           clé requise.
    # "api"   : appel à une API d'embedding externe (OpenAI, NVIDIA, ou
    #           tout autre fournisseur exposant /embeddings au format
    #           OpenAI) — nécessite embedding_api_key et embedding_base_url.
    embedding_provider: EmbeddingProviderName = "local"
    embedding_model: Optional[str] = None  # requis seulement si embedding_provider="api"
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

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

        if not self.token_counter_api_key:
            raise ValueError("token_counter_api_key est obligatoire.")

        if self.token_counter_provider != "anthropic" and not self.token_counter_base_url:
            raise ValueError(
                "token_counter_base_url est obligatoire quand token_counter_provider "
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

        if self.embedding_provider == "api":
            if not self.embedding_api_key:
                raise ValueError(
                    "embedding_api_key est obligatoire quand embedding_provider='api'."
                )
            if not self.embedding_base_url:
                raise ValueError(
                    "embedding_base_url est obligatoire quand embedding_provider='api' "
                    "(indiquez l'URL de base de votre fournisseur d'embeddings, ex: "
                    "'https://api.openai.com/v1')."
                )
            if not self.embedding_model:
                raise ValueError(
                    "embedding_model est obligatoire quand embedding_provider='api'."
                )

        if self.compression_mode == "api":
            if not (self.compression_llm_provider and self.compression_llm_model and self.compression_llm_api_key):
                raise ValueError(
                    "compression_llm_provider, compression_llm_model et "
                    "compression_llm_api_key sont obligatoires quand "
                    "compression_mode='llm'."
                )
            if self.compression_llm_provider != "anthropic" and not self.compression_llm_base_url:
                raise ValueError(
                    "compression_llm_base_url est obligatoire quand "
                    "compression_llm_provider n'est pas 'anthropic'."
                )

        if not (0 < self.compression_target_ratio <= 1):
            raise ValueError("compression_target_ratio doit être entre 0 et 1 (exclu 0).")

        if self.token_budget_max <= 0:
            raise ValueError("token_budget_max doit être un entier positif.")

        if not self.agent_scopes:
            raise ValueError("agent_scopes ne peut pas être une liste vide.")

        if "global" not in self.agent_scopes:
            # "global" doit toujours être un scope valide, même si le
            # développeur définit ses propres scopes d'agents.
            self.agent_scopes = ["global", *self.agent_scopes]