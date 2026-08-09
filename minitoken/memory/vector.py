"""
Mémoire vectorielle (long-term semantic memory).

Génère les embeddings utilisés pour stocker et retrouver des souvenirs par
similarité sémantique (memory_embeddings, recherche via
repository.search_similar_embeddings()).

Par défaut, l'embedding est calculé localement via `sentence-transformers`
(gratuit, aucun appel API, aucun coût) — cohérent avec l'objectif "gratuit
pour tout le monde". Un développeur peut aussi configurer un provider
d'embedding externe (ex: OpenAI) s'il préfère.
"""

from abc import ABC, abstractmethod

from minitoken.config import MinitokenConfig


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension du vecteur produit — nécessaire pour configurer la
        colonne pgvector à la bonne taille lors de la migration."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """
    Embedder local via sentence-transformers — gratuit, aucun appel réseau
    après le premier téléchargement du modèle, aucune clé API requise.
    """

    # Modèle par défaut : léger, rapide, bon rapport qualité/vitesse pour
    # de la recherche sémantique de mémoire conversationnelle.
    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384  # dimension native de all-MiniLM-L6-v2

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME

        # Import différé : sentence-transformers (+ torch) est une
        # dépendance lourde, on ne la charge que si l'embedding local est
        # réellement utilisé.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class OpenAIEmbedder(Embedder):
    """
    Embedder via l'API OpenAI (text-embedding-3-small par défaut).
    Nécessite une clé API — utilisé seulement si le développeur configure
    explicitement embedding_provider="openai".
    """

    DEFAULT_MODEL_NAME = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536

    def __init__(self, api_key: str, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding


def get_embedder(config: MinitokenConfig) -> Embedder:
    """
    Fabrique l'embedder à utiliser selon la config du développeur.
    "local" (défaut) ne nécessite aucune clé API et ne coûte rien.
    """
    if config.embedding_provider == "local":
        return LocalEmbedder()

    if config.embedding_provider == "openai":
        if not config.embedding_api_key:
            raise ValueError("embedding_api_key est requis pour embedding_provider='openai'.")
        return OpenAIEmbedder(api_key=config.embedding_api_key)

    raise ValueError(f"embedding_provider '{config.embedding_provider}' non supporté.")