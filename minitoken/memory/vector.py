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
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class ApiEmbedder(Embedder):
    """
    Embedder via une API externe compatible OpenAI (/embeddings) — OpenAI,
    NVIDIA, ou tout autre fournisseur. base_url est obligatoire et
    détermine le vrai fournisseur appelé, comme pour les autres providers
    du package (aucune liste fermée).
    """

    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        # La dimension dépend du modèle utilisé — on la déduit du premier
        # embedding généré plutôt que de la deviner à l'avance.
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Déclenche un premier appel pour déterminer la dimension
            # réelle du modèle configuré.
            sample = self.embed("dimension probe")
            self._dimension = len(sample)
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model_name, input=text)
        vector = response.data[0].embedding
        if self._dimension is None:
            self._dimension = len(vector)
        return vector


def get_embedder(config: MinitokenConfig) -> Embedder:
    """
    Fabrique l'embedder à utiliser selon la config du développeur.
    "local" (défaut) ne nécessite aucune clé API et ne coûte rien.
    """
    if config.embedding_provider == "local":
        return LocalEmbedder()

    if config.embedding_provider == "api":
        return ApiEmbedder(
            api_key=config.embedding_api_key,
            model_name=config.embedding_model,
            base_url=config.embedding_base_url,
        )

    raise ValueError(f"embedding_provider '{config.embedding_provider}' non supporté.")