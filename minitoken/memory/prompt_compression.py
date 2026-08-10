"""
Compression de texte (résumé trop long, question utilisateur).

Deux implémentations :
  - LocalLLMLinguaCompressor : LLMLingua-2 en local, extractif, gratuit.
  - LLMBasedCompressor : reformulation via un LLM génératif (API).
  - NoOpCompressor : ne fait rien (compression_mode="none").
"""

from abc import ABC, abstractmethod

from minitoken.config import MinitokenConfig
from minitoken.providers.base import ChatMessage, LLMProvider

_LLM_COMPRESSION_PROMPT = """Compresse le texte suivant en gardant TOUS les \
faits, détails et le sens exact. Supprime uniquement les mots de liaison, \
répétitions et formulations inutiles. Réponds uniquement avec le texte \
compressé, sans préambule."""


class Compressor(ABC):
    @abstractmethod
    def compress(self, text: str, target_ratio: float) -> str:
        raise NotImplementedError


class NoOpCompressor(Compressor):
    def compress(self, text: str, target_ratio: float) -> str:
        return text


class LocalLLMLinguaCompressor(Compressor):
    DEFAULT_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME

        # Import différé : llmlingua est une dépendance optionnelle.
        from llmlingua import PromptCompressor

        self._compressor = PromptCompressor(
            model_name=self.model_name, use_llmlingua2=True
        )

    def compress(self, text: str, target_ratio: float) -> str:
        if not text.strip():
            return text
        result = self._compressor.compress_prompt(text, rate=target_ratio)
        return result["compressed_prompt"]


class LLMBasedCompressor(Compressor):
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def compress(self, text: str, target_ratio: float) -> str:
        if not text.strip():
            return text
        result = self.provider.generate(
            messages=[
                ChatMessage(role="system", content=_LLM_COMPRESSION_PROMPT),
                ChatMessage(role="user", content=text),
            ],
            max_tokens=max(50, int(len(text.split()) * target_ratio) + 50),
        )
        return result.content.strip()


def get_compressor(config: MinitokenConfig, extraction_provider: LLMProvider) -> Compressor:
    """
    Fabrique le compressor selon compression_mode. compression_llm_* est
    totalement indépendant de extraction_provider — si compression_mode
    n'est pas configuré avec ses propres champs, on ne réutilise jamais
    extraction_provider par défaut, pour éviter tout couplage implicite.
    """
    if config.compression_mode == "local":
        return LocalLLMLinguaCompressor()

    if config.compression_mode == "api":
        from minitoken.providers import build_provider

        provider = build_provider(
            provider_name=config.compression_llm_provider,
            model=config.compression_llm_model,
            api_key=config.compression_llm_api_key,
            base_url=config.compression_llm_base_url,
        )
        return LLMBasedCompressor(provider=provider)

    return NoOpCompressor()