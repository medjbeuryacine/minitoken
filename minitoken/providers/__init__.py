"""API publique du sous-module providers."""

from minitoken.providers.base import ChatMessage, CompletionResult, LLMProvider


def build_provider(*, provider_name: str, model: str, api_key: str) -> LLMProvider:
    """
    Fabrique un provider concret à partir de son nom. Point d'entrée
    unique utilisé par client.py pour construire response_provider et
    extraction_provider depuis la config.
    """
    if provider_name == "anthropic":
        from minitoken.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model)

    if provider_name in ("groq", "openai", "nvidia"):
        from minitoken.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(api_key=api_key, model=model, provider_name=provider_name)

    raise ValueError(
        f"Provider '{provider_name}' non supporté nativement. "
        "Providers supportés : groq, openai, nvidia, anthropic."
    )


__all__ = ["ChatMessage", "CompletionResult", "LLMProvider", "build_provider"]