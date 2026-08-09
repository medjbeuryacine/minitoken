"""API publique du sous-module providers."""

from minitoken.providers.base import ChatMessage, CompletionResult, LLMProvider


def build_provider(
    *, provider_name: str, model: str, api_key: str, base_url: str | None = None
) -> LLMProvider:
    """
    Fabrique un provider concret.

    - provider_name="anthropic" : utilise l'API native Claude (format non
      OpenAI-compatible), base_url ignorée.
    - tout autre provider_name (ex: "groq", "openai", "nvidia", ou
      n'importe quel autre nom libre) : traité comme OpenAI-compatible,
      `base_url` est alors OBLIGATOIRE — c'est elle qui détermine le vrai
      fournisseur appelé, pas provider_name (qui ne sert qu'à choisir la
      branche Anthropic vs OpenAI-compatible, et pour l'affichage/logs).
    """
    if provider_name == "anthropic":
        from minitoken.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model)

    from minitoken.providers.openai_compatible import OpenAICompatibleProvider

    if not base_url:
        raise ValueError(
            f"base_url est obligatoire pour provider_name='{provider_name}' "
            "(tout provider autre que 'anthropic' est traité comme "
            "OpenAI-compatible et nécessite l'URL de base exacte de "
            "votre fournisseur, ex: 'https://api.groq.com/openai/v1')."
        )

    return OpenAICompatibleProvider(
        api_key=api_key, model=model, base_url=base_url, provider_label=provider_name
    )


__all__ = ["ChatMessage", "CompletionResult", "LLMProvider", "build_provider"]



__all__ = ["ChatMessage", "CompletionResult", "LLMProvider", "build_provider"]