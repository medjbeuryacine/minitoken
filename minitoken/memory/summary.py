"""
Mémoire résumé (conversation summary).

Absorbe les messages sortis de la fenêtre courte (short_term.py) dans un
résumé compact, mis à jour de façon incrémentale : on repart toujours du
résumé précédent + des messages bruts à intégrer, jamais d'un résumé-du-
résumé (pour éviter la dégradation progressive dont on avait parlé).
"""

from minitoken.providers.base import ChatMessage, LLMProvider

_SUMMARY_SYSTEM_PROMPT = """Tu mets à jour le résumé d'une conversation.

Résume en conservant : les faits importants, les décisions prises, les
informations données par l'utilisateur, le contexte nécessaire pour
continuer la discussion. Ignore les formules de politesse et les
reformulations. Sois factuel et dense, pas narratif.

Réponds uniquement avec le résumé mis à jour, sans préambule."""

_RECOMPRESS_SYSTEM_PROMPT = """Le résumé suivant est trop long. Condense-le
davantage, sans perdre les faits, décisions, et informations essentielles.
Réponds uniquement avec le résumé condensé, sans préambule."""


def _format_messages_for_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def update_summary(
    *,
    provider: LLMProvider,
    existing_summary: str,
    messages_to_summarize: list[ChatMessage],
) -> str:
    """
    Produit un résumé mis à jour à partir du résumé existant + des
    nouveaux messages à intégrer. Repart toujours des messages bruts
    (jamais d'un résumé déjà résumé), pour éviter la perte progressive
    d'information.
    """
    if not messages_to_summarize:
        return existing_summary

    user_content = (
        f"Résumé actuel :\n{existing_summary or '(vide, première mise à jour)'}\n\n"
        f"Nouveaux messages à intégrer :\n{_format_messages_for_prompt(messages_to_summarize)}"
    )

    result = provider.generate(
        messages=[
            ChatMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ],
        max_tokens=600,
    )

    return result.content.strip()


def recompress_if_too_long(
    *,
    provider: LLMProvider,
    summary: str,
    max_tokens: int = 1000,
) -> str:
    """
    Si le résumé dépasse `max_tokens` (mesuré avec le vrai tokenizer du
    provider), demande au LLM de le condenser davantage. Garantit que le
    résumé reste petit même après des mois de conversation.
    """
    if provider.count_tokens(summary) <= max_tokens:
        return summary

    result = provider.generate(
        messages=[
            ChatMessage(role="system", content=_RECOMPRESS_SYSTEM_PROMPT),
            ChatMessage(role="user", content=summary),
        ],
        max_tokens=max_tokens,
    )

    return result.content.strip()