"""
Comptage de tokens du contexte assemblé.

Mesure, section par section (recent messages / summary / structured facts
/ vector memories), combien de tokens seront réellement envoyés au LLM —
en utilisant le tokenizer réel du provider (via provider.count_tokens()),
jamais une approximation générique quand le vrai tokenizer est disponible.
"""

from dataclasses import dataclass, field

from minitoken.providers.base import ChatMessage, LLMProvider


@dataclass
class ContextBundle:
    """Le contexte assemblé, prêt à être envoyé au LLM (ou à être coupé
    par trimmer.py si trop volumineux)."""

    recent_messages: list[ChatMessage] = field(default_factory=list)
    summary: str = ""
    structured_facts: list[str] = field(default_factory=list)
    vector_memories: list[str] = field(default_factory=list)


@dataclass
class TokenReport:
    recent_messages_tokens: int
    summary_tokens: int
    structured_facts_tokens: int
    vector_memories_tokens: int

    @property
    def total_tokens(self) -> int:
        return (
            self.recent_messages_tokens
            + self.summary_tokens
            + self.structured_facts_tokens
            + self.vector_memories_tokens
        )


def count_bundle_tokens(*, provider: LLMProvider, bundle: ContextBundle) -> TokenReport:
    """
    Calcule le nombre de tokens de chaque section du bundle, avec le
    tokenizer réel du provider fourni (donc cohérent avec le modèle
    effectivement utilisé pour la requête).
    """
    recent_tokens = sum(provider.count_tokens(m.content) for m in bundle.recent_messages)
    summary_tokens = provider.count_tokens(bundle.summary) if bundle.summary else 0
    structured_tokens = sum(provider.count_tokens(f) for f in bundle.structured_facts)
    vector_tokens = sum(provider.count_tokens(v) for v in bundle.vector_memories)

    return TokenReport(
        recent_messages_tokens=recent_tokens,
        summary_tokens=summary_tokens,
        structured_facts_tokens=structured_tokens,
        vector_memories_tokens=vector_tokens,
    )


def fits_budget(*, report: TokenReport, budget_max: int) -> bool:
    return report.total_tokens <= budget_max