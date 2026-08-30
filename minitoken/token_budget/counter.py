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
    par trimmer.py si trop volumineux).

    conversation_state : état structuré de la tâche en cours (ex: un
    outil propose_* en attente avec ses paramètres déjà connus) --
    contrairement aux autres sections, JAMAIS coupé par trim_to_budget,
    même priorité que recent_messages. Voir client.py set_conversation_state()
    pour l'écriture ; ce champ est générique (dict libre), minitoken ne
    connaît jamais le contenu métier qu'il transporte."""

    recent_messages: list[ChatMessage] = field(default_factory=list)
    summary: str = ""
    structured_facts: list[str] = field(default_factory=list)
    vector_memories: list[str] = field(default_factory=list)
    conversation_state: dict = field(default_factory=dict)


@dataclass
class TokenReport:
    recent_messages_tokens: int
    summary_tokens: int
    structured_facts_tokens: int
    vector_memories_tokens: int
    conversation_state_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.recent_messages_tokens
            + self.summary_tokens
            + self.structured_facts_tokens
            + self.vector_memories_tokens
            + self.conversation_state_tokens
        )


def count_bundle_tokens(*, provider: LLMProvider, bundle: ContextBundle) -> TokenReport:
    """
    Calcule le nombre de tokens de chaque section du bundle, avec le
    tokenizer réel du provider fourni (donc cohérent avec le modèle
    effectivement utilisé pour la requête).
    """
    import json

    recent_tokens = sum(provider.count_tokens(m.content) for m in bundle.recent_messages)
    summary_tokens = provider.count_tokens(bundle.summary) if bundle.summary else 0
    structured_tokens = sum(provider.count_tokens(f) for f in bundle.structured_facts)
    vector_tokens = sum(provider.count_tokens(v) for v in bundle.vector_memories)
    # Sérialisé en JSON compact pour un comptage réaliste -- c'est
    # exactement sous cette forme qu'il sera injecté dans le prompt côté
    # application hôte (voir agent-mvp/graph.py, à venir).
    state_tokens = (
        provider.count_tokens(json.dumps(bundle.conversation_state, ensure_ascii=False))
        if bundle.conversation_state else 0
    )

    return TokenReport(
        recent_messages_tokens=recent_tokens,
        summary_tokens=summary_tokens,
        structured_facts_tokens=structured_tokens,
        vector_memories_tokens=vector_tokens,
        conversation_state_tokens=state_tokens,
    )


def fits_budget(*, report: TokenReport, budget_max: int) -> bool:
    return report.total_tokens <= budget_max