"""
Mémoire court-terme (short-term memory).

Gère la fenêtre glissante des derniers messages d'une conversation, envoyés
en clair au LLM, et décide quand cette fenêtre est trop grande pour être
envoyée telle quelle (auquel cas summary.py doit être déclenché pour
absorber les messages les plus anciens).
"""

from dataclasses import dataclass

from minitoken.providers.base import ChatMessage


@dataclass
class ShortTermWindow:
    recent_messages: list[ChatMessage]
    messages_to_summarize: list[ChatMessage]


def split_recent_messages(
    *,
    all_messages: list[ChatMessage],
    keep_recent_count: int = 8,
) -> ShortTermWindow:
    """
    Sépare les messages d'une conversation en deux groupes :
      - recent_messages : les `keep_recent_count` derniers, envoyés en
        clair au LLM à chaque appel
      - messages_to_summarize : tout ce qui précède, à absorber dans le
        résumé (via memory/summary.py) plutôt que d'être renvoyé en clair

    Si la conversation a moins de `keep_recent_count` messages, tout est
    considéré comme "récent" et rien n'a besoin d'être résumé.
    """
    if len(all_messages) <= keep_recent_count:
        return ShortTermWindow(recent_messages=all_messages, messages_to_summarize=[])

    split_index = len(all_messages) - keep_recent_count
    return ShortTermWindow(
        recent_messages=all_messages[split_index:],
        messages_to_summarize=all_messages[:split_index],
    )


def needs_summarization(
    *,
    total_message_count: int,
    message_count_at_last_summary: int,
    keep_recent_count: int = 8,
    resummarize_every: int = 5,
) -> bool:
    """
    Décide si un nouveau cycle de résumé doit être déclenché.

    On ne résume pas à chaque message (coût en appels LLM inutile) : on
    attend qu'il y ait au moins `keep_recent_count` messages au-delà de la
    fenêtre récente, ET qu'au moins `resummarize_every` nouveaux messages
    soient arrivés depuis le dernier résumé.
    """
    new_messages_since_last_summary = total_message_count - message_count_at_last_summary

    has_enough_old_messages = total_message_count > keep_recent_count
    has_enough_new_messages = new_messages_since_last_summary >= resummarize_every

    return has_enough_old_messages and has_enough_new_messages