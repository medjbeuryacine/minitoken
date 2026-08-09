"""
Interface commune à tous les providers LLM supportés par minitoken.

Chaque provider (Groq, OpenAI, NVIDIA, Anthropic...) doit implémenter cette
interface. Le reste de minitoken (memory/, token_budget/) ne parle jamais
directement à un SDK provider : il passe toujours par cette interface, ce
qui permet d'ajouter un nouveau provider sans toucher au reste du code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    content: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """
    Interface que chaque adaptateur provider (openai_compatible.py,
    anthropic_provider.py, ...) doit implémenter.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, messages: list[ChatMessage], max_tokens: int = 1000) -> CompletionResult:
        """
        Envoie les messages au LLM et retourne sa réponse, avec le nombre
        de tokens réellement consommés (input/output), tel que rapporté
        par le provider.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Compte les tokens d'un texte selon le tokenizer réel de ce
        provider/modèle. Doit être aussi précis que possible ; si le
        tokenizer exact n'est pas disponible, l'implémentation peut
        retomber sur une approximation, mais doit le documenter clairement.
        """
        raise NotImplementedError