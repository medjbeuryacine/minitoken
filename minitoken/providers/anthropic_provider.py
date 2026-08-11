"""
Adaptateur pour Anthropic (Claude).

Contrairement à Groq/OpenAI/NVIDIA, l'API Anthropic n'est pas
OpenAI-compatible : format de requête différent, gestion du system prompt
séparée des messages, et SDK dédié (`anthropic`). Cet adaptateur isole
cette différence du reste de minitoken.
"""

from minitoken.providers.base import ChatMessage, CompletionResult, LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, rate_limiter=None):
        super().__init__(api_key=api_key, model=model, rate_limiter=rate_limiter)

        # Import différé : évite de forcer la dépendance `anthropic` si le
        # développeur n'utilise que des providers OpenAI-compatible.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)

    def generate(self, messages: list[ChatMessage], max_tokens: int = 1000) -> CompletionResult:
        system_prompt, conversation_messages = self._split_system_prompt(messages)

        if self._rate_limiter:
            estimated = sum(self.count_tokens(m.content) for m in messages) + max_tokens
            self._rate_limiter.wait_if_needed(estimated_tokens=estimated)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": m.role, "content": m.content} for m in conversation_messages],
        )

        content = "".join(block.text for block in response.content if block.type == "text")

        if self._rate_limiter:
            self._rate_limiter.record_call(tokens_used=response.usage.input_tokens + response.usage.output_tokens)

        return CompletionResult(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def count_tokens(self, text: str) -> int:
        # L'API Anthropic expose un endpoint de comptage de tokens dédié,
        # qui utilise le tokenizer exact du modèle plutôt qu'une
        # approximation.
        result = self._client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens

    @staticmethod
    def _split_system_prompt(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        conversation_messages = [m for m in messages if m.role != "system"]
        return "\n".join(system_parts), conversation_messages