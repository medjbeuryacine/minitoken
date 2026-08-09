"""
Adaptateur pour tous les providers "OpenAI-compatible" : Groq, OpenAI,
NVIDIA NIM, et tout autre provider exposant une API au format OpenAI
(/chat/completions).

Un seul adaptateur suffit pour ces trois providers car ils partagent le
même format de requête/réponse — seule la base_url (et parfois le style
exact du modèle) change.
"""

from minitoken.providers.base import ChatMessage, CompletionResult, LLMProvider

# Base URL par provider connu. Un développeur peut aussi passer une
# base_url personnalisée pour un provider OpenAI-compatible non listé ici.
_KNOWN_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, provider_name: str, base_url: str | None = None):
        super().__init__(api_key=api_key, model=model)

        resolved_base_url = base_url or _KNOWN_BASE_URLS.get(provider_name)
        if resolved_base_url is None:
            raise ValueError(
                f"Provider '{provider_name}' inconnu et aucune base_url fournie. "
                f"Providers connus : {list(_KNOWN_BASE_URLS)}. "
                "Pour un provider OpenAI-compatible non listé, passez base_url explicitement."
            )

        self.provider_name = provider_name
        self.base_url = resolved_base_url

        # Import différé : on évite de forcer la dépendance `openai` si le
        # développeur n'utilise que le provider Anthropic, par exemple.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=resolved_base_url)

        # Tokenizer réel si disponible pour ce modèle (voir _resolve_tokenizer).
        # Sinon, count_tokens() retombe sur une approximation documentée.
        self._tokenizer = self._resolve_tokenizer(model)

    def _resolve_tokenizer(self, model: str):
        """
        Tente de charger un tokenizer précis pour ce modèle via la lib
        `transformers` (utile pour les modèles open-weight type
        Llama/Qwen hébergés par Groq/NVIDIA). Retourne None si
        indisponible — count_tokens() utilisera alors une approximation.
        """
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(model)
        except Exception:
            return None

    def generate(self, messages: list[ChatMessage], max_tokens: int = 1000) -> CompletionResult:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        choice = response.choices[0]
        usage = response.usage

        return CompletionResult(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else self._estimate_tokens_for_messages(messages),
            output_tokens=usage.completion_tokens if usage else self.count_tokens(choice.message.content or ""),
        )

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        # Approximation universelle documentée : ~4 caractères par token.
        # Utilisée uniquement quand le tokenizer exact n'a pas pu être
        # chargé (modèle propriétaire non exposé via transformers, pas
        # d'accès réseau à HuggingFace, etc.).
        return max(1, len(text) // 4)

    def _estimate_tokens_for_messages(self, messages: list[ChatMessage]) -> int:
        return sum(self.count_tokens(m.content) for m in messages)