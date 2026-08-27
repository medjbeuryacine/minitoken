"""
Adaptateur pour tous les providers "OpenAI-compatible" : Groq, OpenAI,
NVIDIA NIM, ou tout autre service exposant une API au format OpenAI
(/chat/completions).

Un seul adaptateur suffit pour n'importe quel provider de ce type — la
seule chose qui change d'un service à l'autre est la `base_url`. On ne
maintient pas de liste fermée de providers "connus" : le développeur
donne directement l'URL de base de son fournisseur (ex:
"https://api.groq.com/openai/v1"), et minitoken fonctionne avec, tant que
ce fournisseur respecte le format OpenAI.
"""

from minitoken.providers.base import ChatMessage, CompletionResult, LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str, provider_label: str | None = None, rate_limiter=None, extra_params: dict | None = None):
        """
        base_url : l'URL de base de votre fournisseur OpenAI-compatible
                   (ex: "https://api.groq.com/openai/v1",
                   "https://api.openai.com/v1",
                   "https://integrate.api.nvidia.com/v1", ou toute autre
                   API respectant le format OpenAI).
        provider_label : nom libre, uniquement informatif (logs, messages
                   d'erreur) — n'affecte pas le comportement.
        extra_params : paramètres additionnels envoyés tels quels
                   (extra_body) à chaque appel /chat/completions -- utile
                   pour activer une option propre à un fournisseur/modèle
                   précis (ex: un mode raisonnement/thinking), sans
                   jamais avoir à modifier ce fichier.
        """
        super().__init__(api_key=api_key, model=model, rate_limiter=rate_limiter, extra_params=extra_params)

        if not base_url:
            raise ValueError(
                "base_url est obligatoire : indiquez l'URL de base de votre "
                "fournisseur OpenAI-compatible (ex: 'https://api.groq.com/openai/v1')."
            )

        self.base_url = base_url
        self.provider_label = provider_label or base_url

        # Import différé : on évite de forcer la dépendance `openai` si le
        # développeur n'utilise que le provider Anthropic, par exemple.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        # Tokenizer réel si disponible pour ce modèle (voir _resolve_tokenizer).
        # Sinon, count_tokens() retombe sur une approximation documentée.
        self._tokenizer = self._resolve_tokenizer(model)
        self.last_rate_limit_headers = None
        
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
        if self._rate_limiter:
            estimated = self._estimate_tokens_for_messages(messages) + max_tokens
            check = self._rate_limiter.check_limit(estimated_tokens=estimated)
            if not check.allowed:
                from minitoken.token_budget.rate_limiter import RateLimitExceededError
                raise RateLimitExceededError(retry_after_seconds=check.retry_after_seconds)

        # with_raw_response donne accès aux headers HTTP en plus du JSON
        # parsé, pour capturer les vrais compteurs du fournisseur (Groq
        # les expose, d'autres non — on gère les deux cas).
        raw_response = self._client.chat.completions.with_raw_response.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            extra_body=self.extra_params or None,
        )
        response = raw_response.parse()
        headers = raw_response.headers

        rate_limit_headers = self._extract_rate_limit_headers(headers)

        choice = response.choices[0]
        usage = response.usage
        total_tokens = usage.total_tokens if usage else self._estimate_tokens_for_messages(messages) + max_tokens

        if self._rate_limiter:
            self._rate_limiter.record_call(tokens_used=total_tokens)

        self.last_rate_limit_headers = rate_limit_headers

        return CompletionResult(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else self._estimate_tokens_for_messages(messages),
            output_tokens=usage.completion_tokens if usage else self.count_tokens(choice.message.content or ""),
            provider_rate_limit_headers=rate_limit_headers,
        )

    def _extract_rate_limit_headers(self, headers) -> "ProviderRateLimitHeaders":
        """
        Lit les headers x-ratelimit-* standards (format Groq/OpenAI).
        Retourne un objet avec des champs None si un header est absent
        (fournisseur qui ne les expose pas).
        """
        from minitoken.providers.base import ProviderRateLimitHeaders

        def _get_int(name):
            val = headers.get(name)
            return int(val) if val is not None else None

        def _get_seconds(name):
            # Groq renvoie parfois "1.2s" ou "120ms" — on normalise en secondes.
            val = headers.get(name)
            if val is None:
                return None
            val = val.strip().lower()
            try:
                if val.endswith("ms"):
                    return float(val[:-2]) / 1000
                if val.endswith("s"):
                    return float(val[:-1])
                return float(val)
            except ValueError:
                return None

        return ProviderRateLimitHeaders(
            limit_requests=_get_int("x-ratelimit-limit-requests"),
            remaining_requests=_get_int("x-ratelimit-remaining-requests"),
            limit_tokens=_get_int("x-ratelimit-limit-tokens"),
            remaining_tokens=_get_int("x-ratelimit-remaining-tokens"),
            reset_requests_seconds=_get_seconds("x-ratelimit-reset-requests"),
            reset_tokens_seconds=_get_seconds("x-ratelimit-reset-tokens"),
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