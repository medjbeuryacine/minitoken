"""
Point d'entrée public de minitoken.

MinitokenClient assemble tout ce que le développeur a besoin :
  - la config (providers, DB, scopes)
  - les 4 niveaux de mémoire (short-term, summary, structured, vector)
  - le token budgeter

Deux méthodes principales couvrent le flux complet d'un échange :
  - get_context()     : à appeler AVANT la réponse principale, pour
                         obtenir le contexte optimisé à envoyer au LLM
  - record_exchange()  : à appeler APRÈS la réponse principale, en tâche
                         de fond (résumé + extraction de faits + embedding)
"""

import uuid

from minitoken.config import MinitokenConfig
from minitoken.database.repository import MinitokenRepository
from minitoken.memory import short_term, structured, summary
from minitoken.memory.vector import Embedder, get_embedder
from minitoken.memory.prompt_compression import Compressor, get_compressor
from minitoken.providers import ChatMessage, LLMProvider, build_provider
from minitoken.token_budget.rate_limiter import RateLimiter
from minitoken.token_budget.counter import ContextBundle, TokenReport, count_bundle_tokens
from minitoken.token_budget.trimmer import trim_to_budget


class MinitokenClient:
    def __init__(
        self,
        config: MinitokenConfig,
        *,
        repository: MinitokenRepository | None = None,
        token_counter_provider: LLMProvider | None = None,
        extraction_provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
    ):
        """
        Les paramètres optionnels (repository/providers/embedder) permettent
        d'injecter des implémentations alternatives — utile pour les tests,
        ou si un développeur veut fournir un provider personnalisé non
        couvert par build_provider(). Par défaut, tout est construit
        automatiquement à partir de `config`.
        """
        self.config = config

        self.repository = repository or MinitokenRepository(config)

        token_counter_limiter = (
            RateLimiter(repository=self.repository, provider_role="token_counter", config=config.token_counter_rate_limit)
            if config.token_counter_rate_limit else None
        )
        extraction_limiter = (
            RateLimiter(repository=self.repository, provider_role="extraction", config=config.extraction_rate_limit)
            if config.extraction_rate_limit else None
        )

        self.token_counter_provider = token_counter_provider or build_provider(
            provider_name=config.token_counter_provider,
            model=config.token_counter_model,
            api_key=config.token_counter_api_key,
            base_url=config.token_counter_base_url,
            rate_limiter=token_counter_limiter,
        )

        self.extraction_provider = extraction_provider or build_provider(
            provider_name=config.extraction_provider,
            model=config.extraction_model,
            api_key=config.extraction_api_key,
            base_url=config.extraction_base_url,
            rate_limiter=extraction_limiter,
        )

        self.embedder = embedder or get_embedder(config)
        self.compressor: Compressor = get_compressor(config, self.extraction_provider)

    def initialize(self) -> None:
        """À appeler une fois, au démarrage de l'application hôte : active
        pgvector, crée les 3 tables de minitoken si elles n'existent pas
        déjà, et ajoute la colonne embedding à la bonne dimension selon
        l'embedder configuré."""
        from minitoken.database.migrate import apply_migrations

        apply_migrations(repository=self.repository, embedder=self.embedder)

    def check_response_rate_limit(self, *, estimated_tokens: int):
        """
        Vérifie si un appel au LLM de réponse (celui configuré via
        token_counter_provider) est autorisé maintenant, sans jamais
        attendre. À appeler par l'application hôte AVANT de générer sa
        réponse (ex: avant d'invoquer votre graph LangGraph), si vous
        voulez respecter les limites configurées pour ce provider.

        Retourne un RateLimitCheckResult(allowed, retry_after_seconds).
        Si aucun rate limit n'est configuré pour token_counter_provider,
        retourne toujours allowed=True.
        """
        from minitoken.token_budget.rate_limiter import RateLimitCheckResult

        limiter = getattr(self.token_counter_provider, "_rate_limiter", None)
        if limiter is None:
            return RateLimitCheckResult(allowed=True)
        return limiter.check_limit(estimated_tokens=estimated_tokens)

    def get_rate_limit_status(self, *, provider_role: str = "token_counter") -> dict:
        """
        Retourne l'état de rate limit pour affichage (barres de
        progression, %). Combine :
        - les VRAIS headers du fournisseur (si disponibles, ex: Groq)
        - notre propre calcul Postgres (fallback universel), couvrant
          RPM/RPD/TPM/TPD, peu importe lesquels sont configurés — chacun
          avec limite, utilisé, restant, ET pourcentage.
        """
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import func

        provider = self.token_counter_provider if provider_role == "token_counter" else self.extraction_provider
        limiter = getattr(provider, "_rate_limiter", None)
        real_headers = getattr(provider, "last_rate_limit_headers", None)

        status = {"provider_role": provider_role, "source": "estimated"}

        if real_headers and real_headers.limit_tokens is not None:
            tokens_used = real_headers.limit_tokens - (real_headers.remaining_tokens or 0)
            status.update({
                "source": "provider_headers",
                "tokens_limit": real_headers.limit_tokens,
                "tokens_remaining": real_headers.remaining_tokens,
                "tokens_used": tokens_used,
                "tokens_percentage": round(100 * tokens_used / real_headers.limit_tokens, 1)
                    if real_headers.limit_tokens else None,
            })
            if real_headers.limit_requests is not None:
                requests_used = real_headers.limit_requests - (real_headers.remaining_requests or 0)
                status.update({
                    "requests_limit": real_headers.limit_requests,
                    "requests_remaining": real_headers.remaining_requests,
                    "requests_used": requests_used,
                    "requests_percentage": round(100 * requests_used / real_headers.limit_requests, 1)
                        if real_headers.limit_requests else None,
                })
            return status

        if not limiter:
            return status

        def _window_usage(window: timedelta, mode: str):
            window_start = datetime.now(timezone.utc) - window
            with limiter.repository._session() as session:
                table = limiter.repository.models.RateLimitEvent
                if mode == "count":
                    return (
                        session.query(table)
                        .filter(table.provider_role == limiter.provider_role, table.called_at >= window_start)
                        .count()
                    )
                return (
                    session.query(func.coalesce(func.sum(table.tokens_used), 0))
                    .filter(table.provider_role == limiter.provider_role, table.called_at >= window_start)
                    .scalar()
                )

        def _add_metric(prefix: str, limit: int, window: timedelta, mode: str):
            used = _window_usage(window, mode)
            status.update({
                f"{prefix}_limit": limit,
                f"{prefix}_used": used,
                f"{prefix}_remaining": max(0, limit - used),
                f"{prefix}_percentage": round(100 * used / limit, 1) if limit else None,
            })

        cfg = limiter.config

        if cfg.requests_per_minute:
            _add_metric("requests_per_minute", cfg.requests_per_minute, timedelta(minutes=1), "count")

        if cfg.requests_per_day:
            _add_metric("requests_per_day", cfg.requests_per_day, timedelta(days=1), "count")

        if cfg.tokens_per_minute:
            _add_metric("tokens_per_minute", cfg.tokens_per_minute, timedelta(minutes=1), "sum")

        if cfg.tokens_per_day:
            _add_metric("tokens_per_day", cfg.tokens_per_day, timedelta(days=1), "sum")

        return status

    def compress_user_message(self, message: str) -> str:
        """Compresse une question utilisateur avant envoi au LLM principal,
        selon compression_mode configuré. Retourne le texte inchangé si
        compression_mode='none' (défaut) — appel explicite et optionnel,
        jamais automatique dans get_context()."""
        return self.compressor.compress(message, self.config.compression_target_ratio)

    
    # ------------------------------------------------------------------
    # Avant la réponse principale : construire le contexte optimisé
    # ------------------------------------------------------------------

    def get_context(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        agent_scope: str,
        all_messages: list[ChatMessage],
        keep_recent_count: int = 8,
        vector_top_k: int = 3,
        budget_override: int | None = None,
        max_user_facts: int | None = None,
    ) -> tuple[ContextBundle, TokenReport]:
        """
        budget_override : si fourni, utilise ce budget au lieu de
        config.token_budget_max. Utile quand l'appelant a un system
        prompt fixe (non géré par minitoken) qu'il faut soustraire du
        budget total avant d'appeler get_context().
        """
        relevant_scopes = self._resolve_scopes(agent_scope)

        window = short_term.split_recent_messages(
            all_messages=all_messages, keep_recent_count=keep_recent_count
        )

        conversation_memory = self.repository.get_conversation_memory(conversation_id)
        existing_summary = conversation_memory.summary if conversation_memory else ""

        effective_max_facts = max_user_facts if max_user_facts is not None else self.config.max_user_facts
        user_facts = self.repository.get_user_facts(
            user_id=user_id, scopes=relevant_scopes, limit=effective_max_facts
        )

        vector_memories: list[str] = []
        if window.recent_messages:
            last_message = window.recent_messages[-1].content
            query_embedding = self.embedder.embed(last_message)
            rows = self.repository.search_similar_embeddings(
                user_id=user_id,
                scopes=relevant_scopes,
                query_embedding=query_embedding,
                top_k=vector_top_k,
            )
            vector_memories = [row.content for row in rows]

        bundle = ContextBundle(
            recent_messages=window.recent_messages,
            summary=existing_summary,
            structured_facts=[f.fact for f in user_facts],
            vector_memories=vector_memories,
        )

        effective_budget = budget_override if budget_override is not None else self.config.token_budget_max

        return trim_to_budget(
            provider=self.token_counter_provider,
            bundle=bundle,
            budget_max=effective_budget,
        )

    # ------------------------------------------------------------------
    # Après la réponse principale : mise à jour de la mémoire (tâche de fond)
    # ------------------------------------------------------------------

    def record_exchange(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        agent_scope: str,
        all_messages: list[ChatMessage],
        user_message: str,
        assistant_response: str,
        keep_recent_count: int = 8,
        resummarize_every: int = 5,
        summary_max_tokens: int = 500,
    ) -> None:
        """
        À appeler après avoir envoyé la réponse à l'utilisateur, de
        préférence de façon asynchrone/non bloquante côté appelant. Met à
        jour le résumé si nécessaire, extrait les faits durables, et
        stocke l'échange en mémoire vectorielle.

        N'échoue jamais bruyamment sur l'extraction : si le LLM
        d'extraction échoue, l'échange n'est simplement pas ajouté à
        user_memory, sans lever d'exception qui remonterait jusqu'à
        l'appelant (voir memory/structured.py).
        """
        try:
            conversation_memory = self.repository.get_conversation_memory(conversation_id)
            message_count_at_last_summary = (
                conversation_memory.message_count_at_last_summary if conversation_memory else 0
            )
            existing_summary = conversation_memory.summary if conversation_memory else ""
            existing_state = conversation_memory.conversation_state if conversation_memory else {}

            if short_term.needs_summarization(
                total_message_count=len(all_messages),
                message_count_at_last_summary=message_count_at_last_summary,
                keep_recent_count=keep_recent_count,
                resummarize_every=resummarize_every,
            ):
                window = short_term.split_recent_messages(
                    all_messages=all_messages, keep_recent_count=keep_recent_count
                )
                new_summary = summary.update_summary(
                    provider=self.extraction_provider,
                    existing_summary=existing_summary,
                    messages_to_summarize=window.messages_to_summarize,
                )
                if self.config.compression_mode != "none":
                    if self.token_counter_provider.count_tokens(new_summary) > summary_max_tokens:
                        new_summary = self.compressor.compress(
                            new_summary, self.config.compression_target_ratio
                        )
                else:
                    new_summary = summary.recompress_if_too_long(
                        provider=self.extraction_provider,
                        summary=new_summary,
                        max_tokens=summary_max_tokens,
                    )

                self.repository.upsert_conversation_memory(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    summary=new_summary,
                    conversation_state=existing_state,
                    message_count_at_last_summary=len(all_messages),
                )
        except Exception:
            # Le résumé n'a pas pu être mis à jour (panne LLM, réseau...).
            # On continue quand même — ce n'est pas critique, la
            # conversation reste utilisable, le résumé sera retenté au
            # prochain échange.
            pass

        try:
            extracted_facts = structured.extract_facts(
                provider=self.extraction_provider,
                current_agent_scope=agent_scope,
                user_message=user_message,
                assistant_response=assistant_response,
            )
            for fact in extracted_facts:
                self.repository.add_user_fact(
                    user_id=user_id,
                    fact=fact.fact,
                    scope=fact.scope,
                    category=fact.category,
                    source_conversation_id=conversation_id,
                )
        except Exception:
            # L'extraction de faits n'a pas pu se faire (panne LLM...).
            # On continue quand même, même logique que ci-dessus.
            pass
        try:
            exchange_text = f"User: {user_message}\nAssistant: {assistant_response}"
            embedding = self.embedder.embed(exchange_text)
            self.repository.add_memory_embedding(
                user_id=user_id,
                content=exchange_text,
                scope=agent_scope,
                embedding=embedding,
                conversation_id=conversation_id,
            )
        except Exception:
            # Le stockage vectoriel a échoué (embedder local en panne,
            # base indisponible...). On continue quand même.
            pass

    # ------------------------------------------------------------------

    def _resolve_scopes(self, agent_scope: str) -> list[str]:
        """"global" + le scope de l'agent courant — jamais les scopes
        des autres agents."""
        if agent_scope == "global":
            return ["global"]
        return ["global", agent_scope]