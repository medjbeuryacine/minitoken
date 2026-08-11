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

        self.token_counter_provider = token_counter_provider or build_provider(
            provider_name=config.token_counter_provider,
            model=config.token_counter_model,
            api_key=config.token_counter_api_key,
            base_url=config.token_counter_base_url,
        )

        self.extraction_provider = extraction_provider or build_provider(
            provider_name=config.extraction_provider,
            model=config.extraction_model,
            api_key=config.extraction_api_key,
            base_url=config.extraction_base_url,
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

        user_facts = self.repository.get_user_facts(user_id=user_id, scopes=relevant_scopes)

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