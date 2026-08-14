"""
Rate limiting générique (RPM/RPD/TPM/TPD), partagé via Postgres pour
fonctionner correctement même avec plusieurs instances/workers de
l'application hôte.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func


class RateLimitExceededError(Exception):
    """Levée quand check_limit() refuse un appel. Ne bloque jamais elle-
    même — c'est à l'appelant de décider quoi faire (attendre, informer
    l'utilisateur, abandonner silencieusement)."""

    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit dépassé, réessayez dans {retry_after_seconds:.1f}s")


@dataclass
class RateLimitConfig:
    requests_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    tokens_per_day: Optional[int] = None

@dataclass
class RateLimitCheckResult:
    allowed: bool
    retry_after_seconds: float = 0.0

class RateLimiter:
    def __init__(self, *, repository, provider_role: str, config: RateLimitConfig):
        self.repository = repository
        self.provider_role = provider_role
        self.config = config

    def wait_if_needed(self, *, estimated_tokens: int) -> None:
        """Vérifie chaque seuil configuré, attend si nécessaire avant de
        laisser passer l'appel."""
        now = datetime.now(timezone.utc)

        checks = [
            (self.config.requests_per_minute, timedelta(minutes=1), "count"),
            (self.config.requests_per_day, timedelta(days=1), "count"),
            (self.config.tokens_per_minute, timedelta(minutes=1), "sum", estimated_tokens),
            (self.config.tokens_per_day, timedelta(days=1), "sum", estimated_tokens),
        ]

        for check in checks:
            limit = check[0]
            if limit is None:
                continue
            window = check[1]
            mode = check[2]
            extra = check[3] if len(check) > 3 else 0

            wait_seconds = self._compute_wait_seconds(
                window=window, limit=limit, mode=mode, additional=extra, now=now
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = datetime.now(timezone.utc)  # revérifie après l'attente

    def check_limit(self, *, estimated_tokens: int) -> RateLimitCheckResult:
        """
        Vérifie si l'appel est autorisé MAINTENANT, sans jamais attendre.
        À utiliser dans un chemin synchrone visible par l'utilisateur
        (contrairement à wait_if_needed, prévu pour un usage en tâche de
        fond). Retourne le temps d'attente nécessaire si refusé, pour que
        l'appelant puisse l'afficher clairement (ex: "réessaie dans 58s"),
        plutôt que de faire patienter silencieusement.
        """
        now = datetime.now(timezone.utc)

        checks = [
            (self.config.requests_per_minute, timedelta(minutes=1), "count"),
            (self.config.requests_per_day, timedelta(days=1), "count"),
            (self.config.tokens_per_minute, timedelta(minutes=1), "sum", estimated_tokens),
            (self.config.tokens_per_day, timedelta(days=1), "sum", estimated_tokens),
        ]

        max_wait = 0.0
        for check in checks:
            limit = check[0]
            if limit is None:
                continue
            window = check[1]
            mode = check[2]
            extra = check[3] if len(check) > 3 else 0

            wait_seconds = self._compute_wait_seconds(
                window=window, limit=limit, mode=mode, additional=extra, now=now
            )
            max_wait = max(max_wait, wait_seconds)

        if max_wait > 0:
            return RateLimitCheckResult(allowed=False, retry_after_seconds=max_wait)

        return RateLimitCheckResult(allowed=True)

    def record_call(self, *, tokens_used: int) -> None:
        with self.repository._session() as session:
            record = self.repository.models.RateLimitEvent(
                provider_role=self.provider_role,
                tokens_used=tokens_used,
            )
            session.add(record)
            session.commit()

    def check_and_reserve(self, *, estimated_tokens: int):
        """
        Version ATOMIQUE de check_limit() suivi d'une RÉSERVATION
        immédiate du quota estimé — à utiliser à la place de check_limit()
        partout où l'appel réel (ex: appel LLM) se fait APRÈS le check,
        potentiellement plusieurs secondes plus tard.

        check_limit() + record_call() séparés souffrent d'une race
        condition sous forte concurrence : plusieurs threads peuvent lire
        le même compteur avant qu'aucun n'ait encore écrit son événement,
        et donc tous croire que la limite n'est pas encore atteinte.

        Cette méthode utilise un verrou consultatif Postgres
        (pg_advisory_xact_lock), scopé sur provider_role, qui sérialise
        tous les appels concurrents pour ce même rôle : un seul thread/
        process à la fois peut faire "vérifier puis réserver", les
        autres attendent leur tour avant de vérifier à leur tour (avec un
        compteur à jour incluant la réservation qui vient d'être faite).

        La réservation se fait avec estimated_tokens (avant l'appel LLM
        réel, dont le coût exact n'est pas encore connu) — voir
        update_reservation() pour corriger le nombre réel de tokens une
        fois l'appel terminé, sans avoir besoin de garder un verrou ouvert
        pendant tout l'appel LLM (qui peut prendre plusieurs secondes).

        Retourne (RateLimitCheckResult, reservation_id | None). Le
        reservation_id est None si allowed=False (rien n'a été réservé).
        """
        with self.repository._session() as session:
            # Verrou exclusif scopé à ce provider_role — hashtext() donne
            # un entier stable à partir de la string, requis par
            # pg_advisory_xact_lock qui attend un bigint.
            session.execute(
                func.pg_advisory_xact_lock(func.hashtext(self.provider_role))
            )

            now = datetime.now(timezone.utc)
            checks = [
                (self.config.requests_per_minute, timedelta(minutes=1), "count"),
                (self.config.requests_per_day, timedelta(days=1), "count"),
                (self.config.tokens_per_minute, timedelta(minutes=1), "sum", estimated_tokens),
                (self.config.tokens_per_day, timedelta(days=1), "sum", estimated_tokens),
            ]

            max_wait = 0.0
            for check in checks:
                limit = check[0]
                if limit is None:
                    continue
                window = check[1]
                mode = check[2]
                extra = check[3] if len(check) > 3 else 0

                wait_seconds = self._compute_wait_seconds_in_session(
                    session=session, window=window, limit=limit, mode=mode,
                    additional=extra, now=now,
                )
                max_wait = max(max_wait, wait_seconds)

            if max_wait > 0:
                # Rien réservé — le verrou est relâché automatiquement à
                # la fin du bloc `with` (rollback implicite).
                return RateLimitCheckResult(allowed=False, retry_after_seconds=max_wait), None

            record = self.repository.models.RateLimitEvent(
                provider_role=self.provider_role,
                tokens_used=estimated_tokens,
            )
            session.add(record)
            session.commit()  # relâche aussi le verrou advisory
            session.refresh(record)

            return RateLimitCheckResult(allowed=True), record.id

    def update_reservation(self, *, reservation_id, actual_tokens: int) -> None:
        """
        Corrige le nombre de tokens réellement consommés sur une
        réservation faite par check_and_reserve(), une fois l'appel LLM
        réel terminé et son coût exact connu. N'a pas besoin de verrou :
        c'est une simple mise à jour d'une ligne déjà existante et déjà
        comptée par les futurs checks, peu importe la valeur exacte au
        moment de la mise à jour.
        """
        if reservation_id is None:
            return
        with self.repository._session() as session:
            table = self.repository.models.RateLimitEvent
            session.query(table).filter(table.id == reservation_id).update(
                {"tokens_used": actual_tokens}
            )
            session.commit()

    def _compute_wait_seconds(self, *, window: timedelta, limit: int, mode: str, additional: int, now: datetime) -> float:
        with self.repository._session() as session:
            return self._compute_wait_seconds_in_session(
                session=session, window=window, limit=limit, mode=mode,
                additional=additional, now=now,
            )

    def _compute_wait_seconds_in_session(self, *, session, window: timedelta, limit: int, mode: str, additional: int, now: datetime) -> float:
        """Même logique que _compute_wait_seconds, mais réutilise une
        session déjà ouverte (nécessaire pour check_and_record, où le
        calcul doit se faire DANS la transaction verrouillée)."""
        window_start = now - window
        table = self.repository.models.RateLimitEvent

        if mode == "count":
            current = (
                session.query(table)
                .filter(table.provider_role == self.provider_role, table.called_at >= window_start)
                .count()
            )
            would_exceed = (current + 1) > limit
        else:  # "sum"
            current = (
                session.query(func.coalesce(func.sum(table.tokens_used), 0))
                .filter(table.provider_role == self.provider_role, table.called_at >= window_start)
                .scalar()
            )
            would_exceed = (current + additional) > limit

        if not would_exceed:
            return 0.0

        oldest = (
            session.query(func.min(table.called_at))
            .filter(table.provider_role == self.provider_role, table.called_at >= window_start)
            .scalar()
        )
        if oldest is None:
            return 0.0

        free_at = oldest + window
        wait = (free_at - now).total_seconds()
        return max(0.0, wait)