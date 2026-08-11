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

    def _compute_wait_seconds(self, *, window: timedelta, limit: int, mode: str, additional: int, now: datetime) -> float:
        window_start = now - window

        with self.repository._session() as session:
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

            # Trouve le plus ancien événement encore dans la fenêtre, pour
            # savoir dans combien de temps il en "sortira" et libérera du quota.
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