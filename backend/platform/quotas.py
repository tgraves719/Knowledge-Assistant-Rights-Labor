"""Quota metering and enforcement for union-level usage controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.platform.auth import AuthContext
from backend.platform.models import QuotaPolicy, UsageEvent
from backend.platform.settings import PlatformSettings


@dataclass
class QuotaDecision:
    allowed: bool
    reason: str | None = None
    warn: bool = False
    usage_snapshot: dict | None = None


class QuotaService:
    def __init__(self, settings: PlatformSettings):
        self.settings = settings

    def _default_policy(self, union_id: str) -> QuotaPolicy:
        return QuotaPolicy(
            union_id=union_id,
            requests_per_day=self.settings.hard_cap_default_requests_per_day,
            tokens_per_day=self.settings.hard_cap_default_tokens_per_day,
            cost_usd_per_day=self.settings.hard_cap_default_cost_usd_per_day,
            per_user_requests_per_hour=self.settings.request_rate_limit_per_minute,
            warn_threshold_ratio=0.8,
            is_paused=False,
        )

    def get_or_create_policy(self, db: Session, union_id: str) -> QuotaPolicy:
        policy = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == union_id))
        if policy:
            return policy
        policy = self._default_policy(union_id)
        db.add(policy)
        db.flush()
        return policy

    def check_query(self, db: Session | None, auth: AuthContext, estimated_tokens: int) -> QuotaDecision:
        if db is None or not auth.union_id or auth.is_super_admin:
            return QuotaDecision(allowed=True)

        policy = self.get_or_create_policy(db, auth.union_id)
        if policy.is_paused:
            return QuotaDecision(allowed=False, reason="Union access is temporarily paused by an administrator.")

        now = datetime.utcnow()
        since_day = now - timedelta(days=1)
        since_hour = now - timedelta(hours=1)

        totals = db.execute(
            select(
                func.coalesce(func.sum(UsageEvent.request_count), 0),
                func.coalesce(func.sum(UsageEvent.token_count), 0),
                func.coalesce(func.sum(UsageEvent.estimated_cost_usd), 0.0),
            ).where(UsageEvent.union_id == auth.union_id, UsageEvent.created_at >= since_day)
        ).one()
        hourly_requests = db.scalar(
            select(func.coalesce(func.sum(UsageEvent.request_count), 0)).where(
                UsageEvent.union_id == auth.union_id,
                UsageEvent.user_id == auth.user_id,
                UsageEvent.created_at >= since_hour,
            )
        ) or 0

        reqs, tokens, cost = int(totals[0]), int(totals[1]), float(totals[2])
        if reqs + 1 > policy.requests_per_day:
            return QuotaDecision(allowed=False, reason="Union daily request cap exceeded.")
        if tokens + estimated_tokens > policy.tokens_per_day:
            return QuotaDecision(allowed=False, reason="Union daily token cap exceeded.")
        if hourly_requests + 1 > policy.per_user_requests_per_hour:
            return QuotaDecision(allowed=False, reason="Per-user hourly request cap exceeded.")

        warn = (
            (reqs + 1) / max(policy.requests_per_day, 1) >= policy.warn_threshold_ratio
            or (tokens + estimated_tokens) / max(policy.tokens_per_day, 1) >= policy.warn_threshold_ratio
            or cost / max(policy.cost_usd_per_day, 0.01) >= policy.warn_threshold_ratio
        )
        snapshot = {
            "requests_today": reqs,
            "tokens_today": tokens,
            "cost_today": cost,
            "requests_cap": policy.requests_per_day,
            "tokens_cap": policy.tokens_per_day,
            "cost_cap": policy.cost_usd_per_day,
        }
        return QuotaDecision(allowed=True, warn=warn, usage_snapshot=snapshot)

    def record_usage(
        self,
        db: Session | None,
        auth: AuthContext,
        *,
        route: str,
        token_count: int,
        estimated_cost_usd: float,
        metadata: dict | None = None,
    ) -> None:
        if db is None or not auth.union_id:
            return
        db.add(
            UsageEvent(
                union_id=auth.union_id,
                user_id=auth.user_id,
                route=route,
                request_count=1,
                token_count=max(0, int(token_count)),
                estimated_cost_usd=max(0.0, float(estimated_cost_usd)),
                metadata_json=metadata or {},
            )
        )

