"""Sentinel monitoring for abuse, quota, and security events."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.platform.auth import AuthContext
from backend.platform.models import Notification, NotificationStatus, SecurityEvent, SecuritySeverity
from backend.platform.settings import PlatformSettings


@dataclass
class SentinelDecision:
    severity: str
    response_action: str
    notify_union_admins: bool = False
    notify_super_admins: bool = False


class SentinelService:
    def __init__(self, settings: PlatformSettings):
        self.settings = settings

    def classify(self, event_type: str, details: dict | None = None) -> SentinelDecision:
        details = details or {}
        if event_type in {"prompt_blocked", "cross_tenant_denied"}:
            return SentinelDecision("warning", "temporary_rate_limit", notify_super_admins=True)
        if event_type in {"quota_warning", "quota_exceeded"}:
            return SentinelDecision("warning", "union_notify", notify_union_admins=True)
        if event_type == "ingestion_review_required":
            return SentinelDecision("warning", "review_union_document", notify_union_admins=True)
        if event_type == "ingestion_review_escalated":
            return SentinelDecision("critical", "manual_review_super_admin")
        if event_type == "document_sensitive_data_flagged":
            return SentinelDecision("warning", "review_sensitive_document", notify_union_admins=True)
        if event_type == "document_prompt_injection_blocked":
            return SentinelDecision("critical", "quarantine_document_pending_super_admin", notify_union_admins=True, notify_super_admins=True)
        if event_type == "ingestion_failed":
            return SentinelDecision("warning", "investigate_ingestion_failure", notify_union_admins=True, notify_super_admins=True)
        if event_type in {"repeated_login_failures", "provider_secret_change"}:
            return SentinelDecision("critical", "escalate_super_admin", notify_super_admins=True)
        if details.get("risk_score", 0) >= 0.95:
            return SentinelDecision("critical", "temporary_cooldown", notify_super_admins=True)
        return SentinelDecision("info", "record_only")

    def record_event(
        self,
        db: Session | None,
        auth: AuthContext,
        *,
        event_type: str,
        details: dict | None = None,
    ) -> SentinelDecision:
        decision = self.classify(event_type, details)
        if db is None:
            return decision

        db.add(
            SecurityEvent(
                union_id=auth.union_id,
                user_id=auth.user_id,
                event_type=event_type,
                severity=SecuritySeverity(decision.severity),
                response_action=decision.response_action,
                details_json=details or {},
            )
        )
        if decision.notify_union_admins:
            db.add(
                Notification(
                    union_id=auth.union_id,
                    user_id=None,
                    channel="in_app",
                    subject=f"Union alert: {event_type}",
                    body=f"Sentinel detected {event_type}. Details: {details or {}}",
                    status=NotificationStatus.PENDING,
                )
            )
        if decision.notify_super_admins:
            db.add(
                Notification(
                    union_id=None,
                    user_id=None,
                    channel="in_app",
                    subject=f"Security alert: {event_type}",
                    body=f"Sentinel detected {event_type}. Details: {details or {}}",
                    status=NotificationStatus.PENDING,
                )
            )
        return decision
