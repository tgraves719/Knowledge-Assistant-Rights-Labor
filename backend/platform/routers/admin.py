"""Admin and tenant management endpoints for the production foundation."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, insert, or_, select

from backend.platform.db import apply_request_context, apply_service_bootstrap_context
from backend.platform.deps import get_auth_context, get_container, get_db, require_roles
from backend.platform.inference import InferenceConfig, test_inference_config
from backend.platform.models import (
    AuditEvent,
    AuthSession,
    Chat,
    ChunkEmbedding,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    InviteAudience,
    InviteCode,
    LocalAuthCredential,
    Message,
    Notification,
    NotificationStatus,
    ProviderConfig,
    QuotaPolicy,
    RawQueryRecord,
    RawQueryStorageMode,
    Role,
    SecurityEvent,
    SecuritySeverity,
    TelemetryEvent,
    TrackingPolicy,
    Union,
    UnionMembership,
    User,
    UsageEvent,
    UserTrackingPreference,
)
from backend.platform.queueing import estimate_ingestion_runtime_seconds, ingestion_job_priority
from backend.platform.worker import process_pending_ingestion_jobs


router = APIRouter(prefix="/api/admin", tags=["admin"])

ROLE_ORDER = {
    Role.USER.value: 0,
    Role.STEWARD_ADMIN.value: 1,
    Role.UNION_ADMIN.value: 2,
    Role.SUPER_ADMIN.value: 3,
}


def _role_rank(role: Role | str | None) -> int:
    normalized = str(role.value if isinstance(role, Role) else role or "").strip().lower()
    return ROLE_ORDER.get(normalized, -1)


def _can_manage_role(auth, target_role: Role | str | None) -> bool:
    if getattr(auth, "is_super_admin", False):
        return True
    return _role_rank(target_role) <= _role_rank(getattr(auth, "role", None))


def _require_union_scope(auth, union_id: str) -> None:
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")


def _serialize_user_row(user: User, membership: UnionMembership, credential: LocalAuthCredential | None) -> dict:
    return {
        "user_id": user.id,
        "membership_id": membership.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": membership.role.value,
        "is_active": membership.is_active and user.is_active,
        "membership_active": membership.is_active,
        "user_active": user.is_active,
        "username": credential.username if credential is not None else None,
        "has_local_auth": credential is not None and credential.is_active,
        "created_at": user.created_at.isoformat(),
    }


def _user_anonymized_keys(telemetry, *, user_id: str, union_ids, session_ids) -> set[str]:
    """Reconstruct every ``anonymized_user_key`` that could have been written for a user.

    In anonymized tracking mode telemetry/raw-query rows store no ``user_id`` — only the
    HMAC ``anonymized_user_key`` keyed on ``(user_id, union_id, session_id)`` and the plaintext
    ``session_id`` (the ``AuthSession.id`` for server-issued events). To delete those rows on
    purge we recompute the candidate keys from the user's known unions and sessions.
    """
    if telemetry is None:
        return set()
    candidate_union_ids = set(union_ids) | {None}
    candidate_session_ids = set(session_ids) | {None}
    keys: set[str] = set()
    for candidate_union_id in candidate_union_ids:
        for candidate_session_id in candidate_session_ids:
            key = telemetry.anonymized_user_key(
                user_id=user_id,
                union_id=candidate_union_id,
                session_id=candidate_session_id,
            )
            if key:
                keys.add(key)
    return keys


def _emit_identified_raw_query_signal_if_enabled(db, *, union_id, actor_user_id, prior_mode, new_mode) -> None:
    """Emit a dedicated SecurityEvent when raw-query storage transitions to identified mode.

    ``enabled_identified`` is the single most sensitive tracking setting (raw member queries
    stored against a real ``user_id``). It already writes an AuditEvent; this adds an
    unmistakable warning-severity SecurityEvent on the *enable transition* so it stands out in
    the security trail. Re-saving an already-identified policy emits nothing.
    """
    identified = RawQueryStorageMode.ENABLED_IDENTIFIED.value
    if new_mode == identified and prior_mode != identified:
        db.add(
            SecurityEvent(
                union_id=union_id,
                user_id=actor_user_id,
                event_type="raw_query_identified_enabled",
                severity=SecuritySeverity.WARNING,
                response_action="audit",
                details_json={
                    "scope": "global" if union_id is None else "union_override",
                    "union_id": union_id,
                    "prior_raw_query_storage_mode": prior_mode,
                },
            )
        )


def _purge_user_records(db, *, user_id: str, union_id: str | None = None, global_scope: bool = False, telemetry=None) -> None:
    scoped = bool(union_id) and not global_scope

    # Capture the user's sessions and unions *before* deleting anything so we can reconstruct
    # the anonymized telemetry keys that no longer carry a user_id.
    session_stmt = select(AuthSession).where(AuthSession.user_id == user_id)
    if scoped:
        session_stmt = session_stmt.where(AuthSession.union_id == union_id)
    user_sessions = list(db.scalars(session_stmt).all())
    session_ids = {session.id for session in user_sessions}
    if scoped:
        candidate_union_ids = {union_id}
    else:
        candidate_union_ids = set(
            db.scalars(select(UnionMembership.union_id).where(UnionMembership.user_id == user_id)).all()
        )
        candidate_union_ids |= {session.union_id for session in user_sessions}
        candidate_union_ids |= {union_id}
    anonymized_keys = _user_anonymized_keys(
        telemetry,
        user_id=user_id,
        union_ids=candidate_union_ids,
        session_ids=session_ids,
    )

    chat_stmt = select(Chat.id).where(Chat.user_id == user_id)
    if scoped:
        chat_stmt = chat_stmt.where(Chat.union_id == union_id)
    chat_ids = list(db.scalars(chat_stmt).all())
    if chat_ids:
        db.query(Message).filter(Message.chat_id.in_(chat_ids)).delete(synchronize_session=False)
    chat_delete = db.query(Chat).filter(Chat.user_id == user_id)
    usage_delete = db.query(UsageEvent).filter(UsageEvent.user_id == user_id)
    security_delete = db.query(SecurityEvent).filter(SecurityEvent.user_id == user_id)
    notification_delete = db.query(Notification).filter(Notification.user_id == user_id)
    session_delete = db.query(AuthSession).filter(AuthSession.user_id == user_id)
    preference_delete = db.query(UserTrackingPreference).filter(UserTrackingPreference.user_id == user_id)
    document_update = db.query(Document).filter(Document.uploaded_by_user_id == user_id)
    ingestion_update = db.query(IngestionJob).filter(IngestionJob.requested_by_user_id == user_id)

    if scoped:
        chat_delete = chat_delete.filter(Chat.union_id == union_id)
        usage_delete = usage_delete.filter(UsageEvent.union_id == union_id)
        security_delete = security_delete.filter(SecurityEvent.union_id == union_id)
        notification_delete = notification_delete.filter(Notification.union_id == union_id)
        session_delete = session_delete.filter(AuthSession.union_id == union_id)
        preference_delete = preference_delete.filter(UserTrackingPreference.union_id == union_id)
        document_update = document_update.filter(Document.union_id == union_id)
        ingestion_update = ingestion_update.filter(IngestionJob.union_id == union_id)

    chat_delete.delete(synchronize_session=False)
    usage_delete.delete(synchronize_session=False)
    security_delete.delete(synchronize_session=False)
    notification_delete.delete(synchronize_session=False)
    session_delete.delete(synchronize_session=False)
    preference_delete.delete(synchronize_session=False)
    document_update.update({Document.uploaded_by_user_id: None}, synchronize_session=False)
    ingestion_update.update({IngestionJob.requested_by_user_id: None}, synchronize_session=False)

    # Telemetry and raw-query rows: match by direct user_id (identified mode), by reconstructed
    # anonymized key, and by plaintext session_id (server-issued anonymized events).
    for model in (TelemetryEvent, RawQueryRecord):
        conditions = [model.user_id == user_id]
        if session_ids:
            conditions.append(model.session_id.in_(session_ids))
        if anonymized_keys:
            conditions.append(model.anonymized_user_key.in_(anonymized_keys))
        telemetry_delete = db.query(model).filter(or_(*conditions))
        if scoped:
            telemetry_delete = telemetry_delete.filter(model.union_id == union_id)
        telemetry_delete.delete(synchronize_session=False)

    if global_scope:
        db.query(AuditEvent).filter(AuditEvent.actor_user_id == user_id).delete(synchronize_session=False)
        db.query(UnionMembership).filter(UnionMembership.user_id == user_id).delete(synchronize_session=False)
        db.query(LocalAuthCredential).filter(LocalAuthCredential.user_id == user_id).delete(synchronize_session=False)
        user = db.get(User, user_id)
        if user is not None:
            db.delete(user)


def _usage_warning_level(*, requests_ratio: float, tokens_ratio: float, cost_ratio: float, paused: bool) -> str:
    if paused:
        return "paused"
    peak_ratio = max(requests_ratio, tokens_ratio, cost_ratio)
    if peak_ratio >= 1:
        return "limit_reached"
    if peak_ratio >= 0.8:
        return "warning"
    return "healthy"


def _slugify_union_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:120] or "union"


def _generate_unique_union_slug(db, name: str, requested_slug: str | None = None) -> str:
    base = _slugify_union_name(requested_slug or name)
    candidate = base
    suffix = 2
    while db.scalar(select(Union).where(Union.slug == candidate)) is not None:
        candidate = f"{base[:110]}-{suffix}"
        suffix += 1
    return candidate


def _generate_unique_union_local_id(db) -> str:
    while True:
        candidate = f"union-{uuid.uuid4().hex[:12]}"
        if db.scalar(select(Union).where(Union.union_local_id == candidate)) is None:
            return candidate


def _build_platform_operations_summary(db) -> dict:
    unions = db.scalars(select(Union).order_by(Union.name.asc())).all()
    quotas = {item.union_id: item for item in db.scalars(select(QuotaPolicy)).all()}
    providers = {item.union_id: item for item in db.scalars(select(ProviderConfig).where(ProviderConfig.is_active.is_(True))).all()}
    memberships = db.scalars(select(UnionMembership).where(UnionMembership.is_active.is_(True))).all()
    users = {item.id: item for item in db.scalars(select(User)).all()}
    notifications = db.scalars(select(Notification).where(Notification.status == NotificationStatus.PENDING)).all()
    usage_events = db.scalars(select(UsageEvent)).all()

    now = datetime.utcnow()
    day_cutoff = now - timedelta(days=1)
    hour_cutoff = now - timedelta(hours=1)

    usage_by_union: dict[str, dict] = {}
    for item in usage_events:
        bucket = usage_by_union.setdefault(
            item.union_id,
            {
                "requests_last_24h": 0,
                "tokens_last_24h": 0,
                "estimated_cost_last_24h": 0.0,
                "requests_last_hour": 0,
            },
        )
        if item.created_at >= day_cutoff:
            bucket["requests_last_24h"] += int(item.request_count or 0)
            bucket["tokens_last_24h"] += int(item.token_count or 0)
            bucket["estimated_cost_last_24h"] += float(item.estimated_cost_usd or 0.0)
        if item.created_at >= hour_cutoff:
            bucket["requests_last_hour"] += int(item.request_count or 0)

    admins_by_union: dict[str, list[dict]] = {}
    for membership in memberships:
        if membership.role not in {Role.UNION_ADMIN, Role.SUPER_ADMIN}:
            continue
        user = users.get(membership.user_id)
        if user is None:
            continue
        admins_by_union.setdefault(membership.union_id, []).append(
            {
                "user_id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": membership.role.value,
            }
        )

    pending_notification_counts: dict[str, int] = {}
    for notification in notifications:
        if notification.union_id:
            pending_notification_counts[notification.union_id] = pending_notification_counts.get(notification.union_id, 0) + 1

    items = []
    for union in unions:
        quota = quotas.get(union.id)
        provider = providers.get(union.id)
        usage = usage_by_union.get(
            union.id,
            {
                "requests_last_24h": 0,
                "tokens_last_24h": 0,
                "estimated_cost_last_24h": 0.0,
                "requests_last_hour": 0,
            },
        )
        requests_ratio = (usage["requests_last_24h"] / quota.requests_per_day) if quota and quota.requests_per_day else 0
        tokens_ratio = (usage["tokens_last_24h"] / quota.tokens_per_day) if quota and quota.tokens_per_day else 0
        cost_ratio = (usage["estimated_cost_last_24h"] / quota.cost_usd_per_day) if quota and quota.cost_usd_per_day else 0
        provider_status = "configured" if provider and provider.encrypted_api_key else "missing"
        items.append(
            {
                "union_id": union.id,
                "slug": union.slug,
                "name": union.name,
                "is_active": union.is_active,
                "usage": {
                    **usage,
                    "warning_level": _usage_warning_level(
                        requests_ratio=requests_ratio,
                        tokens_ratio=tokens_ratio,
                        cost_ratio=cost_ratio,
                        paused=bool(quota.is_paused) if quota else False,
                    ),
                    "requests_ratio": round(requests_ratio, 4),
                    "tokens_ratio": round(tokens_ratio, 4),
                    "cost_ratio": round(cost_ratio, 4),
                },
                "quota": None
                if quota is None
                else {
                    "requests_per_day": quota.requests_per_day,
                    "tokens_per_day": quota.tokens_per_day,
                    "cost_usd_per_day": quota.cost_usd_per_day,
                    "per_user_requests_per_hour": quota.per_user_requests_per_hour,
                    "warn_threshold_ratio": quota.warn_threshold_ratio,
                    "is_paused": quota.is_paused,
                },
                "provider_health": {
                    "status": provider_status,
                    "provider_name": provider.provider_name if provider is not None else None,
                    "model_name": provider.model_name if provider is not None else None,
                    "has_api_key": bool(provider.encrypted_api_key) if provider is not None else False,
                },
                "admins": admins_by_union.get(union.id, []),
                "pending_notifications": pending_notification_counts.get(union.id, 0),
            }
        )

    return {
        "captured_at": now.isoformat(),
        "items": items,
        "summary": {
            "warning_unions": sum(1 for item in items if item["usage"]["warning_level"] in {"warning", "limit_reached"}),
            "paused_unions": sum(1 for item in items if item["quota"] and item["quota"]["is_paused"]),
            "provider_issues": sum(1 for item in items if item["provider_health"]["status"] != "configured"),
            "unions_without_admins": sum(1 for item in items if not item["admins"]),
        },
    }


def _build_union_export_bundle(db, union_id: str) -> dict:
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    provider = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == union_id, ProviderConfig.is_active.is_(True)))
    quota = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == union_id))
    memberships = db.scalars(select(UnionMembership).where(UnionMembership.union_id == union_id)).all()
    users = {item.id: item for item in db.scalars(select(User)).all()}
    documents = db.scalars(select(Document).where(Document.union_id == union_id).order_by(Document.created_at.desc())).all()
    audits = db.scalars(select(AuditEvent).where(AuditEvent.union_id == union_id).order_by(AuditEvent.created_at.desc()).limit(500)).all()
    notifications = db.scalars(select(Notification).where(Notification.union_id == union_id).order_by(Notification.created_at.desc()).limit(200)).all()
    security_events = db.scalars(select(SecurityEvent).where(SecurityEvent.union_id == union_id).order_by(SecurityEvent.created_at.desc()).limit(200)).all()

    return {
        "exported_at": datetime.utcnow().isoformat(),
        "union": {
            "id": union.id,
            "slug": union.slug,
            "name": union.name,
            "union_local_id": union.union_local_id,
            "is_active": union.is_active,
            "message_retention_enabled": union.message_retention_enabled,
            "metadata": union.metadata_json,
        },
        "quota": None if quota is None else {
            "requests_per_day": quota.requests_per_day,
            "tokens_per_day": quota.tokens_per_day,
            "cost_usd_per_day": quota.cost_usd_per_day,
            "per_user_requests_per_hour": quota.per_user_requests_per_hour,
            "warn_threshold_ratio": quota.warn_threshold_ratio,
            "is_paused": quota.is_paused,
        },
        "provider": None if provider is None else {
            "provider_name": provider.provider_name,
            "model_name": provider.model_name,
            "config": provider.config_json,
            "has_api_key": bool(provider.encrypted_api_key),
        },
        "memberships": [
            {
                "user_id": membership.user_id,
                "role": membership.role.value,
                "is_active": membership.is_active,
                "user": {
                    "full_name": users.get(membership.user_id).full_name if users.get(membership.user_id) else None,
                    "email": users.get(membership.user_id).email if users.get(membership.user_id) else None,
                },
            }
            for membership in memberships
        ],
        "documents": [
            {
                "id": document.id,
                "title": document.title,
                "content_type": document.content_type,
                "bytes_size": document.bytes_size,
                "status": document.status.value,
                "metadata": document.metadata_json,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ],
        "audit_events": [
            {
                "event_type": audit.event_type,
                "event_payload": audit.event_payload,
                "created_at": audit.created_at.isoformat(),
            }
            for audit in audits
        ],
        "notifications": [
            {
                "subject": item.subject,
                "body": item.body,
                "status": item.status.value,
                "created_at": item.created_at.isoformat(),
            }
            for item in notifications
        ],
        "security_events": [
            {
                "event_type": item.event_type,
                "severity": item.severity.value,
                "response_action": item.response_action,
                "details": item.details_json,
                "created_at": item.created_at.isoformat(),
            }
            for item in security_events
        ],
    }


def _estimate_ingestion_ready_seconds(db, union_id: str, document: Document | None, job: IngestionJob) -> int | None:
    if job.status == IngestionJobStatus.SUCCEEDED:
        return 0
    if job.status == IngestionJobStatus.FAILED:
        return None
    pending_jobs = db.scalars(
        select(IngestionJob).where(
            IngestionJob.union_id == union_id,
            IngestionJob.status == IngestionJobStatus.PENDING,
        )
    ).all()
    pending_documents = {
        candidate.document_id: db.get(Document, candidate.document_id)
        for candidate in pending_jobs
        if candidate.document_id
    }
    ordered = sorted(
        pending_jobs,
        key=lambda candidate: ingestion_job_priority(pending_documents.get(candidate.document_id), candidate),
    )
    total_seconds = 0
    for candidate in ordered:
        total_seconds += estimate_ingestion_runtime_seconds(pending_documents.get(candidate.document_id), candidate)
        if candidate.id == job.id:
            return total_seconds
    return estimate_ingestion_runtime_seconds(document, job)


def _serialize_ingestion_job(db, union_id: str, job: IngestionJob, document: Document | None) -> dict:
    return {
        "id": job.id,
        "document_id": job.document_id,
        "document_title": document.title if document is not None else None,
        "status": job.status.value,
        "error_message": job.error_message,
        "metadata": job.metadata_json,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "estimated_ready_seconds": _estimate_ingestion_ready_seconds(db, union_id, document, job),
    }


class UnionCreateRequest(BaseModel):
    name: str
    slug: str | None = None


class UnionUpdateRequest(BaseModel):
    name: str | None = None
    message_retention_enabled: bool | None = None
    is_active: bool | None = None
    member_login_required: bool | None = None
    branding: dict = Field(default_factory=dict)
    custom_domain: str | None = None
    features: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class TrackingPolicyRequest(BaseModel):
    tracking_mode: str
    privacy_mode: str
    member_choice_mode: str
    raw_query_storage_mode: str
    default_member_preference: str
    allow_union_override: bool | None = None


class UserCreateRequest(BaseModel):
    email: str
    full_name: str
    role: str = Role.USER.value
    external_auth_id: str | None = None
    username: str | None = None
    password: str | None = None


class UserUpdateRequest(BaseModel):
    email: str | None = None
    full_name: str | None = None
    role: str | None = None
    username: str | None = None
    is_active: bool | None = None
    local_auth_enabled: bool | None = None


class UserPasswordResetRequest(BaseModel):
    password: str


class ProviderConfigRequest(BaseModel):
    provider_name: str
    model_name: str
    api_key: str | None = None
    config: dict = Field(default_factory=dict)


class ProviderTestRequest(BaseModel):
    provider_name: str
    model_name: str
    api_key: str | None = None
    config: dict = Field(default_factory=dict)


class QuotaPolicyRequest(BaseModel):
    requests_per_day: int
    tokens_per_day: int
    cost_usd_per_day: float
    per_user_requests_per_hour: int
    warn_threshold_ratio: float = 0.8
    is_paused: bool = False


class RetryIngestionRequest(BaseModel):
    ocr_enabled: bool = False


class EscalateIngestionReviewRequest(BaseModel):
    note: str | None = None


class ReviewStateUpdateRequest(BaseModel):
    review_status: str
    note: str | None = None


class DocumentSafetyDecisionRequest(BaseModel):
    decision: str
    note: str | None = None


def _serialize_document(db, doc: Document) -> dict:
    metadata = dict(doc.metadata_json or {})
    review_status = metadata.get("review_status")
    if review_status is None:
        if metadata.get("quality_status") == "needs_review":
            review_status = "needs_review"
        elif metadata.get("ready_for_query"):
            review_status = "not_required"
    latest_job = db.scalar(
        select(IngestionJob)
        .where(IngestionJob.document_id == doc.id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    latest_job_payload = None
    if latest_job is not None:
        latest_job_payload = _serialize_ingestion_job(db, doc.union_id, latest_job, doc)
    return {
        "id": doc.id,
        "title": doc.title,
        "contract_id": doc.contract_id,
        "content_type": doc.content_type,
        "bytes_size": doc.bytes_size,
        "status": doc.status.value,
        "metadata": metadata,
        "quality_status": metadata.get("quality_status"),
        "quality_reason": metadata.get("quality_reason"),
        "ocr_status": metadata.get("ocr_status"),
        "scan_likelihood": metadata.get("scan_likelihood"),
        "ready_for_query": bool(metadata.get("ready_for_query")),
        "recommended_action": metadata.get("recommended_action"),
        "review_status": review_status,
        "document_type": metadata.get("document_type"),
        "document_type_confidence": metadata.get("document_type_confidence"),
        "structure_mode": metadata.get("structure_mode"),
        "structure_extraction_status": metadata.get("structure_extraction_status"),
        "total_articles": metadata.get("total_articles"),
        "total_sections": metadata.get("total_sections"),
        "topic_hints": metadata.get("topic_hints") or [],
        "safety_status": metadata.get("safety_status") or "clear",
        "safety_reasons": metadata.get("safety_reasons") or [],
        "prompt_injection_risk": bool(metadata.get("prompt_injection_risk")),
        "sensitive_data_risk": bool(metadata.get("sensitive_data_risk")),
        "member_visible": bool(metadata.get("member_visible", True)),
        "safety_review_status": metadata.get("safety_review_status") or "not_required",
        "latest_ingestion_job": latest_job_payload,
    }


def _latest_document_job(db, document_id: str) -> IngestionJob | None:
    return db.scalar(
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )


def _load_document_review_preview(request: Request, document: Document, latest_job: IngestionJob | None) -> dict:
    container = get_container(request)
    metadata = dict(document.metadata_json or {})
    artifact_key = metadata.get("artifact_key") or ((latest_job.metadata_json or {}).get("artifact_key") if latest_job is not None else None)
    parsed_text = ""
    pages: list[dict] = []

    if artifact_key:
        artifact_path = container.storage.open(artifact_key)
        if artifact_path.exists():
            try:
                artifact_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
                parsed_text = str(artifact_payload.get("text") or "")
                for page in (artifact_payload.get("pages") or [])[:5]:
                    page_text = str(page.get("text") or "").strip()
                    if page_text:
                        pages.append(
                            {
                                "page_number": page.get("page_number"),
                                "text_excerpt": page_text[:2500],
                            }
                        )
            except Exception:
                parsed_text = ""
                pages = []

    if not parsed_text and str(document.content_type or "").startswith("text/"):
        source_path = container.storage.open(document.storage_key)
        if source_path.exists():
            try:
                parsed_text = source_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                parsed_text = ""

    guardrails = getattr(container, "guardrails", None)
    findings = guardrails.review_findings(parsed_text) if guardrails is not None and parsed_text else []
    redacted_preview = guardrails.redact_sensitive_text(parsed_text).sanitized_text[:4000] if guardrails is not None and parsed_text else ""
    return {
        "text_excerpt": parsed_text[:4000],
        "redacted_excerpt": redacted_preview,
        "page_count": len(pages),
        "pages": pages,
        "safety_findings": findings,
    }


def _set_document_safety_override(db, *, document: Document, latest_job: IngestionJob | None, actor_user_id: str, approved_by_superadmin: bool, note: str | None) -> None:
    metadata = dict(document.metadata_json or {})
    prompt_injection_risk = bool(metadata.get("prompt_injection_risk"))
    sensitive_data_risk = bool(metadata.get("sensitive_data_risk"))

    metadata.update(
        {
            "member_visible": True,
            "ready_for_query": True,
            "review_status": "resolved",
            "review_note": note,
            "safety_status": "reviewed_safe",
            "safety_review_status": "resolved",
            "recommended_action": "Approved for full member access after manual safety review.",
            "safety_reasons": [],
            "prompt_injection_risk": False,
            "sensitive_data_risk": False,
            "safety_override": {
                "approved_at": datetime.utcnow().isoformat(),
                "approved_by_user_id": actor_user_id,
                "approved_by_superadmin": approved_by_superadmin,
                "note": note,
                "previous_prompt_injection_risk": prompt_injection_risk,
                "previous_sensitive_data_risk": sensitive_data_risk,
            },
        }
    )
    document.metadata_json = metadata

    if latest_job is not None:
        latest_job.metadata_json = {
            **(latest_job.metadata_json or {}),
            "member_visible": True,
            "ready_for_query": True,
            "review_status": "resolved",
            "review_note": note,
            "safety_status": "reviewed_safe",
            "safety_review_status": "resolved",
            "recommended_action": "Approved for full member access after manual safety review.",
            "prompt_injection_risk": False,
            "sensitive_data_risk": False,
            "safety_reasons": [],
        }

    chunk_rows = db.scalars(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document.id)).all()
    for chunk in chunk_rows:
        chunk.metadata_json = {
            **(chunk.metadata_json or {}),
            "member_visible": True,
            "prompt_injection_risk": False,
            "sensitive_data_risk": False,
            "safety_status": "reviewed_safe",
            "safety_review_status": "resolved",
            "safety_reasons": [],
        }


def _notify_super_admin_review_escalation(db, *, auth, union_id: str, document: Document, job: IngestionJob, note: str | None) -> None:
    apply_service_bootstrap_context(db)
    try:
        db.execute(
            insert(Notification).values(
                union_id=None,
                user_id=None,
                channel="in_app",
                subject="Security alert: ingestion_review_escalated",
                body=(
                    f"Document '{document.title}' in union {union_id} requires manual review escalation. "
                    f"Job: {job.id}. Note: {note or 'none'}"
                ),
                status=NotificationStatus.PENDING,
            )
        )
    finally:
        apply_request_context(db, auth)


@router.get("/me")
def admin_me(request: Request, auth=Depends(get_auth_context)):
    return {
        "authenticated": auth.is_authenticated,
        "role": auth.role,
        "union_id": auth.union_id,
        "union_slug": auth.union_slug,
        "email": auth.email,
        "full_name": auth.full_name,
    }


@router.get("/unions")
def list_unions(
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.STEWARD_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    stmt = select(Union)
    if not auth.is_super_admin and auth.union_id:
        stmt = stmt.where(Union.id == auth.union_id)
    items = db.scalars(stmt.order_by(Union.name.asc())).all()
    return {"items": [
        {
            "id": item.id,
            "slug": item.slug,
            "name": item.name,
            "union_local_id": item.union_local_id,
            "is_active": item.is_active,
            "message_retention_enabled": item.message_retention_enabled,
            "metadata": item.metadata_json,
        }
        for item in items
    ]}


@router.post("/unions")
def create_union(
    payload: UnionCreateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    union = Union(
        slug=_generate_unique_union_slug(db, payload.name, payload.slug),
        name=payload.name,
        union_local_id=_generate_unique_union_local_id(db),
    )
    db.add(union)
    db.add(AuditEvent(union_id=union.id, actor_user_id=get_auth_context(request).user_id, event_type="union_created", event_payload=payload.model_dump()))
    db.flush()
    return {"id": union.id, "slug": union.slug, "name": union.name}


@router.put("/unions/{union_id}")
def update_union(
    union_id: str,
    payload: UnionUpdateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    union = db.get(Union, union_id)
    if not union:
        raise HTTPException(status_code=404, detail="Union not found.")
    if not auth.is_super_admin and auth.union_id != union.id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    if payload.name is not None:
        union.name = payload.name
    if payload.is_active is not None:
        union.is_active = payload.is_active
    if payload.message_retention_enabled is not None:
        union.message_retention_enabled = payload.message_retention_enabled
    metadata = dict(union.metadata_json or {})
    if payload.member_login_required is not None:
        auth_policy = dict(metadata.get("auth_policy") or {})
        auth_policy["member_login_required"] = payload.member_login_required
        metadata["auth_policy"] = auth_policy
    if payload.branding:
        metadata["branding"] = {**dict(metadata.get("branding") or {}), **payload.branding}
    if payload.custom_domain is not None:
        metadata["custom_domain"] = payload.custom_domain
    if payload.features:
        metadata["features"] = {**dict(metadata.get("features") or {}), **payload.features}
    if payload.metadata:
        metadata = {**metadata, **payload.metadata}
    if metadata:
        union.metadata_json = metadata
    union.updated_at = datetime.utcnow()
    db.add(AuditEvent(union_id=union.id, actor_user_id=auth.user_id, event_type="union_updated", event_payload=payload.model_dump()))
    return {"ok": True}


@router.delete("/unions/{union_id}")
def delete_union(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")

    deleted_documents = db.query(Document).filter(Document.union_id == union_id).count()
    deleted_memberships = db.query(UnionMembership).filter(UnionMembership.union_id == union_id).count()

    db.query(AuthSession).filter(AuthSession.union_id == union_id).delete(synchronize_session=False)
    db.query(Message).filter(Message.union_id == union_id).delete(synchronize_session=False)
    db.query(Chat).filter(Chat.union_id == union_id).delete(synchronize_session=False)
    db.query(ChunkEmbedding).filter(ChunkEmbedding.union_id == union_id).delete(synchronize_session=False)
    db.query(IngestionJob).filter(IngestionJob.union_id == union_id).delete(synchronize_session=False)
    db.query(Document).filter(Document.union_id == union_id).delete(synchronize_session=False)
    db.query(ProviderConfig).filter(ProviderConfig.union_id == union_id).delete(synchronize_session=False)
    db.query(QuotaPolicy).filter(QuotaPolicy.union_id == union_id).delete(synchronize_session=False)
    db.query(UsageEvent).filter(UsageEvent.union_id == union_id).delete(synchronize_session=False)
    db.query(SecurityEvent).filter(SecurityEvent.union_id == union_id).delete(synchronize_session=False)
    db.query(Notification).filter(Notification.union_id == union_id).delete(synchronize_session=False)
    db.query(AuditEvent).filter(AuditEvent.union_id == union_id).delete(synchronize_session=False)
    db.query(UnionMembership).filter(UnionMembership.union_id == union_id).delete(synchronize_session=False)
    db.delete(union)

    try:
        get_container(request).storage.delete_prefix(union.slug)
    except Exception:
        pass

    db.add(
        AuditEvent(
            union_id=None,
            actor_user_id=auth.user_id,
            event_type="union_deleted",
            event_payload={
                "deleted_union_id": union_id,
                "deleted_union_slug": union.slug,
                "deleted_documents": deleted_documents,
                "deleted_memberships": deleted_memberships,
            },
        )
    )
    return {
        "ok": True,
        "deleted_union_id": union_id,
        "deleted_union_slug": union.slug,
        "deleted_documents": deleted_documents,
        "deleted_memberships": deleted_memberships,
    }


@router.post("/unions/offline-all")
def take_all_unions_offline(
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    unions = db.scalars(select(Union)).all()
    changed = 0
    for union in unions:
        if union.is_active:
            union.is_active = False
            union.updated_at = datetime.utcnow()
            changed += 1
    db.add(
        AuditEvent(
            union_id=None,
            actor_user_id=auth.user_id,
            event_type="all_unions_taken_offline",
            event_payload={"changed_unions": changed},
        )
    )
    return {"ok": True, "changed_unions": changed}


@router.get("/unions/{union_id}/users")
def list_union_users(
    union_id: str,
    request: Request,
    q: str | None = Query(default=None),
    q_field: str = Query(default="all"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    sort: str = Query(default="name"),
    direction: str = Query(default="asc"),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    normalized_query = str(q or "").strip().lower()
    if normalized_query == "*":
        normalized_query = ""
    normalized_field = str(q_field or "all").strip().lower()
    normalized_sort = str(sort or "name").strip().lower()
    normalized_direction = str(direction or "asc").strip().lower()
    rows = db.execute(
        select(UnionMembership, User, LocalAuthCredential)
        .join(User, User.id == UnionMembership.user_id)
        .outerjoin(LocalAuthCredential, LocalAuthCredential.user_id == User.id)
        .where(UnionMembership.union_id == union_id)
    ).all()
    items = []
    union_total = 0
    for membership, user, credential in rows:
        if not auth.is_super_admin and membership.role == Role.SUPER_ADMIN:
            continue
        union_total += 1
        searchable_fields = {
            "all": " ".join(
                [
                    str(user.full_name or ""),
                    str(user.email or ""),
                    str(credential.username or "") if credential is not None else "",
                    membership.role.value,
                ]
            ).lower(),
            "name": str(user.full_name or "").lower(),
            "email": str(user.email or "").lower(),
            "username": str(credential.username or "").lower() if credential is not None else "",
            "role": membership.role.value.lower(),
        }
        searchable = searchable_fields.get(normalized_field, searchable_fields["all"])
        if normalized_query and normalized_query not in searchable:
            continue
        items.append(_serialize_user_row(user, membership, credential))

    reverse = normalized_direction == "desc"
    if normalized_sort == "email":
        items.sort(key=lambda item: (item["email"] or "").lower(), reverse=reverse)
    elif normalized_sort == "role":
        items.sort(key=lambda item: (_role_rank(item["role"]), (item["full_name"] or "").lower()), reverse=reverse)
    elif normalized_sort == "created_at":
        items.sort(key=lambda item: item["created_at"], reverse=reverse)
    else:
        items.sort(key=lambda item: (item["full_name"] or "").lower(), reverse=reverse)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "union_total": union_total,
    }


@router.post("/unions/{union_id}/users")
def create_union_user(
    union_id: str,
    payload: UserCreateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    if not _can_manage_role(auth, payload.role):
        raise HTTPException(status_code=403, detail="You cannot assign a role higher than your own.")
    if bool(payload.username) != bool(payload.password):
        raise HTTPException(status_code=400, detail="Username and password must be provided together for local auth.")
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None:
        user = User(email=payload.email, full_name=payload.full_name, external_auth_id=payload.external_auth_id)
        db.add(user)
        db.flush()
    else:
        user.full_name = payload.full_name or user.full_name
    membership = db.scalar(
        select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == user.id)
    )
    if membership is None:
        membership = UnionMembership(union_id=union_id, user_id=user.id, role=Role(payload.role))
        db.add(membership)
    else:
        membership.role = Role(payload.role)
        membership.is_active = True
    if payload.username and payload.password:
        get_container(request).local_auth.create_or_update_credential(
            db,
            user=user,
            username=payload.username,
            password=payload.password,
        )
    db.add(AuditEvent(union_id=union_id, actor_user_id=auth.user_id, event_type="union_user_created", event_payload=payload.model_dump()))
    return {"user_id": user.id, "membership_id": membership.id}


@router.put("/unions/{union_id}/users/{user_id}")
def update_union_user(
    union_id: str,
    user_id: str,
    payload: UserUpdateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    membership = db.scalar(
        select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == user_id)
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="User membership not found.")
    if not auth.is_super_admin and membership.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=404, detail="User membership not found.")
    if payload.role is not None and not _can_manage_role(auth, payload.role):
        raise HTTPException(status_code=403, detail="You cannot assign a role higher than your own.")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.user_id == user_id))

    if payload.email is not None:
        user.email = payload.email.strip().lower()
    if payload.full_name is not None:
        user.full_name = payload.full_name.strip()
    if payload.role is not None:
        membership.role = Role(payload.role)
    if payload.is_active is not None:
        membership.is_active = payload.is_active
        user.is_active = payload.is_active
    if payload.local_auth_enabled is False and credential is not None:
        credential.is_active = False
    elif payload.local_auth_enabled is True:
        if credential is not None:
            credential.is_active = True
        elif payload.username is not None:
            raise HTTPException(status_code=400, detail="Set a password to enable local login for this user.")
    if payload.username is not None:
        normalized_username = str(payload.username or "").strip().lower()
        if not normalized_username and payload.local_auth_enabled is True:
            raise HTTPException(status_code=400, detail="Username is required when enabling local login.")
        if credential is not None and normalized_username:
            credential.username = normalized_username
            credential.is_active = payload.local_auth_enabled is not False

    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="union_user_updated",
            event_payload={"user_id": user_id, **payload.model_dump(exclude_none=True)},
        )
    )
    db.flush()
    return {"user": _serialize_user_row(user, membership, credential)}


@router.post("/unions/{union_id}/users/{user_id}/password")
def reset_union_user_password(
    union_id: str,
    user_id: str,
    payload: UserPasswordResetRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    membership = db.scalar(
        select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == user_id)
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="User membership not found.")
    if not auth.is_super_admin and membership.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=404, detail="User membership not found.")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.user_id == user_id))
    if credential is None:
        raise HTTPException(status_code=400, detail="This user does not have local login enabled yet.")
    get_container(request).local_auth.create_or_update_credential(
        db,
        user=user,
        username=credential.username,
        password=payload.password,
    )
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="union_user_password_reset",
            event_payload={"user_id": user_id},
        )
    )
    return {"ok": True}


@router.delete("/unions/{union_id}/users/{user_id}")
def remove_union_user(
    union_id: str,
    user_id: str,
    request: Request,
    purge_user: bool = Query(default=False),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    if user_id == auth.user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own active admin account from this screen.")
    membership = db.scalar(
        select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == user_id)
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="User membership not found.")
    if not auth.is_super_admin and membership.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=404, detail="User membership not found.")
    other_membership_count = db.query(UnionMembership).filter(
        UnionMembership.user_id == user_id,
        UnionMembership.union_id != union_id,
    ).count()
    if purge_user:
        if other_membership_count and not auth.is_super_admin:
            raise HTTPException(
                status_code=403,
                detail="This user also belongs to another union. Only a superadmin can purge all user data across unions.",
            )
        global_scope = auth.is_super_admin or other_membership_count == 0
        if not global_scope:
            db.delete(membership)
        _purge_user_records(
            db,
            user_id=user_id,
            union_id=union_id,
            global_scope=global_scope,
            telemetry=get_container(request).telemetry,
        )
    else:
        db.delete(membership)
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="union_user_purged" if purge_user else "union_user_removed",
            event_payload={
                "user_id": user_id,
                "purge_user": purge_user,
                "global_scope": bool(purge_user and (auth.is_super_admin or other_membership_count == 0)),
            },
        )
    )
    return {"ok": True, "purged": purge_user}


@router.get("/platform-summary")
def platform_summary(
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {
            "totals": {
                "unions": 0,
                "active_unions": 0,
                "inactive_unions": 0,
                "users": 0,
                "documents": 0,
                "pending_reviews": 0,
                "pending_notifications": 0,
            },
            "union_summaries": [],
        }

    unions = db.scalars(select(Union).order_by(Union.name.asc())).all()
    documents = db.scalars(select(Document)).all()
    notifications = db.scalars(select(Notification).where(Notification.status == NotificationStatus.PENDING)).all()
    memberships = db.scalars(select(UnionMembership)).all()

    document_counts: dict[str, int] = {}
    pending_review_counts: dict[str, int] = {}
    for document in documents:
        document_counts[document.union_id] = document_counts.get(document.union_id, 0) + 1
        metadata = dict(document.metadata_json or {})
        if metadata.get("quality_status") in {"needs_review", "retrying_with_ocr"} or metadata.get("review_status") in {"needs_review", "in_review"}:
            pending_review_counts[document.union_id] = pending_review_counts.get(document.union_id, 0) + 1

    user_counts: dict[str, int] = {}
    for membership in memberships:
        user_counts[membership.union_id] = user_counts.get(membership.union_id, 0) + 1

    pending_notification_counts: dict[str, int] = {}
    for notification in notifications:
        if notification.union_id:
            pending_notification_counts[notification.union_id] = pending_notification_counts.get(notification.union_id, 0) + 1

    union_summaries = [
        {
            "id": union.id,
            "slug": union.slug,
            "name": union.name,
            "is_active": union.is_active,
            "member_login_required": bool((union.metadata_json or {}).get("auth_policy", {}).get("member_login_required", True)),
            "custom_domain": (union.metadata_json or {}).get("custom_domain"),
            "message_retention_enabled": union.message_retention_enabled,
            "user_count": user_counts.get(union.id, 0),
            "document_count": document_counts.get(union.id, 0),
            "pending_review_count": pending_review_counts.get(union.id, 0),
            "pending_notification_count": pending_notification_counts.get(union.id, 0),
        }
        for union in unions
    ]
    return {
        "totals": {
            "unions": len(unions),
            "active_unions": sum(1 for union in unions if union.is_active),
            "inactive_unions": sum(1 for union in unions if not union.is_active),
            "users": len({membership.user_id for membership in memberships}),
            "documents": len(documents),
            "pending_reviews": sum(pending_review_counts.values()),
            "pending_notifications": len(notifications),
        },
        "union_summaries": union_summaries,
        "tracking_policy": get_container(request).telemetry.serialize_policy(
            db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id.is_(None)))
        ),
    }


@router.get("/platform-ops")
def platform_operations_summary(
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"captured_at": datetime.utcnow().isoformat(), "items": [], "summary": {"warning_unions": 0, "paused_unions": 0, "provider_issues": 0, "unions_without_admins": 0}}
    return _build_platform_operations_summary(db)


@router.get("/tracking-policy/global")
def get_global_tracking_policy(
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    policy = container.telemetry.get_or_create_global_policy(db)
    return {"policy": container.telemetry.serialize_policy(policy)}


@router.put("/tracking-policy/global")
def update_global_tracking_policy(
    payload: TrackingPolicyRequest,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    existing = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id.is_(None)))
    prior_mode = existing.raw_query_storage_mode.value if existing is not None else RawQueryStorageMode.DISABLED.value
    actor_user_id = get_auth_context(request).user_id
    policy = container.telemetry.update_policy(db, union_id=None, payload=payload.model_dump())
    db.add(
        AuditEvent(
            union_id=None,
            actor_user_id=actor_user_id,
            event_type="global_tracking_policy_updated",
            event_payload=container.telemetry.serialize_policy(policy),
        )
    )
    _emit_identified_raw_query_signal_if_enabled(
        db,
        union_id=None,
        actor_user_id=actor_user_id,
        prior_mode=prior_mode,
        new_mode=policy.raw_query_storage_mode.value,
    )
    db.flush()
    return {"policy": container.telemetry.serialize_policy(policy)}


@router.get("/unions/{union_id}/tracking-policy")
def get_union_tracking_policy(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    container = get_container(request)
    global_policy = container.telemetry.get_or_create_global_policy(db)
    policy = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id == union_id))
    effective = container.telemetry.resolve_policy(db, union_id=union_id)
    return {
        "global_allow_union_override": bool(global_policy.allow_union_override),
        "override_enabled": policy is not None,
        "policy": None if policy is None else container.telemetry.serialize_policy(policy),
        "effective_policy": effective.to_summary(),
    }


@router.put("/unions/{union_id}/tracking-policy")
def update_union_tracking_policy(
    union_id: str,
    payload: TrackingPolicyRequest,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    container = get_container(request)
    global_policy = container.telemetry.get_or_create_global_policy(db)
    if not global_policy.allow_union_override:
        raise HTTPException(status_code=409, detail="Union-specific tracking overrides are disabled by the global policy.")
    existing = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id == union_id))
    prior_mode = existing.raw_query_storage_mode.value if existing is not None else RawQueryStorageMode.DISABLED.value
    actor_user_id = get_auth_context(request).user_id
    policy = container.telemetry.update_policy(db, union_id=union_id, payload=payload.model_dump(exclude={"allow_union_override"}))
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=actor_user_id,
            event_type="union_tracking_policy_override_updated",
            event_payload=container.telemetry.serialize_policy(policy),
        )
    )
    _emit_identified_raw_query_signal_if_enabled(
        db,
        union_id=union_id,
        actor_user_id=actor_user_id,
        prior_mode=prior_mode,
        new_mode=policy.raw_query_storage_mode.value,
    )
    db.flush()
    return {
        "override_enabled": True,
        "policy": container.telemetry.serialize_policy(policy),
        "effective_policy": container.telemetry.resolve_policy(db, union_id=union_id).to_summary(),
    }


@router.delete("/unions/{union_id}/tracking-policy")
def clear_union_tracking_policy(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    container = get_container(request)
    container.telemetry.clear_union_override(db, union_id=union_id)
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=get_auth_context(request).user_id,
            event_type="union_tracking_policy_override_cleared",
            event_payload={"union_id": union_id},
        )
    )
    db.flush()
    return {"override_enabled": False, "effective_policy": container.telemetry.resolve_policy(db, union_id=union_id).to_summary()}


@router.post("/unions/{union_id}/admin-takeover")
def assign_superadmin_takeover(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    current_user = db.get(User, auth.user_id)
    if current_user is None:
        raise HTTPException(status_code=404, detail="Current user not found.")
    membership = db.scalar(select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == current_user.id))
    if membership is None:
        membership = UnionMembership(union_id=union_id, user_id=current_user.id, role=Role.UNION_ADMIN, is_active=True)
        db.add(membership)
    else:
        membership.role = Role.UNION_ADMIN
        membership.is_active = True
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="union_admin_takeover_assigned",
            event_payload={"user_id": current_user.id, "union_id": union_id},
        )
    )
    db.flush()
    return {
        "ok": True,
        "membership_id": membership.id,
        "user": {
            "user_id": current_user.id,
            "full_name": current_user.full_name,
            "email": current_user.email,
            "role": membership.role.value,
        },
    }


@router.get("/unions/{union_id}/debug-config")
def get_union_debug_config(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"union": None}
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    provider = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == union_id, ProviderConfig.is_active.is_(True)))
    quota = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == union_id))
    return {
        "union": {
            "id": union.id,
            "slug": union.slug,
            "name": union.name,
            "union_local_id": union.union_local_id,
            "is_active": union.is_active,
            "message_retention_enabled": union.message_retention_enabled,
            "metadata": union.metadata_json,
        },
        "provider": None if provider is None else {
            "provider_name": provider.provider_name,
            "model_name": provider.model_name,
            "config": provider.config_json,
            "has_api_key": bool(provider.encrypted_api_key),
        },
        "quota": None if quota is None else {
            "requests_per_day": quota.requests_per_day,
            "tokens_per_day": quota.tokens_per_day,
            "cost_usd_per_day": quota.cost_usd_per_day,
            "per_user_requests_per_hour": quota.per_user_requests_per_hour,
            "warn_threshold_ratio": quota.warn_threshold_ratio,
            "is_paused": quota.is_paused,
        },
    }


@router.get("/unions/{union_id}/export")
def export_union_bundle(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    payload = _build_union_export_bundle(db, union_id)
    slug = str(payload["union"]["slug"])
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{slug}-export.json"'},
    )


@router.get("/unions/{union_id}/provider")
def get_provider_config(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"provider": None}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    item = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == union_id, ProviderConfig.is_active.is_(True)))
    if item is None:
        return {"provider": None}
    return {
        "provider": {
            "id": item.id,
            "provider_name": item.provider_name,
            "model_name": item.model_name,
            "config": item.config_json,
            "has_api_key": bool(item.encrypted_api_key),
        }
    }


@router.put("/unions/{union_id}/provider")
def upsert_provider_config(
    union_id: str,
    payload: ProviderConfigRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    item = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == union_id))
    api_key = str(payload.api_key or "").strip()
    if item is None:
        if not api_key:
            raise HTTPException(status_code=400, detail="Provider API key is required when creating a provider configuration.")
        encrypted = container.secret_cipher.encrypt(api_key)
        item = ProviderConfig(
            union_id=union_id,
            provider_name=payload.provider_name,
            model_name=payload.model_name,
            encrypted_api_key=encrypted,
            config_json=payload.config,
        )
        db.add(item)
    else:
        if api_key:
            item.encrypted_api_key = container.secret_cipher.encrypt(api_key)
        elif not item.encrypted_api_key:
            raise HTTPException(status_code=400, detail="Provider API key is required.")
        item.provider_name = payload.provider_name
        item.model_name = payload.model_name
        item.config_json = payload.config
        item.updated_at = datetime.utcnow()
    db.add(AuditEvent(union_id=union_id, actor_user_id=auth.user_id, event_type="provider_config_updated", event_payload={"provider_name": payload.provider_name, "model_name": payload.model_name}))
    return {"id": item.id}


@router.post("/unions/{union_id}/provider/test")
async def test_provider(
    union_id: str,
    payload: ProviderTestRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    existing = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == union_id, ProviderConfig.is_active.is_(True)))
    api_key = str(payload.api_key or "").strip()
    if not api_key and existing is not None:
        try:
            api_key = container.secret_cipher.decrypt(existing.encrypted_api_key)
        except Exception:
            api_key = ""
    if not api_key:
        raise HTTPException(status_code=400, detail="Provider API key is required to run a live test.")
    inference_config = InferenceConfig(
        provider_name=str(payload.provider_name or "").strip().lower(),
        model_name=str(payload.model_name or "").strip(),
        api_key=api_key,
        base_url=str((payload.config or {}).get("base_url") or "").strip() or None,
        config=dict(payload.config or {}),
    )
    timeout_seconds = max(3, int(get_container(request).settings.inference_request_timeout_seconds))
    result = await test_inference_config(inference_config, timeout_seconds=timeout_seconds)
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="provider_config_tested",
            event_payload={
                "provider_name": inference_config.provider_name,
                "model_name": inference_config.model_name,
                "ok": result.get("ok"),
                "error_type": result.get("error_type"),
                "latency_ms": result.get("latency_ms"),
            },
        )
    )
    return {"result": result}


@router.get("/unions/{union_id}/quota")
def get_quota_policy(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"quota": None}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    policy = get_container(request).quotas.get_or_create_policy(db, union_id)
    now = datetime.utcnow()
    day_cutoff = now - timedelta(days=1)
    hour_cutoff = now - timedelta(hours=1)
    usage_events = db.scalars(select(UsageEvent).where(UsageEvent.union_id == union_id)).all()
    requests_day = 0
    tokens_day = 0
    cost_day = 0.0
    requests_hour = 0
    for item in usage_events:
        if item.created_at >= day_cutoff:
            requests_day += int(item.request_count or 0)
            tokens_day += int(item.token_count or 0)
            cost_day += float(item.estimated_cost_usd or 0.0)
        if item.created_at >= hour_cutoff:
            requests_hour += int(item.request_count or 0)
    return {"quota": {
        "requests_per_day": policy.requests_per_day,
        "tokens_per_day": policy.tokens_per_day,
        "cost_usd_per_day": policy.cost_usd_per_day,
        "per_user_requests_per_hour": policy.per_user_requests_per_hour,
        "warn_threshold_ratio": policy.warn_threshold_ratio,
        "is_paused": policy.is_paused,
        "usage_snapshot": {
            "requests_last_24h": requests_day,
            "tokens_last_24h": tokens_day,
            "estimated_cost_last_24h": round(cost_day, 4),
            "requests_last_hour": requests_hour,
            "captured_at": now.isoformat(),
        },
    }}


@router.put("/unions/{union_id}/quota")
def upsert_quota_policy(
    union_id: str,
    payload: QuotaPolicyRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    policy = get_container(request).quotas.get_or_create_policy(db, union_id)
    policy.requests_per_day = payload.requests_per_day
    policy.tokens_per_day = payload.tokens_per_day
    policy.cost_usd_per_day = payload.cost_usd_per_day
    policy.per_user_requests_per_hour = payload.per_user_requests_per_hour
    policy.warn_threshold_ratio = payload.warn_threshold_ratio
    policy.is_paused = payload.is_paused
    db.add(AuditEvent(union_id=union_id, actor_user_id=auth.user_id, event_type="quota_policy_updated", event_payload=payload.model_dump()))
    return {"ok": True}


@router.get("/unions/{union_id}/documents")
def list_documents(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.STEWARD_ADMIN.value, Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    documents = db.scalars(select(Document).where(Document.union_id == union_id).order_by(Document.created_at.desc())).all()
    return {"items": [_serialize_document(db, doc) for doc in documents]}


@router.post("/unions/{union_id}/documents")
async def upload_document(
    union_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    # Which contract book this document is. Retrieval filters on it so a
    # member is never answered out of another bargaining unit's agreement.
    contract_id: str | None = Form(default=None),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    container = get_container(request)
    payload = await file.read()
    stored = container.storage.save_bytes(union.slug, file.filename, payload)
    try:
        result = container.ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=auth.user_id,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            payload=payload,
            storage_key=stored.key,
        )
        document = result.document
        scoped_contract_id = (contract_id or "").strip()
        if scoped_contract_id:
            document.contract_id = scoped_contract_id
        ingestion_job = result.ingestion_job
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
    db.add(AuditEvent(union_id=union_id, actor_user_id=auth.user_id, event_type="document_uploaded", event_payload={"filename": file.filename, "bytes_size": stored.bytes_size}))
    db.flush()
    job_metadata = dict(ingestion_job.metadata_json or {})
    if job_metadata.get("mode") == "deferred" and job_metadata.get("parser") and ingestion_job.status == IngestionJobStatus.PENDING:
        background_tasks.add_task(process_pending_ingestion_jobs, container, limit=5)
    return {
        "id": document.id,
        "storage_key": stored.key,
        "ingestion_job_id": ingestion_job.id,
        "ingestion_status": ingestion_job.status.value,
        "artifact_key": (ingestion_job.metadata_json or {}).get("artifact_key"),
        "queued_for_background_processing": bool(job_metadata.get("mode") == "deferred"),
    }


@router.delete("/unions/{union_id}/documents/{document_id}")
def delete_document(
    union_id: str,
    document_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    document = db.get(Document, document_id)
    if document is None or document.union_id != union_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    container = get_container(request)
    try:
        metadata = dict(document.metadata_json or {})
        latest_jobs = db.scalars(
            select(IngestionJob).where(IngestionJob.document_id == document.id)
        ).all()
        deleted_chunks = container.retrieval.delete_document(db, document_id=document.id)
        deleted_jobs = len(latest_jobs)
        for job in latest_jobs:
            artifact_key = (job.metadata_json or {}).get("artifact_key")
            if artifact_key:
                container.storage.delete(artifact_key)
            db.delete(job)
        db.flush()

        artifact_key = metadata.get("artifact_key")
        if artifact_key:
            container.storage.delete(artifact_key)
        if document.storage_key:
            container.storage.delete(document.storage_key)
        union_row = db.get(Union, union_id)
        union_slug = auth.union_slug or (union_row.slug if union_row is not None else None)
        if union_slug:
            container.storage.delete_prefix(f"{union_slug}/{document.id}")

        title = document.title
        db.delete(document)
        db.add(
            AuditEvent(
                union_id=union_id,
                actor_user_id=auth.user_id,
                event_type="document_deleted",
                event_payload={
                    "document_id": document_id,
                    "title": title,
                    "deleted_chunks": deleted_chunks,
                    "deleted_jobs": deleted_jobs,
                },
            )
        )
        db.flush()
        return {
            "deleted": True,
            "document_id": document_id,
            "title": title,
            "deleted_chunks": deleted_chunks,
            "deleted_jobs": deleted_jobs,
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Document delete failed: {type(exc).__name__}: {exc}") from exc


@router.get("/unions/{union_id}/documents/{document_id}/content")
def get_admin_document_content(
    union_id: str,
    document_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    document = db.get(Document, document_id)
    if document is None or document.union_id != union_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    path = get_container(request).storage.open(document.storage_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored document file not found.")

    return FileResponse(
        path,
        media_type=document.content_type or "application/octet-stream",
        filename=document.title,
        headers={"Content-Disposition": f'inline; filename="{document.title}"'},
    )


@router.get("/unions/{union_id}/documents/{document_id}/review-detail")
def get_document_review_detail(
    union_id: str,
    document_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    document = db.get(Document, document_id)
    if document is None or document.union_id != union_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    latest_job = _latest_document_job(db, document.id)
    preview = _load_document_review_preview(request, document, latest_job)
    metadata = dict(document.metadata_json or {})
    prompt_injection_risk = bool(metadata.get("prompt_injection_risk"))
    sensitive_data_risk = bool(metadata.get("sensitive_data_risk"))
    can_approve = not prompt_injection_risk or auth.is_super_admin

    return {
        "document": _serialize_document(db, document),
        "latest_job": _serialize_ingestion_job(db, union_id, latest_job, document) if latest_job is not None else None,
        "review_preview": preview,
        "review_actions": {
            "can_mark_in_review": True,
            "can_delete_document": True,
            "can_approve_member_access": can_approve,
            "requires_superadmin_override": prompt_injection_risk and not auth.is_super_admin,
            "approval_effect": (
                "Approve this document for full member access. This removes member redaction and retrieval down-ranking."
                if sensitive_data_risk and not prompt_injection_risk
                else "Release this document for full member access after manual safety review."
            ),
        },
    }


@router.post("/unions/{union_id}/documents/{document_id}/source-pdf")
async def attach_document_source_pdf(
    union_id: str,
    document_id: str,
    request: Request,
    file: UploadFile,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    """Attach the printed contract PDF that a document's text came from.

    Stored only -- deliberately not parsed or ingested. The extracted markdown
    remains the retrieval source; this is purely so citations can open the page
    a member can check against the physical book.
    """
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    document = db.get(Document, document_id)
    if document is None or document.union_id != union_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded source PDF is empty.")
    if not payload.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Source file must be a PDF.")

    container = get_container(request)
    filename = (file.filename or "contract.pdf").strip() or "contract.pdf"
    stored = container.storage.save_bytes(union.slug, f"source-pdf-{document_id}-{filename}", payload)
    document.source_pdf_key = stored.key
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="document_source_pdf_attached",
            event_payload={"document_id": document_id, "filename": filename, "bytes_size": stored.bytes_size},
        )
    )
    db.commit()
    return {
        "document_id": document_id,
        "source_pdf_key": stored.key,
        "bytes_size": stored.bytes_size,
    }


@router.post("/unions/{union_id}/documents/{document_id}/safety-review")
def apply_document_safety_review(
    union_id: str,
    document_id: str,
    payload: DocumentSafetyDecisionRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    document = db.get(Document, document_id)
    if document is None or document.union_id != union_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    decision = str(payload.decision or "").strip().lower()
    if decision not in {"mark_in_review", "approve_member_access"}:
        raise HTTPException(status_code=422, detail="Unsupported safety review decision.")

    latest_job = _latest_document_job(db, document.id)
    metadata = dict(document.metadata_json or {})
    prompt_injection_risk = bool(metadata.get("prompt_injection_risk"))
    note = (payload.note or "").strip() or None

    if decision == "mark_in_review":
        metadata.update(
            {
                "review_status": "in_review",
                "review_note": note,
                "safety_review_status": "in_review" if metadata.get("safety_review_status") else metadata.get("safety_review_status"),
            }
        )
        document.metadata_json = metadata
        if latest_job is not None:
            latest_job.metadata_json = {
                **(latest_job.metadata_json or {}),
                "review_status": "in_review",
                "review_note": note,
                "safety_review_status": "in_review" if metadata.get("safety_review_status") else (latest_job.metadata_json or {}).get("safety_review_status"),
            }
        db.add(
            AuditEvent(
                union_id=union_id,
                actor_user_id=auth.user_id,
                event_type="document_safety_review_marked_in_review",
                event_payload={"document_id": document.id, "job_id": latest_job.id if latest_job is not None else None, "note": note},
            )
        )
        db.flush()
        return {"ok": True, "decision": decision, "document": _serialize_document(db, document)}

    if prompt_injection_risk and not auth.is_super_admin:
        raise HTTPException(status_code=403, detail="Only superadmins can override prompt-injection safety blocks.")

    _set_document_safety_override(
        db,
        document=document,
        latest_job=latest_job,
        actor_user_id=auth.user_id,
        approved_by_superadmin=bool(auth.is_super_admin),
        note=note,
    )
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="document_safety_review_approved",
            event_payload={
                "document_id": document.id,
                "job_id": latest_job.id if latest_job is not None else None,
                "note": note,
                "prompt_injection_override": prompt_injection_risk,
            },
        )
    )
    db.flush()
    return {"ok": True, "decision": decision, "document": _serialize_document(db, document)}


@router.get("/unions/{union_id}/ingestion-jobs")
def list_ingestion_jobs(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    rows = db.execute(
        select(IngestionJob, Document)
        .join(Document, Document.id == IngestionJob.document_id, isouter=True)
        .where(IngestionJob.union_id == union_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(100)
    ).all()
    return {"items": [_serialize_ingestion_job(db, union_id, job, document) for job, document in rows]}


@router.get("/unions/{union_id}/ingestion-jobs/{job_id}")
def get_ingestion_job_detail(
    union_id: str,
    job_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"job": None}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    row = db.execute(
        select(IngestionJob, Document)
        .join(Document, Document.id == IngestionJob.document_id, isouter=True)
        .where(IngestionJob.id == job_id, IngestionJob.union_id == union_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    job, document = row
    return {"job": _serialize_ingestion_job(db, union_id, job, document)}


@router.post("/unions/{union_id}/ingestion-jobs/{job_id}/retry")
def retry_ingestion_job(
    union_id: str,
    job_id: str,
    payload: RetryIngestionRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    union = db.get(Union, union_id)
    if union is None:
        raise HTTPException(status_code=404, detail="Union not found.")
    row = db.execute(
        select(IngestionJob, Document)
        .join(Document, Document.id == IngestionJob.document_id, isouter=True)
        .where(IngestionJob.id == job_id, IngestionJob.union_id == union_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    source_job, document = row
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found for ingestion job.")
    try:
        retry_job = get_container(request).ingestion.enqueue_retry(
            db,
            union=union,
            document=document,
            source_job=source_job,
            requested_by_user_id=auth.user_id,
            ocr_enabled=payload.ocr_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="ingestion_job_retried",
            event_payload={"source_job_id": source_job.id, "retry_job_id": retry_job.id, "document_id": document.id},
        )
    )
    db.flush()
    return {"job": _serialize_ingestion_job(db, union_id, retry_job, document)}


@router.post("/unions/{union_id}/ingestion-jobs/{job_id}/escalate-review")
def escalate_ingestion_review(
    union_id: str,
    job_id: str,
    payload: EscalateIngestionReviewRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    row = db.execute(
        select(IngestionJob, Document)
        .join(Document, Document.id == IngestionJob.document_id, isouter=True)
        .where(IngestionJob.id == job_id, IngestionJob.union_id == union_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    job, document = row
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found for ingestion job.")
    if (job.metadata_json or {}).get("quality_status") != "needs_review":
        raise HTTPException(status_code=409, detail="Only review-required ingestion jobs can be escalated.")

    note = (payload.note or "").strip()
    details = {
        "document_id": document.id,
        "document_title": document.title,
        "job_id": job.id,
        "recommended_action": (job.metadata_json or {}).get("recommended_action"),
        "note": note or None,
    }
    get_container(request).sentinel.record_event(
        db,
        auth,
        event_type="ingestion_review_escalated",
        details=details,
    )
    _notify_super_admin_review_escalation(
        db,
        auth=auth,
        union_id=union_id,
        document=document,
        job=job,
        note=note or None,
    )
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="ingestion_review_escalated",
            event_payload=details,
        )
    )
    job.metadata_json = {
        **(job.metadata_json or {}),
        "escalated_for_review": True,
        "escalated_at": datetime.utcnow().isoformat(),
        "escalation_note": note or None,
        "escalated_by_user_id": auth.user_id,
        "review_status": "escalated",
    }
    document.metadata_json = {
        **(document.metadata_json or {}),
        "review_status": "escalated",
        "review_note": note or None,
    }
    db.flush()
    return {"job": _serialize_ingestion_job(db, union_id, job, document)}


@router.post("/unions/{union_id}/ingestion-jobs/{job_id}/review-state")
def update_ingestion_review_state(
    union_id: str,
    job_id: str,
    payload: ReviewStateUpdateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    review_status = str(payload.review_status or "").strip().lower()
    if review_status not in {"in_review", "resolved"}:
        raise HTTPException(status_code=422, detail="Unsupported review status.")
    row = db.execute(
        select(IngestionJob, Document)
        .join(Document, Document.id == IngestionJob.document_id, isouter=True)
        .where(IngestionJob.id == job_id, IngestionJob.union_id == union_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    job, document = row
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found for ingestion job.")
    document_metadata = dict(document.metadata_json or {})
    if bool(document_metadata.get("prompt_injection_risk")) and not auth.is_super_admin:
        raise HTTPException(status_code=403, detail="Only superadmins can resolve prompt-injection safety reviews.")
    job.metadata_json = {
        **(job.metadata_json or {}),
        "review_status": review_status,
        "review_note": (payload.note or "").strip() or None,
        "review_updated_at": datetime.utcnow().isoformat(),
        "review_updated_by_user_id": auth.user_id,
    }
    next_document_metadata = {
        **document_metadata,
        "review_status": review_status,
        "review_note": (payload.note or "").strip() or None,
    }
    if review_status == "resolved" and bool(document_metadata.get("prompt_injection_risk")):
        _set_document_safety_override(
            db,
            document=document,
            latest_job=job,
            actor_user_id=auth.user_id,
            approved_by_superadmin=True,
            note=(payload.note or "").strip() or None,
        )
        next_document_metadata = dict(document.metadata_json or {})
    elif review_status == "in_review" and document_metadata.get("safety_review_status"):
        next_document_metadata["safety_review_status"] = "in_review"
        job.metadata_json = {
            **(job.metadata_json or {}),
            "safety_review_status": "in_review",
        }
    document.metadata_json = next_document_metadata
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="ingestion_review_state_updated",
            event_payload={"job_id": job.id, "document_id": document.id, "review_status": review_status},
        )
    )
    db.flush()
    return {"job": _serialize_ingestion_job(db, union_id, job, document), "review_status": review_status}


@router.get("/unions/{union_id}/chats")
def list_chats(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    chats = db.scalars(select(Chat).where(Chat.union_id == union_id).order_by(Chat.updated_at.desc()).limit(100)).all()
    return {"items": [
        {
            "id": chat.id,
            "session_id": chat.session_id,
            "user_id": chat.user_id,
            "updated_at": chat.updated_at.isoformat(),
        }
        for chat in chats
    ]}


@router.get("/unions/{union_id}/chats/{chat_id}")
def get_chat_detail(
    union_id: str,
    chat_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"messages": []}
    auth = get_auth_context(request)
    if not auth.is_super_admin and auth.union_id != union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    chat = db.get(Chat, chat_id)
    if chat is None or chat.union_id != union_id:
        raise HTTPException(status_code=404, detail="Chat not found.")
    messages = db.scalars(select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())).all()
    return {"messages": [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata_json,
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]}


# --- Invite codes (QR enrollment) -------------------------------------------------

_INVITE_CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no confusables (i/l/1/o/0)


def _generate_invite_code(db) -> str:
    import secrets as _secrets

    for _ in range(20):
        candidate = "".join(_secrets.choice(_INVITE_CODE_ALPHABET) for _ in range(10))
        if db.scalar(select(InviteCode).where(InviteCode.code == candidate)) is None:
            return candidate
    raise HTTPException(status_code=500, detail="Could not generate a unique join code.")


def _serialize_invite(invite: InviteCode) -> dict:
    now = datetime.utcnow()
    expired = invite.expires_at is not None and invite.expires_at <= now
    exhausted = invite.max_uses is not None and invite.use_count >= invite.max_uses
    if invite.revoked_at is not None:
        status = "revoked"
    elif expired:
        status = "expired"
    elif exhausted:
        status = "exhausted"
    else:
        status = "active"
    return {
        "id": invite.id,
        "code": invite.code,
        "join_path": f"/j/{invite.code}",
        "audience": invite.audience,
        "label": invite.label,
        "contract_id": invite.contract_id,
        "status": status,
        "use_count": invite.use_count,
        "max_uses": invite.max_uses,
        # Per-code token metering — overwritten with real totals in the list
        # endpoint; defaulted here so single-invite responses keep the shape.
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "first_used_at": invite.first_used_at.isoformat() if invite.first_used_at else None,
        "last_used_at": invite.last_used_at.isoformat() if invite.last_used_at else None,
        "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
        "revoked_at": invite.revoked_at.isoformat() if invite.revoked_at else None,
        "created_by": invite.created_by,
        "created_at": invite.created_at.isoformat(),
    }


class InviteCreateRequest(BaseModel):
    audience: str = InviteAudience.MEMBER.value
    label: str = ""
    contract_id: str | None = None
    expires_at: str | None = None
    max_uses: int | None = Field(default=None, ge=1)


@router.get("/unions/{union_id}/invites")
def list_union_invites(
    union_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    invites = db.scalars(
        select(InviteCode).where(InviteCode.union_id == union_id).order_by(InviteCode.created_at.desc())
    ).all()
    # One grouped pass over usage_events keeps this O(1) queries regardless of
    # how many codes a union has printed.
    usage_rows = db.execute(
        select(
            UsageEvent.invite_code_id,
            func.coalesce(func.sum(UsageEvent.request_count), 0),
            func.coalesce(func.sum(UsageEvent.token_count), 0),
            func.coalesce(func.sum(UsageEvent.estimated_cost_usd), 0.0),
        )
        .where(UsageEvent.union_id == union_id, UsageEvent.invite_code_id.is_not(None))
        .group_by(UsageEvent.invite_code_id)
    ).all()
    usage_by_code = {
        code_id: {"total_requests": int(reqs), "total_tokens": int(tokens), "total_cost_usd": round(float(cost), 6)}
        for code_id, reqs, tokens, cost in usage_rows
    }
    items = []
    for invite in invites:
        serialized = _serialize_invite(invite)
        serialized.update(usage_by_code.get(invite.id, {}))
        items.append(serialized)
    return {"items": items}


@router.post("/unions/{union_id}/invites")
def create_union_invite(
    union_id: str,
    payload: InviteCreateRequest,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    union = db.get(Union, union_id)
    if union is None or not union.is_active:
        raise HTTPException(status_code=404, detail="Union not found.")

    audience = str(payload.audience or "").strip().lower() or InviteAudience.MEMBER.value
    if audience not in (InviteAudience.MEMBER.value, InviteAudience.STEWARD.value):
        raise HTTPException(status_code=400, detail="audience must be 'member' or 'steward'.")

    contract_id = str(payload.contract_id or "").strip() or None
    if audience == InviteAudience.MEMBER.value and not contract_id:
        # Member codes isolate to one contract — a member code without a pin
        # would leak every contract in the union to a rank-and-file scanner.
        raise HTTPException(status_code=400, detail="A member code must be pinned to a contract.")
    if audience == InviteAudience.STEWARD.value and contract_id:
        # Stewards see (and switch between) every contract, so a pin is meaningless.
        raise HTTPException(status_code=400, detail="A steward code cannot be pinned to a single contract.")

    expires_at = None
    if payload.expires_at:
        try:
            expires_at = datetime.fromisoformat(str(payload.expires_at).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="expires_at must be an ISO-8601 timestamp.") from exc

    invite = InviteCode(
        union_id=union_id,
        code=_generate_invite_code(db),
        audience=audience,
        label=str(payload.label or "").strip(),
        contract_id=contract_id,
        created_by=auth.user_id,
        expires_at=expires_at,
        max_uses=payload.max_uses,
    )
    db.add(invite)
    db.flush()
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="invite_code_created",
            event_payload={
                "invite_id": invite.id,
                "audience": invite.audience,
                "label": invite.label,
                "max_uses": invite.max_uses,
            },
        )
    )
    return _serialize_invite(invite)


class InviteRevokeRequest(BaseModel):
    disconnect_sessions: bool = False


@router.post("/unions/{union_id}/invites/{invite_id}/revoke")
def revoke_union_invite(
    union_id: str,
    invite_id: str,
    request: Request,
    payload: InviteRevokeRequest | None = None,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    _require_union_scope(auth, union_id)
    invite = db.get(InviteCode, invite_id)
    if invite is None or invite.union_id != union_id:
        raise HTTPException(status_code=404, detail="Invite code not found.")
    disconnect = bool(payload.disconnect_sessions) if payload is not None else False
    disconnected = 0
    now = datetime.utcnow()
    if invite.revoked_at is None:
        invite.revoked_at = now
        invite.updated_at = now
    if disconnect:
        # Misuse response: also cut off everyone who joined through this placement.
        sessions = db.scalars(
            select(AuthSession).where(
                AuthSession.invite_code_id == invite.id,
                AuthSession.revoked_at.is_(None),
            )
        ).all()
        for session in sessions:
            session.revoked_at = now
            disconnected += 1
    db.add(
        AuditEvent(
            union_id=union_id,
            actor_user_id=auth.user_id,
            event_type="invite_code_revoked",
            event_payload={
                "invite_id": invite.id,
                "label": invite.label,
                "use_count": invite.use_count,
                "disconnect_sessions": disconnect,
                "sessions_disconnected": disconnected,
            },
        )
    )
    result = _serialize_invite(invite)
    result["sessions_disconnected"] = disconnected
    return result


def _join_url_for(request: Request, invite: InviteCode) -> str:
    """The absolute URL a printed QR encodes for this placement."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/j/{invite.code}"


def _load_scoped_invite(db, auth, union_id: str, invite_id: str) -> InviteCode:
    _require_union_scope(auth, union_id)
    invite = db.get(InviteCode, invite_id)
    if invite is None or invite.union_id != union_id:
        raise HTTPException(status_code=404, detail="Invite code not found.")
    return invite


@router.get("/unions/{union_id}/invites/{invite_id}/qr")
def invite_qr_image(
    union_id: str,
    invite_id: str,
    request: Request,
    format: str = "svg",
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    """Render the join URL as a scannable QR image (SVG by default, PNG on request).

    The QR encodes the permanent /j/{code} landing URL, so a placement can be
    repointed without reprinting. SVG stays crisp at any print size; PNG is
    offered for tools that want a raster file.
    """
    import segno

    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    invite = _load_scoped_invite(db, auth, union_id, invite_id)

    fmt = str(format or "svg").strip().lower()
    qr = segno.make(_join_url_for(request, invite), error="m")
    filename = f"karl-qr-{invite.code}.{fmt}"
    disposition = f'inline; filename="{filename}"'

    import io

    buffer = io.BytesIO()
    if fmt == "png":
        qr.save(buffer, kind="png", scale=10, border=2)
        media_type = "image/png"
    elif fmt == "svg":
        qr.save(buffer, kind="svg", scale=10, border=2)
        media_type = "image/svg+xml"
    else:
        raise HTTPException(status_code=400, detail="format must be 'svg' or 'png'.")
    return Response(content=buffer.getvalue(), media_type=media_type, headers={"Content-Disposition": disposition})


@router.get("/unions/{union_id}/invites/{invite_id}/card", response_class=HTMLResponse)
def invite_printable_card(
    union_id: str,
    invite_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    """A print-ready card (QR + code + label) for a placement — open and print or save as PDF."""
    import base64
    import io
    import html as _html

    import segno

    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    invite = _load_scoped_invite(db, auth, union_id, invite_id)
    union = db.get(Union, union_id)
    union_name = union.name if union is not None else "Your Union"

    join_url = _join_url_for(request, invite)
    png_buffer = io.BytesIO()
    segno.make(join_url, error="m").save(png_buffer, kind="png", scale=10, border=2)
    png_data_uri = "data:image/png;base64," + base64.b64encode(png_buffer.getvalue()).decode("ascii")

    is_steward = str(invite.audience or "").strip().lower() == InviteAudience.STEWARD.value
    audience_line = "Steward access, all contracts" if is_steward else "Member access"

    # The printable card echoes the member onboarding scene (see join.html):
    # dark union-blue gradient, soft gold/blue glow, a Playfair-italic hero
    # line, and gold reserved for the single accent. Two faces at standard
    # 3.5in x 2in business-card proportions: a scan-forward front (QR + join
    # code + CTA) and a shield-logo back, laid out for clean printing.
    hero_line = "Every contract, one scan." if is_steward else "Know your contract."
    front_sub = (
        "Every contract in your union, answered with citations."
        if is_steward
        else "Cited answers about your union contract, from your phone."
    )
    cta_line = "Scan for steward access" if is_steward else "Scan to join Karl"

    union_name_esc = _html.escape(union_name)
    code_esc = _html.escape(invite.code)
    # Drop the scheme (https://) so the printed URL fits on one line; the QR
    # still encodes the full join_url.
    join_url_display = _html.escape(join_url.split("://", 1)[-1])

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karl QR card · {code_esc}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@1,600&display=swap" rel="stylesheet">
<style>
  :root {{
    color-scheme: light;
    --blue-dark: #0D3B54; --blue-mid: #14506E; --blue-light: #1B6B8A;
    --gold: #D4A029; --gold-light: #E8B84A;
    --ink-100: #f1f5f9; --ink-300: #cbd5e1; --ink-400: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0; background: #0a1922; color: var(--ink-100);
  }}
  .toolbar {{ position: fixed; top: 16px; right: 16px; z-index: 10; display: flex; gap: 9px; }}
  .btn {{
    border: none; border-radius: 999px; padding: 11px 18px; cursor: pointer;
    font-family: inherit; font-size: 13px; font-weight: 700;
  }}
  .btn:hover {{ filter: brightness(1.06); }}
  .btn-gold {{
    background: linear-gradient(180deg, var(--gold-light) 0%, var(--gold) 100%);
    color: #1a1305; box-shadow: 0 8px 26px -12px rgba(212,160,41,.6);
  }}
  .btn-ghost {{
    background: rgba(255,255,255,.08); color: #e8eef2;
    border: 1px solid rgba(255,255,255,.28);
  }}
  .stage {{
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 22px; padding: 44px 16px 64px;
  }}
  .stage-note {{
    font-size: 11px; letter-spacing: .16em; text-transform: uppercase;
    color: rgba(203,213,225,.55); text-align: center;
  }}
  .card {{
    position: relative; width: 3.5in; height: 2in; border-radius: 14px;
    overflow: hidden; color: var(--ink-100);
    background:
      radial-gradient(circle at 20% 16%, rgba(212,160,41,.20), transparent 40%),
      radial-gradient(circle at 88% 4%, rgba(27,107,138,.42), transparent 46%),
      linear-gradient(135deg, #081420 0%, var(--blue-dark) 46%, var(--blue-mid) 100%);
    box-shadow: 0 18px 44px -22px rgba(0,0,0,.75);
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }}
  /* Larger, legible preview on screen; true 3.5x2in when printed. */
  @media screen {{ .card {{ zoom: 1.7; }} }}
  .front::after {{
    content: ""; position: absolute; left: 16px; right: 16px; bottom: 11px;
    height: 1px; background: linear-gradient(90deg, transparent, rgba(212,160,41,.55), transparent);
  }}

  /* --- Front (scan side): QR + join code on the left, hero + CTA on the right --- */
  .front {{ display: flex; align-items: center; gap: 15px; padding: 14px 17px; }}
  .qr-col {{ flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; }}
  .qr-chip {{
    width: 1.34in; height: 1.34in; background: #fff;
    border-radius: 11px; padding: 6px; box-shadow: 0 4px 14px -6px rgba(0,0,0,.5);
  }}
  .qr-chip img {{ width: 100%; height: 100%; display: block; image-rendering: pixelated; }}
  .qr-code-label {{
    font-size: 5.5px; letter-spacing: .2em; text-transform: uppercase;
    color: var(--ink-400); font-weight: 700; margin-top: 6px;
  }}
  .qr-code {{
    font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
    font-size: 11px; font-weight: 700; letter-spacing: .1em; color: var(--gold-light); margin-top: 1px;
  }}
  .front-body {{ flex: 1 1 auto; min-width: 0; }}
  .eyebrow {{
    font-size: 7px; letter-spacing: .19em; text-transform: uppercase;
    color: var(--gold-light); font-weight: 700; margin: 0 0 5px;
  }}
  .hero {{
    font-family: "Playfair Display", Georgia, serif; font-style: italic;
    font-weight: 600; font-size: 19px; line-height: 1.06; margin: 0 0 6px; color: #fff;
  }}
  .front-sub {{ font-size: 8px; line-height: 1.42; color: var(--ink-300); margin: 0 0 8px; }}
  .cta {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 8.5px; font-weight: 700; letter-spacing: .02em; color: var(--gold-light);
  }}
  .cta svg {{ width: 11px; height: 11px; stroke: var(--gold-light); }}
  .url {{
    font-family: ui-monospace, Consolas, monospace; font-size: 6.5px;
    color: var(--ink-400); white-space: nowrap; margin: 7px 0 0;
  }}

  /* --- Back (logo side): shield mark centered, union + access at the top so
     it reads first in a wallet, brand line beneath the shield. --- */
  .back {{
    display: flex; flex-direction: column; align-items: center;
    justify-content: space-between; text-align: center; padding: 15px 18px 13px;
  }}
  .back .union {{ font-size: 13.5px; font-weight: 700; color: #fff; margin: 0 0 3px; }}
  .back .audience {{
    font-size: 7.5px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--gold-light); font-weight: 700; margin: 0;
  }}
  .back-shield {{ width: 1.3in; height: 1.3in; opacity: .8; }}
  .brand {{
    font-family: ui-monospace, Consolas, monospace; font-size: 6.5px;
    letter-spacing: .2em; text-transform: uppercase; color: rgba(203,213,225,.6); margin: 0;
  }}

  @media print {{
    /* Each face prints as its own exact 3.5x2in page (no Letter whitespace),
       square-cornered so the design bleeds to the trim edge. */
    @page {{ size: 3.5in 2in; margin: 0; }}
    body {{ background: #fff; }}
    .toolbar, .stage-note {{ display: none; }}
    .stage {{ display: block; min-height: auto; padding: 0; gap: 0; }}
    .card {{
      zoom: 1; box-shadow: none; border-radius: 0;
      width: 3.5in; height: 2in; page-break-inside: avoid; break-inside: avoid;
    }}
    .card.front {{ page-break-after: always; break-after: page; }}
  }}
</style></head>
<body data-card-code="{code_esc}">
  <div class="toolbar">
    <button class="btn btn-gold" onclick="window.print()">Print / Save PDF</button>
    <button class="btn btn-ghost" onclick="downloadFace('front')">PNG · Front</button>
    <button class="btn btn-ghost" onclick="downloadFace('back')">PNG · Back</button>
  </div>
  <div class="stage">
    <p class="stage-note">Front · scan side</p>
    <div class="card front">
      <div class="qr-col">
        <div class="qr-chip"><img src="{png_data_uri}" alt="QR code for join code {code_esc}"></div>
        <div class="qr-code-label">Join code</div>
        <div class="qr-code">{code_esc}</div>
      </div>
      <div class="front-body">
        <p class="eyebrow">{union_name_esc} · Karl</p>
        <h1 class="hero">{hero_line}</h1>
        <p class="front-sub">{front_sub}</p>
        <span class="cta">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/></svg>
          {cta_line}
        </span>
        <p class="url">{join_url_display}</p>
      </div>
    </div>
    <p class="stage-note">Back · logo side</p>
    <div class="card back">
      <div class="back-top">
        <p class="union">{union_name_esc}</p>
        <p class="audience">{audience_line}</p>
      </div>
      <svg class="back-shield" viewBox="12 22 176 176" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 190C50 190,20 150,20 40Q60 40,100 30Z" fill="#4A7A9F"/>
        <path d="M100 190C150 190,180 150,180 40Q140 40,100 30Z" fill="#EECF6D"/>
      </svg>
      <p class="brand">Powered by Karl · Karl Stewardship</p>
    </div>
  </div>
</body></html>"""

    # Client-side high-DPI PNG export of each face for print shops. Uses
    # html2canvas (vendored same-origin, no CDN), which paints the real DOM to
    # a canvas — reliable across browsers (unlike SVG-foreignObject, which
    # taints the canvas in Firefox/stricter Chrome) and renders the actual
    # Playfair font. Kept as a plain string so its JS braces don't collide with
    # the f-string above.
    download_script = """
<script src="/static/modular/vendor/html2canvas.min.js"></script>
<script>
(function () {
  function fail() {
    window.alert('Could not export a PNG. Use \\u201CPrint / Save PDF\\u201D instead.');
  }
  window.downloadFace = function (which) {
    var card = document.querySelector('.card.' + which);
    if (!card || typeof html2canvas === 'undefined') { fail(); return; }
    var W = 3.5 * 96, H = 2 * 96;
    var holder = document.createElement('div');
    holder.style.cssText = 'position:fixed;left:-99999px;top:0;';
    var clone = card.cloneNode(true);
    clone.style.zoom = '1';
    clone.style.margin = '0';
    holder.appendChild(clone);
    document.body.appendChild(holder);
    var done = false;
    function finish(ok) { if (done) { return; } done = true; holder.remove(); if (!ok) { fail(); } }
    var ready = (document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve();
    ready.then(function () {
      return html2canvas(clone, { scale: 4, backgroundColor: null, width: W, height: H, logging: false });
    }).then(function (canvas) {
      canvas.toBlob(function (blob) {
        if (!blob) { finish(false); return; }
        var code = document.body.getAttribute('data-card-code') || 'card';
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'karl-card-' + code + '-' + which + '.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 1500);
        finish(true);
      }, 'image/png');
    }).catch(function () { finish(false); });
  };
})();
</script>
"""
    page = page.replace("</body></html>", download_script + "</body></html>")
    return HTMLResponse(content=page)
