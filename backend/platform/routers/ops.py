"""Operational visibility endpoints for quota and sentinel state."""

from __future__ import annotations

from collections import defaultdict
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from backend.platform.deps import get_auth_context, get_db, require_roles
from backend.platform.models import AuthSession, Document, IngestionJob, Notification, NotificationStatus, Role, SecurityEvent, TelemetryEvent, Union, UsageEvent


router = APIRouter(prefix="/api/ops", tags=["ops"])

UNRESOLVED_REVIEW_STATES = {
    "pending_ingestion",
    "retry_pending",
    "retrying_with_ocr",
    "needs_review",
    "in_review",
    "escalated",
    "failed",
    "blocked_pending_superadmin",
}


def _serialize_dashboard(
    *,
    documents,
    usage_events,
    telemetry_events,
    notifications,
    security_events,
    sessions,
    union_scope_label: str,
):
    now = datetime.utcnow()
    since_24h = now - timedelta(hours=24)
    since_7d = now - timedelta(days=7)
    day_keys = [(now - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(6, -1, -1)]
    day_labels = [(now - timedelta(days=offset)).strftime("%b %-d") for offset in range(6, -1, -1)]

    request_series = defaultdict(int)
    active_users_series: dict[str, set[str]] = {key: set() for key in day_keys}
    security_series = defaultdict(int)
    notification_series = defaultdict(int)
    session_series = defaultdict(int)
    telemetry_usage_series = defaultdict(int)
    telemetry_journey_series = defaultdict(int)
    query_failure_series = defaultdict(int)

    requests_last_24h = 0
    tokens_last_24h = 0
    estimated_cost_last_24h = 0.0
    active_users_7d: set[str] = set()
    query_failures_7d = 0
    source_opens_7d = 0
    member_workspace_loads_7d = 0
    login_failures_7d = 0

    for item in usage_events:
        if item.created_at < since_7d:
            continue
        day_key = item.created_at.strftime("%Y-%m-%d")
        request_series[day_key] += int(item.request_count or 0)
        if item.user_id:
            active_users_series.setdefault(day_key, set()).add(item.user_id)
            active_users_7d.add(item.user_id)
        if item.created_at >= since_24h:
            requests_last_24h += int(item.request_count or 0)
            tokens_last_24h += int(item.token_count or 0)
            estimated_cost_last_24h += float(item.estimated_cost_usd or 0.0)

    for item in telemetry_events:
        if item.created_at < since_7d:
            continue
        day_key = item.created_at.strftime("%Y-%m-%d")
        if item.category == "usage_ux":
            telemetry_usage_series[day_key] += 1
        elif item.category == "bug_journey":
            telemetry_journey_series[day_key] += 1
        if item.event_type == "query_failed":
            query_failures_7d += 1
            query_failure_series[day_key] += 1
        elif item.event_type == "source_opened":
            source_opens_7d += 1
        elif item.event_type == "member_workspace_loaded":
            member_workspace_loads_7d += 1
        elif item.event_type == "session_login_failed":
            login_failures_7d += 1

    for item in security_events:
        if item.created_at < since_7d:
            continue
        day_key = item.created_at.strftime("%Y-%m-%d")
        security_series[day_key] += 1

    pending_alerts = 0
    for item in notifications:
        if item.status == NotificationStatus.PENDING:
            pending_alerts += 1
        if item.created_at < since_7d:
            continue
        day_key = item.created_at.strftime("%Y-%m-%d")
        notification_series[day_key] += 1

    sign_ins_7d = 0
    for item in sessions:
        if item.created_at < since_7d:
            continue
        sign_ins_7d += 1
        day_key = item.created_at.strftime("%Y-%m-%d")
        session_series[day_key] += 1

    open_review_items = sum(
        1
        for item in documents
        if str((item.metadata_json or {}).get("review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
        or str((item.metadata_json or {}).get("safety_review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
    )
    ready_documents = sum(
        1
        for item in documents
        if item.status.value == "active" and bool((item.metadata_json or {}).get("ready_for_query"))
    )
    documents_needing_attention = sum(
        1
        for item in documents
        if item.status.value == "failed"
        or str((item.metadata_json or {}).get("review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
        or str((item.metadata_json or {}).get("safety_review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
    )
    uploads_7d = sum(1 for item in documents if item.created_at >= since_7d)

    return {
        "captured_at": now.isoformat(),
        "scope_label": union_scope_label,
        "summary": {
            "requests_last_24h": requests_last_24h,
            "tokens_last_24h": tokens_last_24h,
            "estimated_cost_last_24h": round(estimated_cost_last_24h, 4),
            "active_users_7d": len(active_users_7d),
            "sign_ins_7d": sign_ins_7d,
            "login_failures_7d": login_failures_7d,
            "open_review_items": open_review_items,
            "pending_alerts": pending_alerts,
            "security_events_7d": sum(1 for item in security_events if item.created_at >= since_7d),
            "query_failures_7d": query_failures_7d,
            "source_opens_7d": source_opens_7d,
            "member_workspace_loads_7d": member_workspace_loads_7d,
            "journey_events_7d": sum(1 for item in telemetry_events if item.created_at >= since_7d and item.category == "bug_journey"),
            "usage_events_7d": sum(1 for item in telemetry_events if item.created_at >= since_7d and item.category == "usage_ux"),
            "ready_documents": ready_documents,
            "documents_needing_attention": documents_needing_attention,
            "uploads_7d": uploads_7d,
        },
        "trends": {
            "labels": day_labels,
            "requests": [request_series.get(key, 0) for key in day_keys],
            "active_users": [len(active_users_series.get(key, set())) for key in day_keys],
            "security_events": [security_series.get(key, 0) for key in day_keys],
            "notifications": [notification_series.get(key, 0) for key in day_keys],
            "sign_ins": [session_series.get(key, 0) for key in day_keys],
            "journey_events": [telemetry_journey_series.get(key, 0) for key in day_keys],
            "usage_events": [telemetry_usage_series.get(key, 0) for key in day_keys],
            "query_failures": [query_failure_series.get(key, 0) for key in day_keys],
        },
    }


@router.get("/dashboard")
def operations_dashboard(
    request: Request,
    union_id: str | None = Query(default=None),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {
            "captured_at": datetime.utcnow().isoformat(),
            "scope_label": "Unavailable",
            "summary": {},
            "trends": {"labels": [], "requests": [], "active_users": [], "security_events": [], "notifications": [], "sign_ins": []},
        }
    auth = get_auth_context(request)

    effective_union_id = union_id if auth.is_super_admin else auth.union_id
    union_scope_label = "All unions"
    if effective_union_id:
        union = db.get(Union, effective_union_id)
        if union is None:
            raise HTTPException(status_code=404, detail="Union not found.")
        union_scope_label = union.name

    documents_stmt = select(Document)
    usage_stmt = select(UsageEvent)
    notifications_stmt = select(Notification)
    security_stmt = select(SecurityEvent)
    sessions_stmt = select(AuthSession)
    telemetry_stmt = select(TelemetryEvent)

    if effective_union_id:
        documents_stmt = documents_stmt.where(Document.union_id == effective_union_id)
        usage_stmt = usage_stmt.where(UsageEvent.union_id == effective_union_id)
        notifications_stmt = notifications_stmt.where((Notification.union_id == effective_union_id) | (Notification.union_id.is_(None)))
        security_stmt = security_stmt.where(SecurityEvent.union_id == effective_union_id)
        sessions_stmt = sessions_stmt.where(AuthSession.union_id == effective_union_id)
        telemetry_stmt = telemetry_stmt.where(TelemetryEvent.union_id == effective_union_id)

    documents = db.scalars(documents_stmt).all()
    usage_events = db.scalars(usage_stmt).all()
    telemetry_events = db.scalars(telemetry_stmt).all()
    notifications = db.scalars(notifications_stmt).all()
    security_events = db.scalars(security_stmt).all()
    sessions = db.scalars(sessions_stmt).all()

    return _serialize_dashboard(
        documents=documents,
        usage_events=usage_events,
        telemetry_events=telemetry_events,
        notifications=notifications,
        security_events=security_events,
        sessions=sessions,
        union_scope_label=union_scope_label,
    )


@router.get("/security-events")
def list_security_events(
    request: Request,
    union_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=50),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
    auth = get_auth_context(request)
    effective_union_id = union_id if auth.is_super_admin and union_id else auth.union_id
    stmt = select(SecurityEvent).order_by(SecurityEvent.created_at.desc())
    if effective_union_id:
        stmt = stmt.where(SecurityEvent.union_id == effective_union_id)
    items = db.scalars(stmt).all()
    total = len(items)
    start = (page - 1) * page_size
    paged_items = items[start:start + page_size]
    return {"items": [
        {
            "id": item.id,
            "union_id": item.union_id,
            "event_type": item.event_type,
            "severity": item.severity.value,
            "response_action": item.response_action,
            "details": item.details_json,
            "created_at": item.created_at.isoformat(),
        }
        for item in paged_items
    ], "page": page, "page_size": page_size, "total": total}


@router.get("/notifications")
def list_notifications(
    request: Request,
    status: str | None = Query(default=None),
    include_acknowledged: bool = Query(default=False),
    review_only: bool = Query(default=False),
    union_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=50),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
    auth = get_auth_context(request)
    effective_union_id = union_id if auth.is_super_admin and union_id else auth.union_id
    stmt = select(Notification).order_by(Notification.created_at.desc())
    if effective_union_id:
        stmt = stmt.where((Notification.union_id == effective_union_id) | (Notification.union_id.is_(None)))
    normalized_status = str(status or "").strip().lower()
    if normalized_status:
        stmt = stmt.where(Notification.status == NotificationStatus(normalized_status))
    elif not include_acknowledged:
        stmt = stmt.where(Notification.status != NotificationStatus.ACKNOWLEDGED)
    if review_only:
        stmt = stmt.where(
            Notification.subject.ilike("Document %")
            | Notification.subject.ilike("Security alert: ingestion_review_escalated")
            | Notification.subject.ilike("Union alert: ingestion_review_required")
        )
    items = db.scalars(stmt).all()
    total = len(items)
    start = (page - 1) * page_size
    paged_items = items[start:start + page_size]
    return {"items": [
        {
            "id": item.id,
            "union_id": item.union_id,
            "channel": item.channel,
            "subject": item.subject,
            "body": item.body,
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
        }
        for item in paged_items
    ], "page": page, "page_size": page_size, "total": total}


@router.get("/telemetry-events")
def list_telemetry_events(
    request: Request,
    category: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    q: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    union_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=8, ge=1, le=50),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
    auth = get_auth_context(request)
    effective_union_id = union_id if auth.is_super_admin and union_id else auth.union_id

    items = db.scalars(select(TelemetryEvent).order_by(TelemetryEvent.created_at.desc()).limit(800)).all()
    union_names = {item.id: item.name for item in db.scalars(select(Union)).all()} if auth.is_super_admin else {}
    normalized_category = str(category or "").strip().lower()
    normalized_event_type = str(event_type or "").strip().lower()
    normalized_query = str(q or "").strip().lower()
    normalized_session_id = str(session_id or "").strip()

    filtered = []
    for item in items:
        if effective_union_id and item.union_id != effective_union_id:
            continue
        if normalized_category and item.category != normalized_category:
            continue
        if normalized_event_type and normalized_event_type not in str(item.event_type or "").strip().lower():
            continue
        if normalized_session_id and str(item.session_id or "").strip() != normalized_session_id:
            continue
        if normalized_query:
            haystacks = [
                str(item.route or "").lower(),
                str(item.category or "").lower(),
                str(item.event_type or "").lower(),
                str(item.session_id or "").lower(),
                str(item.anonymized_user_key or "").lower(),
                json.dumps(item.metadata_json or {}, sort_keys=True).lower(),
            ]
            if auth.is_super_admin:
                haystacks.append(str(union_names.get(item.union_id, "")).lower())
            if not any(normalized_query in haystack for haystack in haystacks):
                continue
        filtered.append(item)

    total = len(filtered)
    start = (page - 1) * page_size
    paged_items = filtered[start:start + page_size]
    return {
        "items": [
            {
                "id": item.id,
                "union_id": item.union_id,
                "union_name": union_names.get(item.union_id),
                "user_id": item.user_id,
                "anonymized_user_key": item.anonymized_user_key,
                "session_id": item.session_id,
                "route": item.route,
                "category": item.category,
                "event_type": item.event_type,
                "metadata": item.metadata_json or {},
                "created_at": item.created_at.isoformat(),
            }
            for item in paged_items
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.get("/telemetry-events/session/{session_id}")
def get_telemetry_session_timeline(
    session_id: str,
    request: Request,
    union_id: str | None = Query(default=None),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"session_id": session_id, "items": [], "summary": {"total_events": 0}}
    auth = get_auth_context(request)
    effective_union_id = union_id if auth.is_super_admin and union_id else auth.union_id

    stmt = (
        select(TelemetryEvent)
        .where(TelemetryEvent.session_id == session_id)
        .order_by(TelemetryEvent.created_at.asc())
        .limit(500)
    )
    if effective_union_id:
        stmt = stmt.where(TelemetryEvent.union_id == effective_union_id)

    items = db.scalars(stmt).all()
    union_names = {item.id: item.name for item in db.scalars(select(Union)).all()} if auth.is_super_admin else {}
    categories = sorted({item.category for item in items if item.category})
    event_types = [item.event_type for item in items if item.event_type]
    most_recent_union_id = items[-1].union_id if items else effective_union_id

    return {
        "session_id": session_id,
        "union_id": most_recent_union_id,
        "union_name": union_names.get(most_recent_union_id) if most_recent_union_id else None,
        "summary": {
            "total_events": len(items),
            "categories": categories,
            "event_types": event_types,
            "started_at": items[0].created_at.isoformat() if items else None,
            "ended_at": items[-1].created_at.isoformat() if items else None,
        },
        "items": [
            {
                "id": item.id,
                "union_id": item.union_id,
                "union_name": union_names.get(item.union_id),
                "user_id": item.user_id,
                "anonymized_user_key": item.anonymized_user_key,
                "session_id": item.session_id,
                "route": item.route,
                "category": item.category,
                "event_type": item.event_type,
                "metadata": item.metadata_json or {},
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ],
    }


@router.get("/review-queue")
def list_review_queue(
    request: Request,
    include_acknowledged: bool = Query(default=False),
    union_id: str | None = Query(default=None),
    review_status: str | None = Query(default=None),
    status: str | None = Query(default=None),
    q: str | None = Query(default=None),
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        return {"items": [], "summary": {"unresolved_documents": 0, "pending_notifications": 0}}
    auth = get_auth_context(request)

    document_stmt = select(Document).order_by(Document.updated_at.desc()).limit(400)
    if not auth.is_super_admin and auth.union_id:
        document_stmt = document_stmt.where(Document.union_id == auth.union_id)
    elif auth.is_super_admin and union_id:
        document_stmt = document_stmt.where(Document.union_id == union_id)
    documents = db.scalars(document_stmt).all()
    normalized_review_status = str(review_status or "").strip().lower()
    normalized_status = str(status or "").strip().lower()
    normalized_query = str(q or "").strip().lower()
    unresolved_documents = [
        document
        for document in documents
        if (
            str((document.metadata_json or {}).get("review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
            or str((document.metadata_json or {}).get("safety_review_status") or "").strip().lower() in UNRESOLVED_REVIEW_STATES
        )
        and (
            not normalized_review_status
            or str((document.metadata_json or {}).get("review_status") or "").strip().lower() == normalized_review_status
            or str((document.metadata_json or {}).get("safety_review_status") or "").strip().lower() == normalized_review_status
        )
        and (not normalized_status or document.status.value.lower() == normalized_status)
        and (
            not normalized_query
            or normalized_query in str(document.title or "").lower()
            or normalized_query in str((document.metadata_json or {}).get("quality_reason") or "").lower()
            or normalized_query in str((document.metadata_json or {}).get("recommended_action") or "").lower()
            or normalized_query in " ".join(str(reason).lower() for reason in ((document.metadata_json or {}).get("safety_reasons") or []))
        )
    ]

    notification_stmt = select(Notification).order_by(Notification.created_at.desc()).limit(400)
    if not auth.is_super_admin and auth.union_id:
        notification_stmt = notification_stmt.where((Notification.union_id == auth.union_id) | (Notification.union_id.is_(None)))
    elif auth.is_super_admin and union_id:
        notification_stmt = notification_stmt.where((Notification.union_id == union_id) | (Notification.union_id.is_(None)))
    notifications = db.scalars(notification_stmt).all()
    review_notifications = [
        notification
        for notification in notifications
        if (
            notification.subject.startswith("Document ")
            or notification.subject == "Security alert: ingestion_review_escalated"
            or notification.subject == "Union alert: ingestion_review_required"
        )
        and (include_acknowledged or notification.status != NotificationStatus.ACKNOWLEDGED)
    ]
    notifications_by_union: dict[str | None, list[Notification]] = {}
    for notification in review_notifications:
        notifications_by_union.setdefault(notification.union_id, []).append(notification)

    items = []
    union_names = {item.id: item.name for item in db.scalars(select(Union)).all()} if auth.is_super_admin else {}
    for document in unresolved_documents:
        latest_job = db.scalar(
            select(IngestionJob)
            .where(IngestionJob.document_id == document.id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        relevant_notifications = notifications_by_union.get(document.union_id, [])
        items.append(
            {
                "document_id": document.id,
                "union_id": document.union_id,
                "union_name": union_names.get(document.union_id),
                "title": document.title,
                "status": document.status.value,
                "quality_status": (document.metadata_json or {}).get("quality_status"),
                "quality_reason": (document.metadata_json or {}).get("quality_reason"),
                "ocr_status": (document.metadata_json or {}).get("ocr_status"),
                "scan_likelihood": (document.metadata_json or {}).get("scan_likelihood"),
                "review_status": (document.metadata_json or {}).get("review_status"),
                "safety_status": (document.metadata_json or {}).get("safety_status"),
                "safety_reasons": (document.metadata_json or {}).get("safety_reasons") or [],
                "prompt_injection_risk": bool((document.metadata_json or {}).get("prompt_injection_risk")),
                "sensitive_data_risk": bool((document.metadata_json or {}).get("sensitive_data_risk")),
                "member_visible": bool((document.metadata_json or {}).get("member_visible", True)),
                "safety_review_status": (document.metadata_json or {}).get("safety_review_status"),
                "recommended_action": (document.metadata_json or {}).get("recommended_action"),
                "ready_for_query": bool((document.metadata_json or {}).get("ready_for_query")),
                "updated_at": document.updated_at.isoformat(),
                "latest_job": None
                if latest_job is None
                else {
                    "id": latest_job.id,
                    "status": latest_job.status.value,
                    "error_message": latest_job.error_message,
                    "metadata": latest_job.metadata_json,
                    "created_at": latest_job.created_at.isoformat(),
                },
                "notifications": [
                    {
                        "id": notification.id,
                        "subject": notification.subject,
                        "status": notification.status.value,
                        "created_at": notification.created_at.isoformat(),
                    }
                    for notification in relevant_notifications[:5]
                ],
            }
        )

    return {
        "items": items,
        "summary": {
            "unresolved_documents": len(items),
            "pending_notifications": sum(1 for notification in review_notifications if notification.status == NotificationStatus.PENDING),
            "acknowledged_notifications": sum(1 for notification in review_notifications if notification.status == NotificationStatus.ACKNOWLEDGED),
        },
    }


@router.post("/notifications/{notification_id}/acknowledge")
def acknowledge_notification(
    notification_id: str,
    request: Request,
    _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value)),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    notification = db.get(Notification, notification_id)
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found.")
    if not auth.is_super_admin:
        if notification.union_id is None or notification.union_id != auth.union_id:
            raise HTTPException(status_code=404, detail="Notification not found.")
    notification.status = NotificationStatus.ACKNOWLEDGED
    db.flush()
    return {
        "notification": {
            "id": notification.id,
            "status": notification.status.value,
            "acknowledged_at": datetime.utcnow().isoformat(),
        }
    }


@router.get("/usage")
def list_usage(request: Request, _auth=Depends(require_roles(Role.UNION_ADMIN.value, Role.SUPER_ADMIN.value))):
    db = get_db(request)
    if db is None:
        return {"items": []}
    auth = get_auth_context(request)
    stmt = select(UsageEvent).order_by(UsageEvent.created_at.desc()).limit(100)
    if not auth.is_super_admin and auth.union_id:
        stmt = stmt.where(UsageEvent.union_id == auth.union_id)
    items = db.scalars(stmt).all()
    return {"items": [
        {
            "id": item.id,
            "route": item.route,
            "token_count": item.token_count,
            "estimated_cost_usd": item.estimated_cost_usd,
            "metadata": item.metadata_json,
            "created_at": item.created_at.isoformat(),
        }
        for item in items
    ]}
