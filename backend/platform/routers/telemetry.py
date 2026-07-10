from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import or_, select

from backend.platform.deps import get_auth_context, get_container, get_db
from backend.platform.models import Role, Union


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class TelemetryEventRequest(BaseModel):
    category: str
    event_type: str
    route: str | None = None
    metadata: dict | None = None
    session_id: str | None = None
    union_slug: str | None = None
    surface: str | None = None


def _resolve_union(request: Request, union_slug: str | None):
    db = get_db(request)
    if db is None or not union_slug:
        return None
    return db.scalar(select(Union).where(or_(Union.slug == union_slug, Union.union_local_id == union_slug)))


@router.post("/event")
def capture_event(payload: TelemetryEventRequest, request: Request):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    auth = get_auth_context(request)
    category = str(payload.category or "").strip().lower()
    event_type = str(payload.event_type or "").strip().lower()
    if category not in {"bug_journey", "usage_ux"}:
        raise HTTPException(status_code=422, detail="Unsupported telemetry category.")
    if not event_type:
        raise HTTPException(status_code=422, detail="Event type is required.")

    requested_union_slug = (
        str(payload.union_slug or "").strip()
        or str(getattr(request.state, "tenant_slug", None) or "").strip()
        or auth.union_slug
    )
    requested_union = _resolve_union(request, requested_union_slug)
    if auth.is_authenticated and auth.union_id and requested_union is not None and not auth.is_super_admin and requested_union.id != auth.union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")

    effective_union_id = auth.union_id or (requested_union.id if requested_union is not None else None)
    surface = str(payload.surface or "").strip().lower()
    is_member = auth.role == Role.USER.value if auth.is_authenticated else surface == "member"

    container.telemetry.record_event(
        db,
        auth if auth.is_authenticated else None,
        category=category,
        event_type=event_type,
        route=str(payload.route or "").strip() or None,
        metadata=dict(payload.metadata or {}),
        session_id=str(payload.session_id or "").strip() or None,
        union_id=effective_union_id,
        is_member=is_member,
    )
    return {"ok": True, "captured": True}
