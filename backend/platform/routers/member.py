from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select

from backend.platform.auth import AuthContext
from backend.platform.db import apply_request_context, apply_service_bootstrap_context
from backend.platform.deps import get_auth_context, get_container, get_db
from backend.platform.models import ChunkEmbedding, Document, Role


router = APIRouter(prefix="/api/member", tags=["member"])


def _member_document_safety_state(document: Document) -> dict:
    metadata = dict(document.metadata_json or {})
    return {
        "member_visible": bool(metadata.get("member_visible", True)),
        "prompt_injection_risk": bool(metadata.get("prompt_injection_risk")),
        "sensitive_data_risk": bool(metadata.get("sensitive_data_risk")),
        "safety_review_status": str(metadata.get("safety_review_status") or "").strip().lower(),
        "safety_status": str(metadata.get("safety_status") or "").strip().lower(),
    }


def _require_member_safe_document(request: Request, document: Document, *, allow_redacted_selection: bool = False) -> None:
    auth = get_auth_context(request)
    if auth is not None and auth.is_super_admin:
        return
    safety = _member_document_safety_state(document)
    if safety["prompt_injection_risk"] or not safety["member_visible"]:
        raise HTTPException(status_code=409, detail="This document is temporarily unavailable while it is under safety review.")
    if safety["sensitive_data_risk"] and not allow_redacted_selection:
        raise HTTPException(status_code=409, detail="This document contains sensitive data and cannot be opened in full from the member view.")


def _resolve_member_document_auth(request: Request, *, access_token: str | None) -> tuple:
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if access_token:
        apply_service_bootstrap_context(db)
        resolved = get_container(request).auth_adapter.resolve(
            db=db,
            session_cookie=None,
            authorization=f"Bearer {access_token}",
            external_auth_id=None,
            email=None,
            full_name=None,
            requested_role=None,
            union_slug=None,
        )
        if resolved and resolved.is_authenticated:
            auth = resolved
        apply_request_context(db, auth)
        request.state.auth_context = auth
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return db, auth


@router.get("/documents/{document_id}/content")
def get_member_document_content(
    document_id: str,
    request: Request,
    access_token: str | None = Query(default=None),
):
    db, auth = _resolve_member_document_auth(request, access_token=access_token)
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not auth.is_super_admin and auth.union_id != document.union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    _require_member_safe_document(request, document, allow_redacted_selection=False)

    path = get_container(request).storage.open(document.storage_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored document file not found.")

    return FileResponse(
        path,
        media_type=document.content_type or "application/octet-stream",
        filename=document.title,
        headers={"Content-Disposition": f'inline; filename="{document.title}"'},
    )


@router.get("/documents/{document_id}/selection")
def get_member_document_selection(
    document_id: str,
    request: Request,
    article_num: str | None = Query(default=None),
    section_num: str | None = Query(default=None),
    chunk_index: int | None = Query(default=None),
    access_token: str | None = Query(default=None),
):
    db, auth = _resolve_member_document_auth(request, access_token=access_token)
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not auth.is_super_admin and auth.union_id != document.union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    _require_member_safe_document(request, document, allow_redacted_selection=True)

    rows = db.scalars(
        select(ChunkEmbedding)
        .where(ChunkEmbedding.document_id == document_id)
        .order_by(ChunkEmbedding.chunk_index.asc())
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No indexed document selections were found.")

    normalized_article = str(article_num or "").strip() or None
    normalized_section = str(section_num or "").strip() or None
    selected = None
    for row in rows:
        metadata = dict(row.metadata_json or {})
        if normalized_article and str(metadata.get("article_num") or "").strip() != normalized_article:
            continue
        if normalized_section and str(metadata.get("section_num") or "").strip() != normalized_section:
            continue
        if chunk_index is not None and int(row.chunk_index) != int(chunk_index):
            continue
        selected = row
        break
    if selected is None and chunk_index is not None:
        for row in rows:
            if int(row.chunk_index) == int(chunk_index):
                selected = row
                break
    if selected is None:
        selected = rows[0]

    metadata = dict(selected.metadata_json or {})
    guardrails = get_container(request).guardrails
    excerpt = str(selected.chunk_text or "")
    redacted = guardrails.redact_sensitive_text(excerpt)
    summary = str(metadata.get("summary") or "")
    redacted_summary = guardrails.redact_sensitive_text(summary).sanitized_text if summary else None
    return {
        "document_id": document.id,
        "document_title": document.title,
        "article_num": metadata.get("article_num"),
        "article_title": metadata.get("article_title"),
        "section_num": metadata.get("section_num"),
        "section_title": metadata.get("section_title"),
        "page_start": metadata.get("page_start") or metadata.get("page_number"),
        "page_end": metadata.get("page_end"),
        "summary": redacted_summary,
        "topic_tags": metadata.get("topic_tags") or [],
        "excerpt": redacted.sanitized_text,
        "chunk_index": selected.chunk_index,
        "safety_redacted": bool(redacted.reasons),
    }
