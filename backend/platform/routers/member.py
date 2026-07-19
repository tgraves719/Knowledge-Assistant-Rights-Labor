from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from backend.platform.auth import AuthContext
from backend.platform.db import apply_request_context, apply_service_bootstrap_context
from backend.platform.deps import get_auth_context, get_container, get_db
from backend.platform.models import AuditEvent, ChunkEmbedding, Document, DocumentStatus, Role, UnionMembership
from backend.platform.routers.admin import _purge_user_records


router = APIRouter(prefix="/api/member", tags=["member"])


class MemberDataDeletionRequest(BaseModel):
    confirm: bool = False


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


@router.get("/contracts/{contract_id}/outline")
def get_member_contract_outline(
    contract_id: str,
    request: Request,
    access_token: str | None = Query(default=None),
):
    """Article/section table of contents for the contract explorer.

    Built from indexed chunk metadata rather than the on-disk contract packs,
    so a tenant workspace can browse its own uploaded contract. Sections are
    grouped under their article and ordered by chunk position, which is the
    order the contract reads in.
    """
    db, auth = _resolve_member_document_auth(request, access_token=access_token)
    union_id = auth.union_id
    if not union_id and not auth.is_super_admin:
        raise HTTPException(status_code=403, detail="No union scope for this session.")

    document_stmt = select(Document).where(
        Document.status == DocumentStatus.ACTIVE,
        Document.contract_id == contract_id,
    )
    if not auth.is_super_admin:
        document_stmt = document_stmt.where(Document.union_id == union_id)
    documents = db.scalars(document_stmt).all()
    documents = [
        document
        for document in documents
        if bool((document.metadata_json or {}).get("ready_for_query"))
        and bool((document.metadata_json or {}).get("member_visible", True))
    ]
    if not documents:
        raise HTTPException(status_code=404, detail="No readable documents for this contract.")

    document_ids = [document.id for document in documents]
    rows = db.scalars(
        select(ChunkEmbedding)
        .where(ChunkEmbedding.document_id.in_(document_ids))
        .order_by(ChunkEmbedding.chunk_index.asc())
    ).all()

    articles: dict[str, dict] = {}
    seen_sections: set[tuple[str, str]] = set()
    for row in rows:
        metadata = dict(row.metadata_json or {})
        article_num = str(metadata.get("article_num") or "").strip()
        section_num = str(metadata.get("section_num") or "").strip()
        if not article_num:
            continue
        article = articles.setdefault(
            article_num,
            {
                "article_num": article_num,
                "article_title": str(metadata.get("article_title") or "").strip() or f"Article {article_num}",
                "sections": [],
            },
        )
        if not section_num or (article_num, section_num) in seen_sections:
            continue
        seen_sections.add((article_num, section_num))
        article["sections"].append(
            {
                "section_num": section_num,
                "section_label": str(metadata.get("section_label") or metadata.get("section_title") or "").strip(),
                "anchor_id": str(metadata.get("anchor_id") or "").strip() or None,
                "page": metadata.get("source_page"),
                "document_id": row.document_id,
            }
        )

    def _sort_key(value: str) -> tuple:
        try:
            return (0, int(value), "")
        except (TypeError, ValueError):
            return (1, 0, str(value))

    outline = sorted(articles.values(), key=lambda item: _sort_key(item["article_num"]))
    for article in outline:
        article["sections"].sort(key=lambda item: _sort_key(item["section_num"]))

    primary = documents[0]
    return {
        "contract_id": contract_id,
        "document_id": primary.id,
        "document_title": primary.title,
        "has_source_pdf": bool(primary.source_pdf_key),
        "source_pdf_url": f"/api/member/documents/{primary.id}/source-pdf" if primary.source_pdf_key else None,
        "total_articles": len(outline),
        "total_sections": len(seen_sections),
        "articles": outline,
    }


@router.get("/contracts/{contract_id}/section")
def get_member_contract_section(
    contract_id: str,
    request: Request,
    article_num: str | None = Query(default=None),
    section_num: str | None = Query(default=None),
    anchor_id: str | None = Query(default=None),
    access_token: str | None = Query(default=None),
):
    """Full text of one article or section, for the explorer's reading pane."""
    db, auth = _resolve_member_document_auth(request, access_token=access_token)
    union_id = auth.union_id
    if not union_id and not auth.is_super_admin:
        raise HTTPException(status_code=403, detail="No union scope for this session.")

    document_stmt = select(Document).where(
        Document.status == DocumentStatus.ACTIVE,
        Document.contract_id == contract_id,
    )
    if not auth.is_super_admin:
        document_stmt = document_stmt.where(Document.union_id == union_id)
    documents = [
        document
        for document in db.scalars(document_stmt).all()
        if bool((document.metadata_json or {}).get("ready_for_query"))
        and bool((document.metadata_json or {}).get("member_visible", True))
    ]
    if not documents:
        raise HTTPException(status_code=404, detail="No readable documents for this contract.")

    rows = db.scalars(
        select(ChunkEmbedding)
        .where(ChunkEmbedding.document_id.in_([document.id for document in documents]))
        .order_by(ChunkEmbedding.chunk_index.asc())
    ).all()

    matches = []
    for row in rows:
        metadata = dict(row.metadata_json or {})
        if anchor_id:
            if str(metadata.get("anchor_id") or "").strip() != anchor_id:
                continue
        else:
            if article_num and str(metadata.get("article_num") or "").strip() != str(article_num).strip():
                continue
            if section_num and str(metadata.get("section_num") or "").strip() != str(section_num).strip():
                continue
        matches.append((row, metadata))

    if not matches:
        raise HTTPException(status_code=404, detail="No matching contract section was found.")

    first_metadata = matches[0][1]
    pages = sorted({m.get("source_page") for _, m in matches if isinstance(m.get("source_page"), int)})
    return {
        "contract_id": contract_id,
        "article_num": str(first_metadata.get("article_num") or "").strip() or None,
        "article_title": str(first_metadata.get("article_title") or "").strip() or None,
        "section_num": str(first_metadata.get("section_num") or "").strip() or None,
        "section_label": str(first_metadata.get("section_label") or "").strip() or None,
        "anchor_id": str(first_metadata.get("anchor_id") or "").strip() or None,
        "page": pages[0] if pages else None,
        "pages": pages,
        "document_id": matches[0][0].document_id,
        "content": "\n\n".join(str(row.chunk_text or "").strip() for row, _ in matches if row.chunk_text),
    }


@router.get("/documents/{document_id}/source-pdf")
def get_member_document_source_pdf(
    document_id: str,
    request: Request,
    access_token: str | None = Query(default=None),
):
    """Serve the printed contract PDF behind a document's citations.

    Same tenant scoping and member-safety checks as the extracted content --
    the PDF is the same material, so it must not become a way around them.
    """
    db, auth = _resolve_member_document_auth(request, access_token=access_token)
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not auth.is_super_admin and auth.union_id != document.union_id:
        raise HTTPException(status_code=403, detail="Union scope mismatch.")
    _require_member_safe_document(request, document, allow_redacted_selection=False)

    if not document.source_pdf_key:
        raise HTTPException(status_code=404, detail="No source PDF is attached to this document.")

    path = get_container(request).storage.open(document.source_pdf_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored source PDF was not found.")

    filename = str(document.source_pdf_key).rsplit("/", 1)[-1] or "contract.pdf"
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
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


@router.delete("/me/data")
def delete_my_data(payload: MemberDataDeletionRequest, request: Request):
    """Member self-service erasure of their personal data within their current union.

    Removes the member's chats, messages, usage, notifications, sessions, telemetry,
    raw-query records, and tracking preference for the union they are signed into. Full
    cross-union account deletion remains an admin-driven action (see DEPLOYMENT-POLICY.md
    deletion SLA). Member account login is left intact; only the personal data is purged.
    """
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    auth = get_auth_context(request)
    if auth is None or not auth.is_authenticated or not auth.user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if not auth.union_id:
        raise HTTPException(status_code=400, detail="A union context is required to delete member data.")
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete your personal data.")

    container = get_container(request)
    _purge_user_records(
        db,
        user_id=auth.user_id,
        union_id=auth.union_id,
        global_scope=False,
        telemetry=container.telemetry,
    )
    db.add(
        AuditEvent(
            union_id=auth.union_id,
            actor_user_id=auth.user_id,
            event_type="member_self_service_data_deleted",
            event_payload={"user_id": auth.user_id, "union_id": auth.union_id, "scope": "union"},
        )
    )
    db.flush()
    return {"ok": True, "deleted": True, "scope": "union", "union_id": auth.union_id}
