"""Helpers for contract effective-snapshot resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR
from backend.source_docs import (
    source_doc_applies_to_contract,
    resolve_source_doc_pdf_by_name,
    resolve_source_doc_pdf_path,
)


LATEST_POINTER_FILENAME = "latest.json"


def contract_dir(contract_id: str) -> Path:
    return DATA_DIR / "contracts" / str(contract_id or "")


def effective_root(contract_id: str) -> Path:
    return contract_dir(contract_id) / "effective"


def latest_pointer_path(contract_id: str) -> Path:
    return effective_root(contract_id) / LATEST_POINTER_FILENAME


def _is_moa_name(name: str) -> bool:
    return "moa" in str(name or "").lower()


def load_latest_effective_pointer(contract_id: str) -> dict:
    pointer = latest_pointer_path(contract_id)
    if not pointer.exists():
        return {}
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def resolve_latest_effective_version_id(contract_id: str) -> Optional[str]:
    root = effective_root(contract_id)
    if not root.exists():
        return None

    pointer = load_latest_effective_pointer(contract_id)
    candidate = str(pointer.get("effective_version_id") or "").strip()
    if candidate and (root / candidate).exists():
        return candidate

    versions = sorted(
        p.name for p in root.iterdir()
        if p.is_dir()
    )
    if not versions:
        return None
    return versions[-1]


def resolve_latest_effective_content_hash(contract_id: str) -> Optional[str]:
    pointer = load_latest_effective_pointer(contract_id)
    value = str(pointer.get("effective_content_hash") or "").strip()
    if value:
        return value
    return None


def resolve_effective_version_dir(
    contract_id: str,
    effective_version_id: Optional[str] = None,
) -> Optional[Path]:
    version_id = str(effective_version_id or "").strip() or resolve_latest_effective_version_id(contract_id)
    if not version_id:
        return None
    path = effective_root(contract_id) / version_id
    if not path.exists():
        return None
    return path


def resolve_effective_index_input(contract_id: str, filename: str) -> Optional[Path]:
    version_dir = resolve_effective_version_dir(contract_id)
    if not version_dir:
        return None
    path = version_dir / "index_inputs" / str(filename or "")
    if not path.exists():
        return None
    return path


def load_effective_contract(
    contract_id: str,
    effective_version_id: Optional[str] = None,
) -> Optional[dict]:
    version_dir = resolve_effective_version_dir(contract_id, effective_version_id=effective_version_id)
    if not version_dir:
        return None
    path = version_dir / "effective_contract.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_latest_effective_pointer(
    contract_id: str,
    effective_version_id: str,
    effective_content_hash: Optional[str] = None,
) -> Path:
    out = {
        "effective_version_id": str(effective_version_id or "").strip(),
    }
    hash_value = str(effective_content_hash or "").strip()
    if hash_value:
        out["effective_content_hash"] = hash_value
    path = latest_pointer_path(contract_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def resolve_contract_source_pdf_path(
    contract_id: str,
    source_pdf: Optional[str] = None,
    prefer_moa: bool = False,
    source_doc_id: Optional[str] = None,
) -> Optional[Path]:
    source_doc_hint = str(source_doc_id or "").strip()
    if source_doc_hint:
        if not source_doc_applies_to_contract(source_doc_hint, contract_id):
            return None
        path = resolve_source_doc_pdf_path(source_doc_hint)
        if path and path.exists():
            return path

    source_dir = contract_dir(contract_id) / "source"
    candidates = sorted(source_dir.glob("*.pdf")) if source_dir.exists() else []

    source_hint = str(source_pdf or "").strip().lower()
    if source_hint:
        if source_hint in {"moa", "amendment"}:
            moa = [p for p in candidates if _is_moa_name(p.name)]
            if moa:
                return moa[0]
            fallback = _resolve_effective_source_doc_pdf_for_contract(contract_id, prefer_moa=True)
            if fallback:
                return fallback
        if source_hint in {"base", "cba", "contract"}:
            base = [p for p in candidates if not _is_moa_name(p.name)]
            if base:
                return base[0]
        for p in candidates:
            if p.name.lower() == source_hint:
                return p
        for p in candidates:
            if source_hint in p.name.lower():
                return p
        by_name = resolve_source_doc_pdf_by_name(source_hint, contract_id=contract_id)
        if by_name and by_name.exists():
            return by_name

    if prefer_moa:
        moa = [p for p in candidates if _is_moa_name(p.name)]
        if moa:
            return moa[0]
        fallback = _resolve_effective_source_doc_pdf_for_contract(contract_id, prefer_moa=True)
        if fallback:
            return fallback

    base = [p for p in candidates if not _is_moa_name(p.name)]
    if base:
        return base[0]
    if candidates:
        return candidates[0]

    if source_hint:
        by_name = resolve_source_doc_pdf_by_name(source_hint, contract_id=contract_id)
        if by_name and by_name.exists():
            return by_name
    return _resolve_effective_source_doc_pdf_for_contract(contract_id, prefer_moa=prefer_moa)


def _resolve_effective_source_doc_pdf_for_contract(contract_id: str, prefer_moa: bool) -> Optional[Path]:
    payload = load_effective_contract(contract_id=contract_id)
    if not isinstance(payload, dict):
        return None
    documents = payload.get("source_documents")
    documents = documents if isinstance(documents, dict) else {}
    source_doc_ids = [
        str(row or "").strip()
        for row in (documents.get("amendment_source_doc_ids") or [])
        if str(row or "").strip()
    ]
    for source_doc_id in sorted(source_doc_ids):
        path = resolve_source_doc_pdf_path(source_doc_id)
        if not path or not path.exists():
            continue
        if prefer_moa and not _is_moa_name(path.name):
            continue
        return path
    return None
