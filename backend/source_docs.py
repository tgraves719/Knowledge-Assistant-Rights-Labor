"""Shared source-document registry helpers (MOAs, CBAs, LOUs, etc.)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR


SOURCE_DOCS_DIR = DATA_DIR / "source_docs"
SOURCE_DOC_SCHEMA_VERSION = "source_doc_v0_9_0"


def _norm(value: str) -> str:
    return str(value or "").strip()


def _norm_lower(value: str) -> str:
    return _norm(value).lower()


def source_doc_type_dir(doc_type: str) -> Path:
    return SOURCE_DOCS_DIR / _norm_lower(doc_type)


def source_doc_dir(source_doc_id: str, doc_type: str) -> Path:
    return source_doc_type_dir(doc_type) / _norm(source_doc_id)


def resolve_source_doc_dir(source_doc_id: str) -> Optional[Path]:
    doc_id = _norm(source_doc_id)
    if not doc_id:
        return None
    if not SOURCE_DOCS_DIR.exists():
        return None
    for type_dir in sorted(SOURCE_DOCS_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        candidate = type_dir / doc_id
        if candidate.exists():
            return candidate
    return None


def resolve_source_doc_type(source_doc_id: str) -> Optional[str]:
    path = resolve_source_doc_dir(source_doc_id)
    if not path:
        return None
    return path.parent.name


def metadata_path_for_source_doc(source_doc_id: str) -> Optional[Path]:
    path = resolve_source_doc_dir(source_doc_id)
    if not path:
        return None
    out = path / "metadata.json"
    if out.exists():
        return out
    return None


def load_source_doc_metadata(source_doc_id: str) -> dict:
    path = metadata_path_for_source_doc(source_doc_id)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def source_doc_applicable_contract_ids(source_doc_id: str) -> list[str]:
    """
    Return stable contract applicability list for a shared source doc.

    Backward compatible with older metadata that only stored `contract_ids`.
    """
    meta = load_source_doc_metadata(source_doc_id)
    if not isinstance(meta, dict):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for key in ("applies_to_contract_ids", "contract_ids", "linked_contract_ids"):
        raw = meta.get(key)
        if not isinstance(raw, list):
            continue
        for value in raw:
            token = _norm(value)
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    single = _norm(meta.get("contract_id"))
    if single and single not in seen:
        out.append(single)
    return sorted(out)


def source_doc_applies_to_contract(source_doc_id: str, contract_id: Optional[str]) -> bool:
    """
    Applicability guard for shared MOA/source docs.

    If metadata declares target contracts, enforce membership.
    If no targets are declared (legacy/incomplete metadata), allow.
    """
    target = _norm(contract_id)
    if not target:
        return True
    allowed = source_doc_applicable_contract_ids(source_doc_id)
    if not allowed:
        return True
    return target in allowed


def _pdf_filename_from_metadata(metadata: dict) -> Optional[str]:
    value = _norm(metadata.get("source_pdf_filename"))
    return value or None


def resolve_source_doc_pdf_path(source_doc_id: str) -> Optional[Path]:
    root = resolve_source_doc_dir(source_doc_id)
    if not root:
        return None

    meta = load_source_doc_metadata(source_doc_id)
    filename = _pdf_filename_from_metadata(meta)
    candidates: list[Path] = []
    if filename:
        candidates.append(root / filename)
    candidates.append(root / "original.pdf")
    candidates.extend(sorted(root.glob("*.pdf")))

    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def resolve_source_doc_pdf_name(source_doc_id: str) -> Optional[str]:
    meta = load_source_doc_metadata(source_doc_id)
    declared = _pdf_filename_from_metadata(meta)
    if declared:
        return declared
    path = resolve_source_doc_pdf_path(source_doc_id)
    if path and path.exists():
        return path.name
    return None


def _iter_source_doc_dirs(doc_type: Optional[str] = None):
    if not SOURCE_DOCS_DIR.exists():
        return
    if doc_type:
        type_dir = source_doc_type_dir(doc_type)
        if not type_dir.exists():
            return
        roots = [type_dir]
    else:
        roots = [p for p in sorted(SOURCE_DOCS_DIR.iterdir()) if p.is_dir()]

    for root in roots:
        for child in sorted(root.iterdir()):
            if child.is_dir():
                yield child


def resolve_source_doc_pdf_by_name(name_or_hint: str, contract_id: Optional[str] = None) -> Optional[Path]:
    hint = _norm_lower(name_or_hint)
    if not hint:
        return None

    exact: list[Path] = []
    fuzzy: list[Path] = []
    for doc_dir in _iter_source_doc_dirs():
        source_doc_id = doc_dir.name
        if not source_doc_applies_to_contract(source_doc_id, contract_id):
            continue
        meta = load_source_doc_metadata(source_doc_id)
        declared_name = _norm_lower(meta.get("source_pdf_filename"))
        pdf_path = resolve_source_doc_pdf_path(source_doc_id)
        if not pdf_path:
            continue

        path_name = _norm_lower(pdf_path.name)
        names = [path_name]
        if declared_name:
            names.append(declared_name)

        if hint in names:
            exact.append(pdf_path)
            continue

        if any(hint in n for n in names):
            fuzzy.append(pdf_path)

    if exact:
        return exact[0]
    if fuzzy:
        return fuzzy[0]
    return None


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def register_source_doc(
    *,
    source_doc_id: str,
    doc_type: str,
    title: Optional[str] = None,
    document_date: Optional[str] = None,
    ratified_date: Optional[str] = None,
    source_pdf_path: Optional[Path] = None,
    extracted_json_path: Optional[Path] = None,
    extracted_md_path: Optional[Path] = None,
    parties: Optional[list[str]] = None,
    contract_ids: Optional[list[str]] = None,
    applies_to_contract_ids: Optional[list[str]] = None,
    overwrite: bool = False,
) -> dict:
    """
    Register/update a shared source document under data/source_docs/.

    Copies incoming artifacts into deterministic filenames:
    - original.pdf
    - extracted.json
    - extracted.md
    - metadata.json
    """
    doc_id = _norm(source_doc_id)
    doc_type_norm = _norm_lower(doc_type)
    if not doc_id:
        raise ValueError("source_doc_id is required")
    if not doc_type_norm:
        raise ValueError("doc_type is required")

    out_dir = source_doc_dir(doc_id, doc_type_norm)
    if out_dir.exists() and not overwrite:
        raise ValueError(f"Source doc already exists: {out_dir}. Re-run with overwrite=True to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / "original.pdf"
    out_json = out_dir / "extracted.json"
    out_md = out_dir / "extracted.md"

    if source_pdf_path and source_pdf_path.exists():
        shutil.copyfile(source_pdf_path, out_pdf)
    if extracted_json_path and extracted_json_path.exists():
        shutil.copyfile(extracted_json_path, out_json)
    if extracted_md_path and extracted_md_path.exists():
        shutil.copyfile(extracted_md_path, out_md)

    source_pdf_filename = _norm(source_pdf_path.name) if source_pdf_path and source_pdf_path.exists() else (
        out_pdf.name if out_pdf.exists() else None
    )

    normalized_contract_ids = sorted({_norm(c) for c in (contract_ids or []) if _norm(c)})
    normalized_applies = sorted(
        {
            _norm(c)
            for c in (
                list(contract_ids or [])
                + list(applies_to_contract_ids or [])
            )
            if _norm(c)
        }
    )

    metadata = {
        "schema_version": SOURCE_DOC_SCHEMA_VERSION,
        "source_doc_id": doc_id,
        "doc_type": doc_type_norm,
        "title": _norm(title) or None,
        "document_date": _norm(document_date) or None,
        "ratified_date": _norm(ratified_date) or None,
        "source_pdf_filename": source_pdf_filename,
        "source_pdf_sha256": _sha256_file(out_pdf),
        "extracted_json_sha256": _sha256_file(out_json),
        "extracted_md_sha256": _sha256_file(out_md),
        "parties": sorted({_norm(p) for p in (parties or []) if _norm(p)}),
        # `contract_ids` retained for backward compatibility with existing tooling.
        "contract_ids": normalized_contract_ids,
        # Canonical applicability field for shared MOAs spanning multiple contracts.
        "applies_to_contract_ids": normalized_applies,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "source_doc_id": doc_id,
        "doc_type": doc_type_norm,
        "dir": str(out_dir),
        "pdf_path": str(out_pdf) if out_pdf.exists() else None,
        "json_path": str(out_json) if out_json.exists() else None,
        "md_path": str(out_md) if out_md.exists() else None,
        "metadata_path": str(out_dir / "metadata.json"),
    }
