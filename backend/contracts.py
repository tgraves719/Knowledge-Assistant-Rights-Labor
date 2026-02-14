"""Contract manifest helpers for multi-contract runtime selection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from backend.config import MANIFESTS_DIR, CONTRACT_ID


def _clean_text(value: str) -> str:
    """Normalize whitespace from OCR/PDF-derived strings."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def list_manifest_paths() -> list[Path]:
    """Return sorted manifest files."""
    return sorted(MANIFESTS_DIR.glob("*.json"))


def load_manifest(contract_id: str) -> Optional[dict]:
    """Load a single manifest by contract_id, or None if missing."""
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_region_id(contract_id: str, manifest: Optional[dict] = None) -> str:
    """
    Deterministically derive region_id when not explicitly present.

    Region is used as a hard tenancy retrieval filter.
    """
    manifest = manifest or {}
    explicit = _clean_text(manifest.get("region_id"))
    if explicit:
        return re.sub(r"[^a-z0-9]+", "-", explicit.lower()).strip("-")

    location = _clean_text(manifest.get("location"))
    if location:
        return re.sub(r"[^a-z0-9]+", "-", location.lower()).strip("-")

    token = ""
    for t in re.findall(r"[a-z0-9]+", str(contract_id or "").lower()):
        if re.fullmatch(r"\d{4}", t):
            continue
        if re.fullmatch(r"local\d+", t):
            continue
        if t in {"local", "ufcw", "safeway", "kingsoopers", "clerks", "meat"}:
            continue
        token = t
        break
    if token:
        return f"region-{token}"
    fallback = re.sub(r"[^a-z0-9]+", "-", str(contract_id or "").lower()).strip("-")
    return f"region-{fallback or 'unknown'}"


def resolve_contract_region_id(contract_id: str) -> str:
    """Resolve region_id for a contract (manifest value or deterministic fallback)."""
    manifest = load_manifest(contract_id) or {}
    return infer_region_id(contract_id=contract_id, manifest=manifest)


def resolve_default_contract_id() -> Optional[str]:
    """
    Resolve runtime default contract_id.

    Prefers configured CONTRACT_ID when that manifest exists, then first manifest.
    Returns None when no manifests are present.
    """
    catalog = list_contract_catalog()
    if not catalog:
        return None

    configured = _clean_text(CONTRACT_ID)
    if configured:
        for c in catalog:
            if c.get("contract_id") == configured:
                return configured

    # If configured id was deduped out, prefer the canonical row with matching employer/terms.
    configured_manifest = load_manifest(configured) if configured else None
    if configured_manifest:
        key = (
            _clean_text(configured_manifest.get("union_local")),
            _clean_text(infer_region_id(contract_id=configured, manifest=configured_manifest)),
            _clean_text(configured_manifest.get("employer")),
            _clean_text(configured_manifest.get("term_start")),
            _clean_text(configured_manifest.get("term_end")),
        )
        for c in catalog:
            row_key = (
                _clean_text(c.get("union_local_id")),
                _clean_text(c.get("region_id")),
                _clean_text(c.get("employer")),
                _clean_text(c.get("term_start")),
                _clean_text(c.get("term_end")),
            )
            if row_key == key:
                return c.get("contract_id")

    if catalog:
        return catalog[0].get("contract_id")
    return None


def list_contract_catalog() -> list[dict]:
    """List all contract manifests with normalized runtime metadata."""
    contracts = []
    for path in list_manifest_paths():
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        contract_id = _clean_text(manifest.get("contract_id") or path.stem) or path.stem
        union_local = _clean_text(manifest.get("union_local"))
        employer = _clean_text(manifest.get("employer"))
        term_start = _clean_text(manifest.get("term_start"))
        term_end = _clean_text(manifest.get("term_end"))
        contract_version = _clean_text(manifest.get("contract_version"))
        if not contract_version and term_start and term_end:
            contract_version = f"{term_start}__{term_end}"

        contracts.append(
            {
                "contract_id": contract_id,
                "union_local_id": union_local,
                "region_id": infer_region_id(contract_id=contract_id, manifest=manifest),
                "contract_version": contract_version,
                "employer": employer,
                "term_start": term_start,
                "term_end": term_end,
            }
        )

    # Deduplicate equivalent manifests (same union/employer/term), preferring local-prefixed IDs.
    deduped: dict[tuple[str, str, str, str, str], dict] = {}
    for contract in contracts:
        key = (
            _clean_text(contract.get("union_local_id")),
            _clean_text(contract.get("region_id")),
            _clean_text(contract.get("employer")),
            _clean_text(contract.get("term_start")),
            _clean_text(contract.get("term_end")),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = contract
            continue

        existing_id = existing.get("contract_id", "")
        candidate_id = contract.get("contract_id", "")
        existing_score = 1 if existing_id.lower().startswith("local") else 0
        candidate_score = 1 if candidate_id.lower().startswith("local") else 0
        if candidate_score > existing_score:
            deduped[key] = contract
        elif candidate_score == existing_score and len(candidate_id) > len(existing_id):
            # Prefer more descriptive IDs when priority ties.
            deduped[key] = contract

    contracts = list(deduped.values())
    contracts.sort(key=lambda c: c["contract_id"])
    return contracts


def get_contract_catalog_entry(contract_id: str) -> Optional[dict]:
    """Get normalized catalog row for a contract_id."""
    catalog = list_contract_catalog()
    for contract in catalog:
        if contract["contract_id"] == contract_id:
            return contract

    # If this id is an equivalent deduped alias, map by union/employer/term key.
    manifest = load_manifest(contract_id)
    if not manifest:
        return None

    key = (
        _clean_text(manifest.get("union_local")),
        _clean_text(infer_region_id(contract_id=contract_id, manifest=manifest)),
        _clean_text(manifest.get("employer")),
        _clean_text(manifest.get("term_start")),
        _clean_text(manifest.get("term_end")),
    )
    for contract in catalog:
        row_key = (
            _clean_text(contract.get("union_local_id")),
            _clean_text(contract.get("region_id")),
            _clean_text(contract.get("employer")),
            _clean_text(contract.get("term_start")),
            _clean_text(contract.get("term_end")),
        )
        if row_key == key:
            return contract
    return None
