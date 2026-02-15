"""Entitlement artifact path helpers."""

from pathlib import Path
from typing import Optional

from backend.config import ENTITLEMENTS_DIR, MANIFESTS_DIR


def candidate_entitlement_files(contract_id: Optional[str] = None) -> list[Path]:
    names: list[str] = []
    if contract_id:
        names.extend(
            [
                f"entitlement_tables_{contract_id}.json",
                f"{contract_id}_entitlement_tables.json",
            ]
        )
    names.append("entitlement_tables.json")
    return [ENTITLEMENTS_DIR / name for name in names]


def resolve_entitlement_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_shared = allow_shared_fallback and manifest_count <= 1
    for path in candidate_entitlement_files(contract_id=contract_id):
        if path.name == "entitlement_tables.json" and not effective_shared:
            continue
        if path.exists():
            return path
    return None
