"""Entitlement artifact path helpers."""

from pathlib import Path
from typing import Optional

from backend.config import ENTITLEMENTS_DIR, MANIFESTS_DIR
from backend.effective_contracts import resolve_effective_index_input


def candidate_entitlement_files(contract_id: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []
    if contract_id:
        for name in (
            f"entitlement_tables_{contract_id}.json",
            f"{contract_id}_entitlement_tables.json",
        ):
            effective_path = resolve_effective_index_input(contract_id=contract_id, filename=name)
            if effective_path:
                candidates.append(effective_path)

        candidates.extend(
            ENTITLEMENTS_DIR / name
            for name in (
                f"entitlement_tables_{contract_id}.json",
                f"{contract_id}_entitlement_tables.json",
            )
        )

    candidates.append(ENTITLEMENTS_DIR / "entitlement_tables.json")
    return candidates


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
