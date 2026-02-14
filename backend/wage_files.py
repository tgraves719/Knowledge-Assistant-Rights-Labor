"""
Wage artifact path helpers.

Resolves per-contract wage tables first, then shared fallback.
"""

from pathlib import Path
from typing import Optional

from backend.config import WAGES_DIR, MANIFESTS_DIR


def candidate_wage_files(contract_id: Optional[str] = None) -> list[Path]:
    """Return candidate wage-table files in priority order."""
    names: list[str] = []

    if contract_id:
        names.extend(
            [
                f"wage_tables_{contract_id}.json",
                f"{contract_id}_wage_tables.json",
            ]
        )

    names.append("wage_tables.json")
    return [WAGES_DIR / n for n in names]


def resolve_wage_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    """Resolve the best available wage-table file."""
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_allow_shared = allow_shared_fallback and manifest_count <= 1

    for path in candidate_wage_files(contract_id=contract_id):
        is_shared = path.name == "wage_tables.json"
        if is_shared and not effective_allow_shared:
            continue
        if path.exists():
            return path
    return None
