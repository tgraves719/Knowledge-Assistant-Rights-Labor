"""
Concept-index artifact path helpers.

Resolves per-contract concept-index files first, then shared fallback files.
"""

from pathlib import Path
from typing import Optional

from backend.config import CHUNKS_DIR, MANIFESTS_DIR


def candidate_concept_index_files(contract_id: Optional[str] = None) -> list[Path]:
    """Return candidate concept-index files in priority order."""
    names: list[str] = []

    if contract_id:
        names.extend(
            [
                f"concept_index_{contract_id}.json",
                f"{contract_id}_concept_index.json",
            ]
        )

    names.append("concept_index.json")
    return [CHUNKS_DIR / n for n in names]


def resolve_concept_index_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    """
    Resolve the best available concept-index artifact path.

    Args:
        contract_id: Contract context to prefer contract-specific files.
        allow_shared_fallback: If False, only contract-specific candidates are allowed.
    """
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_allow_shared = allow_shared_fallback and manifest_count <= 1

    for path in candidate_concept_index_files(contract_id=contract_id):
        is_shared = path.name == "concept_index.json"
        if is_shared and not effective_allow_shared:
            continue
        if path.exists():
            return path
    return None

