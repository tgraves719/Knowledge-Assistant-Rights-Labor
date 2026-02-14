"""
Chunk artifact path helpers.

Resolves per-contract chunk files first, then shared fallback files.
"""

from pathlib import Path
from typing import Optional

from backend.config import CHUNKS_DIR, MANIFESTS_DIR


def candidate_chunk_files(contract_id: Optional[str] = None) -> list[Path]:
    """Return candidate chunk files in priority order."""
    names: list[str] = []

    if contract_id:
        # Preferred per-contract naming (suffix form).
        names.extend(
            [
                f"contract_chunks_enriched_{contract_id}.json",
                f"contract_chunks_smart_{contract_id}.json",
                f"contract_chunks_{contract_id}.json",
            ]
        )
        # Alternate per-contract naming (prefix form) for compatibility.
        names.extend(
            [
                f"{contract_id}_contract_chunks_enriched.json",
                f"{contract_id}_contract_chunks_smart.json",
                f"{contract_id}_contract_chunks.json",
            ]
        )

    # Shared fallbacks.
    names.extend(
        [
            "contract_chunks_enriched.json",
            "contract_chunks_smart.json",
            "contract_chunks.json",
        ]
    )

    return [CHUNKS_DIR / n for n in names]


def resolve_chunk_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    """
    Resolve the best available chunk artifact path.

    Args:
        contract_id: Contract context to prefer contract-specific files.
        allow_shared_fallback: If False, only contract-specific candidates are allowed.
    """
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_allow_shared = allow_shared_fallback and manifest_count <= 1

    candidates = candidate_chunk_files(contract_id=contract_id)
    for path in candidates:
        name = path.name
        is_shared = (
            name == "contract_chunks_enriched.json"
            or name == "contract_chunks_smart.json"
            or name == "contract_chunks.json"
        )
        if is_shared and not effective_allow_shared:
            continue
        if path.exists():
            return path
    return None
