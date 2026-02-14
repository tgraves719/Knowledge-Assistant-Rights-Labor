"""Language-lexicon artifact path helpers."""

from pathlib import Path
from typing import Optional

from backend.config import ONTOLOGIES_DIR, MANIFESTS_DIR


def candidate_language_lexicon_files(contract_id: Optional[str] = None) -> list[Path]:
    names: list[str] = []
    if contract_id:
        names.extend(
            [
                f"language_lexicon_{contract_id}.json",
                f"{contract_id}_language_lexicon.json",
            ]
        )
    names.append("language_lexicon.json")
    return [ONTOLOGIES_DIR / name for name in names]


def resolve_language_lexicon_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_shared = allow_shared_fallback and manifest_count <= 1
    for path in candidate_language_lexicon_files(contract_id=contract_id):
        if path.name == "language_lexicon.json" and not effective_shared:
            continue
        if path.exists():
            return path
    return None
