"""PDF navigation artifact path helpers."""

from pathlib import Path
from typing import Optional

from backend.config import ONTOLOGIES_DIR, MANIFESTS_DIR


def candidate_pdf_nav_index_files(contract_id: Optional[str] = None) -> list[Path]:
    names: list[str] = []
    if contract_id:
        names.extend(
            [
                f"pdf_nav_index_{contract_id}.json",
                f"{contract_id}_pdf_nav_index.json",
            ]
        )
    names.append("pdf_nav_index.json")
    return [ONTOLOGIES_DIR / n for n in names]


def resolve_pdf_nav_index_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_allow_shared = allow_shared_fallback and manifest_count <= 1
    for path in candidate_pdf_nav_index_files(contract_id=contract_id):
        is_shared = path.name == "pdf_nav_index.json"
        if is_shared and not effective_allow_shared:
            continue
        if path.exists():
            return path
    return None

