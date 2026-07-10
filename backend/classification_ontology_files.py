"""
Classification ontology artifact path helpers.

Resolves per-contract ontology files first, then shared fallback.
"""

from pathlib import Path
from typing import Optional

from backend.config import ONTOLOGIES_DIR


def candidate_classification_ontology_files(contract_id: Optional[str] = None) -> list[Path]:
    names: list[str] = []
    if contract_id:
        names.extend(
            [
                f"classification_ontology_{contract_id}.json",
                f"{contract_id}_classification_ontology.json",
            ]
        )
    names.append("classification_ontology.json")
    return [ONTOLOGIES_DIR / n for n in names]


def resolve_classification_ontology_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    for path in candidate_classification_ontology_files(contract_id=contract_id):
        is_shared = path.name == "classification_ontology.json"
        if is_shared and not allow_shared_fallback:
            continue
        if path.exists():
            return path
    return None

