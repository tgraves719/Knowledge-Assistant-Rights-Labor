"""
Deterministic query-routing synthesis smoke checks.

Validates ingestion-owned routing generation from concept index + lexicon.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import CHUNKS_DIR, MANIFESTS_DIR, ONTOLOGIES_DIR
from backend.ingest.query_routing import synthesize_query_routing


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _test_clerks_query_routing_generation() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    manifest = _load_json(MANIFESTS_DIR / f"{contract_id}.json")
    concept_index = _load_json(CHUNKS_DIR / f"concept_index_{contract_id}.json")
    lexicon = _load_json(ONTOLOGIES_DIR / f"language_lexicon_{contract_id}.json")

    routing, stats = synthesize_query_routing(
        manifest=manifest,
        concept_index=concept_index,
        language_lexicon=lexicon,
        classification_ontology=None,
    )

    topic_map = routing.get("topic_to_articles") or {}
    class_map = routing.get("classification_to_articles") or {}
    slang_map = routing.get("slang_to_contract") or {}
    topic_patterns = routing.get("topic_patterns") or {}

    assert stats.get("topic_entries", 0) >= 8, f"Expected >=8 topic entries, got {stats}"
    assert stats.get("slang_entries", 0) >= 20, f"Expected >=20 slang entries, got {stats}"
    assert "term" in topic_map and 58 in set(topic_map.get("term", [])), (
        f"Expected term topic to include Article 58. Got: {topic_map.get('term')}"
    )
    assert "vacation" in topic_map and 17 in set(topic_map.get("vacation", [])), (
        f"Expected vacation topic to include Article 17. Got: {topic_map.get('vacation')}"
    )
    breaks_articles = set(topic_map.get("breaks", []))
    assert (24 in breaks_articles) or (25 in breaks_articles), (
        f"Expected breaks topic to include Article 24 or 25. Got: {topic_map.get('breaks')}"
    )
    apc_articles = set(class_map.get("all_purpose_clerk", []))
    assert 8 in apc_articles, f"Expected all_purpose_clerk to include Article 8. Got: {apc_articles}"
    assert "start and end" in slang_map, "Expected start/end alias in generated slang map"
    assert "term" in topic_patterns, "Expected generated term topic pattern"


def main() -> None:
    _test_clerks_query_routing_generation()
    print("[OK] Query-routing synthesis checks passed")


if __name__ == "__main__":
    main()

