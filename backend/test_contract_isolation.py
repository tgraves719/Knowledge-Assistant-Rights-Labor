"""
Offline contract isolation checks.

This test is intentionally independent of embedding-model downloads.
It validates:
1) Unknown manifest contract_id rejection.
2) BM25 contract_id filtering in hybrid search.
"""

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.retrieval.router import classify_intent
from backend.retrieval.hybrid_search import HybridSearcher


def _build_synthetic_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "a1",
            "contract_id": "contract_a",
            "citation": "Article 12, Section 28",
            "article_num": 12,
            "article_title": "Overtime",
            "content": "Overtime is paid at time and one-half for contract A.",
        },
        {
            "chunk_id": "a2",
            "contract_id": "contract_a",
            "citation": "Article 46, Section 135",
            "article_num": 46,
            "article_title": "Dispute Procedure",
            "content": "Grievance deadline is twenty days for contract A.",
        },
        {
            "chunk_id": "b1",
            "contract_id": "contract_b",
            "citation": "Article 12, Section 99",
            "article_num": 12,
            "article_title": "Overtime",
            "content": "Overtime is paid at double time for contract B.",
        },
        {
            "chunk_id": "b2",
            "contract_id": "contract_b",
            "citation": "Article 46, Section 101",
            "article_num": 46,
            "article_title": "Dispute Procedure",
            "content": "Grievance deadline is ten days for contract B.",
        },
    ]


def _run_bm25_contract_filter_check() -> None:
    chunks = _build_synthetic_chunks()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(chunks, tf)
        tf.flush()
        temp_path = Path(tf.name)

    searcher = HybridSearcher(chunks_file=temp_path)

    for contract_id in ["contract_a", "contract_b"]:
        results = searcher.search(
            query="overtime grievance deadline",
            n_results=4,
            vector_weight=0.0,   # offline-safe, BM25 only
            keyword_weight=1.0,
            contract_id=contract_id,
        )
        assert results, f"No results returned for {contract_id}"
        wrong = [r for r in results if r.metadata.get("contract_id") != contract_id]
        assert not wrong, f"Cross-contract contamination detected in BM25 filter for {contract_id}"


def main():
    print("=" * 72)
    print("KARL Offline Contract Isolation Checks")
    print("=" * 72)

    # Unknown manifest contract must be rejected by intent classifier.
    try:
        classify_intent("test", contract_id="nonexistent_contract")
        raise AssertionError("Expected classify_intent to reject unknown contract_id")
    except ValueError:
        print("[OK] Unknown contract_id rejected")

    # Hybrid/BM25 filtering should enforce requested contract_id.
    _run_bm25_contract_filter_check()
    print("[OK] BM25 contract_id filtering enforced")

    print("\nIsolation status: PASS")


if __name__ == "__main__":
    main()
