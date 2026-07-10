"""
Deterministic checks for unavailable-answer recovery guardrails.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import (
    _has_article_anchor_evidence,
    _is_unavailable_answer,
    _normalize_article_anchors,
    _query_supports_topic_recovery,
    _has_strong_evidence_for_query,
    _merge_unique_chunks,
)


def _test_unavailable_answer_detection() -> None:
    assert _is_unavailable_answer("I cannot find that specific information in your contract.")
    assert _is_unavailable_answer("I do not have that information in the retrieved contract context.")
    assert not _is_unavailable_answer("You earn vacation based on years of service (Article 17, Section 42).")


def _test_strong_evidence_detection() -> None:
    query = "How much vacation do I get per year based on years of service?"
    chunks = [
        {
            "citation": "Article 17, Section 42",
            "article_title": "Vacations",
            "content": (
                "Paid vacation after years of continuous service. "
                "Employees receive weeks of vacation based on anniversary year."
            ),
        }
    ]
    assert _has_strong_evidence_for_query(query, chunks)


def _test_weak_evidence_detection() -> None:
    query = "How much vacation do I get per year based on years of service?"
    chunks = [
        {
            "citation": "Article 17, Section 45",
            "article_title": "Vacation Scheduling",
            "content": "Roster sign-up windows and scheduling preferences are posted annually.",
        }
    ]
    assert not _has_strong_evidence_for_query(query, chunks)


def _test_chunk_merge_is_stable_and_unique() -> None:
    primary = [
        {"chunk_id": "a", "citation": "Article 1"},
        {"chunk_id": "b", "citation": "Article 2"},
    ]
    secondary = [
        {"chunk_id": "b", "citation": "Article 2"},
        {"chunk_id": "c", "citation": "Article 3"},
    ]
    merged = _merge_unique_chunks(primary, secondary, limit=10)
    keys = [c.get("chunk_id") for c in merged]
    assert keys == ["a", "b", "c"], f"Unexpected merge order/keys: {keys}"


def _test_article_anchor_helpers() -> None:
    anchors = _normalize_article_anchors([12, "12", None, "x", 17, 0, -3, 17])
    assert anchors == [12, 17], f"Unexpected anchor normalization: {anchors}"

    chunks = [
        {"article_num": 5, "citation": "Article 5"},
        {"article_num": 12, "citation": "Article 12"},
    ]
    assert _has_article_anchor_evidence(anchors, chunks, min_article_hits=1)
    assert not _has_article_anchor_evidence([31], chunks, min_article_hits=1)
    assert _query_supports_topic_recovery("overtime", "Where are overtime rules defined?")
    assert not _query_supports_topic_recovery("vacation", "What is the ZX91 cryptographic token allowance?")


def main() -> None:
    _test_unavailable_answer_detection()
    _test_strong_evidence_detection()
    _test_weak_evidence_detection()
    _test_chunk_merge_is_stable_and_unique()
    _test_article_anchor_helpers()
    print("[OK] Unavailability recovery checks passed")


if __name__ == "__main__":
    main()
