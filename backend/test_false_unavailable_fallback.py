"""
Deterministic fallback-ranking checks for false-unavailable recovery.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import _fallback_relevant_chunks, _has_contiguous_clause_phrase_evidence  # noqa: E402


def _test_definition_query_prefers_topic_title_anchor() -> None:
    question = "Where are overtime rules defined in this agreement?"
    chunks = [
        {
            "citation": "Article 30, Section 88",
            "article_num": 30,
            "section_num": 88,
            "article_title": "UNSCHEDULED OVERTIME",
            "content": "Overtime rules for unscheduled assignments.",
        },
        {
            "citation": "Article 12, Section 32",
            "article_num": 12,
            "section_num": 32,
            "article_title": "OVERTIME",
            "content": "Overtime compensation at time and one-half for hours worked over threshold.",
        },
        {
            "citation": "Article 57, Section 168",
            "article_num": 57,
            "section_num": 168,
            "article_title": "JOB STEWARD",
            "content": "General language about steward rights.",
        },
    ]

    ranked = _fallback_relevant_chunks(
        question=question,
        chunks=chunks,
        min_token_hits=2,
        preferred_articles=[12, 30],
    )
    citations = [str(c.get("citation") or "") for c in ranked]
    assert citations, "Expected non-empty fallback chunk selection."
    assert any(c.startswith("Article 12") for c in citations[:2]), (
        f"Expected overtime-defining anchor in top-2 fallback chunks, got: {citations}"
    )


def _test_preferred_article_boost_for_shift_gap_query() -> None:
    question = "What are minimum hours between shifts?"
    chunks = [
        {
            "citation": "Article 10, Section 21",
            "article_num": 10,
            "section_num": 21,
            "article_title": "WORKWEEK",
            "content": "Scheduling and assignment language with shifts.",
        },
        {
            "citation": "Article 24, Section 60",
            "article_num": 24,
            "section_num": 60,
            "article_title": "INTER-SHIFT REST",
            "content": "Minimum hours between end of shift and start of next shift.",
        },
    ]

    ranked = _fallback_relevant_chunks(
        question=question,
        chunks=chunks,
        min_token_hits=2,
        preferred_articles=[24, 25],
    )
    citations = [str(c.get("citation") or "") for c in ranked]
    assert any(c.startswith("Article 24") for c in citations), (
        f"Expected Article 24 to be retained in fallback chunks. Got: {citations}"
    )


def _test_clause_phrase_evidence_requires_contiguous_match() -> None:
    question = (
        "The Sunday premium shall not be averaged into the employee's straight-time rate "
        "for determining overtime. Is that in the current effective agreement?"
    )
    noisy_chunks = [
        {
            "citation": "Article 13, Section 29",
            "article_num": 13,
            "section_num": 29,
            "article_title": "SUNDAY PREMIUM",
            "content": "Employees are entitled to one and one-quarter times their straight-time hourly rate on Sunday.",
        },
        {
            "citation": "Article 15, Section 34",
            "article_num": 15,
            "section_num": 34,
            "article_title": "NIGHT PREMIUMS",
            "content": "Night premium shall not apply where the employee is working at overtime or on Sunday or on a holiday.",
        },
    ]
    assert _has_contiguous_clause_phrase_evidence(question, noisy_chunks) is False

    matching_chunks = [
        {
            "citation": "Article 13, Section 29",
            "article_num": 13,
            "section_num": 29,
            "article_title": "SUNDAY PREMIUM",
            "content": (
                "The Sunday premium shall not be averaged into the employee's straight-time rate "
                "for the purpose of determining the rate upon which daily or weekly overtime is based."
            ),
        }
    ]
    assert _has_contiguous_clause_phrase_evidence(question, matching_chunks) is True


def main() -> None:
    _test_definition_query_prefers_topic_title_anchor()
    _test_preferred_article_boost_for_shift_gap_query()
    _test_clause_phrase_evidence_requires_contiguous_match()
    print("[OK] False-unavailable fallback checks passed")


if __name__ == "__main__":
    main()
