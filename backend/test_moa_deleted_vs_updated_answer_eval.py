"""Deterministic checks for MOA deleted-vs-updated answer eval scoring."""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_moa_deleted_vs_updated_answer as mod


def _test_source_type_matching_prefers_matching_citation() -> None:
    sources = [
        {"citation": "Article 8, Section 19", "source_type": "base"},
        {"citation": "Article 15, Section 34", "source_type": "moa"},
    ]
    ok, observed = mod._source_type_match("moa", sources, ["Article 15, Section 34"])
    assert ok is True
    assert observed == "moa"


def _test_source_type_matching_fails_without_moa_source() -> None:
    sources = [
        {"citation": "Article 15, Section 34", "source_type": "base"},
    ]
    ok, observed = mod._source_type_match("moa", sources, ["Article 15, Section 34"])
    assert ok is False
    assert observed == "base"


def main() -> None:
    _test_source_type_matching_prefers_matching_citation()
    _test_source_type_matching_fails_without_moa_source()
    print("[OK] moa deleted-vs-updated answer eval helpers passed")


if __name__ == "__main__":
    main()
