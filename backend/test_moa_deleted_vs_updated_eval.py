"""Deterministic checks for deleted-vs-updated MOA eval aggregation."""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_moa_deleted_vs_updated as mod


def _test_bucket_metrics_and_gate() -> None:
    results = [
        {
            "id": "u1",
            "case_class": "updated",
            "expectation": "present",
            "pass": True,
            "citation_hit": True,
            "expected_source_type": "moa",
            "source_type_ok": True,
        },
        {
            "id": "u2",
            "case_class": "updated",
            "expectation": "present",
            "pass": False,
            "citation_hit": False,
            "expected_source_type": "moa",
            "source_type_ok": False,
        },
        {
            "id": "d1",
            "case_class": "deleted",
            "expectation": "absent",
            "pass": True,
            "citation_hit": False,
            "forbidden_ok": True,
        },
        {
            "id": "d2",
            "case_class": "deleted",
            "expectation": "absent",
            "pass": True,
            "citation_hit": False,
            "forbidden_ok": True,
        },
    ]
    buckets = mod._build_bucket_metrics(results)
    assert float((buckets.get("updated") or {}).get("pass_rate") or 0.0) == 0.5
    assert float((buckets.get("updated") or {}).get("moa_source_type_match_rate") or 0.0) == 0.5
    assert float((buckets.get("deleted") or {}).get("pass_rate") or 0.0) == 1.0
    assert float((buckets.get("deleted") or {}).get("forbidden_ok_rate") or 0.0) == 1.0

    gate = mod._build_gate(
        overall={"pass_rate": 0.75},
        buckets=buckets,
        min_overall_pass_rate=0.75,
        min_updated_pass_rate=0.5,
        min_deleted_pass_rate=1.0,
        min_updated_moa_source_type_match_rate=0.5,
    )
    assert bool(gate.get("pass")) is True

    strict_gate = mod._build_gate(
        overall={"pass_rate": 0.75},
        buckets=buckets,
        min_overall_pass_rate=0.80,
        min_updated_pass_rate=1.0,
        min_deleted_pass_rate=1.0,
        min_updated_moa_source_type_match_rate=1.0,
    )
    assert bool(strict_gate.get("pass")) is False


def main() -> None:
    _test_bucket_metrics_and_gate()
    print("[OK] moa deleted-vs-updated eval aggregation checks passed")


if __name__ == "__main__":
    main()
