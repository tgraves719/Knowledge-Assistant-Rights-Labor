"""Focused MOA regression slice: deleted clauses vs updated clauses.

Wraps `backend.evaluate_moa_effective` with a tighter dataset and emits
bucket metrics that can be used as explicit release gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
import backend.evaluate_moa_effective as moa_effective


OUT_PATH = DATA_DIR / "test_set" / "moa_deleted_vs_updated_results.json"
DEFAULT_INPUT = DATA_DIR / "test_set" / "moa_deleted_vs_updated_test.json"


def _rate(num: int, den: int) -> float:
    return round((num / den) if den else 0.0, 4)


def _case_bucket(row: dict[str, Any]) -> str:
    case_class = str(row.get("case_class") or "").strip().lower()
    if case_class in {"updated", "deleted"}:
        return case_class
    expectation = str(row.get("expectation") or "").strip().lower()
    if expectation == "absent":
        return "deleted"
    return "updated"


def _augment_case_classes(results: list[dict[str, Any]], dataset_path: Path) -> None:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return
    by_id = {}
    for case in list(payload.get("test_cases") or []):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "").strip()
        if case_id:
            by_id[case_id] = str(case.get("case_class") or "").strip().lower()
    for row in results:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id") or "").strip()
        if case_id and case_id in by_id and by_id[case_id]:
            row["case_class"] = by_id[case_id]


def _build_bucket_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bucket_rows: dict[str, list[dict[str, Any]]] = {"updated": [], "deleted": []}
    for row in results:
        bucket_rows.setdefault(_case_bucket(row), []).append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for bucket in ("updated", "deleted"):
        rows = bucket_rows.get(bucket) or []
        total = len(rows)
        passed = sum(1 for row in rows if bool(row.get("pass")))
        detail: dict[str, Any] = {
            "passed": passed,
            "total": total,
            "pass_rate": _rate(passed, total),
        }
        if bucket == "updated":
            source_rows = [r for r in rows if str(r.get("expected_source_type") or "").strip()]
            source_total = len(source_rows)
            source_hits = sum(1 for r in source_rows if bool(r.get("source_type_ok")))
            detail["source_type_cases"] = source_total
            detail["moa_source_type_match_rate"] = _rate(source_hits, source_total) if source_total else None
            citation_hits = sum(1 for r in rows if bool(r.get("citation_hit")))
            detail["citation_hit_rate"] = _rate(citation_hits, total) if total else 0.0
        else:
            forbidden_ok = sum(1 for r in rows if bool(r.get("forbidden_ok")))
            detail["forbidden_ok_rate"] = _rate(forbidden_ok, total) if total else 0.0
        metrics[bucket] = detail
    return metrics


def _build_gate(
    *,
    overall: dict[str, Any],
    buckets: dict[str, dict[str, Any]],
    min_overall_pass_rate: float,
    min_updated_pass_rate: float,
    min_deleted_pass_rate: float,
    min_updated_moa_source_type_match_rate: float,
) -> dict[str, Any]:
    overall_rate = float(overall.get("pass_rate") or 0.0)
    updated = buckets.get("updated") or {}
    deleted = buckets.get("deleted") or {}
    updated_rate = float(updated.get("pass_rate") or 0.0)
    deleted_rate = float(deleted.get("pass_rate") or 0.0)
    updated_source_rate = updated.get("moa_source_type_match_rate")
    updated_source_rate_f = float(updated_source_rate or 0.0)
    updated_source_total = int(updated.get("source_type_cases") or 0)

    checks = {
        "overall_pass_rate": {
            "pass": overall_rate >= min_overall_pass_rate,
            "observed": overall_rate,
            "threshold": min_overall_pass_rate,
        },
        "updated_clause_pass_rate": {
            "pass": updated_rate >= min_updated_pass_rate,
            "observed": updated_rate,
            "threshold": min_updated_pass_rate,
        },
        "deleted_clause_pass_rate": {
            "pass": deleted_rate >= min_deleted_pass_rate,
            "observed": deleted_rate,
            "threshold": min_deleted_pass_rate,
        },
        "updated_moa_source_type_match_rate": {
            "pass": (updated_source_total == 0) or (updated_source_rate_f >= min_updated_moa_source_type_match_rate),
            "observed": updated_source_rate if updated_source_total else None,
            "threshold": min_updated_moa_source_type_match_rate,
            "cases": updated_source_total,
        },
    }
    gate_pass = all(bool((row or {}).get("pass")) for row in checks.values())
    return {
        "pass": gate_pass,
        "checks": checks,
    }


def run(
    *,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_results: int = 8,
    bm25_only: bool = False,
    min_overall_pass_rate: float = 1.00,
    min_updated_pass_rate: float = 1.00,
    min_deleted_pass_rate: float = 1.00,
    min_updated_moa_source_type_match_rate: float = 1.00,
) -> dict[str, Any]:
    dataset_path = input_path or DEFAULT_INPUT
    base_report = moa_effective.run(test_file=dataset_path, n_results=n_results, bm25_only=bm25_only)
    results = [dict(row) for row in list(base_report.get("results") or [])]
    _augment_case_classes(results, dataset_path)

    overall = dict(base_report.get("overall") or {})
    buckets = _build_bucket_metrics(results)
    gate = _build_gate(
        overall=overall,
        buckets=buckets,
        min_overall_pass_rate=min_overall_pass_rate,
        min_updated_pass_rate=min_updated_pass_rate,
        min_deleted_pass_rate=min_deleted_pass_rate,
        min_updated_moa_source_type_match_rate=min_updated_moa_source_type_match_rate,
    )

    report = {
        "schema_version": "moa_deleted_vs_updated_eval_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_schema_version": str(base_report.get("dataset_schema_version") or ""),
        "test_file": str(dataset_path),
        "n_results": int(base_report.get("n_results") or n_results),
        "bm25_only": bool(base_report.get("bm25_only") if "bm25_only" in base_report else bm25_only),
        "overall": overall,
        "buckets": buckets,
        "gate": gate,
        "results": results,
    }
    _write_report(report, output_path or OUT_PATH)
    return report


def _write_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deleted-vs-updated MOA regression slice.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(OUT_PATH))
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--min-overall-pass-rate", type=float, default=1.00)
    parser.add_argument("--min-updated-pass-rate", type=float, default=1.00)
    parser.add_argument("--min-deleted-pass-rate", type=float, default=1.00)
    parser.add_argument("--min-updated-moa-source-type-match-rate", type=float, default=1.00)
    args = parser.parse_args()

    report = run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        n_results=int(args.n_results),
        bm25_only=bool(args.bm25_only),
        min_overall_pass_rate=float(args.min_overall_pass_rate),
        min_updated_pass_rate=float(args.min_updated_pass_rate),
        min_deleted_pass_rate=float(args.min_deleted_pass_rate),
        min_updated_moa_source_type_match_rate=float(args.min_updated_moa_source_type_match_rate),
    )
    out_path = Path(args.output)
    overall = report.get("overall") or {}
    buckets = report.get("buckets") or {}
    gate = report.get("gate") or {}
    print("=" * 72)
    print("KARL MOA Deleted-vs-Updated Eval")
    print("=" * 72)
    print(
        f"Overall: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    for bucket_name in ("updated", "deleted"):
        b = buckets.get(bucket_name) or {}
        print(
            f"- {bucket_name}: {int(b.get('passed', 0))}/{int(b.get('total', 0))} "
            f"({float(b.get('pass_rate', 0.0)):.1%})"
        )
    print(f"Gate: {bool(gate.get('pass'))}")
    print(f"Results: {out_path}")
    return 0 if bool(gate.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
