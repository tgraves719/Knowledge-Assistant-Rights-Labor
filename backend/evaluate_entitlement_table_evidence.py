"""
Canonical entitlement-table evidence evaluator.

Runs deterministic contract-scoped vacation entitlement lookups and verifies
ingestion-owned entitlement schedule artifacts resolve expected accrual values.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.contracts import get_contract_catalog_entry
from backend.retrieval.router import HybridRetriever


def _canonical_contract_id(contract_id: str) -> str:
    entry = get_contract_catalog_entry(contract_id)
    if entry and entry.get("contract_id"):
        return str(entry["contract_id"])
    return str(contract_id)


def run(
    test_file: Optional[Path] = None,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "entitlement_table_evidence_test.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    test_cases = list(payload.get("test_cases") or [])

    retriever = HybridRetriever(vector_store=None)
    rows = []
    by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
    weeks_passes = 0
    source_method_passes = 0
    evidence_passes = 0

    for case in test_cases:
        case_id = str(case.get("id") or "")
        contract_id = _canonical_contract_id(str(case.get("contract_id") or ""))
        months_employed = int(case.get("months_employed") or 0)
        hours_worked = int(case.get("hours_worked") or 0)
        hire_date = str(case.get("hire_date") or "").strip() or None
        expected_weeks = int(case.get("expected_weeks_per_year") or 0)
        min_evidence_rows = int(case.get("min_evidence_rows") or 1)

        entitlement_info = retriever.lookup_vacation_entitlement(
            months_employed=months_employed,
            hours_worked=hours_worked,
            hire_date=hire_date,
            contract_id=contract_id,
        )

        found = entitlement_info is not None
        source_method = str((entitlement_info or {}).get("source_method") or "")
        estimated_weeks = (entitlement_info or {}).get("estimated_weeks_per_year")
        selected_schedule = (entitlement_info or {}).get("selected_schedule") or {}
        evidence = list((entitlement_info or {}).get("entitlement_evidence") or [])
        citation = str((entitlement_info or {}).get("citation") or "")

        weeks_ok = found and isinstance(estimated_weeks, int) and int(estimated_weeks) == expected_weeks
        source_method_ok = found and source_method == "vacation_entitlement_tiers"
        selected_ok = found and isinstance(selected_schedule, dict) and bool(selected_schedule)
        evidence_ok = found and len(evidence) >= min_evidence_rows
        citation_ok = found and ("article" in citation.lower())
        passed = bool(weeks_ok and source_method_ok and selected_ok and evidence_ok and citation_ok)

        by_contract[contract_id]["total"] += 1
        if passed:
            by_contract[contract_id]["passed"] += 1
        if weeks_ok:
            weeks_passes += 1
        if source_method_ok:
            source_method_passes += 1
        if evidence_ok:
            evidence_passes += 1

        rows.append(
            {
                "id": case_id,
                "contract_id": contract_id,
                "months_employed": months_employed,
                "hours_worked": hours_worked,
                "hire_date": hire_date,
                "expected_weeks_per_year": expected_weeks,
                "found": found,
                "estimated_weeks_per_year": estimated_weeks,
                "weeks_ok": weeks_ok,
                "source_method": source_method,
                "source_method_ok": source_method_ok,
                "selected_schedule_id": selected_schedule.get("schedule_id"),
                "selected_ok": selected_ok,
                "evidence_count": len(evidence),
                "evidence_ok": evidence_ok,
                "citation": citation,
                "citation_ok": citation_ok,
                "pass": passed,
            }
        )

    total = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    by_contract_summary = {}
    for cid, stats in sorted(by_contract.items()):
        t = int(stats["total"])
        p = int(stats["passed"])
        by_contract_summary[cid] = {
            "passed": p,
            "total": t,
            "pass_rate": round((p / t) if t else 0.0, 4),
        }

    report = {
        "schema_version": "entitlement_table_evidence_eval_v1",
        "dataset_schema_version": str(payload.get("schema_version") or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_file": str(test_file),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
            "weeks_resolution_pass_rate": round((weeks_passes / total) if total else 0.0, 4),
            "source_method_pass_rate": round((source_method_passes / total) if total else 0.0, 4),
            "evidence_presence_rate": round((evidence_passes / total) if total else 0.0, 4),
        },
        "by_contract": by_contract_summary,
        "results": rows,
    }
    return report


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "entitlement_table_evidence_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate canonical entitlement-table evidence slice.")
    parser.add_argument("--input", type=str, default=None, help="Path to entitlement-table-evidence benchmark JSON")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file)
    out_path = _write_report(report)

    overall = report["overall"]
    print("=" * 72)
    print("KARL Entitlement-Table Evidence Evaluation")
    print("=" * 72)
    print(f"Overall: {overall['passed']}/{overall['total']} ({overall['pass_rate']:.1%})")
    print(f"Weeks-resolution pass rate: {overall['weeks_resolution_pass_rate']:.1%}")
    print(f"Source-method pass rate: {overall['source_method_pass_rate']:.1%}")
    print(f"Evidence presence rate: {overall['evidence_presence_rate']:.1%}")
    for cid, stats in report["by_contract"].items():
        print(f"- {cid}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
