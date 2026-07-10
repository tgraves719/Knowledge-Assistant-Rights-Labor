"""
Canonical wage-table evidence evaluator.

Runs deterministic contract-scoped wage lookups and verifies canonical-row
table evidence is present in runtime wage results.
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

import backend.config as runtime_config
from backend.config import DATA_DIR
import backend.retrieval.router as router_module
from backend.contracts import get_contract_catalog_entry
from backend.retrieval.router import HybridRetriever


def _canonical_contract_id(contract_id: str) -> str:
    entry = get_contract_catalog_entry(contract_id)
    if entry and entry.get("contract_id"):
        return str(entry["contract_id"])
    return str(contract_id)


def run(
    test_file: Optional[Path] = None,
    bm25_only: bool = False,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "wage_table_evidence_test.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    test_cases = list(payload.get("test_cases") or [])

    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_keyword = runtime_config.HYBRID_KEYWORD_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_router_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    try:
        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)

        rows = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        source_method_passes = 0
        table_evidence_passes = 0
        table_id_passes = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = _canonical_contract_id(str(case.get("contract_id") or ""))
            classification = str(case.get("classification") or "")
            hours_worked = int(case.get("hours_worked") or 0)
            months_employed = int(case.get("months_employed") or 0)
            expected_source_method = str(case.get("expected_source_method") or "canonical_rows")
            min_table_rows = int(case.get("min_table_evidence_rows") or 1)

            wage_info = retriever.lookup_wage(
                classification=classification,
                hours_worked=hours_worked,
                months_employed=months_employed,
                contract_id=contract_id,
            )

            found = wage_info is not None
            source_method = str((wage_info or {}).get("source_method") or "")
            table_evidence = list((wage_info or {}).get("table_evidence") or [])
            citation = str((wage_info or {}).get("citation") or "")
            table_ids = [
                str((row or {}).get("table_id") or "").strip()
                for row in table_evidence
                if isinstance(row, dict)
            ]
            table_ids_non_empty = [tid for tid in table_ids if tid]

            source_method_ok = found and source_method == expected_source_method
            table_evidence_ok = found and len(table_evidence) >= min_table_rows
            table_id_ok = found and len(table_ids_non_empty) >= 1
            citation_ok = found and ("appendix a" in citation.lower())
            passed = bool(found and source_method_ok and table_evidence_ok and table_id_ok and citation_ok)

            by_contract[contract_id]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1
            if source_method_ok:
                source_method_passes += 1
            if table_evidence_ok:
                table_evidence_passes += 1
            if table_id_ok:
                table_id_passes += 1

            rows.append(
                {
                    "id": case_id,
                    "contract_id": contract_id,
                    "classification": classification,
                    "hours_worked": hours_worked,
                    "months_employed": months_employed,
                    "expected_source_method": expected_source_method,
                    "min_table_evidence_rows": min_table_rows,
                    "found": found,
                    "source_method": source_method,
                    "source_method_ok": source_method_ok,
                    "table_evidence_count": len(table_evidence),
                    "table_evidence_ok": table_evidence_ok,
                    "table_ids": table_ids_non_empty[:5],
                    "table_id_ok": table_id_ok,
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
            "schema_version": "wage_table_evidence_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": round((passed / total) if total else 0.0, 4),
                "source_method_pass_rate": round((source_method_passes / total) if total else 0.0, 4),
                "table_evidence_presence_rate": round((table_evidence_passes / total) if total else 0.0, 4),
                "table_id_presence_rate": round((table_id_passes / total) if total else 0.0, 4),
            },
            "by_contract": by_contract_summary,
            "results": rows,
        }
        return report
    finally:
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        runtime_config.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_router_keyword


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "wage_table_evidence_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate canonical wage-table evidence slice.")
    parser.add_argument("--input", type=str, default=None, help="Path to wage-table-evidence benchmark JSON")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, bm25_only=args.bm25_only)
    out_path = _write_report(report)

    overall = report["overall"]
    print("=" * 72)
    print("KARL Wage-Table Evidence Evaluation")
    print("=" * 72)
    print(f"Overall: {overall['passed']}/{overall['total']} ({overall['pass_rate']:.1%})")
    print(f"Source-method pass rate: {overall['source_method_pass_rate']:.1%}")
    print(f"Table evidence presence rate: {overall['table_evidence_presence_rate']:.1%}")
    print(f"Table ID presence rate: {overall['table_id_presence_rate']:.1%}")
    for cid, stats in report["by_contract"].items():
        print(f"- {cid}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
