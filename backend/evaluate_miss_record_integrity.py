"""Canonical evaluator for structured local miss-record integrity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.evaluate_real_user_regressions import case_ids as canonical_regression_case_ids
from backend.miss_records import load_miss_record


DEFAULT_INPUT_DIR = DATA_DIR / "miss_records" / "records"
DEFAULT_OUTPUT = DATA_DIR / "test_set" / "miss_record_integrity_results.json"


def _iter_record_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*.json")
        if path.is_file()
    )


def _validate_record(record: dict[str, Any], known_case_ids: set[str]) -> list[str]:
    errors: list[str] = []
    review_state = str(record.get("classification_review_state") or "")
    clarification_wage_keys = list(record.get("clarification_wage_keys") or [])
    regression_status = str(record.get("regression_status") or "")
    regression_case_id = str(record.get("regression_case_id") or "")
    retrieval_strategy = str(record.get("retrieval_strategy") or "")
    retrieval_anchor_count = record.get("retrieval_anchor_count")

    if review_state == "needs_clarification" and not clarification_wage_keys:
        errors.append("needs_clarification records must include clarification_wage_keys")

    if record.get("followup_context_used") is True:
        if not retrieval_strategy:
            errors.append("followup_context_used records must include retrieval_strategy")
        if not isinstance(retrieval_anchor_count, int) or retrieval_anchor_count < 1:
            errors.append("followup_context_used records must include retrieval_anchor_count >= 1")

    if regression_case_id and regression_case_id not in known_case_ids:
        errors.append(f"regression_case_id is not a known canonical case: {regression_case_id}")

    if regression_status == "regression_added" and not regression_case_id:
        errors.append("regression_added records must include regression_case_id")

    return errors


def run(input_dir: Path) -> dict[str, Any]:
    record_paths = _iter_record_paths(input_dir)
    known_case_ids = canonical_regression_case_ids()
    rows: list[dict[str, Any]] = []
    passed = 0
    regression_added = 0
    regression_linked = 0
    by_contract: dict[str, int] = {}

    for path in record_paths:
        try:
            record = load_miss_record(path)
            errors = _validate_record(record, known_case_ids)
            contract_id = str(record.get("contract_id") or "")
            if contract_id:
                by_contract[contract_id] = by_contract.get(contract_id, 0) + 1
            if str(record.get("regression_status") or "") == "regression_added":
                regression_added += 1
                if str(record.get("regression_case_id") or "") in known_case_ids:
                    regression_linked += 1
            row = {
                "path": str(path),
                "miss_id": str(record.get("miss_id") or ""),
                "contract_id": contract_id,
                "operator_label": str(record.get("operator_label") or ""),
                "regression_status": str(record.get("regression_status") or ""),
                "regression_case_id": str(record.get("regression_case_id") or "") or None,
                "pass": not errors,
                "errors": errors,
            }
            if not errors:
                passed += 1
        except Exception as exc:
            row = {
                "path": str(path),
                "miss_id": None,
                "contract_id": None,
                "operator_label": None,
                "regression_status": None,
                "regression_case_id": None,
                "pass": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        rows.append(row)

    total = len(rows)
    return {
        "schema_version": "miss_record_integrity_eval_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "known_canonical_regression_case_ids": sorted(known_case_ids),
        "overall": {
            "records_total": total,
            "records_present": total > 0,
            "passed": passed,
            "pass_rate": round((passed / total) if total else 1.0, 4),
            "pass": passed == total,
            "regression_added_total": regression_added,
            "regression_linked_total": regression_linked,
            "regression_link_coverage_rate": round((regression_linked / regression_added) if regression_added else 1.0, 4),
        },
        "by_contract": [
            {"contract_id": contract_id, "records": count}
            for contract_id, count in sorted(by_contract.items())
        ],
        "results": rows,
    }


def _write_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate structured KARL miss-record integrity.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run(Path(args.input_dir))
    out_path = _write_report(report, Path(args.output))
    overall = report.get("overall") or {}

    print("=" * 72)
    print("KARL Miss-Record Integrity Evaluation")
    print("=" * 72)
    print(
        f"Overall: {overall.get('passed', 0)}/{overall.get('records_total', 0)} "
        f"({float(overall.get('pass_rate') or 0.0):.1%})"
    )
    print(f"Records present: {overall.get('records_present')}")
    print(f"Results: {out_path}")
    return 0 if overall.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
