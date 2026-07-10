"""Deterministic checks for canonical miss-record integrity evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_miss_record_integrity as evaluate_miss_record_integrity


def _workspace_tempdir(prefix: str) -> Path:
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _valid_record() -> dict:
    return {
        "schema_version": "real_miss_record_v3",
        "taxonomy_type": "trigger_intent_defect",
        "root_cause_type": "deterministic_logic_defect",
        "contract_id": "local7_safeway_pueblo_clerks_2022",
        "question": "what about 4 years",
        "operator_label": "vacation_followup_4_years_clerks",
        "miss_summary": "Short vacation follow-up should reuse deterministic entitlement resolution.",
        "regression_status": "regression_added",
        "regression_case_id": "vacation_followup_4_years",
        "topic": "vacation",
        "retrieval_strategy": "followup_anchor_seeded",
        "followup_context_used": True,
        "retrieval_anchor_count": 2,
        "classification_review_state": "resolved",
        "final_citations": ["Article 13"],
    }


def _test_eval_passes_for_linked_valid_records() -> None:
    tmp_dir = _workspace_tempdir("miss_eval_pass_")
    try:
        records_dir = tmp_dir / "records"
        _write_json(records_dir / "vacation_followup_4_years_clerks.json", _valid_record())
        report = evaluate_miss_record_integrity.run(records_dir)
        overall = report.get("overall") or {}
        assert bool(overall.get("pass")) is True
        assert int(overall.get("records_total") or 0) == 1
        assert float(overall.get("regression_link_coverage_rate") or 0.0) == 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _test_eval_fails_for_unlinked_regression_added_record() -> None:
    tmp_dir = _workspace_tempdir("miss_eval_unlinked_")
    try:
        records_dir = tmp_dir / "records"
        payload = _valid_record()
        payload["regression_case_id"] = "not_a_real_case"
        _write_json(records_dir / "bad_record.json", payload)
        report = evaluate_miss_record_integrity.run(records_dir)
        overall = report.get("overall") or {}
        row = (report.get("results") or [])[0]
        assert bool(overall.get("pass")) is False
        assert any("not a known canonical case" in err for err in list(row.get("errors") or []))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _test_eval_fails_for_followup_record_without_anchor_metadata() -> None:
    tmp_dir = _workspace_tempdir("miss_eval_followup_")
    try:
        records_dir = tmp_dir / "records"
        payload = _valid_record()
        payload["retrieval_strategy"] = ""
        payload["retrieval_anchor_count"] = 0
        _write_json(records_dir / "bad_followup.json", payload)
        report = evaluate_miss_record_integrity.run(records_dir)
        overall = report.get("overall") or {}
        row = (report.get("results") or [])[0]
        assert bool(overall.get("pass")) is False
        assert any("followup_context_used records must include retrieval_strategy" in err for err in list(row.get("errors") or []))
        assert any("retrieval_anchor_count >= 1" in err for err in list(row.get("errors") or []))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    _test_eval_passes_for_linked_valid_records()
    _test_eval_fails_for_unlinked_regression_added_record()
    _test_eval_fails_for_followup_record_without_anchor_metadata()
    print("[OK] miss-record integrity eval checks passed")


if __name__ == "__main__":
    main()
