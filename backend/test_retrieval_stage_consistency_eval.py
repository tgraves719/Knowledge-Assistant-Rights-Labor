"""Regression checks for retrieval stage consistency evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_retrieval_stage_consistency as evaluator


def _test_retrieval_stage_consistency_eval_passes_bm25_only() -> None:
    report = evaluator.run(bm25_only=True)
    overall = dict(report.get("overall") or {})
    assert report.get("dataset_schema_version") == "retrieval_stage_consistency_test_v1", (
        f"Unexpected dataset schema version: {report.get('dataset_schema_version')}"
    )
    assert int(overall.get("passed") or 0) == int(overall.get("total") or 0), (
        f"Expected all retrieval stage consistency cases to pass. Overall: {overall}"
    )
    assert float(overall.get("planned_strategy_match_rate") or 0.0) >= 1.0, (
        f"Expected planned strategy matches to stay perfect. Overall: {overall}"
    )
    assert float(overall.get("policy_strategy_match_rate") or 0.0) >= 1.0, (
        f"Expected policy strategy matches to stay perfect. Overall: {overall}"
    )
    assert float(overall.get("required_plan_flag_rate") or 0.0) >= 1.0, (
        f"Expected required plan flags to stay perfect. Overall: {overall}"
    )
    assert float(overall.get("required_executed_stage_rate") or 0.0) >= 1.0, (
        f"Expected required executed stages to stay perfect. Overall: {overall}"
    )
    assert float(overall.get("plan_to_execution_alignment_rate") or 0.0) >= 1.0, (
        f"Expected plan-to-execution alignment to stay perfect. Overall: {overall}"
    )


def main() -> None:
    _test_retrieval_stage_consistency_eval_passes_bm25_only()
    print("[OK] Retrieval stage consistency evaluation checks passed")


if __name__ == "__main__":
    main()
