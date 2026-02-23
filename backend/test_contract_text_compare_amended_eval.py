"""Deterministic checks for amended-section text compare regression evaluator."""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_contract_text_compare_amended as compare_eval


def _test_amended_compare_targets_pass() -> None:
    report = compare_eval.run()
    overall = report.get("overall") or {}
    coverage = report.get("coverage") or {}
    assert str(report.get("schema_version") or "") == "contract_text_compare_amended_eval_v1"
    assert str(report.get("dataset_schema_version") or "") == "contract_text_compare_amended_targets_test_v1"
    assert int(overall.get("total") or 0) >= 2
    assert bool(overall.get("pass")) is True
    assert "contract_coverage_rate" in coverage
    assert "operation_coverage_rate" in coverage
    assert float(coverage.get("contract_coverage_rate") or 0.0) >= 1.0
    assert float(coverage.get("operation_coverage_rate") or 0.0) >= 1.0


def main() -> None:
    _test_amended_compare_targets_pass()
    print("[OK] contract_text_compare_amended evaluator checks passed")


if __name__ == "__main__":
    main()
