"""Deterministic checks for v0.9.0 readiness scorecard evaluator."""

from __future__ import annotations

import json
import shutil
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_release_090 as release_090


@contextmanager
def _workspace_tempdir(prefix: str):
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _test_release_scorecard_passes_on_valid_artifacts() -> None:
    with _workspace_tempdir("release090_pass_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        base_chunk_lineage_results = tmp / "base_chunk_lineage_report.json"
        contract_text_compare_amended_results = tmp / "contract_text_compare_amended_results.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_role_catalog_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_retrieval_stage_consistency --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_real_user_regressions", "return_code": 0},
                    {"command": "python -m backend.evaluate_miss_record_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_readiness", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_deep_suite", "return_code": 0},
                ],
            },
        )
        _write_json(
            v3_results,
            {
                "overall": {"pass": True, "pass_rate": 1.0, "components_total": 17},
                "components": {"retrieval_stage_consistency": {"pass": True, "details": {}}},
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 2,
                    "records_present": True,
                    "regression_added_total": 2,
                    "regression_linked_total": 2,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 2,
                    "records_present": True,
                    "regression_added_total": 2,
                    "regression_linked_total": 2,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(
            moa_readiness_results,
            {
                "gate_pass": True,
                "gates": {
                    "tests_and_commands_ok": {"pass": True},
                    "moa_effective_pass_rate": {"pass": True},
                    "moa_effective_deep_pass_rate": {"pass": True},
                },
            },
        )
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 1.0}})
        _write_json(
            moa_delupd_results,
            {
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0},
                "buckets": {
                    "updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 2},
                    "deleted": {"pass_rate": 1.0},
                },
            },
        )
        _write_json(
            moa_delupd_answer_results,
            {
                "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
                "dataset_schema_version": "moa_deleted_vs_updated_test_v1",
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0},
                "by_bucket": {
                    "updated": {"pass_rate": 1.0},
                    "deleted": {"pass_rate": 1.0},
                },
            },
        )
        _write_json(
            base_chunk_lineage_results,
            {
                "schema_version": "base_chunk_lineage_report_v1",
                "summary": {"total_contracts": 1, "high_risk": 0, "medium_risk": 0, "low_risk": 1},
                "contracts": [],
            },
        )
        _write_json(
            contract_text_compare_amended_results,
            {
                "schema_version": "contract_text_compare_amended_eval_v1",
                "dataset_schema_version": "contract_text_compare_amended_targets_test_v1",
                "overall": {"pass": True, "pass_rate": 1.0, "passed": 2, "total": 2},
                "coverage": {
                    "contract_coverage_rate": 1.0,
                    "operation_coverage_rate": 1.0,
                    "approved_replace_section_ops_total": 2,
                    "approved_replace_section_ops_covered": 2,
                    "uncovered_contracts": [],
                    "missing_targets_count": 0,
                },
            },
        )

        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text(
            "payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n",
            encoding="utf-8",
        )

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(base_chunk_lineage_results),
            contract_text_compare_amended_results=str(contract_text_compare_amended_results),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        assert bool((report.get("overall") or {}).get("pass")) is True
        lineage = (report.get("advisories") or {}).get("base_chunk_lineage") or {}
        assert lineage.get("status") == "ok"
        assert bool(lineage.get("warning")) is False
        compare_adv = (report.get("advisories") or {}).get("contract_text_compare_amended") or {}
        assert compare_adv.get("status") == "ok"
        assert bool(compare_adv.get("warning")) is False


def _test_release_scorecard_fails_on_low_provenance_rate() -> None:
    with _workspace_tempdir("release090_fail_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        base_chunk_lineage_results = tmp / "base_chunk_lineage_report.json"
        contract_text_compare_amended_results = tmp / "contract_text_compare_amended_results.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_role_catalog_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_retrieval_stage_consistency --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_real_user_regressions", "return_code": 0},
                    {"command": "python -m backend.evaluate_miss_record_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_readiness", "return_code": 0},
                ],
            },
        )
        _write_json(
            v3_results,
            {
                "overall": {"pass": True, "pass_rate": 1.0, "components_total": 17},
                "components": {"retrieval_stage_consistency": {"pass": True, "details": {}}},
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(
            moa_readiness_results,
            {
                "gate_pass": True,
                "gates": {
                    "tests_and_commands_ok": {"pass": True},
                    "moa_effective_pass_rate": {"pass": True},
                },
            },
        )
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 0.25}})
        _write_json(
            moa_delupd_results,
            {
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0},
                "buckets": {
                    "updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 2},
                    "deleted": {"pass_rate": 1.0},
                },
            },
        )
        _write_json(
            moa_delupd_answer_results,
            {
                "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
                "dataset_schema_version": "moa_deleted_vs_updated_test_v1",
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0},
                "by_bucket": {
                    "updated": {"pass_rate": 1.0},
                    "deleted": {"pass_rate": 1.0},
                },
            },
        )
        _write_json(
            base_chunk_lineage_results,
            {
                "schema_version": "base_chunk_lineage_report_v1",
                "summary": {"total_contracts": 2, "high_risk": 1, "medium_risk": 0, "low_risk": 1},
                "contracts": [
                    {"contract_id": "x", "risk_level": "high", "findings": ["base_chunk_missing"]},
                    {"contract_id": "y", "risk_level": "low", "findings": []},
                ],
            },
        )
        _write_json(
            contract_text_compare_amended_results,
            {
                "schema_version": "contract_text_compare_amended_eval_v1",
                "dataset_schema_version": "contract_text_compare_amended_targets_test_v1",
                "overall": {"pass": False, "pass_rate": 0.5, "passed": 1, "total": 2},
                "coverage": {
                    "contract_coverage_rate": 1.0,
                    "operation_coverage_rate": 0.5,
                    "approved_replace_section_ops_total": 2,
                    "approved_replace_section_ops_covered": 1,
                    "uncovered_contracts": [],
                    "missing_targets_count": 1,
                },
            },
        )

        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text(
            "payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n",
            encoding="utf-8",
        )

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(base_chunk_lineage_results),
            contract_text_compare_amended_results=str(contract_text_compare_amended_results),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        overall = report.get("overall") or {}
        assert bool(overall.get("pass")) is False
        component = (report.get("components") or {}).get("moa_citation_provenance_source_type") or {}
        assert bool(component.get("pass")) is False
        lineage = (report.get("advisories") or {}).get("base_chunk_lineage") or {}
        assert lineage.get("status") == "warn"
        assert bool(lineage.get("warning")) is True
        compare_adv = (report.get("advisories") or {}).get("contract_text_compare_amended") or {}
        assert compare_adv.get("status") == "warn"
        assert bool(compare_adv.get("warning")) is True


def _test_release_scorecard_missing_lineage_artifact_is_non_gating_warning() -> None:
    with _workspace_tempdir("release090_lineage_missing_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        contract_text_compare_amended_results = tmp / "contract_text_compare_amended_results.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_role_catalog_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_retrieval_stage_consistency --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_real_user_regressions", "return_code": 0},
                    {"command": "python -m backend.evaluate_miss_record_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_readiness", "return_code": 0},
                ],
            },
        )
        _write_json(
            v3_results,
            {
                "overall": {"pass": True, "pass_rate": 1.0, "components_total": 17},
                "components": {"retrieval_stage_consistency": {"pass": True, "details": {}}},
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(
            moa_readiness_results,
            {"gate_pass": True, "gates": {"tests_and_commands_ok": {"pass": True}, "moa_effective_pass_rate": {"pass": True}}},
        )
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 1.0}})
        _write_json(
            moa_delupd_results,
            {
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0},
                "buckets": {"updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 1}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(
            moa_delupd_answer_results,
            {
                "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
                "dataset_schema_version": "moa_deleted_vs_updated_test_v1",
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0},
                "by_bucket": {"updated": {"pass_rate": 1.0}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(
            contract_text_compare_amended_results,
            {
                "schema_version": "contract_text_compare_amended_eval_v1",
                "dataset_schema_version": "contract_text_compare_amended_targets_test_v1",
                "overall": {"pass": True, "pass_rate": 1.0, "passed": 1, "total": 1},
                "coverage": {
                    "contract_coverage_rate": 1.0,
                    "operation_coverage_rate": 1.0,
                    "approved_replace_section_ops_total": 1,
                    "approved_replace_section_ops_covered": 1,
                    "uncovered_contracts": [],
                    "missing_targets_count": 0,
                },
            },
        )
        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text(
            "payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n",
            encoding="utf-8",
        )

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(tmp / "missing_lineage.json"),
            contract_text_compare_amended_results=str(contract_text_compare_amended_results),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        assert bool((report.get("overall") or {}).get("pass")) is True
        lineage = (report.get("advisories") or {}).get("base_chunk_lineage") or {}
        assert lineage.get("status") == "missing"
        assert bool(lineage.get("warning")) is True
        compare_adv = (report.get("advisories") or {}).get("contract_text_compare_amended") or {}
        assert compare_adv.get("status") == "ok"
        assert bool(compare_adv.get("warning")) is False


def _test_release_scorecard_missing_compare_artifact_is_non_gating_warning() -> None:
    with _workspace_tempdir("release090_compare_missing_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        base_chunk_lineage_results = tmp / "base_chunk_lineage_report.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_role_catalog_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_retrieval_stage_consistency --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_real_user_regressions", "return_code": 0},
                    {"command": "python -m backend.evaluate_miss_record_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_readiness", "return_code": 0},
                ],
            },
        )
        _write_json(
            v3_results,
            {
                "overall": {"pass": True, "pass_rate": 1.0, "components_total": 17},
                "components": {"retrieval_stage_consistency": {"pass": True, "details": {}}},
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(moa_readiness_results, {"gate_pass": True, "gates": {"tests_and_commands_ok": {"pass": True}, "moa_effective_pass_rate": {"pass": True}}})
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 1.0}})
        _write_json(moa_delupd_results, {"gate": {"pass": True}, "overall": {"pass_rate": 1.0}, "buckets": {"updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 1}, "deleted": {"pass_rate": 1.0}}})
        _write_json(moa_delupd_answer_results, {"schema_version": "moa_deleted_vs_updated_answer_eval_v1", "dataset_schema_version": "moa_deleted_vs_updated_test_v1", "gate": {"pass": True}, "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0}, "by_bucket": {"updated": {"pass_rate": 1.0}, "deleted": {"pass_rate": 1.0}}})
        _write_json(base_chunk_lineage_results, {"schema_version": "base_chunk_lineage_report_v1", "summary": {"total_contracts": 1, "high_risk": 0, "medium_risk": 0, "low_risk": 1}, "contracts": []})
        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text("payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n", encoding="utf-8")

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(base_chunk_lineage_results),
            contract_text_compare_amended_results=str(tmp / "missing_compare.json"),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        assert bool((report.get("overall") or {}).get("pass")) is True
        compare_adv = (report.get("advisories") or {}).get("contract_text_compare_amended") or {}
        assert compare_adv.get("status") == "missing"
        assert bool(compare_adv.get("warning")) is True


def _test_release_scorecard_fails_when_v3_missing_retrieval_stage_component() -> None:
    with _workspace_tempdir("release090_v3_missing_component_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        base_chunk_lineage_results = tmp / "base_chunk_lineage_report.json"
        contract_text_compare_amended_results = tmp / "contract_text_compare_amended_results.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_role_catalog_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_retrieval_stage_consistency --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_real_user_regressions", "return_code": 0},
                    {"command": "python -m backend.evaluate_miss_record_integrity", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_readiness", "return_code": 0},
                ],
            },
        )
        _write_json(v3_results, {"overall": {"pass": True, "pass_rate": 1.0, "components_total": 16}, "components": {}})
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(
            moa_readiness_results,
            {"gate_pass": True, "gates": {"tests_and_commands_ok": {"pass": True}, "moa_effective_pass_rate": {"pass": True}}},
        )
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 1.0}})
        _write_json(
            moa_delupd_results,
            {
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0},
                "buckets": {"updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 1}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(
            moa_delupd_answer_results,
            {
                "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
                "dataset_schema_version": "moa_deleted_vs_updated_test_v1",
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0},
                "by_bucket": {"updated": {"pass_rate": 1.0}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(base_chunk_lineage_results, {"schema_version": "base_chunk_lineage_report_v1", "summary": {"total_contracts": 1, "high_risk": 0, "medium_risk": 0, "low_risk": 1}, "contracts": []})
        _write_json(
            contract_text_compare_amended_results,
            {
                "schema_version": "contract_text_compare_amended_eval_v1",
                "dataset_schema_version": "contract_text_compare_amended_targets_test_v1",
                "overall": {"pass": True, "pass_rate": 1.0, "passed": 1, "total": 1},
                "coverage": {
                    "contract_coverage_rate": 1.0,
                    "operation_coverage_rate": 1.0,
                    "approved_replace_section_ops_total": 1,
                    "approved_replace_section_ops_covered": 1,
                    "uncovered_contracts": [],
                    "missing_targets_count": 0,
                },
            },
        )
        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text("payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n", encoding="utf-8")

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(base_chunk_lineage_results),
            contract_text_compare_amended_results=str(contract_text_compare_amended_results),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        overall = report.get("overall") or {}
        assert bool(overall.get("pass")) is False
        v3_component = (report.get("components") or {}).get("v3_green") or {}
        details = v3_component.get("details") or {}
        assert bool(v3_component.get("pass")) is False
        assert bool(details.get("retrieval_stage_consistency_present")) is False


def _test_release_scorecard_fails_when_track_all_metadata_misses_required_canonical_commands() -> None:
    with _workspace_tempdir("release090_track_all_missing_commands_") as tmp:
        eval_metadata = tmp / "eval_run_metadata_all_20260222T120000Z.json"
        v3_results = tmp / "v3_results.json"
        miss_record_integrity_results = tmp / "miss_record_integrity_results.json"
        moa_deep_results = tmp / "moa_deep_eval_suite_results.json"
        moa_readiness_results = tmp / "moa_readiness_results.json"
        moa_effective_results = tmp / "moa_effective_results.json"
        moa_delupd_results = tmp / "moa_deleted_vs_updated_results.json"
        moa_delupd_answer_results = tmp / "moa_deleted_vs_updated_answer_results.json"
        base_chunk_lineage_results = tmp / "base_chunk_lineage_report.json"
        contract_text_compare_amended_results = tmp / "contract_text_compare_amended_results.json"
        api_file = tmp / "api.py"
        verifier_file = tmp / "verifier.py"

        _write_json(
            eval_metadata,
            {
                "track": "all",
                "manifest_validation_return_code": 0,
                "results": [
                    {"command": "python -m backend.evaluate_v3 --bm25-only", "return_code": 0},
                    {"command": "python -m backend.evaluate_moa_deep_suite", "return_code": 0},
                ],
            },
        )
        _write_json(
            v3_results,
            {
                "overall": {"pass": True, "pass_rate": 1.0, "components_total": 17},
                "components": {"retrieval_stage_consistency": {"pass": True, "details": {}}},
            },
        )
        _write_json(
            miss_record_integrity_results,
            {
                "overall": {
                    "pass": True,
                    "records_total": 1,
                    "records_present": True,
                    "regression_added_total": 1,
                    "regression_linked_total": 1,
                    "regression_link_coverage_rate": 1.0,
                },
            },
        )
        _write_json(moa_deep_results, {"overall": {"all_passed": True, "pass_rate": 1.0, "commands_total": 16}})
        _write_json(
            moa_readiness_results,
            {"gate_pass": True, "gates": {"tests_and_commands_ok": {"pass": True}, "moa_effective_pass_rate": {"pass": True}}},
        )
        _write_json(moa_effective_results, {"overall": {"source_type_match_rate": 1.0}})
        _write_json(
            moa_delupd_results,
            {
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0},
                "buckets": {"updated": {"pass_rate": 1.0, "moa_source_type_match_rate": 1.0, "source_type_cases": 1}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(
            moa_delupd_answer_results,
            {
                "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
                "dataset_schema_version": "moa_deleted_vs_updated_test_v1",
                "gate": {"pass": True},
                "overall": {"pass_rate": 1.0, "source_type_match_rate": 1.0},
                "by_bucket": {"updated": {"pass_rate": 1.0}, "deleted": {"pass_rate": 1.0}},
            },
        )
        _write_json(base_chunk_lineage_results, {"schema_version": "base_chunk_lineage_report_v1", "summary": {"total_contracts": 1, "high_risk": 0, "medium_risk": 0, "low_risk": 1}, "contracts": []})
        _write_json(
            contract_text_compare_amended_results,
            {
                "schema_version": "contract_text_compare_amended_eval_v1",
                "dataset_schema_version": "contract_text_compare_amended_targets_test_v1",
                "overall": {"pass": True, "pass_rate": 1.0, "passed": 1, "total": 1},
                "coverage": {
                    "contract_coverage_rate": 1.0,
                    "operation_coverage_rate": 1.0,
                    "approved_replace_section_ops_total": 1,
                    "approved_replace_section_ops_covered": 1,
                    "uncovered_contracts": [],
                    "missing_targets_count": 0,
                },
            },
        )
        api_file.write_text(
            "\n".join(
                [
                    "from pydantic import BaseModel, Field",
                    "from typing import Optional",
                    "class QueryResponse(BaseModel):",
                    "    answer: str",
                    "    citations: list[str]",
                    "    sources: list[dict]",
                    "    effective_version_id: Optional[str] = None",
                    "    amendments_applied: list[str] = Field(default_factory=list)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verifier_file.write_text("payload = {'source_type': 'moa', 'effective_version_id': 'v1', 'amendments_applied': []}\n", encoding="utf-8")

        args = Namespace(
            eval_runner_metadata=str(eval_metadata),
            v3_results=str(v3_results),
            miss_record_integrity_results=str(miss_record_integrity_results),
            moa_deep_results=str(moa_deep_results),
            moa_readiness_results=str(moa_readiness_results),
            moa_effective_results=str(moa_effective_results),
            moa_deleted_vs_updated_results=str(moa_delupd_results),
            moa_deleted_vs_updated_answer_results=str(moa_delupd_answer_results),
            base_chunk_lineage_results=str(base_chunk_lineage_results),
            contract_text_compare_amended_results=str(contract_text_compare_amended_results),
            api_file=str(api_file),
            verifier_file=str(verifier_file),
            skip_manifest_validation=True,
            skip_track_all_metadata=False,
            min_moa_source_type_match_rate=0.95,
            min_moa_deleted_vs_updated_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_updated_moa_source_type_match_rate=1.0,
            min_moa_deleted_vs_updated_answer_overall_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_updated_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_deleted_pass_rate=1.0,
            min_moa_deleted_vs_updated_answer_source_type_match_rate=1.0,
        )
        report = release_090.run(args)
        overall = report.get("overall") or {}
        component = (report.get("components") or {}).get("runner_track_all_green") or {}
        details = component.get("details") or {}
        assert bool(overall.get("pass")) is False
        assert bool(component.get("pass")) is False
        assert "backend.evaluate_miss_record_integrity" in set(details.get("missing_required_commands") or [])


def main() -> None:
    _test_release_scorecard_passes_on_valid_artifacts()
    _test_release_scorecard_fails_on_low_provenance_rate()
    _test_release_scorecard_missing_lineage_artifact_is_non_gating_warning()
    _test_release_scorecard_missing_compare_artifact_is_non_gating_warning()
    _test_release_scorecard_fails_when_v3_missing_retrieval_stage_component()
    _test_release_scorecard_fails_when_track_all_metadata_misses_required_canonical_commands()
    print("[OK] release_090 scorecard checks passed")


if __name__ == "__main__":
    main()
