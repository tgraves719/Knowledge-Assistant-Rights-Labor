"""
Deterministic v0.9.0 readiness scorecard.

Aggregates must-have release gates from existing evaluator artifacts.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.validate_manifests import main as validate_manifests_main


OUT_PATH = DATA_DIR / "test_set" / "release_0_9_0_scorecard.json"
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_API_FILE = PROJECT_ROOT / "backend" / "api.py"
DEFAULT_VERIFIER_FILE = PROJECT_ROOT / "backend" / "generation" / "verifier.py"
DEFAULT_BASE_CHUNK_LINEAGE_FILE = DATA_DIR / "test_set" / "base_chunk_lineage_report.json"
DEFAULT_CONTRACT_TEXT_COMPARE_AMENDED_FILE = DATA_DIR / "test_set" / "contract_text_compare_amended_results.json"
DEFAULT_MISS_RECORD_INTEGRITY_FILE = DATA_DIR / "test_set" / "miss_record_integrity_results.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_all_eval_metadata() -> Path | None:
    root = DATA_DIR / "test_set"
    candidates = sorted(root.glob("eval_run_metadata_all_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def _query_response_field_set(api_file: Path) -> set[str]:
    with open(api_file, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QueryResponse":
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
            return fields
    return set()


def _check_manifest_validation(skip: bool) -> tuple[bool, dict]:
    if skip:
        return True, {"skipped": True, "pass": True}
    rc = int(validate_manifests_main())
    return rc == 0, {"skipped": False, "return_code": rc, "pass": rc == 0}


def _check_track_all_metadata(path: Path) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}

    track = str(payload.get("track") or "")
    results = payload.get("results") or []
    manifest_rc = payload.get("manifest_validation_return_code")
    command_count = len(results)
    failing = [r for r in results if int(r.get("return_code", 1)) != 0]
    commands = [str(r.get("command") or "") for r in results]
    required_command_tokens = [
        "backend.evaluate_v3",
        "backend.evaluate_role_catalog_integrity",
        "backend.evaluate_retrieval_stage_consistency",
        "backend.evaluate_real_user_regressions",
        "backend.evaluate_miss_record_integrity",
        "backend.evaluate_moa_readiness",
    ]
    missing_required_commands = [
        token for token in required_command_tokens
        if not any(token in command for command in commands)
    ]
    pass_all = (
        track == "all"
        and manifest_rc == 0
        and command_count > 0
        and not failing
        and not missing_required_commands
    )
    return pass_all, {
        "path": str(path),
        "track": track,
        "manifest_validation_return_code": manifest_rc,
        "command_count": command_count,
        "missing_required_commands": missing_required_commands,
        "failing_commands": [str(r.get("command") or "") for r in failing[:10]],
        "pass": pass_all,
    }


def _check_v3(path: Path) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    components = payload.get("components") or {}
    retrieval_component = components.get("retrieval_stage_consistency") or {}
    passed = bool(overall.get("pass"))
    pass_rate = float(overall.get("pass_rate") or 0.0)
    total = int(overall.get("components_total") or 0)
    retrieval_component_present = bool(retrieval_component)
    retrieval_component_pass = bool(retrieval_component.get("pass")) if retrieval_component_present else False
    ok = passed and retrieval_component_present and retrieval_component_pass
    return ok, {
        "path": str(path),
        "pass": passed,
        "pass_rate": pass_rate,
        "components_total": total,
        "retrieval_stage_consistency_present": retrieval_component_present,
        "retrieval_stage_consistency_pass": retrieval_component_pass,
    }


def _check_miss_record_integrity(path: Path) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    passed = bool(overall.get("pass"))
    return passed, {
        "path": str(path),
        "pass": passed,
        "records_total": int(overall.get("records_total") or 0),
        "records_present": bool(overall.get("records_present")),
        "regression_added_total": int(overall.get("regression_added_total") or 0),
        "regression_linked_total": int(overall.get("regression_linked_total") or 0),
        "regression_link_coverage_rate": float(overall.get("regression_link_coverage_rate") or 0.0),
    }


def _check_moa_deep(path: Path) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    all_passed = bool(overall.get("all_passed"))
    pass_rate = float(overall.get("pass_rate") or 0.0)
    return all_passed, {
        "path": str(path),
        "pass": all_passed,
        "pass_rate": pass_rate,
        "commands_total": int(overall.get("commands_total") or 0),
    }


def _check_moa_readiness(path: Path) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    gates = payload.get("gates") or {}
    gate_pass = bool(payload.get("gate_pass"))
    tests_ok = bool((gates.get("tests_and_commands_ok") or {}).get("pass"))
    moa_rate_ok = bool((gates.get("moa_effective_pass_rate") or {}).get("pass"))
    delupd_gate = gates.get("moa_deleted_vs_updated_gate")
    delupd_ok = True if delupd_gate is None else bool(delupd_gate.get("pass"))
    delupd_answer_gate = gates.get("moa_deleted_vs_updated_answer_gate")
    delupd_answer_ok = True if delupd_answer_gate is None else bool(delupd_answer_gate.get("pass"))
    deep_gate = gates.get("moa_effective_deep_pass_rate")
    deep_ok = True if deep_gate is None else bool(deep_gate.get("pass"))
    ok = gate_pass and tests_ok and moa_rate_ok and delupd_ok and delupd_answer_ok and deep_ok
    return ok, {
        "path": str(path),
        "gate_pass": gate_pass,
        "tests_and_commands_ok": tests_ok,
        "moa_effective_pass_rate_gate": moa_rate_ok,
        "moa_deleted_vs_updated_gate": delupd_ok,
        "moa_deleted_vs_updated_answer_gate": delupd_answer_ok,
        "moa_effective_deep_pass_rate_gate": deep_ok,
        "pass": ok,
    }


def _check_moa_provenance(path: Path, min_source_type_match_rate: float) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    observed = float(overall.get("source_type_match_rate") or 0.0)
    ok = observed >= min_source_type_match_rate
    return ok, {
        "path": str(path),
        "source_type_match_rate": observed,
        "threshold": min_source_type_match_rate,
        "pass": ok,
    }


def _check_moa_deleted_vs_updated(
    path: Path,
    *,
    min_overall_pass_rate: float,
    min_updated_pass_rate: float,
    min_deleted_pass_rate: float,
    min_updated_moa_source_type_match_rate: float,
) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    buckets = payload.get("buckets") or {}
    gate = payload.get("gate") or {}
    updated = buckets.get("updated") or {}
    deleted = buckets.get("deleted") or {}
    observed_overall = float(overall.get("pass_rate") or 0.0)
    observed_updated = float(updated.get("pass_rate") or 0.0)
    observed_deleted = float(deleted.get("pass_rate") or 0.0)
    source_rate_raw = updated.get("moa_source_type_match_rate")
    source_cases = int(updated.get("source_type_cases") or 0)
    observed_source_rate = (float(source_rate_raw) if source_rate_raw is not None else None)
    source_ok = (source_cases == 0) or ((observed_source_rate or 0.0) >= min_updated_moa_source_type_match_rate)
    ok = (
        bool(gate.get("pass"))
        and observed_overall >= min_overall_pass_rate
        and observed_updated >= min_updated_pass_rate
        and observed_deleted >= min_deleted_pass_rate
        and source_ok
    )
    return ok, {
        "path": str(path),
        "gate_pass": bool(gate.get("pass")),
        "overall_pass_rate": observed_overall,
        "updated_clause_pass_rate": observed_updated,
        "deleted_clause_pass_rate": observed_deleted,
        "updated_moa_source_type_match_rate": observed_source_rate,
        "updated_moa_source_type_cases": source_cases,
        "thresholds": {
            "overall_pass_rate": min_overall_pass_rate,
            "updated_clause_pass_rate": min_updated_pass_rate,
            "deleted_clause_pass_rate": min_deleted_pass_rate,
            "updated_moa_source_type_match_rate": min_updated_moa_source_type_match_rate,
        },
        "pass": ok,
    }


def _check_moa_deleted_vs_updated_answer(
    path: Path,
    *,
    min_overall_pass_rate: float,
    min_updated_pass_rate: float,
    min_deleted_pass_rate: float,
    min_source_type_match_rate: float,
) -> tuple[bool, dict]:
    payload = _load_json(path)
    if not payload:
        return False, {"reason": "artifact_missing", "path": str(path)}
    overall = payload.get("overall") or {}
    by_bucket = payload.get("by_bucket") or {}
    gate = payload.get("gate") or {}
    updated = by_bucket.get("updated") or {}
    deleted = by_bucket.get("deleted") or {}
    observed_overall = float(overall.get("pass_rate") or 0.0)
    observed_updated = float(updated.get("pass_rate") or 0.0)
    observed_deleted = float(deleted.get("pass_rate") or 0.0)
    source_rate_raw = overall.get("source_type_match_rate")
    observed_source_rate = float(source_rate_raw) if source_rate_raw is not None else None
    source_ok = (observed_source_rate is not None) and (observed_source_rate >= min_source_type_match_rate)
    gate_ok = bool(gate.get("pass")) if isinstance(gate, dict) and gate else None
    ok = (
        (True if gate_ok is None else gate_ok)
        and observed_overall >= min_overall_pass_rate
        and observed_updated >= min_updated_pass_rate
        and observed_deleted >= min_deleted_pass_rate
        and source_ok
    )
    return ok, {
        "path": str(path),
        "schema_version": str(payload.get("schema_version") or ""),
        "dataset_schema_version": str(payload.get("dataset_schema_version") or ""),
        "gate_pass": gate_ok,
        "overall_pass_rate": observed_overall,
        "updated_clause_pass_rate": observed_updated,
        "deleted_clause_pass_rate": observed_deleted,
        "source_type_match_rate": observed_source_rate,
        "thresholds": {
            "overall_pass_rate": min_overall_pass_rate,
            "updated_clause_pass_rate": min_updated_pass_rate,
            "deleted_clause_pass_rate": min_deleted_pass_rate,
            "source_type_match_rate": min_source_type_match_rate,
        },
        "pass": ok,
    }


def _check_api_payload_contract_history(api_file: Path, verifier_file: Path) -> tuple[bool, dict]:
    if not api_file.exists():
        return False, {"reason": "api_file_missing", "path": str(api_file)}
    if not verifier_file.exists():
        return False, {"reason": "verifier_file_missing", "path": str(verifier_file)}

    fields = _query_response_field_set(api_file)
    required_fields = {"effective_version_id", "amendments_applied", "sources"}
    missing_fields = sorted(required_fields - fields)

    verifier_text = verifier_file.read_text(encoding="utf-8")
    verifier_required_tokens = [
        "source_type",
        "effective_version_id",
        "amendments_applied",
    ]
    missing_tokens = [t for t in verifier_required_tokens if t not in verifier_text]
    ok = not missing_fields and not missing_tokens
    return ok, {
        "api_file": str(api_file),
        "verifier_file": str(verifier_file),
        "query_response_fields": sorted(fields),
        "missing_query_response_fields": missing_fields,
        "missing_verifier_tokens": missing_tokens,
        "pass": ok,
    }


def _check_base_chunk_lineage_advisory(path: Path) -> dict:
    payload = _load_json(path)
    if not payload:
        return {
            "status": "missing",
            "warning": True,
            "reason": "artifact_missing",
            "path": str(path),
        }

    summary = payload.get("summary") or {}
    contracts = payload.get("contracts") or []
    high_risk = int(summary.get("high_risk") or 0)
    medium_risk = int(summary.get("medium_risk") or 0)
    missing_base = int(summary.get("missing_base_chunk") or 0)
    sampled_high = []
    for row in contracts:
        if not isinstance(row, dict):
            continue
        if str(row.get("risk_level") or "") != "high":
            continue
        sampled_high.append(
            {
                "contract_id": str(row.get("contract_id") or ""),
                "findings": list(row.get("findings") or []),
            }
        )
        if len(sampled_high) >= 5:
            break

    warning = high_risk > 0 or medium_risk > 0
    return {
        "status": "warn" if warning else "ok",
        "warning": warning,
        "path": str(path),
        "schema_version": str(payload.get("schema_version") or ""),
        "summary": {
            "total_contracts": int(summary.get("total_contracts") or 0),
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": int(summary.get("low_risk") or 0),
            "missing_base_chunk": missing_base,
            "base_equals_effective": int(summary.get("base_equals_effective") or 0),
            "patched_section_equal_all": int(summary.get("patched_section_equal_all") or 0),
        },
        "sample_high_risk_contracts": sampled_high,
    }


def _check_contract_text_compare_amended_advisory(path: Path) -> dict:
    payload = _load_json(path)
    if not payload:
        return {
            "status": "missing",
            "warning": True,
            "reason": "artifact_missing",
            "path": str(path),
        }

    overall = payload.get("overall") or {}
    coverage = payload.get("coverage") or {}
    overall_pass = bool(overall.get("pass"))
    total_cases = int(overall.get("total") or 0)
    pass_rate = float(overall.get("pass_rate") or 0.0)
    contract_cov = coverage.get("contract_coverage_rate")
    op_cov = coverage.get("operation_coverage_rate")
    contract_cov_f = float(contract_cov) if contract_cov is not None else None
    op_cov_f = float(op_cov) if op_cov is not None else None
    warning = (
        not overall_pass
        or total_cases <= 0
        or contract_cov_f is None
        or op_cov_f is None
        or contract_cov_f < 1.0
        or op_cov_f < 1.0
    )
    return {
        "status": "warn" if warning else "ok",
        "warning": warning,
        "path": str(path),
        "schema_version": str(payload.get("schema_version") or ""),
        "dataset_schema_version": str(payload.get("dataset_schema_version") or ""),
        "overall": {
            "pass": overall_pass,
            "pass_rate": pass_rate,
            "total": total_cases,
            "passed": int(overall.get("passed") or 0),
        },
        "coverage": {
            "contract_coverage_rate": contract_cov_f,
            "operation_coverage_rate": op_cov_f,
            "approved_replace_section_ops_total": int(coverage.get("approved_replace_section_ops_total") or 0),
            "approved_replace_section_ops_covered": int(coverage.get("approved_replace_section_ops_covered") or 0),
            "uncovered_contracts": list(coverage.get("uncovered_contracts") or []),
            "missing_targets_count": int(coverage.get("missing_targets_count") or 0),
        },
    }


def run(args) -> dict:
    eval_metadata_path = Path(args.eval_runner_metadata) if args.eval_runner_metadata else _latest_all_eval_metadata()
    components: dict[str, dict] = {}
    advisories: dict[str, dict] = {}
    pass_count = 0

    manifest_ok, manifest_details = _check_manifest_validation(skip=bool(args.skip_manifest_validation))
    components["manifest_validation_clean"] = {"pass": manifest_ok, "details": manifest_details}
    pass_count += int(manifest_ok)

    if bool(args.skip_track_all_metadata):
        track_ok, track_details = True, {"skipped": True, "pass": True}
    else:
        track_ok, track_details = _check_track_all_metadata(eval_metadata_path) if eval_metadata_path else (
            False,
            {"reason": "artifact_missing", "path": None},
        )
    components["runner_track_all_green"] = {"pass": track_ok, "details": track_details}
    pass_count += int(track_ok)

    v3_ok, v3_details = _check_v3(Path(args.v3_results))
    components["v3_green"] = {"pass": v3_ok, "details": v3_details}
    pass_count += int(v3_ok)

    miss_record_ok, miss_record_details = _check_miss_record_integrity(Path(args.miss_record_integrity_results))
    components["miss_record_integrity_green"] = {"pass": miss_record_ok, "details": miss_record_details}
    pass_count += int(miss_record_ok)

    moa_deep_ok, moa_deep_details = _check_moa_deep(Path(args.moa_deep_results))
    components["moa_deep_suite_green"] = {"pass": moa_deep_ok, "details": moa_deep_details}
    pass_count += int(moa_deep_ok)

    moa_ready_ok, moa_ready_details = _check_moa_readiness(Path(args.moa_readiness_results))
    components["moa_readiness_green"] = {"pass": moa_ready_ok, "details": moa_ready_details}
    pass_count += int(moa_ready_ok)

    moa_delupd_ok, moa_delupd_details = _check_moa_deleted_vs_updated(
        Path(args.moa_deleted_vs_updated_results),
        min_overall_pass_rate=float(args.min_moa_deleted_vs_updated_overall_pass_rate),
        min_updated_pass_rate=float(args.min_moa_deleted_vs_updated_updated_pass_rate),
        min_deleted_pass_rate=float(args.min_moa_deleted_vs_updated_deleted_pass_rate),
        min_updated_moa_source_type_match_rate=float(args.min_moa_deleted_vs_updated_updated_moa_source_type_match_rate),
    )
    components["moa_deleted_vs_updated_regression"] = {"pass": moa_delupd_ok, "details": moa_delupd_details}
    pass_count += int(moa_delupd_ok)

    moa_delupd_answer_ok, moa_delupd_answer_details = _check_moa_deleted_vs_updated_answer(
        Path(args.moa_deleted_vs_updated_answer_results),
        min_overall_pass_rate=float(args.min_moa_deleted_vs_updated_answer_overall_pass_rate),
        min_updated_pass_rate=float(args.min_moa_deleted_vs_updated_answer_updated_pass_rate),
        min_deleted_pass_rate=float(args.min_moa_deleted_vs_updated_answer_deleted_pass_rate),
        min_source_type_match_rate=float(args.min_moa_deleted_vs_updated_answer_source_type_match_rate),
    )
    components["moa_deleted_vs_updated_answer_regression"] = {"pass": moa_delupd_answer_ok, "details": moa_delupd_answer_details}
    pass_count += int(moa_delupd_answer_ok)

    provenance_ok, provenance_details = _check_moa_provenance(
        Path(args.moa_effective_results),
        min_source_type_match_rate=float(args.min_moa_source_type_match_rate),
    )
    components["moa_citation_provenance_source_type"] = {"pass": provenance_ok, "details": provenance_details}
    pass_count += int(provenance_ok)

    payload_ok, payload_details = _check_api_payload_contract_history(
        api_file=Path(args.api_file),
        verifier_file=Path(args.verifier_file),
    )
    components["api_effective_payload_contract_history"] = {"pass": payload_ok, "details": payload_details}
    pass_count += int(payload_ok)

    lineage_path_raw = getattr(args, "base_chunk_lineage_results", str(DEFAULT_BASE_CHUNK_LINEAGE_FILE))
    advisories["base_chunk_lineage"] = _check_base_chunk_lineage_advisory(Path(lineage_path_raw))
    compare_path_raw = getattr(
        args,
        "contract_text_compare_amended_results",
        str(DEFAULT_CONTRACT_TEXT_COMPARE_AMENDED_FILE),
    )
    advisories["contract_text_compare_amended"] = _check_contract_text_compare_amended_advisory(Path(compare_path_raw))

    total_components = len(components)
    overall_pass = pass_count == total_components
    return {
        "schema_version": "release_090_scorecard_v2",
        "inputs": {
            "eval_runner_metadata": str(eval_metadata_path) if eval_metadata_path else None,
            "v3_results": str(Path(args.v3_results)),
            "miss_record_integrity_results": str(Path(args.miss_record_integrity_results)),
            "moa_deep_results": str(Path(args.moa_deep_results)),
            "moa_readiness_results": str(Path(args.moa_readiness_results)),
            "moa_effective_results": str(Path(args.moa_effective_results)),
            "moa_deleted_vs_updated_results": str(Path(args.moa_deleted_vs_updated_results)),
            "moa_deleted_vs_updated_answer_results": str(Path(args.moa_deleted_vs_updated_answer_results)),
            "base_chunk_lineage_results": str(Path(lineage_path_raw)),
            "contract_text_compare_amended_results": str(Path(compare_path_raw)),
            "api_file": str(Path(args.api_file)),
            "verifier_file": str(Path(args.verifier_file)),
            "skip_manifest_validation": bool(args.skip_manifest_validation),
            "skip_track_all_metadata": bool(args.skip_track_all_metadata),
            "min_moa_source_type_match_rate": float(args.min_moa_source_type_match_rate),
            "min_moa_deleted_vs_updated_overall_pass_rate": float(args.min_moa_deleted_vs_updated_overall_pass_rate),
            "min_moa_deleted_vs_updated_updated_pass_rate": float(args.min_moa_deleted_vs_updated_updated_pass_rate),
            "min_moa_deleted_vs_updated_deleted_pass_rate": float(args.min_moa_deleted_vs_updated_deleted_pass_rate),
            "min_moa_deleted_vs_updated_updated_moa_source_type_match_rate": float(args.min_moa_deleted_vs_updated_updated_moa_source_type_match_rate),
            "min_moa_deleted_vs_updated_answer_overall_pass_rate": float(args.min_moa_deleted_vs_updated_answer_overall_pass_rate),
            "min_moa_deleted_vs_updated_answer_updated_pass_rate": float(args.min_moa_deleted_vs_updated_answer_updated_pass_rate),
            "min_moa_deleted_vs_updated_answer_deleted_pass_rate": float(args.min_moa_deleted_vs_updated_answer_deleted_pass_rate),
            "min_moa_deleted_vs_updated_answer_source_type_match_rate": float(args.min_moa_deleted_vs_updated_answer_source_type_match_rate),
        },
        "components": components,
        "advisories": advisories,
        "overall": {
            "components_passed": pass_count,
            "components_total": total_components,
            "pass_rate": round((pass_count / total_components) if total_components else 0.0, 4),
            "pass": overall_pass,
        },
    }


def _write_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic v0.9.0 readiness scorecard.")
    parser.add_argument(
        "--eval-runner-metadata",
        default="",
        help="Path to eval_run_metadata_all_*.json. Default: newest available.",
    )
    parser.add_argument("--v3-results", default=str(DATA_DIR / "test_set" / "v3_results.json"))
    parser.add_argument("--miss-record-integrity-results", default=str(DEFAULT_MISS_RECORD_INTEGRITY_FILE))
    parser.add_argument("--moa-deep-results", default=str(DATA_DIR / "test_set" / "moa_deep_eval_suite_results.json"))
    parser.add_argument("--moa-readiness-results", default=str(DATA_DIR / "test_set" / "moa_readiness_results.json"))
    parser.add_argument("--moa-effective-results", default=str(DATA_DIR / "test_set" / "moa_effective_results.json"))
    parser.add_argument(
        "--moa-deleted-vs-updated-results",
        default=str(DATA_DIR / "test_set" / "moa_deleted_vs_updated_results.json"),
    )
    parser.add_argument(
        "--moa-deleted-vs-updated-answer-results",
        default=str(DATA_DIR / "test_set" / "moa_deleted_vs_updated_answer_results.json"),
    )
    parser.add_argument(
        "--base-chunk-lineage-results",
        default=str(DEFAULT_BASE_CHUNK_LINEAGE_FILE),
        help="Advisory-only lineage audit artifact (non-gating).",
    )
    parser.add_argument(
        "--contract-text-compare-amended-results",
        default=str(DEFAULT_CONTRACT_TEXT_COMPARE_AMENDED_FILE),
        help="Advisory-only amended section compare eval artifact (non-gating).",
    )
    parser.add_argument("--api-file", default=str(DEFAULT_API_FILE))
    parser.add_argument("--verifier-file", default=str(DEFAULT_VERIFIER_FILE))
    parser.add_argument("--skip-manifest-validation", action="store_true")
    parser.add_argument(
        "--skip-track-all-metadata",
        action="store_true",
        help="Skip track-all metadata gate (useful for CI jobs that run slices independently).",
    )
    parser.add_argument("--min-moa-source-type-match-rate", type=float, default=0.95)
    parser.add_argument("--min-moa-deleted-vs-updated-overall-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-updated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-deleted-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-updated-moa-source-type-match-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-answer-overall-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-answer-updated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-answer-deleted-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-answer-source-type-match-rate", type=float, default=1.0)
    parser.add_argument("--output", default=str(OUT_PATH))
    args = parser.parse_args()

    report = run(args)
    out_path = _write_report(report, Path(args.output))
    overall = report.get("overall") or {}

    print("=" * 72)
    print("KARL v0.9.0 Readiness Scorecard")
    print("=" * 72)
    print(
        f"Components: {overall.get('components_passed', 0)}/"
        f"{overall.get('components_total', 0)} ({overall.get('pass_rate', 0.0):.1%})"
    )
    print(f"Pass: {overall.get('pass')}")
    print(f"Results: {out_path}")
    return 0 if overall.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
