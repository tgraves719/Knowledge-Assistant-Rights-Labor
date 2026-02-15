"""
Canonical v3 evaluation suite (multi-contract phase).

Runs/validates deterministic multi-contract integrity slices and writes a
single v3 artifact for release-gate consumption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.contracts import list_contract_catalog


OUT_PATH = DATA_DIR / "test_set" / "v3_results.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _active_contract_ids() -> list[str]:
    return sorted(str(c.get("contract_id")) for c in list_contract_catalog() if c.get("contract_id"))


def _run_cmd(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    return {
        "command": " ".join(cmd),
        "return_code": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _check_multi_contract(results: dict | None, min_overall: float, min_per_contract: float) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = float((results.get("overall") or {}).get("pass_rate") or 0.0)
    by_contract = results.get("by_contract") or {}
    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        ok = rate >= min_per_contract
        per_contract[cid] = {"pass_rate": rate, "pass": ok}
        if not ok:
            per_contract_ok = False
    ok = overall >= min_overall and bool(by_contract) and per_contract_ok
    return ok, {
        "overall_pass_rate": overall,
        "min_overall": min_overall,
        "min_per_contract": min_per_contract,
        "per_contract": per_contract,
    }


def _check_cross_contamination(results: dict | None) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    skipped = bool(results.get("skipped"))
    failures_count = int(results.get("failures_count") or 0)
    passed = bool(results.get("pass"))
    ok = (not skipped) and passed and failures_count == 0
    return ok, {
        "skipped": skipped,
        "pass": passed,
        "failures_count": failures_count,
    }


def _check_paraphrase(
    results: dict | None,
    min_family: float,
    min_worker_slang: float,
    min_formal: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    family = float(overall.get("family_pass_rate") or 0.0)
    slang = float(overall.get("worker_slang_pass_rate") or 0.0)
    formal = float(overall.get("formal_rewrite_pass_rate") or 0.0)
    ok = family >= min_family and slang >= min_worker_slang and formal >= min_formal
    return ok, {
        "family_pass_rate": family,
        "worker_slang_pass_rate": slang,
        "formal_rewrite_pass_rate": formal,
        "thresholds": {
            "family": min_family,
            "worker_slang": min_worker_slang,
            "formal_rewrite": min_formal,
        },
    }


def _check_adversarial(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
    min_precedence_rate: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    precedence_rate_raw = overall.get("precedence_pass_rate")
    precedence_rate = float(precedence_rate_raw) if precedence_rate_raw is not None else 0.0
    precedence_total = int(overall.get("precedence_total") or 0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and precedence_total > 0
        and precedence_rate >= min_precedence_rate
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "precedence_pass_rate": precedence_rate,
        "precedence_total": precedence_total,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_precedence_rate": min_precedence_rate,
        },
    }


def _check_unanswerable(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    abstention_rate = float(overall.get("abstention_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and abstention_rate >= min_overall
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "abstention_rate": abstention_rate,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
        },
    }


def _check_cross_contract_mentions(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
    min_no_citation_rate: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    no_citation_rate = float(overall.get("no_citation_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and no_citation_rate >= min_no_citation_rate
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "no_citation_rate": no_citation_rate,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_no_citation_rate": min_no_citation_rate,
        },
    }


def _check_false_unavailable(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_recover_cases: int,
    min_uncertain_cases: int,
    min_overall: float,
    min_per_contract: float,
    min_recovered: float,
    min_uncertainty: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    recover_total = int(overall.get("recover_total") or 0)
    uncertain_total = int(overall.get("uncertain_total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    recovered_rate = float(overall.get("false_unavailable_recovered_rate") or 0.0)
    uncertainty_rate = float(overall.get("proper_uncertainty_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        ok = rate >= min_per_contract
        per_contract[cid] = {"pass_rate": rate, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and recover_total >= min_recover_cases
        and uncertain_total >= min_uncertain_cases
        and overall_rate >= min_overall
        and recovered_rate >= min_recovered
        and uncertainty_rate >= min_uncertainty
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "recovered_rate": recovered_rate,
        "proper_uncertainty_rate": uncertainty_rate,
        "total_cases": total,
        "recover_cases": recover_total,
        "uncertain_cases": uncertain_total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_recover_cases": min_recover_cases,
            "min_uncertain_cases": min_uncertain_cases,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_recovered": min_recovered,
            "min_uncertainty": min_uncertainty,
        },
    }


def _check_needle(results: dict | None, min_pass_rate: float, min_position_rate: float) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_position = results.get("by_position") or {}
    overall_rate = float(overall.get("pass_rate") or 0.0)
    position_rates = {}
    pos_ok = True
    for pos in ("top", "middle", "bottom"):
        rate = float(((by_position.get(pos) or {}).get("pass_rate")) or 0.0)
        ok = rate >= min_position_rate
        position_rates[pos] = {"pass_rate": rate, "pass": ok}
        if not ok:
            pos_ok = False
    ok = overall_rate >= min_pass_rate and pos_ok
    return ok, {
        "overall_pass_rate": overall_rate,
        "positions": position_rates,
        "thresholds": {"overall": min_pass_rate, "position": min_position_rate},
    }


def _check_wage_table_evidence(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
    min_source_method_pass_rate: float,
    min_table_evidence_presence_rate: float,
    min_table_id_presence_rate: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    source_method_rate = float(overall.get("source_method_pass_rate") or 0.0)
    table_evidence_rate = float(overall.get("table_evidence_presence_rate") or 0.0)
    table_id_rate = float(overall.get("table_id_presence_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and source_method_rate >= min_source_method_pass_rate
        and table_evidence_rate >= min_table_evidence_presence_rate
        and table_id_rate >= min_table_id_presence_rate
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "source_method_pass_rate": source_method_rate,
        "table_evidence_presence_rate": table_evidence_rate,
        "table_id_presence_rate": table_id_rate,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_source_method_pass_rate": min_source_method_pass_rate,
            "min_table_evidence_presence_rate": min_table_evidence_presence_rate,
            "min_table_id_presence_rate": min_table_id_presence_rate,
        },
    }


def _check_entitlement_table_evidence(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
    min_weeks_resolution_pass_rate: float,
    min_source_method_pass_rate: float,
    min_evidence_presence_rate: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")
    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    weeks_rate = float(overall.get("weeks_resolution_pass_rate") or 0.0)
    source_method_rate = float(overall.get("source_method_pass_rate") or 0.0)
    evidence_rate = float(overall.get("evidence_presence_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and weeks_rate >= min_weeks_resolution_pass_rate
        and source_method_rate >= min_source_method_pass_rate
        and evidence_rate >= min_evidence_presence_rate
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "weeks_resolution_pass_rate": weeks_rate,
        "source_method_pass_rate": source_method_rate,
        "evidence_presence_rate": evidence_rate,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_weeks_resolution_pass_rate": min_weeks_resolution_pass_rate,
            "min_source_method_pass_rate": min_source_method_pass_rate,
            "min_evidence_presence_rate": min_evidence_presence_rate,
        },
    }


def _check_role_catalog_integrity(
    results: dict | None,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_overall: float,
    min_per_contract: float,
    min_dataset_case_pass_rate: float,
    min_default_wage_ready_rate: float,
    min_unresolved_not_default_rate: float,
    min_default_wage_key_unique_rate: float,
) -> tuple[bool, dict]:
    if not results:
        return False, {"reason": "artifact missing"}
    overall = results.get("overall") or {}
    by_contract = results.get("by_contract") or {}
    dataset_schema = str(results.get("dataset_schema_version") or "")

    total = int(overall.get("total") or 0)
    overall_rate = float(overall.get("pass_rate") or 0.0)
    dataset_case_rate = float(overall.get("dataset_case_pass_rate") or 0.0)
    default_ready_rate = float(overall.get("default_wage_ready_rate") or 0.0)
    unresolved_not_default_rate = float(overall.get("unresolved_not_default_rate") or 0.0)
    unique_rate = float(overall.get("default_wage_key_unique_rate") or 0.0)

    per_contract = {}
    per_contract_ok = True
    for cid, stats in sorted(by_contract.items()):
        rate = float((stats or {}).get("pass_rate") or 0.0)
        c_total = int((stats or {}).get("total") or 0)
        ok = rate >= min_per_contract and c_total >= min_cases_per_contract
        per_contract[cid] = {"pass_rate": rate, "total": c_total, "pass": ok}
        if not ok:
            per_contract_ok = False

    ok = (
        dataset_schema == required_dataset_schema_version
        and total >= min_total_cases
        and overall_rate >= min_overall
        and dataset_case_rate >= min_dataset_case_pass_rate
        and default_ready_rate >= min_default_wage_ready_rate
        and unresolved_not_default_rate >= min_unresolved_not_default_rate
        and unique_rate >= min_default_wage_key_unique_rate
        and bool(by_contract)
        and per_contract_ok
    )
    return ok, {
        "dataset_schema_version": dataset_schema,
        "overall_pass_rate": overall_rate,
        "dataset_case_pass_rate": dataset_case_rate,
        "default_wage_ready_rate": default_ready_rate,
        "unresolved_not_default_rate": unresolved_not_default_rate,
        "default_wage_key_unique_rate": unique_rate,
        "total_cases": total,
        "per_contract": per_contract,
        "thresholds": {
            "required_dataset_schema_version": required_dataset_schema_version,
            "min_total_cases": min_total_cases,
            "min_cases_per_contract": min_cases_per_contract,
            "min_overall": min_overall,
            "min_per_contract": min_per_contract,
            "min_dataset_case_pass_rate": min_dataset_case_pass_rate,
            "min_default_wage_ready_rate": min_default_wage_ready_rate,
            "min_unresolved_not_default_rate": min_unresolved_not_default_rate,
            "min_default_wage_key_unique_rate": min_default_wage_key_unique_rate,
        },
    }


def _evaluate_from_artifacts(args) -> dict:
    components = {}
    pass_count = 0

    fu_dataset_result = _run_cmd([sys.executable, "-m", "backend.validate_false_unavailable_dataset"])
    fu_dataset_ok = fu_dataset_result["return_code"] == 0
    components["false_unavailable_dataset_validation"] = {"pass": fu_dataset_ok, "details": fu_dataset_result}
    pass_count += int(fu_dataset_ok)

    adv_dataset_result = _run_cmd([sys.executable, "-m", "backend.validate_adversarial_dataset"])
    adv_dataset_ok = adv_dataset_result["return_code"] == 0
    components["adversarial_dataset_validation"] = {"pass": adv_dataset_ok, "details": adv_dataset_result}
    pass_count += int(adv_dataset_ok)

    un_dataset_result = _run_cmd([sys.executable, "-m", "backend.validate_unanswerable_dataset"])
    un_dataset_ok = un_dataset_result["return_code"] == 0
    components["unanswerable_dataset_validation"] = {"pass": un_dataset_ok, "details": un_dataset_result}
    pass_count += int(un_dataset_ok)

    ccm_dataset_result = _run_cmd([sys.executable, "-m", "backend.validate_cross_contract_mentions_dataset"])
    ccm_dataset_ok = ccm_dataset_result["return_code"] == 0
    components["cross_contract_mentions_dataset_validation"] = {"pass": ccm_dataset_ok, "details": ccm_dataset_result}
    pass_count += int(ccm_dataset_ok)

    cross_ok, cross_details = _check_cross_contamination(
        _load_json(DATA_DIR / "test_set" / "cross_contamination_results.json")
    )
    components["cross_contamination"] = {"pass": cross_ok, "details": cross_details}
    pass_count += int(cross_ok)

    multi_ok, multi_details = _check_multi_contract(
        _load_json(DATA_DIR / "test_set" / "multi_contract_v2_results.json"),
        min_overall=args.min_multi_contract_accuracy,
        min_per_contract=args.min_multi_contract_per_contract,
    )
    components["multi_contract"] = {"pass": multi_ok, "details": multi_details}
    pass_count += int(multi_ok)

    para_ok, para_details = _check_paraphrase(
        _load_json(DATA_DIR / "test_set" / "paraphrase_results.json"),
        min_family=args.min_paraphrase_family_pass_rate,
        min_worker_slang=args.min_paraphrase_worker_slang_pass_rate,
        min_formal=args.min_paraphrase_formal_rewrite_pass_rate,
    )
    components["paraphrase"] = {"pass": para_ok, "details": para_details}
    pass_count += int(para_ok)

    adv_ok, adv_details = _check_adversarial(
        _load_json(DATA_DIR / "test_set" / "adversarial_results.json"),
        required_dataset_schema_version=args.required_adversarial_dataset_schema_version,
        min_total_cases=args.min_adversarial_total_cases,
        min_cases_per_contract=args.min_adversarial_cases_per_contract,
        min_overall=args.min_adversarial_pass_rate,
        min_per_contract=args.min_adversarial_per_contract,
        min_precedence_rate=args.min_adversarial_precedence_pass_rate,
    )
    components["adversarial_precedence"] = {"pass": adv_ok, "details": adv_details}
    pass_count += int(adv_ok)

    un_ok, un_details = _check_unanswerable(
        _load_json(DATA_DIR / "test_set" / "unanswerable_results.json"),
        required_dataset_schema_version=args.required_unanswerable_dataset_schema_version,
        min_total_cases=args.min_unanswerable_total_cases,
        min_cases_per_contract=args.min_unanswerable_cases_per_contract,
        min_overall=args.min_unanswerable_pass_rate,
        min_per_contract=args.min_unanswerable_per_contract,
    )
    components["unanswerable"] = {"pass": un_ok, "details": un_details}
    pass_count += int(un_ok)

    ccm_ok, ccm_details = _check_cross_contract_mentions(
        _load_json(DATA_DIR / "test_set" / "cross_contract_mentions_results.json"),
        required_dataset_schema_version=args.required_cross_contract_mentions_dataset_schema_version,
        min_total_cases=args.min_cross_contract_mentions_total_cases,
        min_cases_per_contract=args.min_cross_contract_mentions_cases_per_contract,
        min_overall=args.min_cross_contract_mentions_pass_rate,
        min_per_contract=args.min_cross_contract_mentions_per_contract,
        min_no_citation_rate=args.min_cross_contract_mentions_no_citation_rate,
    )
    components["cross_contract_mentions"] = {"pass": ccm_ok, "details": ccm_details}
    pass_count += int(ccm_ok)

    fu_ok, fu_details = _check_false_unavailable(
        _load_json(DATA_DIR / "test_set" / "false_unavailable_results.json"),
        required_dataset_schema_version=args.required_false_unavailable_dataset_schema_version,
        min_total_cases=args.min_false_unavailable_total_cases,
        min_recover_cases=args.min_false_unavailable_recover_cases,
        min_uncertain_cases=args.min_false_unavailable_uncertain_cases,
        min_overall=args.min_false_unavailable_pass_rate,
        min_per_contract=args.min_false_unavailable_per_contract,
        min_recovered=args.min_false_unavailable_recovered_rate,
        min_uncertainty=args.min_false_unavailable_proper_uncertainty_rate,
    )
    components["false_unavailable"] = {"pass": fu_ok, "details": fu_details}
    pass_count += int(fu_ok)

    needle_ok, needle_details = _check_needle(
        _load_json(DATA_DIR / "test_set" / "needle_results.json"),
        min_pass_rate=args.min_needle_pass_rate,
        min_position_rate=args.min_needle_position_pass_rate,
    )
    components["needle"] = {"pass": needle_ok, "details": needle_details}
    pass_count += int(needle_ok)

    wage_table_ok, wage_table_details = _check_wage_table_evidence(
        _load_json(DATA_DIR / "test_set" / "wage_table_evidence_results.json"),
        required_dataset_schema_version=args.required_wage_table_evidence_dataset_schema_version,
        min_total_cases=args.min_wage_table_evidence_total_cases,
        min_cases_per_contract=args.min_wage_table_evidence_cases_per_contract,
        min_overall=args.min_wage_table_evidence_pass_rate,
        min_per_contract=args.min_wage_table_evidence_per_contract,
        min_source_method_pass_rate=args.min_wage_table_evidence_source_method_pass_rate,
        min_table_evidence_presence_rate=args.min_wage_table_evidence_presence_rate,
        min_table_id_presence_rate=args.min_wage_table_evidence_table_id_presence_rate,
    )
    components["wage_table_evidence"] = {"pass": wage_table_ok, "details": wage_table_details}
    pass_count += int(wage_table_ok)

    entitlement_ok, entitlement_details = _check_entitlement_table_evidence(
        _load_json(DATA_DIR / "test_set" / "entitlement_table_evidence_results.json"),
        required_dataset_schema_version=args.required_entitlement_table_evidence_dataset_schema_version,
        min_total_cases=args.min_entitlement_table_evidence_total_cases,
        min_cases_per_contract=args.min_entitlement_table_evidence_cases_per_contract,
        min_overall=args.min_entitlement_table_evidence_pass_rate,
        min_per_contract=args.min_entitlement_table_evidence_per_contract,
        min_weeks_resolution_pass_rate=args.min_entitlement_table_evidence_weeks_resolution_pass_rate,
        min_source_method_pass_rate=args.min_entitlement_table_evidence_source_method_pass_rate,
        min_evidence_presence_rate=args.min_entitlement_table_evidence_presence_rate,
    )
    components["entitlement_table_evidence"] = {"pass": entitlement_ok, "details": entitlement_details}
    pass_count += int(entitlement_ok)

    role_integrity_ok, role_integrity_details = _check_role_catalog_integrity(
        _load_json(DATA_DIR / "test_set" / "role_catalog_integrity_results.json"),
        required_dataset_schema_version=args.required_role_catalog_integrity_dataset_schema_version,
        min_total_cases=args.min_role_catalog_integrity_total_cases,
        min_cases_per_contract=args.min_role_catalog_integrity_cases_per_contract,
        min_overall=args.min_role_catalog_integrity_pass_rate,
        min_per_contract=args.min_role_catalog_integrity_per_contract,
        min_dataset_case_pass_rate=args.min_role_catalog_integrity_dataset_case_pass_rate,
        min_default_wage_ready_rate=args.min_role_catalog_integrity_default_wage_ready_rate,
        min_unresolved_not_default_rate=args.min_role_catalog_integrity_unresolved_not_default_rate,
        min_default_wage_key_unique_rate=args.min_role_catalog_integrity_default_wage_key_unique_rate,
    )
    components["role_catalog_integrity"] = {"pass": role_integrity_ok, "details": role_integrity_details}
    pass_count += int(role_integrity_ok)

    total_components = len(components)
    overall_pass = pass_count == total_components
    return {
        "components": components,
        "overall": {
            "components_passed": pass_count,
            "components_total": total_components,
            "pass_rate": round((pass_count / total_components) if total_components else 0.0, 4),
            "pass": overall_pass,
        },
    }


def run(args) -> dict:
    commands: list[dict] = []
    if not args.from_artifacts:
        run_list = [
            [sys.executable, "-m", "backend.evaluate_multi_contract", "--bm25-only"],
            [sys.executable, "-m", "backend.evaluate_cross_contamination", "--require-multi-contract"],
            [sys.executable, "-m", "backend.evaluate_paraphrase", "--bm25-only"],
            [sys.executable, "-m", "backend.validate_adversarial_dataset"],
            [sys.executable, "-m", "backend.evaluate_adversarial_precedence", "--bm25-only"],
            [sys.executable, "-m", "backend.validate_unanswerable_dataset"],
            [sys.executable, "-m", "backend.evaluate_unanswerable", "--bm25-only"],
            [sys.executable, "-m", "backend.validate_cross_contract_mentions_dataset"],
            [sys.executable, "-m", "backend.evaluate_cross_contract_mentions", "--bm25-only"],
            [sys.executable, "-m", "backend.validate_false_unavailable_dataset"],
            [sys.executable, "-m", "backend.evaluate_false_unavailable", "--bm25-only"],
            [sys.executable, "-m", "backend.evaluate_needle", "--bm25-only"],
            [sys.executable, "-m", "backend.evaluate_wage_table_evidence", "--bm25-only"],
            [sys.executable, "-m", "backend.evaluate_entitlement_table_evidence"],
            [sys.executable, "-m", "backend.evaluate_role_catalog_integrity"],
        ]
        for cmd in run_list:
            result = _run_cmd(cmd)
            commands.append(result)
            if result["return_code"] != 0:
                break

    artifact_eval = _evaluate_from_artifacts(args)
    report = {
        "schema_version": "v3_eval_v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "from_artifacts" if args.from_artifacts else "full",
        "active_contract_ids": _active_contract_ids(),
        "commands": commands,
        "thresholds": {
            "min_multi_contract_accuracy": args.min_multi_contract_accuracy,
            "min_multi_contract_per_contract": args.min_multi_contract_per_contract,
            "min_paraphrase_family_pass_rate": args.min_paraphrase_family_pass_rate,
            "min_paraphrase_worker_slang_pass_rate": args.min_paraphrase_worker_slang_pass_rate,
            "min_paraphrase_formal_rewrite_pass_rate": args.min_paraphrase_formal_rewrite_pass_rate,
            "required_adversarial_dataset_schema_version": args.required_adversarial_dataset_schema_version,
            "min_adversarial_total_cases": args.min_adversarial_total_cases,
            "min_adversarial_cases_per_contract": args.min_adversarial_cases_per_contract,
            "min_adversarial_pass_rate": args.min_adversarial_pass_rate,
            "min_adversarial_per_contract": args.min_adversarial_per_contract,
            "min_adversarial_precedence_pass_rate": args.min_adversarial_precedence_pass_rate,
            "required_unanswerable_dataset_schema_version": args.required_unanswerable_dataset_schema_version,
            "min_unanswerable_total_cases": args.min_unanswerable_total_cases,
            "min_unanswerable_cases_per_contract": args.min_unanswerable_cases_per_contract,
            "min_unanswerable_pass_rate": args.min_unanswerable_pass_rate,
            "min_unanswerable_per_contract": args.min_unanswerable_per_contract,
            "required_cross_contract_mentions_dataset_schema_version": args.required_cross_contract_mentions_dataset_schema_version,
            "min_cross_contract_mentions_total_cases": args.min_cross_contract_mentions_total_cases,
            "min_cross_contract_mentions_cases_per_contract": args.min_cross_contract_mentions_cases_per_contract,
            "min_cross_contract_mentions_pass_rate": args.min_cross_contract_mentions_pass_rate,
            "min_cross_contract_mentions_per_contract": args.min_cross_contract_mentions_per_contract,
            "min_cross_contract_mentions_no_citation_rate": args.min_cross_contract_mentions_no_citation_rate,
            "required_false_unavailable_dataset_schema_version": args.required_false_unavailable_dataset_schema_version,
            "min_false_unavailable_total_cases": args.min_false_unavailable_total_cases,
            "min_false_unavailable_recover_cases": args.min_false_unavailable_recover_cases,
            "min_false_unavailable_uncertain_cases": args.min_false_unavailable_uncertain_cases,
            "min_false_unavailable_pass_rate": args.min_false_unavailable_pass_rate,
            "min_false_unavailable_per_contract": args.min_false_unavailable_per_contract,
            "min_false_unavailable_recovered_rate": args.min_false_unavailable_recovered_rate,
            "min_false_unavailable_proper_uncertainty_rate": args.min_false_unavailable_proper_uncertainty_rate,
            "min_needle_pass_rate": args.min_needle_pass_rate,
            "min_needle_position_pass_rate": args.min_needle_position_pass_rate,
            "required_wage_table_evidence_dataset_schema_version": args.required_wage_table_evidence_dataset_schema_version,
            "min_wage_table_evidence_total_cases": args.min_wage_table_evidence_total_cases,
            "min_wage_table_evidence_cases_per_contract": args.min_wage_table_evidence_cases_per_contract,
            "min_wage_table_evidence_pass_rate": args.min_wage_table_evidence_pass_rate,
            "min_wage_table_evidence_per_contract": args.min_wage_table_evidence_per_contract,
            "min_wage_table_evidence_source_method_pass_rate": args.min_wage_table_evidence_source_method_pass_rate,
            "min_wage_table_evidence_presence_rate": args.min_wage_table_evidence_presence_rate,
            "min_wage_table_evidence_table_id_presence_rate": args.min_wage_table_evidence_table_id_presence_rate,
            "required_entitlement_table_evidence_dataset_schema_version": args.required_entitlement_table_evidence_dataset_schema_version,
            "min_entitlement_table_evidence_total_cases": args.min_entitlement_table_evidence_total_cases,
            "min_entitlement_table_evidence_cases_per_contract": args.min_entitlement_table_evidence_cases_per_contract,
            "min_entitlement_table_evidence_pass_rate": args.min_entitlement_table_evidence_pass_rate,
            "min_entitlement_table_evidence_per_contract": args.min_entitlement_table_evidence_per_contract,
            "min_entitlement_table_evidence_weeks_resolution_pass_rate": args.min_entitlement_table_evidence_weeks_resolution_pass_rate,
            "min_entitlement_table_evidence_source_method_pass_rate": args.min_entitlement_table_evidence_source_method_pass_rate,
            "min_entitlement_table_evidence_presence_rate": args.min_entitlement_table_evidence_presence_rate,
            "required_role_catalog_integrity_dataset_schema_version": args.required_role_catalog_integrity_dataset_schema_version,
            "min_role_catalog_integrity_total_cases": args.min_role_catalog_integrity_total_cases,
            "min_role_catalog_integrity_cases_per_contract": args.min_role_catalog_integrity_cases_per_contract,
            "min_role_catalog_integrity_pass_rate": args.min_role_catalog_integrity_pass_rate,
            "min_role_catalog_integrity_per_contract": args.min_role_catalog_integrity_per_contract,
            "min_role_catalog_integrity_dataset_case_pass_rate": args.min_role_catalog_integrity_dataset_case_pass_rate,
            "min_role_catalog_integrity_default_wage_ready_rate": args.min_role_catalog_integrity_default_wage_ready_rate,
            "min_role_catalog_integrity_unresolved_not_default_rate": args.min_role_catalog_integrity_unresolved_not_default_rate,
            "min_role_catalog_integrity_default_wage_key_unique_rate": args.min_role_catalog_integrity_default_wage_key_unique_rate,
        },
        "components": artifact_eval["components"],
        "overall": artifact_eval["overall"],
    }
    return report


def _write_report(report: dict) -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return OUT_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical v3 multi-contract evaluation suite.")
    parser.add_argument("--from-artifacts", action="store_true", help="Validate from existing artifacts only.")
    parser.add_argument("--bm25-only", action="store_true", help="Reserved for compatibility; suite is deterministic BM25 slices.")
    parser.add_argument("--min-multi-contract-accuracy", type=float, default=0.80)
    parser.add_argument("--min-multi-contract-per-contract", type=float, default=0.75)
    parser.add_argument("--min-paraphrase-family-pass-rate", type=float, default=0.85)
    parser.add_argument("--min-paraphrase-worker-slang-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-paraphrase-formal-rewrite-pass-rate", type=float, default=0.90)
    parser.add_argument("--required-adversarial-dataset-schema-version", default="adversarial_precedence_test_v1")
    parser.add_argument("--min-adversarial-total-cases", type=int, default=12)
    parser.add_argument("--min-adversarial-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-adversarial-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-adversarial-per-contract", type=float, default=0.80)
    parser.add_argument("--min-adversarial-precedence-pass-rate", type=float, default=0.90)
    parser.add_argument("--required-unanswerable-dataset-schema-version", default="unanswerable_multi_contract_test_v1")
    parser.add_argument("--min-unanswerable-total-cases", type=int, default=12)
    parser.add_argument("--min-unanswerable-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-unanswerable-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-unanswerable-per-contract", type=float, default=0.80)
    parser.add_argument("--required-cross-contract-mentions-dataset-schema-version", default="cross_contract_mentions_test_v1")
    parser.add_argument("--min-cross-contract-mentions-total-cases", type=int, default=9)
    parser.add_argument("--min-cross-contract-mentions-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-cross-contract-mentions-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-cross-contract-mentions-per-contract", type=float, default=0.80)
    parser.add_argument("--min-cross-contract-mentions-no-citation-rate", type=float, default=0.90)
    parser.add_argument("--required-false-unavailable-dataset-schema-version", default="false_unavailable_test_v1")
    parser.add_argument("--min-false-unavailable-total-cases", type=int, default=12)
    parser.add_argument("--min-false-unavailable-recover-cases", type=int, default=9)
    parser.add_argument("--min-false-unavailable-uncertain-cases", type=int, default=3)
    parser.add_argument("--min-false-unavailable-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-false-unavailable-per-contract", type=float, default=0.80)
    parser.add_argument("--min-false-unavailable-recovered-rate", type=float, default=0.90)
    parser.add_argument("--min-false-unavailable-proper-uncertainty-rate", type=float, default=0.90)
    parser.add_argument("--min-needle-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-needle-position-pass-rate", type=float, default=0.80)
    parser.add_argument("--required-wage-table-evidence-dataset-schema-version", default="wage_table_evidence_test_v1")
    parser.add_argument("--min-wage-table-evidence-total-cases", type=int, default=12)
    parser.add_argument("--min-wage-table-evidence-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-wage-table-evidence-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-wage-table-evidence-per-contract", type=float, default=0.80)
    parser.add_argument("--min-wage-table-evidence-source-method-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-wage-table-evidence-presence-rate", type=float, default=0.95)
    parser.add_argument("--min-wage-table-evidence-table-id-presence-rate", type=float, default=0.95)
    parser.add_argument("--required-entitlement-table-evidence-dataset-schema-version", default="entitlement_table_evidence_test_v1")
    parser.add_argument("--min-entitlement-table-evidence-total-cases", type=int, default=12)
    parser.add_argument("--min-entitlement-table-evidence-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-entitlement-table-evidence-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-entitlement-table-evidence-per-contract", type=float, default=0.80)
    parser.add_argument("--min-entitlement-table-evidence-weeks-resolution-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-entitlement-table-evidence-source-method-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-entitlement-table-evidence-presence-rate", type=float, default=0.95)
    parser.add_argument("--required-role-catalog-integrity-dataset-schema-version", default="role_catalog_integrity_test_v1")
    parser.add_argument("--min-role-catalog-integrity-total-cases", type=int, default=12)
    parser.add_argument("--min-role-catalog-integrity-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-role-catalog-integrity-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-role-catalog-integrity-per-contract", type=float, default=0.90)
    parser.add_argument("--min-role-catalog-integrity-dataset-case-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-role-catalog-integrity-default-wage-ready-rate", type=float, default=1.0)
    parser.add_argument("--min-role-catalog-integrity-unresolved-not-default-rate", type=float, default=1.0)
    parser.add_argument("--min-role-catalog-integrity-default-wage-key-unique-rate", type=float, default=1.0)
    args = parser.parse_args()

    report = run(args)
    out_path = _write_report(report)
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL Canonical v3 Evaluation Suite")
    print("=" * 72)
    print(f"Mode: {report.get('mode')}")
    print(
        f"Components: {overall.get('components_passed', 0)}/{overall.get('components_total', 0)} "
        f"({overall.get('pass_rate', 0.0):.1%})"
    )
    print(f"Pass: {overall.get('pass')}")
    print(f"Results: {out_path}")
    return 0 if overall.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
