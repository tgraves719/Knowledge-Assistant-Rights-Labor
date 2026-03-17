"""Regression checks for clarification-aware v3/gate follow-up wage scoring."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_gate_check as gate_check
import backend.evaluate_v3 as evaluate_v3


def _sample_followup_role_wage_results() -> dict:
    return {
        "dataset_schema_version": "followup_role_wage_test_v1",
        "overall": {
            "pass_rate": 1.0,
            "total": 14,
            "target_resolution_rate": 0.9286,
            "resolved_target_resolution_rate": 1.0,
            "table_evidence_presence_rate": 0.9286,
            "resolved_table_evidence_presence_rate": 1.0,
            "appendix_citation_rate": 0.9286,
            "resolved_appendix_citation_rate": 1.0,
            "intent_wage_rate": 1.0,
            "no_unavailable_rate": 1.0,
            "explicit_override_rate": 1.0,
            "profile_fallback_rate": 1.0,
            "role_clarification_total": 1,
            "resolved_case_total": 13,
        },
        "by_contract": {
            "local7_kingsoopers_loveland_meat_2019": {"pass_rate": 1.0, "total": 4},
            "local7_safeway_pueblo_clerks_2022": {"pass_rate": 1.0, "total": 6},
            "local7_safeway_pueblo_meat_2022": {"pass_rate": 1.0, "total": 4},
        },
    }


def _sample_retrieval_stage_consistency_results() -> dict:
    return {
        "dataset_schema_version": "retrieval_stage_consistency_test_v1",
        "overall": {
            "pass_rate": 1.0,
            "total": 5,
            "followup_plan_pass_rate": 1.0,
            "planned_strategy_match_rate": 1.0,
            "policy_strategy_match_rate": 1.0,
            "required_plan_flag_rate": 1.0,
            "required_executed_stage_rate": 1.0,
            "plan_to_execution_alignment_rate": 1.0,
        },
        "by_contract": {
            "local7_kingsoopers_loveland_meat_2019": {"pass_rate": 1.0, "total": 1},
            "local7_safeway_pueblo_clerks_2022": {"pass_rate": 1.0, "total": 2},
            "local7_safeway_pueblo_meat_2022": {"pass_rate": 1.0, "total": 2},
        },
    }


def _test_v3_followup_role_wage_uses_resolved_case_rates() -> None:
    ok, details = evaluate_v3._check_followup_role_wage(
        _sample_followup_role_wage_results(),
        required_dataset_schema_version="followup_role_wage_test_v1",
        min_total_cases=12,
        min_cases_per_contract=3,
        min_overall=0.9,
        min_per_contract=0.8,
        min_target_resolution_rate=0.95,
        min_table_evidence_presence_rate=0.95,
        min_appendix_citation_rate=0.95,
        min_intent_wage_rate=0.95,
        min_no_unavailable_rate=0.95,
        min_explicit_override_rate=0.9,
        min_profile_fallback_rate=0.9,
    )
    assert ok, f"Expected v3 follow-up wage gate to pass with resolved-case metrics. Details: {details}"
    assert float(details.get("target_resolution_rate") or 0.0) == 1.0
    assert float(details.get("table_evidence_presence_rate") or 0.0) == 1.0
    assert float(details.get("appendix_citation_rate") or 0.0) == 1.0


def _test_v3_retrieval_stage_consistency_gate_accepts_green_artifact() -> None:
    ok, details = evaluate_v3._check_retrieval_stage_consistency(
        _sample_retrieval_stage_consistency_results(),
        required_dataset_schema_version="retrieval_stage_consistency_test_v1",
        min_total_cases=5,
        min_cases_per_contract=1,
        min_overall=1.0,
        min_per_contract=1.0,
        min_followup_plan_pass_rate=1.0,
        min_planned_strategy_match_rate=1.0,
        min_policy_strategy_match_rate=1.0,
        min_required_plan_flag_rate=1.0,
        min_required_executed_stage_rate=1.0,
        min_plan_to_execution_alignment_rate=1.0,
    )
    assert ok, f"Expected retrieval stage consistency gate to pass. Details: {details}"
    assert float(details.get("planned_strategy_match_rate") or 0.0) == 1.0
    assert float(details.get("policy_strategy_match_rate") or 0.0) == 1.0
    assert float(details.get("plan_to_execution_alignment_rate") or 0.0) == 1.0


def _test_gate_check_followup_role_wage_uses_resolved_case_rates() -> None:
    checks = gate_check._check_followup_role_wage(
        _sample_followup_role_wage_results(),
        required_dataset_schema_version="followup_role_wage_test_v1",
        min_total_cases=12,
        min_cases_per_contract=3,
        min_pass_rate=0.9,
        min_per_contract_pass_rate=0.8,
        min_target_resolution_rate=0.95,
        min_table_evidence_presence_rate=0.95,
        min_appendix_citation_rate=0.95,
        min_intent_wage_rate=0.95,
        min_no_unavailable_rate=0.95,
        min_explicit_override_rate=0.9,
        min_profile_fallback_rate=0.9,
    )
    failed = [msg for ok, msg in checks if not ok]
    assert not failed, f"Expected gate-check follow-up wage checks to pass. Failures: {failed}"


def main() -> None:
    _test_v3_followup_role_wage_uses_resolved_case_rates()
    _test_v3_retrieval_stage_consistency_gate_accepts_green_artifact()
    _test_gate_check_followup_role_wage_uses_resolved_case_rates()
    print("[OK] v3 clarification-aware follow-up wage checks passed")


if __name__ == "__main__":
    main()
