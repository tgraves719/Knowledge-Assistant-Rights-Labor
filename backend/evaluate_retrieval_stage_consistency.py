"""
Canonical evaluator for router-owned retrieval planning and execution consistency.

This track verifies two scale-oriented properties across multiple contracts:
- follow-up routing plans preserve prior topic/article context deterministically
- retrieval plans and executed retrieval stages stay aligned contract-by-contract
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
import backend.retrieval.router as router_module
from backend.retrieval.router import (
    FollowupRoutingPlan,
    HybridRetriever,
    RetrievalPlanRecord,
    RetrievalPolicyRecord,
    build_followup_routing_plan,
    classify_intent,
)


_PLAN_FLAG_TO_STAGE = {
    "apply_topic_seed_coverage": "topic_seed_coverage",
    "apply_article_prioritization": "article_prioritization",
    "apply_side_letter_promotion": "side_letter_promotion",
    "apply_full_article_expansion": "full_article_expansion",
    "apply_related_section_expansion": "related_section_expansion",
    "apply_vacation_entitlement_coverage": "vacation_entitlement_coverage",
    "apply_holiday_premium_coverage": "holiday_premium_coverage",
}


def _rate(num: int, den: int) -> float:
    return round((num / den) if den else 0.0, 4)


def _load_payload(test_file: Path) -> dict:
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _serialize_followup(plan: FollowupRoutingPlan) -> dict:
    return plan.to_dict()


def _serialize_plan(plan: RetrievalPlanRecord) -> dict:
    return plan.to_dict()


def _serialize_policy(policy: RetrievalPolicyRecord) -> dict:
    return policy.to_dict()


def _evaluate_followup_case(case: dict) -> tuple[dict, dict]:
    case_id = str(case.get("id") or "")
    question = str(case.get("question") or "")
    plan = build_followup_routing_plan(
        question=question,
        prior_topic=str(case.get("prior_topic") or ""),
        prior_citations=[str(v) for v in list(case.get("prior_citations") or [])],
        prior_article_anchors=[int(v) for v in list(case.get("prior_article_anchors") or [])],
    )
    typed_plan = FollowupRoutingPlan.from_dict(plan.to_dict())

    expected_strategy = str(case.get("expected_followup_strategy") or "").strip()
    expected_context = bool(case.get("expected_followup_context_used"))
    expected_anchor_count = int(case.get("expected_followup_anchor_count") or 0)
    expected_tokens = [str(v).strip().lower() for v in list(case.get("expected_routing_query_tokens") or []) if str(v).strip()]

    strategy_ok = (not expected_strategy) or typed_plan.strategy == expected_strategy
    context_ok = typed_plan.followup_context_used is expected_context
    anchor_count_ok = len(list(typed_plan.article_anchors or [])) >= expected_anchor_count
    routing_query = str(typed_plan.routing_query or "").strip().lower()
    token_checks = {token: token in routing_query for token in expected_tokens}
    routing_query_ok = all(token_checks.values()) if token_checks else True
    passed = all([strategy_ok, context_ok, anchor_count_ok, routing_query_ok])

    row = {
        "id": case_id,
        "mode": "followup_plan",
        "contract_id": str(case.get("contract_id") or ""),
        "question": question,
        "expected_followup_strategy": expected_strategy,
        "resolved_followup_strategy": typed_plan.strategy,
        "followup_context_used": typed_plan.followup_context_used,
        "article_anchor_count": len(list(typed_plan.article_anchors or [])),
        "strategy_ok": strategy_ok,
        "context_ok": context_ok,
        "anchor_count_ok": anchor_count_ok,
        "routing_query_ok": routing_query_ok,
        "routing_query_token_checks": token_checks,
        "followup_plan": _serialize_followup(typed_plan),
        "pass": passed,
    }
    metrics = {
        "strategy_total": 1 if expected_strategy else 0,
        "strategy_passed": 1 if strategy_ok and expected_strategy else 0,
        "followup_total": 1,
        "followup_passed": 1 if passed else 0,
    }
    return row, metrics


def _evaluate_retrieve_case(case: dict, retriever: HybridRetriever) -> tuple[dict, dict]:
    case_id = str(case.get("id") or "")
    contract_id = str(case.get("contract_id") or "")
    question = str(case.get("question") or "")
    intent = classify_intent(question, contract_id=contract_id)
    result = retriever.retrieve(
        query=question,
        intent=intent,
        n_results=8,
        use_hybrid=True,
        contract_id=contract_id,
    )

    plan = RetrievalPlanRecord.from_dict(result.get("retrieval_plan"))
    policy = RetrievalPolicyRecord.from_dict(result.get("retrieval_policy"))

    expected_planned_strategy = str(case.get("expected_planned_strategy") or "").strip()
    expected_policy_strategy = str(case.get("expected_policy_strategy") or "").strip()
    required_plan_flags = [str(v).strip() for v in list(case.get("required_plan_flags") or []) if str(v).strip()]
    required_executed_stages = [str(v).strip() for v in list(case.get("required_executed_stages") or []) if str(v).strip()]

    planned_strategy_ok = (not expected_planned_strategy) or plan.planned_strategy == expected_planned_strategy
    policy_strategy_ok = (not expected_policy_strategy) or policy.strategy == expected_policy_strategy

    plan_flag_checks = {flag: bool(getattr(plan, flag, False)) for flag in required_plan_flags}
    executed_stages = set(policy.executed_stages or [])
    executed_stage_checks = {stage: stage in executed_stages for stage in required_executed_stages}

    expected_from_plan = {
        stage_name
        for flag_name, stage_name in _PLAN_FLAG_TO_STAGE.items()
        if bool(getattr(plan, flag_name, False))
    }
    alignment_missing = sorted(stage for stage in expected_from_plan if stage not in executed_stages)
    alignment_ok = not alignment_missing

    passed = all(
        [
            planned_strategy_ok,
            policy_strategy_ok,
            all(plan_flag_checks.values()) if plan_flag_checks else True,
            all(executed_stage_checks.values()) if executed_stage_checks else True,
            alignment_ok,
        ]
    )

    row = {
        "id": case_id,
        "mode": "retrieve",
        "contract_id": contract_id,
        "question": question,
        "intent_type": str(getattr(intent, "intent_type", "") or ""),
        "intent_topic": str(getattr(intent, "topic", "") or ""),
        "expected_planned_strategy": expected_planned_strategy,
        "resolved_planned_strategy": plan.planned_strategy,
        "expected_policy_strategy": expected_policy_strategy,
        "resolved_policy_strategy": policy.strategy,
        "planned_strategy_ok": planned_strategy_ok,
        "policy_strategy_ok": policy_strategy_ok,
        "required_plan_flags": required_plan_flags,
        "required_plan_flag_checks": plan_flag_checks,
        "required_executed_stages": required_executed_stages,
        "required_executed_stage_checks": executed_stage_checks,
        "plan_enabled_stage_expectations": sorted(expected_from_plan),
        "alignment_missing_stages": alignment_missing,
        "alignment_ok": alignment_ok,
        "retrieval_plan": _serialize_plan(plan),
        "retrieval_policy": _serialize_policy(policy),
        "pass": passed,
    }
    metrics = {
        "planned_strategy_total": 1 if expected_planned_strategy else 0,
        "planned_strategy_passed": 1 if planned_strategy_ok and expected_planned_strategy else 0,
        "policy_strategy_total": 1 if expected_policy_strategy else 0,
        "policy_strategy_passed": 1 if policy_strategy_ok and expected_policy_strategy else 0,
        "required_plan_flag_total": len(plan_flag_checks),
        "required_plan_flag_passed": sum(1 for ok in plan_flag_checks.values() if ok),
        "required_executed_stage_total": len(executed_stage_checks),
        "required_executed_stage_passed": sum(1 for ok in executed_stage_checks.values() if ok),
        "alignment_total": 1,
        "alignment_passed": 1 if alignment_ok else 0,
        "retrieve_total": 1,
        "retrieve_passed": 1 if passed else 0,
    }
    return row, metrics


def run(test_file: Optional[Path] = None, bm25_only: bool = False) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "retrieval_stage_consistency_test.json"
    payload = _load_payload(test_file)
    test_cases = list(payload.get("test_cases") or [])

    original_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    original_reranker = router_module.CAG_ENABLE_RERANKER
    try:
        router_module.CAG_ENABLE_RERANKER = False
        if bm25_only:
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)
        rows: list[dict] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})

        metric_totals = defaultdict(int)

        for case in test_cases:
            mode = str(case.get("mode") or "").strip().lower()
            contract_id = str(case.get("contract_id") or "")
            if mode == "followup_plan":
                row, metrics = _evaluate_followup_case(case)
            elif mode == "retrieve":
                row, metrics = _evaluate_retrieve_case(case, retriever)
            else:
                row = {
                    "id": str(case.get("id") or ""),
                    "mode": mode or None,
                    "contract_id": contract_id,
                    "question": str(case.get("question") or ""),
                    "pass": False,
                    "error": f"Unsupported mode: {mode}",
                }
                metrics = {}

            by_contract[contract_id]["total"] += 1
            if row.get("pass"):
                by_contract[contract_id]["passed"] += 1

            for key, value in metrics.items():
                metric_totals[key] += int(value)

            rows.append(row)

        total = len(rows)
        passed = sum(1 for row in rows if row.get("pass"))
        by_contract_summary = {}
        for contract_id, stats in sorted(by_contract.items()):
            c_total = int(stats["total"])
            c_passed = int(stats["passed"])
            by_contract_summary[contract_id] = {
                "passed": c_passed,
                "total": c_total,
                "pass_rate": _rate(c_passed, c_total),
            }

        return {
            "schema_version": "retrieval_stage_consistency_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "followup_plan_total": int(metric_totals["followup_total"]),
                "followup_plan_pass_rate": _rate(metric_totals["followup_passed"], metric_totals["followup_total"]),
                "planned_strategy_match_rate": _rate(metric_totals["planned_strategy_passed"], metric_totals["planned_strategy_total"]),
                "policy_strategy_match_rate": _rate(metric_totals["policy_strategy_passed"], metric_totals["policy_strategy_total"]),
                "followup_strategy_match_rate": _rate(metric_totals["strategy_passed"], metric_totals["strategy_total"]),
                "required_plan_flag_rate": _rate(metric_totals["required_plan_flag_passed"], metric_totals["required_plan_flag_total"]),
                "required_executed_stage_rate": _rate(metric_totals["required_executed_stage_passed"], metric_totals["required_executed_stage_total"]),
                "plan_to_execution_alignment_rate": _rate(metric_totals["alignment_passed"], metric_totals["alignment_total"]),
            },
            "by_contract": by_contract_summary,
            "results": rows,
        }
    finally:
        router_module.HYBRID_VECTOR_WEIGHT = original_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.CAG_ENABLE_RERANKER = original_reranker


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "retrieval_stage_consistency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval planning and executed stage consistency.")
    parser.add_argument("--input", type=str, default=None, help="Path to retrieval stage consistency test JSON")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, bm25_only=args.bm25_only)
    out_path = _write_report(report)
    overall = dict(report.get("overall") or {})

    print("=" * 72)
    print("KARL Retrieval Stage Consistency Evaluation")
    print("=" * 72)
    print(
        f"Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} "
        f"({overall.get('pass_rate', 0.0):.1%})"
    )
    print(f"Planned strategy match rate: {overall.get('planned_strategy_match_rate', 0.0):.1%}")
    print(f"Policy strategy match rate: {overall.get('policy_strategy_match_rate', 0.0):.1%}")
    print(f"Follow-up strategy match rate: {overall.get('followup_strategy_match_rate', 0.0):.1%}")
    print(f"Required plan flag rate: {overall.get('required_plan_flag_rate', 0.0):.1%}")
    print(f"Required executed stage rate: {overall.get('required_executed_stage_rate', 0.0):.1%}")
    print(f"Plan-to-execution alignment rate: {overall.get('plan_to_execution_alignment_rate', 0.0):.1%}")
    print(f"Results: {out_path}")
    return 0 if int(overall.get("passed", 0)) == int(overall.get("total", 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
