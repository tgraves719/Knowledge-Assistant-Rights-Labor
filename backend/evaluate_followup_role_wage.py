"""
Follow-up role-targeted wage evaluator.

Runs deterministic end-to-end checks through /api/query to verify that wage
follow-ups honor explicit role mentions in the query while preserving profile
fallback when no explicit target role is present.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.api as api_module
import backend.config as runtime_config
from backend.api import QueryRequest, query_contract
from backend.config import DATA_DIR
import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever


def _load_manifest(contract_id: str) -> dict:
    path = DATA_DIR / "manifests" / f"{contract_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _run_async(payload: dict, bm25_only: bool) -> dict:
    test_cases = list(payload.get("test_cases") or [])

    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_keyword = runtime_config.HYBRID_KEYWORD_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_router_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    original_reranker = runtime_config.CAG_ENABLE_RERANKER
    original_router_reranker = router_module.CAG_ENABLE_RERANKER
    original_retriever = api_module.retriever
    original_generate = api_module.generate_response
    try:
        runtime_config.CAG_ENABLE_RERANKER = False
        router_module.CAG_ENABLE_RERANKER = False
        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        api_module.retriever = HybridRetriever(vector_store=None)

        async def _deterministic_generate(*_args, **_kwargs) -> str:
            return "I will use the contract wage table evidence for this question."

        api_module.generate_response = _deterministic_generate

        rows: list[dict] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        by_expectation = defaultdict(lambda: {"passed": 0, "total": 0})

        target_resolution_passes = 0
        table_evidence_passes = 0
        appendix_citation_passes = 0
        intent_wage_passes = 0
        not_unavailable_passes = 0
        clarification_passes = 0
        clarification_total = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = str(case.get("contract_id") or "")
            question = str(case.get("question") or "")
            user_classification = str(case.get("user_classification") or "").strip() or None
            expected_target = str(case.get("expected_target_classification") or "").strip().lower()
            expectation = str(case.get("expectation") or "explicit_role_override").strip().lower()
            min_rows = int(case.get("min_table_evidence_rows") or 1)
            expected_role_family = str(case.get("expected_role_family") or "").strip().lower()
            expected_clarification_values = [
                str(v or "").strip().lower()
                for v in (case.get("expected_clarification_values") or [])
                if str(v or "").strip()
            ]

            manifest = _load_manifest(contract_id)
            contract_version = str(
                manifest.get("contract_version")
                or f"{manifest.get('term_start', '')}__{manifest.get('term_end', '')}"
            )
            union_local_id = str(manifest.get("union_local") or "")

            request = QueryRequest(
                question=question,
                union_local_id=union_local_id,
                contract_id=contract_id,
                contract_version=contract_version,
                user_classification=user_classification,
                hours_worked=0,
                months_employed=0,
                session_id=None,
            )
            response = await query_contract(request)
            wage_info = dict(response.wage_info or {})
            citations = list(response.citations or [])
            role_clarification = dict(response.role_clarification or {})

            intent_is_wage = str(response.intent_type or "").strip().lower() == "wage"
            target_key = str(wage_info.get("classification_key") or "").strip().lower()
            target_ok = bool(target_key and expected_target and target_key == expected_target)
            table_count = len(list(wage_info.get("table_evidence") or []))
            table_ok = table_count >= min_rows
            appendix_ok = any("appendix a" in str(c or "").lower() for c in citations)
            not_unavailable = not api_module._is_unavailable_answer(response.answer)
            escalation_ok = not bool(response.escalation_required)
            clarification_values = {
                str(opt.get("value") or "").strip().lower()
                for opt in (role_clarification.get("options") or [])
                if str(opt.get("value") or "").strip()
            }
            clarification_ok = (
                bool(role_clarification.get("needs_clarification"))
                and (not expected_role_family or str(role_clarification.get("role_family") or "").strip().lower() == expected_role_family)
                and (
                    not expected_clarification_values
                    or set(expected_clarification_values).issubset(clarification_values)
                )
                and not wage_info
                and not citations
            )

            if expectation == "needs_role_clarification":
                clarification_total += 1
                clarification_passes += int(clarification_ok)
                passed = all([
                    intent_is_wage,
                    clarification_ok,
                    not_unavailable,
                    escalation_ok,
                ])
            else:
                passed = all([
                    intent_is_wage,
                    target_ok,
                    table_ok,
                    appendix_ok,
                    not_unavailable,
                    escalation_ok,
                ])

            by_contract[contract_id]["total"] += 1
            by_expectation[expectation]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1
                by_expectation[expectation]["passed"] += 1

            target_resolution_passes += int(target_ok)
            table_evidence_passes += int(table_ok)
            appendix_citation_passes += int(appendix_ok)
            intent_wage_passes += int(intent_is_wage)
            not_unavailable_passes += int(not_unavailable)

            rows.append(
                {
                    "id": case_id,
                    "expectation": expectation,
                    "contract_id": contract_id,
                    "question": question,
                    "user_classification": user_classification,
                    "expected_target_classification": expected_target,
                    "intent_type": response.intent_type,
                    "intent_is_wage": intent_is_wage,
                    "resolved_target_classification": target_key,
                    "target_resolution_ok": target_ok,
                    "table_evidence_count": table_count,
                    "table_evidence_ok": table_ok,
                    "appendix_citation_present": appendix_ok,
                    "role_clarification_required": bool(role_clarification.get("needs_clarification")),
                    "role_clarification_ok": clarification_ok,
                    "role_clarification_family": str(role_clarification.get("role_family") or "").strip().lower(),
                    "role_clarification_values": sorted(clarification_values),
                    "contains_unavailable_language": not not_unavailable,
                    "escalation_required": bool(response.escalation_required),
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for row in rows if row.get("pass"))
        explicit_total = int(by_expectation["explicit_role_override"]["total"])
        explicit_passed = int(by_expectation["explicit_role_override"]["passed"])
        profile_total = int(by_expectation["profile_fallback"]["total"])
        profile_passed = int(by_expectation["profile_fallback"]["passed"])
        clarification_expectation_total = int(by_expectation["needs_role_clarification"]["total"])
        clarification_expectation_passed = int(by_expectation["needs_role_clarification"]["passed"])
        resolved_case_total = max(0, total - clarification_expectation_total)

        def _rate(num: int, den: int) -> float:
            return round((num / den) if den else 0.0, 4)

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
            "schema_version": "followup_role_wage_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(DATA_DIR / "test_set" / "followup_role_wage_test.json"),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "resolved_case_total": resolved_case_total,
                "target_resolution_rate": _rate(target_resolution_passes, total),
                "resolved_target_resolution_rate": _rate(target_resolution_passes, resolved_case_total),
                "table_evidence_presence_rate": _rate(table_evidence_passes, total),
                "resolved_table_evidence_presence_rate": _rate(table_evidence_passes, resolved_case_total),
                "appendix_citation_rate": _rate(appendix_citation_passes, total),
                "resolved_appendix_citation_rate": _rate(appendix_citation_passes, resolved_case_total),
                "intent_wage_rate": _rate(intent_wage_passes, total),
                "no_unavailable_rate": _rate(not_unavailable_passes, total),
                "explicit_override_passed": explicit_passed,
                "explicit_override_total": explicit_total,
                "explicit_override_rate": _rate(explicit_passed, explicit_total),
                "profile_fallback_passed": profile_passed,
                "profile_fallback_total": profile_total,
                "profile_fallback_rate": _rate(profile_passed, profile_total),
                "role_clarification_passed": clarification_expectation_passed,
                "role_clarification_total": clarification_expectation_total,
                "role_clarification_rate": _rate(clarification_expectation_passed, clarification_expectation_total),
            },
            "by_contract": by_contract_summary,
            "results": rows,
        }
    finally:
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        runtime_config.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_router_keyword
        runtime_config.CAG_ENABLE_RERANKER = original_reranker
        router_module.CAG_ENABLE_RERANKER = original_router_reranker
        api_module.retriever = original_retriever
        api_module.generate_response = original_generate


def run(test_file: Optional[Path] = None, bm25_only: bool = False) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "followup_role_wage_test.json"
    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return asyncio.run(_run_async(payload=payload, bm25_only=bm25_only))


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "followup_role_wage_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate role-targeted wage follow-up behavior.")
    parser.add_argument("--input", type=str, default=None, help="Path to follow-up role wage test JSON")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, bm25_only=args.bm25_only)
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL Follow-Up Role Wage Evaluation")
    print("=" * 72)
    print(
        f"Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} "
        f"({overall.get('pass_rate', 0.0):.1%})"
    )
    print(f"Target resolution rate: {overall.get('target_resolution_rate', 0.0):.1%}")
    print(f"Intent wage rate: {overall.get('intent_wage_rate', 0.0):.1%}")
    print(f"Appendix citation rate: {overall.get('appendix_citation_rate', 0.0):.1%}")
    print(f"Table evidence presence rate: {overall.get('table_evidence_presence_rate', 0.0):.1%}")
    print(f"Explicit override rate: {overall.get('explicit_override_rate', 0.0):.1%}")
    print(f"Profile fallback rate: {overall.get('profile_fallback_rate', 0.0):.1%}")
    if overall.get("role_clarification_total", 0):
        print(f"Role clarification rate: {overall.get('role_clarification_rate', 0.0):.1%}")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
