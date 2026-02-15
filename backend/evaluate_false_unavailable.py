"""
False-unavailable guard evaluator.

Runs deterministic end-to-end checks through /api/query logic with a forced
unavailable first-pass model response to ensure recovery path prevents false
"not available" answers when strong evidence exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.api as api_module
import backend.config as runtime_config
from backend.config import DATA_DIR
import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever
from backend.api import QueryRequest, query_contract


def _load_manifest(contract_id: str) -> dict:
    path = DATA_DIR / "manifests" / f"{contract_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _citation_match(expected: str, found: list[str]) -> bool:
    expected_norm = _norm(expected).replace(" ", "")
    for citation in found:
        got_norm = _norm(citation).replace(" ", "")
        if expected_norm and (expected_norm in got_norm or got_norm in expected_norm):
            return True
    return False


async def _run_async(
    payload: dict,
    bm25_only: bool,
) -> dict:
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

        async def _forced_unavailable(_: str, __: str, ___: Optional[list] = None) -> str:
            return "I cannot find that specific information in your contract."

        api_module.generate_response = _forced_unavailable

        rows: list[dict] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        by_expectation = defaultdict(lambda: {"passed": 0, "total": 0})

        for case in test_cases:
            case_id = str(case.get("id") or "")
            expectation = str(case.get("expectation") or "recover").strip().lower()
            contract_id = str(case.get("contract_id") or "")
            question = str(case.get("question") or "")
            expected_citations = list(case.get("expected_citations") or [])
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
                user_classification=None,
                hours_worked=0,
                months_employed=0,
                session_id=None,
            )
            response = await query_contract(request)
            citations = list(response.citations or [])
            unavailable = api_module._is_unavailable_answer(response.answer)
            citation_hit = any(_citation_match(exp, citations) for exp in expected_citations)
            if expectation == "uncertain":
                passed = unavailable
            else:
                passed = (not unavailable) and citation_hit

            by_contract[contract_id]["total"] += 1
            by_expectation[expectation]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1
                by_expectation[expectation]["passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "expectation": expectation,
                    "contract_id": contract_id,
                    "question": question,
                    "expected_citations": expected_citations,
                    "retrieved_citations": citations,
                    "contains_unavailable_language": unavailable,
                    "citation_hit": citation_hit,
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for row in rows if row.get("pass"))
        recover_total = int(by_expectation["recover"]["total"])
        recover_passed = int(by_expectation["recover"]["passed"])
        uncertain_total = int(by_expectation["uncertain"]["total"])
        uncertain_passed = int(by_expectation["uncertain"]["passed"])

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
            "schema_version": "false_unavailable_eval_v2",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(DATA_DIR / "test_set" / "false_unavailable_test.json"),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "recover_passed": recover_passed,
                "recover_total": recover_total,
                "false_unavailable_recovered_rate": _rate(recover_passed, recover_total),
                "uncertain_passed": uncertain_passed,
                "uncertain_total": uncertain_total,
                "proper_uncertainty_rate": _rate(uncertain_passed, uncertain_total),
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
        test_file = DATA_DIR / "test_set" / "false_unavailable_test.json"
    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return asyncio.run(_run_async(payload=payload, bm25_only=bm25_only))


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "false_unavailable_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate false-unavailable recovery behavior.")
    parser.add_argument("--input", type=str, default=None, help="Path to false unavailable test JSON")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, bm25_only=args.bm25_only)
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL False-Unavailable Guard Evaluation")
    print("=" * 72)
    print(
        "Overall: "
        f"{overall.get('passed', 0)}/{overall.get('total', 0)} "
        f"({overall.get('pass_rate', 0.0):.1%})"
    )
    print(
        "Recovered (evidence-present): "
        f"{overall.get('recover_passed', 0)}/{overall.get('recover_total', 0)} "
        f"({overall.get('false_unavailable_recovered_rate', 0.0):.1%})"
    )
    print(
        "Proper uncertainty (evidence-absent): "
        f"{overall.get('uncertain_passed', 0)}/{overall.get('uncertain_total', 0)} "
        f"({overall.get('proper_uncertainty_rate', 0.0):.1%})"
    )
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
