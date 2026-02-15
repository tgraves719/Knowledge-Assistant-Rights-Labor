"""
Cross-contract entity mention regression evaluator.

Runs deterministic end-to-end checks through /api/query with normal generation
enabled and verifies that explicit references to a different known contract
preserve uncertainty in the active contract context.
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
from backend.config import DATA_DIR
import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever
from backend.api import QueryRequest, query_contract


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
    try:
        runtime_config.CAG_ENABLE_RERANKER = False
        router_module.CAG_ENABLE_RERANKER = False
        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        api_module.retriever = HybridRetriever(vector_store=None)

        rows: list[dict] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = str(case.get("contract_id") or "")
            question = str(case.get("question") or "")
            expected_escalation = bool(case.get("expected_escalation") or False)
            expected_max_citations = int(case.get("expected_max_citations") or 0)

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

            unavailable = api_module._is_unavailable_answer(response.answer)
            escalation_ok = bool(response.escalation_required) == expected_escalation
            citation_count = len(list(response.citations or []))
            citation_ok = citation_count <= expected_max_citations
            passed = unavailable and escalation_ok and citation_ok

            by_contract[contract_id]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "contract_id": contract_id,
                    "question": question,
                    "contains_unavailable_language": unavailable,
                    "expected_escalation": expected_escalation,
                    "escalation_required": bool(response.escalation_required),
                    "escalation_ok": escalation_ok,
                    "citation_count": citation_count,
                    "expected_max_citations": expected_max_citations,
                    "citation_ok": citation_ok,
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for row in rows if row.get("pass"))
        no_citation_passes = sum(1 for row in rows if row.get("citation_ok"))

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
            "schema_version": "cross_contract_mentions_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(DATA_DIR / "test_set" / "cross_contract_mentions_test.json"),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "no_citation_rate": _rate(no_citation_passes, total),
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


def run(test_file: Optional[Path] = None, bm25_only: bool = False) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "cross_contract_mentions_test.json"
    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return asyncio.run(_run_async(payload=payload, bm25_only=bm25_only))


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "cross_contract_mentions_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate cross-contract entity mention behavior.")
    parser.add_argument("--input", type=str, default=None, help="Path to cross-contract mention test JSON")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, bm25_only=args.bm25_only)
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL Cross-Contract Mention Evaluation")
    print("=" * 72)
    print(
        f"Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} "
        f"({overall.get('pass_rate', 0.0):.1%})"
    )
    print(f"No-citation rate: {overall.get('no_citation_rate', 0.0):.1%}")
    for cid, stats in sorted((report.get("by_contract") or {}).items()):
        print(f"- {cid}: {stats.get('passed', 0)}/{stats.get('total', 0)} ({stats.get('pass_rate', 0.0):.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
