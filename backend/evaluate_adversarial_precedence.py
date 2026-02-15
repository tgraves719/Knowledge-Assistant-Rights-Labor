"""
Adversarial formal-precedence evaluator.

Runs deterministic retrieval checks over formal-rewrite near-miss prompts and
verifies that contract-specific exception evidence is retrieved (and, when
configured, ranked ahead of competing general-rule evidence).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.config as runtime_config
from backend.config import DATA_DIR, CONTRACT_ID
import backend.retrieval.router as router_module
from backend.contracts import get_contract_catalog_entry, resolve_default_contract_id
from backend.retrieval.router import HybridRetriever, classify_intent


def _canonical_contract_id(contract_id: Optional[str]) -> str:
    raw = str(contract_id or CONTRACT_ID)
    entry = get_contract_catalog_entry(raw)
    if entry and entry.get("contract_id"):
        return str(entry["contract_id"])
    return resolve_default_contract_id() or raw


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _citation_match(expected: str, found: str) -> bool:
    expected_norm = _norm(expected).replace(" ", "")
    found_norm = _norm(found).replace(" ", "")
    if not expected_norm or not found_norm:
        return False
    return expected_norm in found_norm or found_norm in expected_norm


def _first_citation_rank(expected_citations: list[str], retrieved_citations: list[str]) -> Optional[int]:
    for idx, citation in enumerate(retrieved_citations):
        if any(_citation_match(expected, citation) for expected in expected_citations):
            return idx
    return None


def _keyword_hits(text: str, expected_keywords: list[str]) -> int:
    if not expected_keywords:
        return 0
    haystack = _norm(text)
    return sum(1 for kw in expected_keywords if _norm(kw) and _norm(kw) in haystack)


def run(
    test_file: Optional[Path] = None,
    n_results: int = 8,
    bm25_only: bool = False,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "adversarial_test.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    test_cases = list(payload.get("test_cases") or [])
    default_contract = _canonical_contract_id(payload.get("contract_id"))

    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_keyword = runtime_config.HYBRID_KEYWORD_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_router_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    original_reranker = runtime_config.CAG_ENABLE_RERANKER
    original_router_reranker = router_module.CAG_ENABLE_RERANKER
    original_hypothesis = runtime_config.CAG_ENABLE_HYPOTHESIS_LAYER
    original_router_hypothesis = router_module.CAG_ENABLE_HYPOTHESIS_LAYER
    original_interpreter = runtime_config.CAG_ENABLE_QUERY_INTERPRETER
    original_router_interpreter = router_module.CAG_ENABLE_QUERY_INTERPRETER
    try:
        # Keep this slice deterministic and offline-safe.
        runtime_config.CAG_ENABLE_RERANKER = False
        router_module.CAG_ENABLE_RERANKER = False
        runtime_config.CAG_ENABLE_HYPOTHESIS_LAYER = False
        router_module.CAG_ENABLE_HYPOTHESIS_LAYER = False
        runtime_config.CAG_ENABLE_QUERY_INTERPRETER = False
        router_module.CAG_ENABLE_QUERY_INTERPRETER = False

        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)

        rows: list[dict] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        precedence_total = 0
        precedence_passed = 0
        citation_hits = 0
        keyword_hits = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = _canonical_contract_id(case.get("contract_id") or default_contract)
            question = str(case.get("question") or "")
            user_context = dict(case.get("user_context") or {})
            classification = user_context.get("classification")

            expected_citations = [str(c) for c in list(case.get("expected_citations") or [])]
            competing_citations = [str(c) for c in list(case.get("competing_citations") or [])]
            expected_keywords = [str(k) for k in list(case.get("expected_keywords") or [])]
            min_expected_keyword_hits = int(case.get("min_expected_keyword_hits") or 2)
            require_specific_precedence = bool(case.get("require_specific_precedence"))
            max_specific_rank_gap_vs_competing = int(case.get("max_specific_rank_gap_vs_competing") or 0)

            intent = classify_intent(
                question,
                user_classification=str(classification) if classification is not None else None,
                contract_id=contract_id,
            )
            retrieval = retriever.retrieve(
                query=question,
                intent=intent,
                n_results=n_results,
                use_hybrid=True,
                contract_id=contract_id,
            )
            chunks = list(retrieval.get("chunks") or [])
            retrieved_citations = [str(c.get("citation") or "") for c in chunks[:n_results]]

            expected_rank = _first_citation_rank(expected_citations, retrieved_citations)
            competing_rank = _first_citation_rank(competing_citations, retrieved_citations) if competing_citations else None
            citation_hit = expected_rank is not None
            if citation_hit:
                citation_hits += 1

            best_keyword_hits = 0
            for chunk in chunks[:n_results]:
                text = str(chunk.get("content_with_tables") or chunk.get("content") or "")
                best_keyword_hits = max(best_keyword_hits, _keyword_hits(text, expected_keywords))
            keyword_ok = True if not expected_keywords else best_keyword_hits >= min_expected_keyword_hits
            if keyword_ok:
                keyword_hits += 1

            precedence_ok = True
            if require_specific_precedence:
                precedence_total += 1
                if competing_rank is not None:
                    precedence_ok = bool(
                        citation_hit
                        and expected_rank is not None
                        and expected_rank <= (competing_rank + max_specific_rank_gap_vs_competing)
                    )
                if precedence_ok:
                    precedence_passed += 1

            leaked_contract_ids = sorted(
                {
                    str(c.get("contract_id"))
                    for c in chunks[:n_results]
                    if c.get("contract_id") and str(c.get("contract_id")) != contract_id
                }
            )
            tenancy_ok = not leaked_contract_ids

            passed = citation_hit and keyword_ok and precedence_ok and tenancy_ok
            by_contract[contract_id]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "contract_id": contract_id,
                    "question": question,
                    "expected_citations": expected_citations,
                    "competing_citations": competing_citations,
                    "retrieved_citations": retrieved_citations,
                    "expected_rank": expected_rank,
                    "competing_rank": competing_rank,
                    "citation_hit": citation_hit,
                    "expected_keywords": expected_keywords,
                    "min_expected_keyword_hits": min_expected_keyword_hits,
                    "best_expected_keyword_hits": best_keyword_hits,
                    "keyword_ok": keyword_ok,
                    "require_specific_precedence": require_specific_precedence,
                    "max_specific_rank_gap_vs_competing": max_specific_rank_gap_vs_competing,
                    "precedence_ok": precedence_ok,
                    "tenancy_ok": tenancy_ok,
                    "leaked_contract_ids": leaked_contract_ids,
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for row in rows if row.get("pass"))

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
            "schema_version": "adversarial_precedence_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "n_results": n_results,
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "citation_hit_rate": _rate(citation_hits, total),
                "keyword_hit_rate": _rate(keyword_hits, total),
                "precedence_passed": precedence_passed,
                "precedence_total": precedence_total,
                "precedence_pass_rate": _rate(precedence_passed, precedence_total) if precedence_total else None,
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
        runtime_config.CAG_ENABLE_HYPOTHESIS_LAYER = original_hypothesis
        router_module.CAG_ENABLE_HYPOTHESIS_LAYER = original_router_hypothesis
        runtime_config.CAG_ENABLE_QUERY_INTERPRETER = original_interpreter
        router_module.CAG_ENABLE_QUERY_INTERPRETER = original_router_interpreter


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "adversarial_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic adversarial formal-precedence retrieval.")
    parser.add_argument("--input", type=str, default=None, help="Path to adversarial test JSON")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, n_results=args.n_results, bm25_only=args.bm25_only)
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL Adversarial Formal-Precedence Evaluation")
    print("=" * 72)
    print(f"Pass: {overall.get('passed', 0)}/{overall.get('total', 0)} ({overall.get('pass_rate', 0.0):.1%})")
    print(f"Citation hit rate: {overall.get('citation_hit_rate')}")
    print(f"Keyword hit rate: {overall.get('keyword_hit_rate')}")
    print(f"Precedence pass rate: {overall.get('precedence_pass_rate')}")
    for contract_id, stats in sorted((report.get("by_contract") or {}).items()):
        print(f"- {contract_id}: {stats.get('passed', 0)}/{stats.get('total', 0)} ({stats.get('pass_rate', 0.0):.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
