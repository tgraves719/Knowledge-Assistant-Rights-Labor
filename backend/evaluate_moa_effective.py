"""
MOA effective-state retrieval evaluator.

Focus:
- Does retrieval prioritize current effective language for amended clauses?
- Do removed/absent phrases avoid false positive retrieval artifacts?
- Is wage metadata suppressed for non-wage contract questions?
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


def _extract_source_type_from_chunk(chunk: dict) -> Optional[str]:
    source_type_field = str(chunk.get("source_type") or "").strip().lower()
    if source_type_field:
        if "moa" in source_type_field or "amend" in source_type_field:
            return "moa"
        if source_type_field in {"base", "cba", "contract"}:
            return "base"

    for ref in (chunk.get("provenance") or []):
        if not isinstance(ref, dict):
            continue
        source_type = str(ref.get("source_type") or "").strip().lower()
        pdf_name = str(ref.get("pdf") or "").strip().lower()
        if "moa" in source_type or "amend" in source_type or "moa" in pdf_name:
            return "moa"
        if source_type in {"base", "cba", "contract"}:
            return "base"
    return None


def _keyword_hits(text: str, keywords: list[str]) -> int:
    if not keywords:
        return 0
    haystack = _norm(text)
    return sum(1 for kw in keywords if _norm(kw) and _norm(kw) in haystack)


def run(
    test_file: Optional[Path] = None,
    n_results: int = 8,
    bm25_only: bool = False,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "moa_effective_test.json"

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
        citation_hits = 0
        source_type_hits = 0
        intent_hits = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = _canonical_contract_id(case.get("contract_id") or default_contract)
            question = str(case.get("question") or "")
            expectation = str(case.get("expectation") or "present").strip().lower()

            expected_citations = [str(c) for c in list(case.get("expected_citations") or [])]
            expected_keywords = [str(k) for k in list(case.get("expected_keywords") or [])]
            forbidden_keywords = [str(k) for k in list(case.get("forbidden_keywords") or [])]
            expected_source_type = str(case.get("expected_source_type") or "").strip().lower() or None
            expected_intent_type = str(case.get("expected_intent_type") or "").strip().lower() or None
            min_expected_keyword_hits = int(case.get("min_expected_keyword_hits") or 1)
            require_no_wage_info = bool(case.get("require_no_wage_info", False))

            intent = classify_intent(question, contract_id=contract_id)
            retrieval = retriever.retrieve(
                query=question,
                intent=intent,
                n_results=n_results,
                use_hybrid=True,
                contract_id=contract_id,
            )
            chunks = list(retrieval.get("chunks") or [])
            top_chunks = chunks[:n_results]
            retrieved_citations = [str(c.get("citation") or "") for c in top_chunks]

            expected_rank = _first_citation_rank(expected_citations, retrieved_citations) if expected_citations else None
            citation_hit = expected_rank is not None if expected_citations else True
            if citation_hit:
                citation_hits += 1

            matched_chunk = None
            if expected_rank is not None and expected_rank < len(top_chunks):
                matched_chunk = top_chunks[expected_rank]
            source_type_observed = _extract_source_type_from_chunk(matched_chunk or {})
            source_type_ok = True
            if expected_source_type:
                source_type_ok = source_type_observed == expected_source_type
                if source_type_ok:
                    source_type_hits += 1

            best_keyword_hits = 0
            for chunk in top_chunks:
                text = str(chunk.get("content_with_tables") or chunk.get("content") or "")
                best_keyword_hits = max(best_keyword_hits, _keyword_hits(text, expected_keywords))
            keywords_ok = True if not expected_keywords else best_keyword_hits >= min_expected_keyword_hits

            forbidden_hit = False
            for chunk in top_chunks:
                text = str(chunk.get("content_with_tables") or chunk.get("content") or "")
                if _keyword_hits(text, forbidden_keywords) > 0:
                    forbidden_hit = True
                    break
            forbidden_ok = not forbidden_hit

            intent_ok = True
            if expected_intent_type:
                intent_ok = str(intent.intent_type or "").strip().lower() == expected_intent_type
            if intent_ok:
                intent_hits += 1

            wage_info = retrieval.get("wage_info")
            wage_info_ok = (wage_info is None) if require_no_wage_info else True

            if expectation == "absent":
                passed = forbidden_ok and intent_ok and wage_info_ok and (not expected_citations or not citation_hit)
            else:
                passed = citation_hit and keywords_ok and forbidden_ok and source_type_ok and intent_ok and wage_info_ok

            by_contract[contract_id]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "contract_id": contract_id,
                    "question": question,
                    "expectation": expectation,
                    "expected_citations": expected_citations,
                    "retrieved_citations": retrieved_citations,
                    "expected_rank": expected_rank,
                    "citation_hit": citation_hit,
                    "expected_keywords": expected_keywords,
                    "min_expected_keyword_hits": min_expected_keyword_hits,
                    "best_expected_keyword_hits": best_keyword_hits,
                    "keywords_ok": keywords_ok,
                    "forbidden_keywords": forbidden_keywords,
                    "forbidden_ok": forbidden_ok,
                    "expected_source_type": expected_source_type,
                    "source_type_observed": source_type_observed,
                    "source_type_ok": source_type_ok,
                    "expected_intent_type": expected_intent_type,
                    "observed_intent_type": intent.intent_type,
                    "intent_ok": intent_ok,
                    "require_no_wage_info": require_no_wage_info,
                    "wage_info_present": bool(wage_info),
                    "wage_info_ok": wage_info_ok,
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

        source_cases_total = sum(1 for row in rows if row.get("expected_source_type"))
        intent_cases_total = sum(1 for row in rows if row.get("expected_intent_type"))
        return {
            "schema_version": "moa_effective_eval_v1",
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
                "source_type_match_rate": _rate(source_type_hits, source_cases_total) if source_cases_total else None,
                "intent_match_rate": _rate(intent_hits, intent_cases_total) if intent_cases_total else None,
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


def _write_report(report: dict, out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        out_path = DATA_DIR / "test_set" / "moa_effective_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MOA effective-state retrieval behavior.")
    parser.add_argument("--input", type=str, default=None, help="Path to MOA-effective test JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to write results JSON")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file, n_results=args.n_results, bm25_only=args.bm25_only)
    out_path = _write_report(report, out_path=Path(args.output) if args.output else None)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL MOA Effective Evaluation")
    print("=" * 72)
    print(f"Pass: {overall.get('passed', 0)}/{overall.get('total', 0)} ({overall.get('pass_rate', 0.0):.1%})")
    print(f"Citation hit rate: {overall.get('citation_hit_rate')}")
    print(f"Source-type match rate: {overall.get('source_type_match_rate')}")
    print(f"Intent match rate: {overall.get('intent_match_rate')}")
    for contract_id, stats in sorted((report.get("by_contract") or {}).items()):
        print(f"- {contract_id}: {stats.get('passed', 0)}/{stats.get('total', 0)} ({stats.get('pass_rate', 0.0):.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
