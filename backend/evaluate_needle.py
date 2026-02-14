"""
Needle retrieval evaluator.

Runs deterministic retrieval checks over synthetic KARL_NEEDLE cases and reports:
- expected synthetic citation retrieval rate
- keyword coverage on matched synthetic chunk text
- by-position pass rates (top/middle/bottom)
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


def _citation_match(expected_citation: str, citations: list[str]) -> bool:
    expected_norm = _norm(expected_citation)
    for citation in citations:
        got_norm = _norm(citation)
        if expected_norm and (expected_norm in got_norm or got_norm in expected_norm):
            return True
    return False


def _keyword_hit_count(text: str, expected_keywords: list[str]) -> int:
    if not expected_keywords:
        return 0
    haystack = _norm(text)
    return sum(1 for kw in expected_keywords if _norm(kw) and _norm(kw) in haystack)


def run(
    test_file: Optional[Path] = None,
    contract_id: Optional[str] = None,
    n_results: int = 8,
    bm25_only: bool = False,
    min_keyword_hits: int = 2,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "needle_test.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    test_cases = payload.get("test_cases", [])
    target_contract = _canonical_contract_id(contract_id or payload.get("contract_id"))

    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_keyword = runtime_config.HYBRID_KEYWORD_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_router_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    original_reranker = runtime_config.CAG_ENABLE_RERANKER
    original_router_reranker = router_module.CAG_ENABLE_RERANKER
    try:
        # Keep slice deterministic.
        runtime_config.CAG_ENABLE_RERANKER = False
        router_module.CAG_ENABLE_RERANKER = False

        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)

        rows: list[dict] = []
        by_position = defaultdict(lambda: {"passed": 0, "total": 0})
        citation_matches = 0
        keyword_passes = 0

        for case in test_cases:
            test_id = str(case.get("id") or "")
            needle_id = str(case.get("needle_id") or "")
            question = str(case.get("question") or "")
            expected_citation = str(case.get("expected_citation") or "")
            expected_keywords = list(case.get("expected_answer_keywords") or [])
            position = str(case.get("injection_position") or "unknown")

            intent = classify_intent(question, contract_id=target_contract)
            result = retriever.retrieve(
                query=question,
                intent=intent,
                n_results=n_results,
                use_hybrid=True,
                contract_id=target_contract,
            )
            chunks = result.get("chunks", [])
            citations = [str(c.get("citation") or "") for c in chunks[:n_results]]

            citation_ok = _citation_match(expected_citation, citations)
            if citation_ok:
                citation_matches += 1

            matched_chunk_text = ""
            if citation_ok:
                for c in chunks[:n_results]:
                    citation = str(c.get("citation") or "")
                    if _citation_match(expected_citation, [citation]):
                        matched_chunk_text = str(c.get("content_with_tables") or c.get("content") or "")
                        break

            keyword_hits = _keyword_hit_count(matched_chunk_text, expected_keywords)
            keyword_ok = keyword_hits >= max(0, min_keyword_hits)
            if keyword_ok:
                keyword_passes += 1

            passed = citation_ok and keyword_ok
            by_position[position]["total"] += 1
            if passed:
                by_position[position]["passed"] += 1

            rows.append(
                {
                    "id": test_id,
                    "needle_id": needle_id,
                    "question": question,
                    "injection_position": position,
                    "expected_citation": expected_citation,
                    "retrieved_citations": citations,
                    "citation_match": citation_ok,
                    "keyword_hits": keyword_hits,
                    "keyword_total": len(expected_keywords),
                    "keyword_pass": keyword_ok,
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for r in rows if r["pass"])
        by_position_summary = {}
        for pos, stats in sorted(by_position.items()):
            pos_total = int(stats["total"])
            pos_passed = int(stats["passed"])
            by_position_summary[pos] = {
                "passed": pos_passed,
                "total": pos_total,
                "pass_rate": round((pos_passed / pos_total) if pos_total else 0.0, 4),
            }

        return {
            "schema_version": "needle_eval_v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "contract_id": target_contract,
            "n_results": n_results,
            "bm25_only": bm25_only,
            "min_keyword_hits": min_keyword_hits,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": round((passed / total) if total else 0.0, 4),
                "citation_match_rate": round((citation_matches / total) if total else 0.0, 4),
                "keyword_pass_rate": round((keyword_passes / total) if total else 0.0, 4),
            },
            "by_position": by_position_summary,
            "results": rows,
        }
    finally:
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        runtime_config.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_router_keyword
        runtime_config.CAG_ENABLE_RERANKER = original_reranker
        router_module.CAG_ENABLE_RERANKER = original_router_reranker


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "needle_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic needle retrieval slice.")
    parser.add_argument("--input", type=str, default=None, help="Path to needle test JSON")
    parser.add_argument("--contract-id", type=str, default=None, help="Contract ID override")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--min-keyword-hits", type=int, default=2)
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(
        test_file=test_file,
        contract_id=args.contract_id,
        n_results=args.n_results,
        bm25_only=args.bm25_only,
        min_keyword_hits=args.min_keyword_hits,
    )
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL Needle Retrieval Evaluation")
    print("=" * 72)
    print(f"Pass: {overall.get('passed', 0)}/{overall.get('total', 0)} ({overall.get('pass_rate', 0.0):.1%})")
    print(f"Citation match rate: {overall.get('citation_match_rate')}")
    print(f"Keyword pass rate: {overall.get('keyword_pass_rate')}")
    for position, stats in sorted((report.get("by_position") or {}).items()):
        print(f"- {position}: {stats.get('passed', 0)}/{stats.get('total', 0)} ({stats.get('pass_rate', 0.0):.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
