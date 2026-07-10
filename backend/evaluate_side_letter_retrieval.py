"""
Side-letter retrieval evaluator.

Validates retrieval quality for LOA/LOU-focused prompts on contracts where
side-letter chunks must be represented with stable doc_type buckets.
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
from backend.config import CONTRACT_ID, DATA_DIR
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
    return expected_norm in found_norm


def _first_citation_rank(expected_citations: list[str], retrieved_citations: list[str]) -> Optional[int]:
    for idx, citation in enumerate(retrieved_citations):
        if any(_citation_match(expected, citation) for expected in expected_citations):
            return idx
    return None


def _chunk_citation_hit(chunk: dict, expected_citations: list[str]) -> bool:
    citation = str((chunk or {}).get("citation") or "")
    return any(_citation_match(expected, citation) for expected in expected_citations)


def _keyword_hits(text: str, keywords: list[str]) -> int:
    if not keywords:
        return 0
    haystack = _norm(text)
    return sum(1 for kw in keywords if _norm(kw) and _norm(kw) in haystack)


def _evaluate_window(
    chunks: list[dict],
    *,
    window_size: int,
    expected_citations: list[str],
    expected_doc_types: list[str],
    expected_keywords: list[str],
    min_expected_keyword_hits: int,
) -> dict:
    window_chunks = list(chunks[:max(1, int(window_size))])
    retrieved_citations = [str((c or {}).get("citation") or "") for c in window_chunks]
    expected_rank = _first_citation_rank(expected_citations, retrieved_citations) if expected_citations else None
    citation_hit = expected_rank is not None if expected_citations else True

    citation_hit_chunks = [c for c in window_chunks if _chunk_citation_hit(c, expected_citations)] if expected_citations else list(window_chunks)
    matched_chunk = window_chunks[expected_rank] if expected_rank is not None and expected_rank < len(window_chunks) else None
    observed_doc_type = str((matched_chunk or {}).get("doc_type") or "").strip().lower()
    observed_doc_types_for_citation = sorted(
        {
            str((chunk or {}).get("doc_type") or "").strip().lower()
            for chunk in citation_hit_chunks
            if str((chunk or {}).get("doc_type") or "").strip()
        }
    )
    doc_type_ok = True
    if expected_doc_types:
        doc_type_ok = any(dt in expected_doc_types for dt in observed_doc_types_for_citation)

    best_expected_keyword_hits = 0
    keyword_chunks = citation_hit_chunks if citation_hit_chunks else window_chunks
    for chunk in keyword_chunks:
        text = str((chunk or {}).get("content_with_tables") or (chunk or {}).get("content") or "")
        best_expected_keyword_hits = max(best_expected_keyword_hits, _keyword_hits(text, expected_keywords))
    keywords_ok = True if not expected_keywords else best_expected_keyword_hits >= min_expected_keyword_hits

    passed = citation_hit and doc_type_ok and keywords_ok
    return {
        "window_size": max(1, int(window_size)),
        "retrieved_citations": retrieved_citations,
        "expected_rank": expected_rank,
        "citation_hit": citation_hit,
        "observed_doc_type": observed_doc_type,
        "observed_doc_types_for_citation": observed_doc_types_for_citation,
        "doc_type_ok": doc_type_ok,
        "best_expected_keyword_hits": best_expected_keyword_hits,
        "keywords_ok": keywords_ok,
        "pass": passed,
    }


def run(
    test_file: Optional[Path] = None,
    n_results: int = 8,
    depth_n_results: int = 20,
    bm25_only: bool = False,
    topk_pass_threshold: float = 0.8,
    depth_pass_threshold: float = 0.95,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "side_letter_retrieval_test.json"

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
        by_contract = defaultdict(lambda: {"topk_passed": 0, "depth_passed": 0, "combined_passed": 0, "total": 0})
        topk_citation_hits = 0
        topk_doc_type_hits = 0
        topk_keyword_hits = 0
        depth_citation_hits = 0
        depth_doc_type_hits = 0
        depth_keyword_hits = 0
        topk_passed = 0
        depth_passed = 0
        combined_passed = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = _canonical_contract_id(case.get("contract_id") or default_contract)
            question = str(case.get("question") or "")
            expected_citations = [str(v) for v in list(case.get("expected_citations") or [])]
            expected_doc_types = [str(v).strip().lower() for v in list(case.get("expected_doc_types") or []) if str(v).strip()]
            expected_keywords = [str(v) for v in list(case.get("expected_keywords") or [])]
            min_expected_keyword_hits = int(case.get("min_expected_keyword_hits") or 1)

            intent = classify_intent(question, contract_id=contract_id)
            retrieval = retriever.retrieve(
                query=question,
                intent=intent,
                n_results=max(int(n_results), int(depth_n_results)),
                use_hybrid=True,
                contract_id=contract_id,
            )
            chunks = list(retrieval.get("chunks") or [])
            topk_eval = _evaluate_window(
                chunks,
                window_size=n_results,
                expected_citations=expected_citations,
                expected_doc_types=expected_doc_types,
                expected_keywords=expected_keywords,
                min_expected_keyword_hits=min_expected_keyword_hits,
            )
            depth_eval = _evaluate_window(
                chunks,
                window_size=depth_n_results,
                expected_citations=expected_citations,
                expected_doc_types=expected_doc_types,
                expected_keywords=expected_keywords,
                min_expected_keyword_hits=min_expected_keyword_hits,
            )

            topk_pass = bool(topk_eval.get("pass"))
            depth_pass = bool(depth_eval.get("pass"))
            combined_pass = topk_pass or depth_pass

            if bool(topk_eval.get("citation_hit")):
                topk_citation_hits += 1
            if bool(topk_eval.get("doc_type_ok")):
                topk_doc_type_hits += 1
            if bool(topk_eval.get("keywords_ok")):
                topk_keyword_hits += 1
            if bool(depth_eval.get("citation_hit")):
                depth_citation_hits += 1
            if bool(depth_eval.get("doc_type_ok")):
                depth_doc_type_hits += 1
            if bool(depth_eval.get("keywords_ok")):
                depth_keyword_hits += 1

            if topk_pass:
                topk_passed += 1
            if depth_pass:
                depth_passed += 1
            if combined_pass:
                combined_passed += 1

            by_contract[contract_id]["total"] += 1
            if topk_pass:
                by_contract[contract_id]["topk_passed"] += 1
            if depth_pass:
                by_contract[contract_id]["depth_passed"] += 1
            if combined_pass:
                by_contract[contract_id]["combined_passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "contract_id": contract_id,
                    "question": question,
                    "expected_citations": expected_citations,
                    "retrieved_citations_topk": topk_eval.get("retrieved_citations"),
                    "retrieved_citations_depth": depth_eval.get("retrieved_citations"),
                    "topk_window_size": topk_eval.get("window_size"),
                    "depth_window_size": depth_eval.get("window_size"),
                    "expected_rank_topk": topk_eval.get("expected_rank"),
                    "expected_rank_depth": depth_eval.get("expected_rank"),
                    "citation_hit_topk": bool(topk_eval.get("citation_hit")),
                    "citation_hit_depth": bool(depth_eval.get("citation_hit")),
                    "expected_doc_types": expected_doc_types,
                    "observed_doc_type_topk": topk_eval.get("observed_doc_type"),
                    "observed_doc_type_depth": depth_eval.get("observed_doc_type"),
                    "observed_doc_types_for_citation_topk": topk_eval.get("observed_doc_types_for_citation"),
                    "observed_doc_types_for_citation_depth": depth_eval.get("observed_doc_types_for_citation"),
                    "doc_type_ok_topk": bool(topk_eval.get("doc_type_ok")),
                    "doc_type_ok_depth": bool(depth_eval.get("doc_type_ok")),
                    "expected_keywords": expected_keywords,
                    "min_expected_keyword_hits": min_expected_keyword_hits,
                    "best_expected_keyword_hits_topk": topk_eval.get("best_expected_keyword_hits"),
                    "best_expected_keyword_hits_depth": depth_eval.get("best_expected_keyword_hits"),
                    "keywords_ok_topk": bool(topk_eval.get("keywords_ok")),
                    "keywords_ok_depth": bool(depth_eval.get("keywords_ok")),
                    "topk_pass": topk_pass,
                    "depth_pass": depth_pass,
                    "pass": combined_pass,
                }
            )

        total = len(rows)
        both_fail_count = sum(1 for row in rows if not bool(row.get("topk_pass")) and not bool(row.get("depth_pass")))

        def _rate(num: int, den: int) -> float:
            return round((num / den) if den else 0.0, 4)

        by_contract_summary = {}
        for contract_id, stats in sorted(by_contract.items()):
            c_total = int(stats["total"])
            c_topk_passed = int(stats["topk_passed"])
            c_depth_passed = int(stats["depth_passed"])
            c_combined_passed = int(stats["combined_passed"])
            by_contract_summary[contract_id] = {
                "topk_passed": c_topk_passed,
                "depth_passed": c_depth_passed,
                "combined_passed": c_combined_passed,
                "total": c_total,
                "topk_pass_rate": _rate(c_topk_passed, c_total),
                "depth_pass_rate": _rate(c_depth_passed, c_total),
                "combined_pass_rate": _rate(c_combined_passed, c_total),
            }

        topk_pass_rate = _rate(topk_passed, total)
        depth_pass_rate = _rate(depth_passed, total)
        gate_pass = not (
            topk_pass_rate < float(topk_pass_threshold)
            and depth_pass_rate < float(depth_pass_threshold)
        )

        return {
            "schema_version": "side_letter_retrieval_eval_v2",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "n_results": n_results,
            "depth_n_results": depth_n_results,
            "bm25_only": bm25_only,
            "gate": {
                "pass": gate_pass,
                "rule": "fail_only_if_topk_and_depth_degrade",
                "topk_pass_threshold": float(topk_pass_threshold),
                "depth_pass_threshold": float(depth_pass_threshold),
            },
            "overall": {
                "passed": topk_passed,
                "total": total,
                "pass_rate": topk_pass_rate,
                "topk_passed": topk_passed,
                "topk_pass_rate": topk_pass_rate,
                "depth_passed": depth_passed,
                "depth_pass_rate": depth_pass_rate,
                "combined_passed": combined_passed,
                "combined_pass_rate": _rate(combined_passed, total),
                "both_fail_case_count": both_fail_count,
                "topk_citation_hit_rate": _rate(topk_citation_hits, total),
                "topk_doc_type_match_rate": _rate(topk_doc_type_hits, total),
                "topk_keyword_match_rate": _rate(topk_keyword_hits, total),
                "depth_citation_hit_rate": _rate(depth_citation_hits, total),
                "depth_doc_type_match_rate": _rate(depth_doc_type_hits, total),
                "depth_keyword_match_rate": _rate(depth_keyword_hits, total),
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
    target = out_path or (DATA_DIR / "test_set" / "side_letter_retrieval_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate side-letter retrieval behavior.")
    parser.add_argument("--input", type=str, default=None, help="Path to side-letter retrieval test JSON")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--depth-results", type=int, default=20)
    parser.add_argument("--topk-pass-threshold", type=float, default=0.8)
    parser.add_argument("--depth-pass-threshold", type=float, default=0.95)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    report = run(
        test_file=Path(args.input) if args.input else None,
        n_results=max(1, int(args.n_results)),
        depth_n_results=max(1, int(args.depth_results)),
        bm25_only=bool(args.bm25_only),
        topk_pass_threshold=float(args.topk_pass_threshold),
        depth_pass_threshold=float(args.depth_pass_threshold),
    )
    out_path = _write_report(report, out_path=Path(args.output) if args.output else None)

    gate = report.get("gate") or {}
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL Side-Letter Retrieval Evaluation")
    print("=" * 72)
    print(
        f"Top-{int(report.get('n_results', 0))} pass: {int(overall.get('topk_passed', overall.get('passed', 0)))}"
        f"/{int(overall.get('total', 0))} ({float(overall.get('topk_pass_rate', overall.get('pass_rate', 0.0))):.1%})"
    )
    print(
        f"Depth-{int(report.get('depth_n_results', 0))} pass: {int(overall.get('depth_passed', 0))}"
        f"/{int(overall.get('total', 0))} ({float(overall.get('depth_pass_rate', 0.0)):.1%})"
    )
    print(
        f"Combined pass: {int(overall.get('combined_passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('combined_pass_rate', 0.0)):.1%})"
    )
    print(f"Both-fail cases: {int(overall.get('both_fail_case_count', 0))}")
    print(
        "Gate: "
        f"{bool(gate.get('pass'))} "
        f"(topk_threshold={float(gate.get('topk_pass_threshold', 0.0)):.2f}, "
        f"depth_threshold={float(gate.get('depth_pass_threshold', 0.0)):.2f})"
    )
    for contract_id, stats in sorted((report.get("by_contract") or {}).items()):
        print(
            f"- {contract_id}: topk={int(stats.get('topk_passed', 0))}/{int(stats.get('total', 0))} "
            f"({float(stats.get('topk_pass_rate', 0.0)):.1%}), "
            f"depth={int(stats.get('depth_passed', 0))}/{int(stats.get('total', 0))} "
            f"({float(stats.get('depth_pass_rate', 0.0)):.1%})"
        )
    print(f"Results: {out_path}")
    return 0 if bool(gate.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
