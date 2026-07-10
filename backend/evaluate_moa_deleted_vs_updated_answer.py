"""End-to-end MOA deleted-vs-updated answer evaluator.

Runs through `query_contract` (API logic) and scores answer behavior for:
- updated clauses: synthesized answer should resolve to effective/MOA-backed language
- deleted clauses: system should preserve uncertainty/abstention for removed language
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.api as api_module
import backend.config as runtime_config
from backend.api import QueryRequest, query_contract
from backend.config import DATA_DIR
import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever


OUT_PATH = DATA_DIR / "test_set" / "moa_deleted_vs_updated_answer_results.json"
DEFAULT_INPUT = DATA_DIR / "test_set" / "moa_deleted_vs_updated_test.json"


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _load_manifest(contract_id: str) -> dict:
    path = DATA_DIR / "manifests" / f"{contract_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _citation_match(expected: str, found: str) -> bool:
    expected_norm = _norm(expected).replace(" ", "")
    found_norm = _norm(found).replace(" ", "")
    if not expected_norm or not found_norm:
        return False
    return expected_norm in found_norm or found_norm in expected_norm


def _keyword_hits(text: str, keywords: list[str]) -> int:
    haystack = _norm(text)
    return sum(1 for kw in keywords if _norm(kw) and _norm(kw) in haystack)


def _source_type_match(expected_source_type: Optional[str], sources: list[dict[str, Any]], expected_citations: list[str]) -> tuple[bool, Optional[str]]:
    expected = str(expected_source_type or "").strip().lower() or None
    if not expected:
        return True, None

    observed_types: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        src_type = str(source.get("source_type") or "").strip().lower()
        citation = str(source.get("citation") or "")
        if src_type:
            observed_types.append(src_type)
        if expected_citations and citation:
            if any(_citation_match(exp, citation) for exp in expected_citations):
                if expected in src_type:
                    return True, src_type

    # Fallback: any source type contains expected token.
    for src_type in observed_types:
        if expected in src_type:
            return True, src_type
    return False, (observed_types[0] if observed_types else None)


def _case_bucket(case: dict[str, Any]) -> str:
    token = str(case.get("case_class") or "").strip().lower()
    if token in {"updated", "deleted"}:
        return token
    if str(case.get("expectation") or "").strip().lower() == "absent":
        return "deleted"
    return "updated"


async def _run_async(payload: dict, bm25_only: bool) -> dict:
    test_cases = list(payload.get("test_cases") or [])

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
    original_retriever = api_module.retriever
    original_generate = api_module.generate_response
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

        api_module.retriever = HybridRetriever(vector_store=None)

        async def _forced_unavailable(*_args, **_kwargs) -> str:
            return "I cannot find that specific information in your contract."

        # Deterministic mode: force first-pass unavailable so runtime fallback/
        # abstention behavior is what gets evaluated (no network/LLM dependency).
        api_module.generate_response = _forced_unavailable

        rows: list[dict[str, Any]] = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        by_bucket = defaultdict(lambda: {"passed": 0, "total": 0})
        unavailable_hits = 0
        source_type_hits = 0
        citation_hits = 0

        for case in test_cases:
            case_id = str(case.get("id") or "")
            contract_id = str(case.get("contract_id") or "")
            question = str(case.get("question") or "")
            expectation = str(case.get("expectation") or "present").strip().lower()
            bucket = _case_bucket(case)
            expected_citations = [str(v) for v in list(case.get("expected_citations") or [])]
            expected_keywords = [str(v) for v in list(case.get("expected_keywords") or [])]
            forbidden_keywords = [str(v) for v in list(case.get("forbidden_keywords") or [])]
            expected_source_type = str(case.get("expected_source_type") or "").strip().lower() or None
            expected_intent_type = str(case.get("expected_intent_type") or "").strip().lower() or None
            min_expected_keyword_hits = int(case.get("min_expected_keyword_hits") or 1)
            require_no_wage_info = bool(case.get("require_no_wage_info", False))
            max_citations_for_absent = int(case.get("max_citations_for_absent") or 0)

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

            answer = str(response.answer or "")
            citations = [str(v) for v in list(response.citations or [])]
            sources = [dict(v) for v in list(response.sources or []) if isinstance(v, dict)]
            unavailable = bool(api_module._is_unavailable_answer(answer))
            intent_observed = str(response.intent_type or "").strip().lower()
            intent_ok = True if not expected_intent_type else intent_observed == expected_intent_type
            wage_info_ok = (response.wage_info is None) if require_no_wage_info else True
            citation_hit = True if not expected_citations else any(
                _citation_match(exp, got) for exp in expected_citations for got in citations
            )
            source_type_ok, source_type_observed = _source_type_match(expected_source_type, sources, expected_citations)
            answer_keyword_hits = _keyword_hits(answer, expected_keywords)
            answer_keywords_ok = True if not expected_keywords else answer_keyword_hits >= min_expected_keyword_hits
            answer_forbidden_ok = _keyword_hits(answer, forbidden_keywords) == 0
            citation_count = len(citations)
            absent_citation_ok = (citation_count <= max_citations_for_absent)

            if unavailable:
                unavailable_hits += 1
            if citation_hit:
                citation_hits += 1
            if expected_source_type and source_type_ok:
                source_type_hits += 1

            if expectation == "absent":
                passed = unavailable and answer_forbidden_ok and intent_ok and wage_info_ok and absent_citation_ok
            else:
                passed = (
                    (not unavailable)
                    and citation_hit
                    and source_type_ok
                    and answer_keywords_ok
                    and answer_forbidden_ok
                    and intent_ok
                    and wage_info_ok
                )

            by_contract[contract_id]["total"] += 1
            by_bucket[bucket]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1
                by_bucket[bucket]["passed"] += 1

            rows.append(
                {
                    "id": case_id,
                    "case_class": bucket,
                    "contract_id": contract_id,
                    "question": question,
                    "expectation": expectation,
                    "answer_excerpt": answer[:800],
                    "contains_unavailable_language": unavailable,
                    "expected_citations": expected_citations,
                    "citations": citations,
                    "citation_hit": citation_hit,
                    "citation_count": citation_count,
                    "max_citations_for_absent": max_citations_for_absent if expectation == "absent" else None,
                    "absent_citation_ok": absent_citation_ok if expectation == "absent" else None,
                    "expected_source_type": expected_source_type,
                    "source_type_observed": source_type_observed,
                    "source_type_ok": source_type_ok,
                    "expected_keywords": expected_keywords,
                    "answer_expected_keyword_hits": answer_keyword_hits,
                    "min_expected_keyword_hits": min_expected_keyword_hits,
                    "answer_keywords_ok": answer_keywords_ok,
                    "forbidden_keywords": forbidden_keywords,
                    "answer_forbidden_ok": answer_forbidden_ok,
                    "expected_intent_type": expected_intent_type,
                    "observed_intent_type": intent_observed,
                    "intent_ok": intent_ok,
                    "require_no_wage_info": require_no_wage_info,
                    "wage_info_present": bool(response.wage_info),
                    "wage_info_ok": wage_info_ok,
                    "effective_version_id": response.effective_version_id,
                    "amendments_applied": list(response.amendments_applied or []),
                    "pass": passed,
                }
            )

        def _rate(num: int, den: int) -> float:
            return round((num / den) if den else 0.0, 4)

        total = len(rows)
        passed = sum(1 for r in rows if bool(r.get("pass")))
        source_cases_total = sum(1 for r in rows if r.get("expected_source_type"))

        by_contract_summary = {
            cid: {
                "passed": int(stats["passed"]),
                "total": int(stats["total"]),
                "pass_rate": _rate(int(stats["passed"]), int(stats["total"])),
            }
            for cid, stats in sorted(by_contract.items())
        }
        by_bucket_summary = {
            bucket: {
                "passed": int(stats["passed"]),
                "total": int(stats["total"]),
                "pass_rate": _rate(int(stats["passed"]), int(stats["total"])),
            }
            for bucket, stats in sorted(by_bucket.items())
        }

        return {
            "schema_version": "moa_deleted_vs_updated_answer_eval_v1",
            "dataset_schema_version": str(payload.get("schema_version") or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(DEFAULT_INPUT),
            "bm25_only": bm25_only,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": _rate(passed, total),
                "unavailable_rate": _rate(unavailable_hits, total),
                "citation_hit_rate": _rate(citation_hits, total),
                "source_type_match_rate": _rate(source_type_hits, source_cases_total) if source_cases_total else None,
            },
            "by_contract": by_contract_summary,
            "by_bucket": by_bucket_summary,
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
        api_module.retriever = original_retriever
        api_module.generate_response = original_generate


def run(test_file: Optional[Path] = None, bm25_only: bool = False) -> dict:
    in_path = test_file or DEFAULT_INPUT
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return asyncio.run(_run_async(payload=payload, bm25_only=bm25_only))


def _write_report(report: dict, output_path: Optional[Path] = None) -> Path:
    out_path = output_path or OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end MOA deleted-vs-updated answer behavior.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(OUT_PATH))
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--min-overall-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-updated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-deleted-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-source-type-match-rate", type=float, default=1.0)
    args = parser.parse_args()

    report = run(test_file=Path(args.input), bm25_only=bool(args.bm25_only))
    overall = report.get("overall") or {}
    by_bucket = report.get("by_bucket") or {}

    def _bucket_rate(name: str) -> float:
        return float(((by_bucket.get(name) or {}).get("pass_rate")) or 0.0)

    gate_checks = {
        "overall_pass_rate": float(overall.get("pass_rate") or 0.0) >= float(args.min_overall_pass_rate),
        "updated_pass_rate": _bucket_rate("updated") >= float(args.min_updated_pass_rate),
        "deleted_pass_rate": _bucket_rate("deleted") >= float(args.min_deleted_pass_rate),
        "source_type_match_rate": float(overall.get("source_type_match_rate") or 0.0) >= float(args.min_source_type_match_rate),
    }
    gate_pass = all(gate_checks.values())
    report["thresholds"] = {
        "min_overall_pass_rate": float(args.min_overall_pass_rate),
        "min_updated_pass_rate": float(args.min_updated_pass_rate),
        "min_deleted_pass_rate": float(args.min_deleted_pass_rate),
        "min_source_type_match_rate": float(args.min_source_type_match_rate),
    }
    report["gate"] = {
        "pass": bool(gate_pass),
        "checks": gate_checks,
    }
    out_path = _write_report(report, output_path=Path(args.output))

    print("=" * 72)
    print("KARL MOA Deleted-vs-Updated Answer Evaluation")
    print("=" * 72)
    print(
        f"Overall: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    for bucket in ("updated", "deleted"):
        stats = by_bucket.get(bucket) or {}
        print(f"- {bucket}: {int(stats.get('passed', 0))}/{int(stats.get('total', 0))} ({float(stats.get('pass_rate', 0.0)):.1%})")
    print(f"Source-type match rate: {overall.get('source_type_match_rate')}")
    print(f"Gate: {gate_pass}")
    print(f"Results: {out_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
