"""
Paraphrase robustness evaluator.

Runs retrieval-only checks over paraphrase families to ensure that terse/slang
variants retrieve the same expected contract evidence as canonical wording.
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


def _extract_retrieved_articles(chunks: list[dict]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    for chunk in chunks or []:
        citation = str(chunk.get("citation") or "").lower()
        for match in re.findall(r"article\s*(\d+)", citation):
            ref = f"Article {match}"
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)

        if "appendix" in citation and "Appendix A" not in seen:
            refs.append("Appendix A")
            seen.add("Appendix A")

        if "letter of understanding" in citation and "Letter of Understanding" not in seen:
            refs.append("Letter of Understanding")
            seen.add("Letter of Understanding")

    return refs


def _matches_expected(expected_articles: list[str], retrieved_articles: list[str]) -> bool:
    expected = [str(a) for a in (expected_articles or [])]
    if not expected:
        return True

    for exp in expected:
        exp_norm = exp.lower().replace(" ", "")
        for got in retrieved_articles:
            got_norm = got.lower().replace(" ", "")
            if exp_norm in got_norm or got_norm in exp_norm:
                return True
    return False


def run(
    test_file: Optional[Path] = None,
    contract_id: Optional[str] = None,
    n_results: int = 5,
    bm25_only: bool = False,
    disable_reranker: bool = True,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "paraphrase_test.json"

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
        if disable_reranker:
            runtime_config.CAG_ENABLE_RERANKER = False
            router_module.CAG_ENABLE_RERANKER = False

        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)

        family_rows: list[dict] = []
        by_variant_type = defaultdict(lambda: {"passed": 0, "total": 0})
        by_bucket = defaultdict(lambda: {"passed": 0, "total": 0})

        for case in test_cases:
            family_id = str(case.get("family_id") or "")
            bucket = str(case.get("bucket") or "unknown")
            expected_articles = list(case.get("expected_articles") or [])
            variants = list(case.get("variants") or [])

            variant_rows: list[dict] = []
            variant_passes = 0
            retrieved_sets: list[set[str]] = []

            for variant in variants:
                variant_id = str(variant.get("id") or "")
                variant_type = str(variant.get("variant_type") or "unknown")
                question = str(variant.get("question") or "")

                intent = classify_intent(question, contract_id=target_contract)
                retrieval = retriever.multi_angle_retrieve(
                    query=question,
                    intent=intent,
                    n_results=n_results,
                    contract_id=target_contract,
                )
                retrieved_articles = _extract_retrieved_articles(retrieval.get("chunks", []))
                passed = _matches_expected(expected_articles, retrieved_articles)
                if passed:
                    variant_passes += 1

                by_variant_type[variant_type]["total"] += 1
                if passed:
                    by_variant_type[variant_type]["passed"] += 1

                variant_rows.append(
                    {
                        "id": variant_id,
                        "variant_type": variant_type,
                        "question": question,
                        "retrieved_articles": retrieved_articles[:10],
                        "pass": passed,
                    }
                )
                retrieved_sets.append(set(retrieved_articles))

            family_pass = variant_passes == len(variants) if variants else False
            by_bucket[bucket]["total"] += 1
            if family_pass:
                by_bucket[bucket]["passed"] += 1

            common_articles = set.intersection(*retrieved_sets) if retrieved_sets else set()

            family_rows.append(
                {
                    "family_id": family_id,
                    "bucket": bucket,
                    "canonical_question": str(case.get("canonical_question") or ""),
                    "expected_articles": expected_articles,
                    "variants_total": len(variants),
                    "variants_passed": variant_passes,
                    "family_pass": family_pass,
                    "shared_retrieved_articles": sorted(common_articles),
                    "variants": variant_rows,
                }
            )

        families_total = len(family_rows)
        families_passed = sum(1 for row in family_rows if row["family_pass"])
        variants_total = sum(int(row["variants_total"]) for row in family_rows)
        variants_passed = sum(int(row["variants_passed"]) for row in family_rows)

        by_variant_summary = {}
        for variant_type, stats in sorted(by_variant_type.items()):
            total = int(stats["total"])
            passed = int(stats["passed"])
            by_variant_summary[variant_type] = {
                "passed": passed,
                "total": total,
                "pass_rate": round((passed / total) if total else 0.0, 4),
            }

        by_bucket_summary = {}
        for bucket, stats in sorted(by_bucket.items()):
            total = int(stats["total"])
            passed = int(stats["passed"])
            by_bucket_summary[bucket] = {
                "passed": passed,
                "total": total,
                "pass_rate": round((passed / total) if total else 0.0, 4),
            }

        worker_slang = by_variant_summary.get("worker_slang", {})
        report = {
            "schema_version": "paraphrase_eval_v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "contract_id": target_contract,
            "n_results": n_results,
            "bm25_only": bm25_only,
            "overall": {
                "families_passed": families_passed,
                "families_total": families_total,
                "family_pass_rate": round((families_passed / families_total) if families_total else 0.0, 4),
                "family_consistency_rate": round((families_passed / families_total) if families_total else 0.0, 4),
                "variants_passed": variants_passed,
                "variants_total": variants_total,
                "variant_pass_rate": round((variants_passed / variants_total) if variants_total else 0.0, 4),
                "retrieval_recall_at_k": round((variants_passed / variants_total) if variants_total else 0.0, 4),
                "worker_slang_pass_rate": worker_slang.get("pass_rate"),
            },
            "by_variant_type": by_variant_summary,
            "by_bucket": by_bucket_summary,
            "results": family_rows,
        }
        return report
    finally:
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        runtime_config.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_router_keyword
        runtime_config.CAG_ENABLE_RERANKER = original_reranker
        router_module.CAG_ENABLE_RERANKER = original_router_reranker


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "paraphrase_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate paraphrase/slang retrieval robustness.")
    parser.add_argument("--input", type=str, default=None, help="Path to paraphrase test JSON")
    parser.add_argument("--contract-id", type=str, default=None, help="Contract ID override")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument(
        "--allow-reranker",
        action="store_true",
        help="Use runtime reranker during evaluation (default is disabled for deterministic checks).",
    )
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(
        test_file=test_file,
        contract_id=args.contract_id,
        n_results=args.n_results,
        bm25_only=args.bm25_only,
        disable_reranker=not args.allow_reranker,
    )
    out_path = _write_report(report)

    overall = report.get("overall", {})
    print("=" * 72)
    print("KARL Paraphrase Robustness Evaluation")
    print("=" * 72)
    print(
        "Families: "
        f"{overall.get('families_passed', 0)}/{overall.get('families_total', 0)} "
        f"({overall.get('family_pass_rate', 0.0):.1%})"
    )
    print(
        "Variants: "
        f"{overall.get('variants_passed', 0)}/{overall.get('variants_total', 0)} "
        f"({overall.get('variant_pass_rate', 0.0):.1%})"
    )
    print(f"Worker slang pass rate: {overall.get('worker_slang_pass_rate')}")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
