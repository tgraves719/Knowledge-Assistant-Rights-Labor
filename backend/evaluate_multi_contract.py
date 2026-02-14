"""
Multi-contract benchmark evaluator.

Runs contract-scoped retrieval checks from data/test_set/multi_contract_v2.json
and reports both overall and per-contract pass rates.
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

import backend.config as runtime_config
from backend.config import DATA_DIR
from backend.contracts import get_contract_catalog_entry
import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever, classify_intent


def _canonical_contract_id(contract_id: str) -> str:
    entry = get_contract_catalog_entry(contract_id)
    if entry and entry.get("contract_id"):
        return str(entry["contract_id"])
    return str(contract_id)


def _check_retrieval(expected: list[str], chunks: list[dict]) -> tuple[bool, list[str]]:
    import re

    expected = expected or []
    if not expected:
        return True, []

    retrieved_articles: list[str] = []
    for chunk in chunks:
        citation = str(chunk.get("citation") or "").lower()

        article_matches = re.findall(r"article\s*(\d+)", citation)
        for m in article_matches:
            ref = f"Article {m}"
            if ref not in retrieved_articles:
                retrieved_articles.append(ref)
        if "appendix" in citation and "Appendix A" not in retrieved_articles:
            retrieved_articles.append("Appendix A")
        if "letter of understanding" in citation and "Letter of Understanding" not in retrieved_articles:
            retrieved_articles.append("Letter of Understanding")

    for expected_article in expected:
        expected_lower = str(expected_article).lower()
        for retrieved in retrieved_articles:
            retrieved_lower = retrieved.lower()
            if expected_lower in retrieved_lower or retrieved_lower in expected_lower:
                return True, retrieved_articles
    return False, retrieved_articles


def run(
    test_file: Optional[Path] = None,
    n_results: int = 5,
    bm25_only: bool = False,
) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "multi_contract_v2.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    test_cases = payload.get("test_cases", [])

    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_keyword = runtime_config.HYBRID_KEYWORD_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_router_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    try:
        if bm25_only:
            runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
            runtime_config.HYBRID_KEYWORD_WEIGHT = 1.0
            router_module.HYBRID_VECTOR_WEIGHT = 0.0
            router_module.HYBRID_KEYWORD_WEIGHT = 1.0

        retriever = HybridRetriever(vector_store=None)

        rows = []
        by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
        for case in test_cases:
            contract_id = _canonical_contract_id(str(case.get("contract_id") or ""))
            question = str(case.get("question") or "")
            expected = list(case.get("expected_articles") or [])
            test_id = str(case.get("id") or "")

            intent = classify_intent(question, contract_id=contract_id)
            result = retriever.retrieve(
                query=question,
                intent=intent,
                n_results=n_results,
                use_hybrid=True,
                contract_id=contract_id,
            )
            chunks = result.get("chunks", [])
            passed, retrieved = _check_retrieval(expected, chunks)

            by_contract[contract_id]["total"] += 1
            if passed:
                by_contract[contract_id]["passed"] += 1

            rows.append(
                {
                    "id": test_id,
                    "contract_id": contract_id,
                    "question": question,
                    "expected_articles": expected,
                    "retrieved_articles": retrieved[:8],
                    "pass": passed,
                }
            )

        total = len(rows)
        passed = sum(1 for r in rows if r["pass"])
        by_contract_summary = {}
        for cid, stats in sorted(by_contract.items()):
            t = stats["total"]
            p = stats["passed"]
            by_contract_summary[cid] = {
                "passed": p,
                "total": t,
                "pass_rate": round((p / t) if t else 0.0, 4),
            }

        report = {
            "schema_version": "multi_contract_eval_v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_file),
            "bm25_only": bm25_only,
            "n_results": n_results,
            "overall": {
                "passed": passed,
                "total": total,
                "pass_rate": round((passed / total) if total else 0.0, 4),
            },
            "by_contract": by_contract_summary,
            "results": rows,
        }
        return report
    finally:
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        runtime_config.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_router_keyword


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "multi_contract_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-contract benchmark slice.")
    parser.add_argument("--input", type=str, default=None, help="Path to benchmark JSON")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(
        test_file=test_file,
        n_results=args.n_results,
        bm25_only=args.bm25_only,
    )
    out_path = _write_report(report)

    print("=" * 72)
    print("KARL Multi-Contract Evaluation")
    print("=" * 72)
    overall = report["overall"]
    print(f"Overall: {overall['passed']}/{overall['total']} ({overall['pass_rate']:.1%})")
    for cid, stats in report["by_contract"].items():
        print(f"- {cid}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
