"""
Cross-contamination evaluator scaffold for multi-contract runtime.

In single-contract mode, this script reports SKIPPED unless --require-multi-contract is set.
"""

import argparse
from itertools import permutations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.contracts import list_contract_catalog
from backend.retrieval.hybrid_search import HybridSearcher


TEST_QUERIES = [
    "What is the overtime rate?",
    "How does the grievance filing deadline work?",
    "What are my Weingarten rights?",
    "What is the Sunday premium?",
]


def get_contract_ids() -> list[str]:
    return [c["contract_id"] for c in list_contract_catalog() if c.get("contract_id")]


def main():
    parser = argparse.ArgumentParser(description="Cross-contract contamination checks.")
    parser.add_argument("--require-multi-contract", action="store_true")
    args = parser.parse_args()

    contract_ids = get_contract_ids()
    if len(contract_ids) < 2:
        msg = f"Only {len(contract_ids)} contract manifest(s) found. Need >=2 for contamination evaluation."
        if args.require_multi_contract:
            print(f"[XX] {msg}")
            raise SystemExit(1)
        print(f"[SKIP] {msg}")
        raise SystemExit(0)

    # Offline-safe contamination checks: BM25-only retrieval with explicit weights
    # to avoid embedding-model downloads.
    try:
        searcher = HybridSearcher(vector_store=None)

        failures = []
        pairs = list(permutations(contract_ids, 2))
        for target_contract, _other in pairs:
            for query in TEST_QUERIES:
                chunks = searcher.search_to_chunks(
                    query=query,
                    n_results=5,
                    vector_weight=0.0,
                    keyword_weight=1.0,
                    contract_id=target_contract,
                )
                for chunk in chunks:
                    chunk_contract = chunk.get("contract_id")
                    if chunk_contract and chunk_contract != target_contract:
                        failures.append(
                            {
                                "target_contract": target_contract,
                                "query": query,
                                "chunk_id": chunk.get("chunk_id"),
                                "chunk_contract_id": chunk_contract,
                            }
                        )

        print("=" * 72)
        print("KARL Cross-Contamination Evaluation")
        print("=" * 72)
        print(f"Contracts tested: {contract_ids}")
        print(f"Pairs tested: {len(pairs)}")
        print(f"Queries per pair: {len(TEST_QUERIES)}")

        if failures:
            print(f"[XX] Contamination failures: {len(failures)}")
            for f in failures[:20]:
                print(
                    f"- target={f['target_contract']} chunk_contract={f['chunk_contract_id']} "
                    f"query='{f['query']}' chunk_id={f['chunk_id']}"
                )
            raise SystemExit(1)

        print("[OK] No cross-contract contamination detected.")
    finally:
        pass


if __name__ == "__main__":
    main()
