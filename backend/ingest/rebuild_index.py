"""
Rebuild the full search index with concept-indexed metadata.

Phase 4 of CAG Architecture: After enrichment is complete, this script:
1. Builds the concept index (worker_questions, alternative_names aggregation)
2. Re-embeds all chunks into the vector store with new metadata
3. Validates the new index

Usage:
    python -m backend.ingest.rebuild_index
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import CHUNKS_DIR


def rebuild_index(chunks_file: Path = None, skip_vector_store: bool = False):
    """
    Rebuild the full search index.

    Args:
        chunks_file: Path to enriched chunks JSON
        skip_vector_store: If True, only build concept index
    """
    if chunks_file is None:
        chunks_file = CHUNKS_DIR / "contract_chunks_enriched.json"

    print(f"Loading enriched chunks from {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Step 1: Build concept index
    print("\n" + "=" * 60)
    print("Step 1: Building Concept Index")
    print("=" * 60)

    from backend.ingest.toc_index import build_concept_index
    concept_index = build_concept_index(
        chunks_path=chunks_file,
        output_path=CHUNKS_DIR / "concept_index.json"
    )

    # Step 2: Rebuild vector store (optional)
    if not skip_vector_store:
        print("\n" + "=" * 60)
        print("Step 2: Rebuilding Vector Store")
        print("=" * 60)

        from backend.retrieval.vector_store import ContractVectorStore
        vector_store = ContractVectorStore()

        print("Resetting collection...")
        vector_store.reset_collection()

        print("Adding chunks with new metadata...")
        added = vector_store.add_chunks(chunks)
        print(f"Added {added} chunks to vector store")

    # Step 3: Validate
    print("\n" + "=" * 60)
    print("Step 3: Validation")
    print("=" * 60)

    # Check concept index
    test_queries = [
        "When do I get a break?",
        "Can I get fired?",
        "How much vacation time do I get?",
    ]

    print("\nTesting concept matching:")
    for query in test_queries:
        articles = concept_index.find_articles_by_concept(query)
        questions = concept_index.find_articles_by_question(query)
        print(f"  '{query}'")
        print(f"    By concept: {articles[:3]}")
        print(f"    By question: {questions[:3]}")

    # Check a sample chunk for new fields
    print("\nSample chunk metadata:")
    sample = chunks[0]
    print(f"  chunk_id: {sample.get('chunk_id')}")
    print(f"  worker_questions: {sample.get('worker_questions', [])[:2]}")
    print(f"  alternative_names: {sample.get('alternative_names', [])[:5]}")

    print("\n" + "=" * 60)
    print("Index rebuild complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild search index with concept metadata")
    parser.add_argument("--input", type=str, default=None, help="Path to enriched chunks JSON")
    parser.add_argument("--skip-vector-store", action="store_true", help="Only build concept index")

    args = parser.parse_args()

    chunks_file = Path(args.input) if args.input else None
    rebuild_index(chunks_file=chunks_file, skip_vector_store=args.skip_vector_store)
