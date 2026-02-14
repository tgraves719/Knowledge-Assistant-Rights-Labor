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

from backend.config import CHUNKS_DIR, ONTOLOGIES_DIR, MANIFESTS_DIR
from backend.chunk_files import resolve_chunk_file
from backend.contracts import list_contract_catalog
from backend.ingest.language_lexicon import (
    apply_deterministic_language_enrichment,
    build_language_lexicon,
    save_language_lexicon,
)


def _discover_contract_chunk_inputs(contract_id: str = None, explicit_chunks_file: Path = None) -> list[tuple[str, Path]]:
    """
    Resolve contract->chunk-file inputs for concept/vector index rebuild.

    Returns:
        list of (contract_id, chunks_path)
    """
    if explicit_chunks_file is not None:
        inferred_contract_id = contract_id
        if inferred_contract_id is None:
            try:
                with open(explicit_chunks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    inferred_contract_id = str(data[0].get("contract_id") or "unknown_contract")
            except Exception:
                inferred_contract_id = "unknown_contract"
        return [(inferred_contract_id, explicit_chunks_file)]

    if contract_id:
        chunks_path = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
        if chunks_path is None:
            raise FileNotFoundError(f"No chunks artifact found for contract_id={contract_id}")
        return [(contract_id, chunks_path)]

    contract_inputs: list[tuple[str, Path]] = []
    for row in list_contract_catalog():
        cid = row.get("contract_id")
        if not cid:
            continue
        path = resolve_chunk_file(contract_id=cid, allow_shared_fallback=True)
        if path and path.exists():
            contract_inputs.append((cid, path))

    if contract_inputs:
        return contract_inputs

    # Legacy single-contract fallback only.
    legacy_path = resolve_chunk_file(contract_id=None, allow_shared_fallback=True)
    if legacy_path is None:
        raise FileNotFoundError("No chunks artifact found for rebuild_index")
    return [("legacy_shared", legacy_path)]


def rebuild_index(
    chunks_file: Path = None,
    skip_vector_store: bool = False,
    contract_id: str = None,
):
    """
    Rebuild the full search index.

    Args:
        chunks_file: Path to enriched chunks JSON
        skip_vector_store: If True, only build concept index
        contract_id: Optional contract id (recommended in multi-contract mode)
    """
    contract_inputs = _discover_contract_chunk_inputs(
        contract_id=contract_id,
        explicit_chunks_file=chunks_file,
    )

    all_chunks: list[dict] = []
    built_indexes: dict[str, object] = {}

    # Step 1: Build concept index
    print("\n" + "=" * 60)
    print("Step 1: Building Concept Index")
    print("=" * 60)

    from backend.ingest.toc_index import build_concept_index
    for cid, cpath in contract_inputs:
        print(f"\nLoading chunks for {cid} from {cpath}")
        with open(cpath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks for {cid}")

        manifest = {}
        manifest_path = MANIFESTS_DIR / f"{cid}.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
            except Exception:
                manifest = {}

        chunks, lang_stats = apply_deterministic_language_enrichment(
            chunks,
            contract_id=cid,
            manifest=manifest,
        )
        print(
            f"Language enrichment [{cid}]: "
            f"alts={lang_stats.get('chunks_with_alternative_names')}/"
            f"{lang_stats.get('chunk_count')} "
            f"questions={lang_stats.get('chunks_with_worker_questions')}/"
            f"{lang_stats.get('chunk_count')}"
        )

        # Persist repaired metadata so concept index + runtime stay consistent.
        with open(cpath, "w", encoding="utf-8") as wf:
            json.dump(chunks, wf, indent=2, ensure_ascii=False)

        all_chunks.extend(chunks)

        out_path = CHUNKS_DIR / f"concept_index_{cid}.json"
        built_indexes[cid] = build_concept_index(
            chunks_path=cpath,
            output_path=out_path,
            manifest=manifest,
        )

        lexicon = build_language_lexicon(
            chunks,
            contract_id=cid,
            manifest=manifest,
        )
        lex_path = ONTOLOGIES_DIR / f"language_lexicon_{cid}.json"
        save_language_lexicon(lex_path, lexicon)
        print(f"Saved language lexicon: {lex_path}")

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
        added = vector_store.add_chunks(all_chunks)
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
        first_index = next(iter(built_indexes.values())) if built_indexes else None
        if first_index is None:
            break
        articles = first_index.find_articles_by_concept(query)
        questions = first_index.find_articles_by_question(query)
        print(f"  '{query}'")
        print(f"    By concept: {articles[:3]}")
        print(f"    By question: {questions[:3]}")

    # Check a sample chunk for new fields
    if all_chunks:
        print("\nSample chunk metadata:")
        sample = all_chunks[0]
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
    parser.add_argument("--contract-id", type=str, default=None, help="Contract ID for contract-scoped rebuild")
    parser.add_argument("--skip-vector-store", action="store_true", help="Only build concept index")

    args = parser.parse_args()

    chunks_file = Path(args.input) if args.input else None
    rebuild_index(
        chunks_file=chunks_file,
        skip_vector_store=args.skip_vector_store,
        contract_id=args.contract_id,
    )
