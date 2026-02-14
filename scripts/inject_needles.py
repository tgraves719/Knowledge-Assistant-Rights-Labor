"""
Needle Injection Script — Injects/removes synthetic KARL_NEEDLE_xxx chunks
into ChromaDB and the BM25 chunks JSON for the Needle-in-a-Haystack test.

Usage:
    # Inject needles (must run BEFORE needle_test evaluation)
    python scripts/inject_needles.py --inject

    # Verify needles are present
    python scripts/inject_needles.py --verify

    # Remove needles (run AFTER needle_test evaluation)
    python scripts/inject_needles.py --remove

    # Dry run (show what would be injected without writing)
    python scripts/inject_needles.py --inject --dry-run
"""

import json
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR, CHUNKS_DIR, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, EMBEDDING_MODEL, CONTRACT_ID,
)

NEEDLE_TEST_FILE = DATA_DIR / "test_set" / "needle_test.json"
NEEDLE_ID_PREFIX = "needle_synthetic_"

# BM25 chunks files — we patch whichever one the HybridSearcher would load
BM25_CHUNKS_FILES = [
    CHUNKS_DIR / "contract_chunks_enriched.json",
    CHUNKS_DIR / "contract_chunks_smart.json",
    CHUNKS_DIR / "contract_chunks.json",
]


def _get_bm25_target() -> Path:
    """Return the chunks file the HybridSearcher would actually load."""
    for f in BM25_CHUNKS_FILES:
        if f.exists():
            return f
    raise FileNotFoundError("No chunks JSON file found in data/chunks/")


def _load_needle_definitions() -> list:
    """Load the 5 synthetic needle definitions from needle_test.json."""
    with open(NEEDLE_TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["synthetic_needles"]


def _needle_to_chunk(needle: dict) -> dict:
    """Convert a needle definition into a full chunk dict matching the enriched schema."""
    meta = needle["injection_metadata"]
    return {
        "chunk_id": f"{NEEDLE_ID_PREFIX}{needle['needle_id'].lower()}",
        "contract_id": CONTRACT_ID,
        "doc_type": "cba",
        "article_num": meta["article_num"],
        "article_title": f"Synthetic Needle ({needle['needle_id']})",
        "section_num": meta["section_num"],
        "subsection": None,
        "subsection_title": None,
        "citation": meta["citation"],
        "parent_context": f"Synthetic injection for needle test — {needle['needle_id']}",
        "content": needle["injection_content"],
        "content_with_tables": needle["injection_content"],
        "char_count": len(needle["injection_content"]),
        "applies_to": ["all"],
        "topics": ["needle_test"],
        "cross_references": [],
        "summary": f"Synthetic needle: {needle['needle_id']}",
        "is_definition": False,
        "is_exception": False,
        "hire_date_sensitive": False,
        "is_high_stakes": False,
        "worker_questions": [],
        "alternative_names": [],
    }


# ========================================================================
# INJECT
# ========================================================================

def inject_needles(dry_run: bool = False):
    """Inject synthetic needle chunks into ChromaDB and BM25 JSON."""
    needles = _load_needle_definitions()
    chunks = [_needle_to_chunk(n) for n in needles]

    print(f"Preparing {len(chunks)} needle chunks for injection...")
    for c in chunks:
        print(f"  {c['chunk_id']:45s}  {c['citation']}")

    if dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # ---- 1. Inject into ChromaDB ----
    print("\n[1/2] Injecting into ChromaDB...")
    from backend.retrieval.vector_store import ContractVectorStore

    vs = ContractVectorStore()
    existing_count = vs.count()
    print(f"  Collection '{vs.collection_name}' has {existing_count} documents before injection.")

    # Check for already-injected needles
    try:
        existing = vs.collection.get(
            ids=[c["chunk_id"] for c in chunks],
            include=[],
        )
        if existing and existing["ids"]:
            print(f"  WARNING: {len(existing['ids'])} needle(s) already present — will upsert (overwrite).")
    except Exception:
        pass  # collection.get may fail if IDs not found; that's fine

    added = vs.add_chunks(chunks)
    print(f"  Upserted {added} needle chunks into ChromaDB. New total: {vs.count()}")

    # ---- 2. Inject into BM25 JSON ----
    print("\n[2/2] Injecting into BM25 chunks JSON...")
    bm25_file = _get_bm25_target()
    print(f"  Target file: {bm25_file}")

    # Backup first
    backup_path = bm25_file.with_suffix(".json.pre_needle_backup")
    if not backup_path.exists():
        shutil.copy2(bm25_file, backup_path)
        print(f"  Backup saved: {backup_path.name}")
    else:
        print(f"  Backup already exists: {backup_path.name} (skipping)")

    with open(bm25_file, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    # Remove any old needles (idempotent)
    existing_chunks = [c for c in existing_chunks if not c["chunk_id"].startswith(NEEDLE_ID_PREFIX)]

    # Add new needles
    existing_chunks.extend(chunks)

    with open(bm25_file, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)

    print(f"  Wrote {len(existing_chunks)} chunks (including {len(chunks)} needles) to {bm25_file.name}")

    print(f"\n{'='*60}")
    print("INJECTION COMPLETE. You may now run the needle test:")
    print("  python -m backend.evaluate_comprehensive --test-set needle")
    print(f"{'='*60}")


# ========================================================================
# REMOVE
# ========================================================================

def remove_needles(dry_run: bool = False):
    """Remove synthetic needle chunks from ChromaDB and BM25 JSON."""
    needles = _load_needle_definitions()
    needle_ids = [f"{NEEDLE_ID_PREFIX}{n['needle_id'].lower()}" for n in needles]

    print(f"Removing {len(needle_ids)} needle chunks...")
    for nid in needle_ids:
        print(f"  {nid}")

    if dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # ---- 1. Remove from ChromaDB ----
    print("\n[1/2] Removing from ChromaDB...")
    from backend.retrieval.vector_store import ContractVectorStore

    vs = ContractVectorStore()
    before = vs.count()

    try:
        vs.collection.delete(ids=needle_ids)
        after = vs.count()
        print(f"  Removed {before - after} documents. Collection now has {after} documents.")
    except Exception as e:
        print(f"  Warning: Could not delete from ChromaDB: {e}")

    # ---- 2. Remove from BM25 JSON ----
    print("\n[2/2] Removing from BM25 chunks JSON...")
    bm25_file = _get_bm25_target()

    with open(bm25_file, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    before_count = len(existing_chunks)
    cleaned_chunks = [c for c in existing_chunks if not c["chunk_id"].startswith(NEEDLE_ID_PREFIX)]
    removed_count = before_count - len(cleaned_chunks)

    with open(bm25_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, indent=2, ensure_ascii=False)

    print(f"  Removed {removed_count} needle chunks. File now has {len(cleaned_chunks)} chunks.")

    # Restore backup if it exists
    backup_path = bm25_file.with_suffix(".json.pre_needle_backup")
    if backup_path.exists():
        print(f"  Note: Original backup still at {backup_path.name} (safe to delete manually)")

    print(f"\n{'='*60}")
    print("REMOVAL COMPLETE. Needles purged from both ChromaDB and BM25.")
    print(f"{'='*60}")


# ========================================================================
# VERIFY
# ========================================================================

def verify_needles():
    """Verify that needle chunks are present in both ChromaDB and BM25."""
    needles = _load_needle_definitions()
    needle_ids = [f"{NEEDLE_ID_PREFIX}{n['needle_id'].lower()}" for n in needles]

    print(f"Verifying {len(needle_ids)} needle chunks...\n")

    # ---- 1. Check ChromaDB ----
    print("[ChromaDB]")
    from backend.retrieval.vector_store import ContractVectorStore

    vs = ContractVectorStore()
    chroma_found = 0
    for nid in needle_ids:
        chunk = vs.get_chunk(nid)
        if chunk:
            chroma_found += 1
            content_preview = chunk.get("content", "")[:80]
            print(f"  [OK] {nid} — {content_preview}...")
        else:
            print(f"  [MISSING] {nid}")

    # ---- 2. Check BM25 JSON ----
    print("\n[BM25 Chunks JSON]")
    bm25_file = _get_bm25_target()
    with open(bm25_file, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    bm25_ids = {c["chunk_id"] for c in existing_chunks}
    bm25_found = 0
    for nid in needle_ids:
        if nid in bm25_ids:
            bm25_found += 1
            print(f"  [OK] {nid}")
        else:
            print(f"  [MISSING] {nid}")

    # ---- Summary ----
    total = len(needle_ids)
    print(f"\n{'='*60}")
    print(f"ChromaDB:  {chroma_found}/{total} needles present")
    print(f"BM25 JSON: {bm25_found}/{total} needles present")

    if chroma_found == total and bm25_found == total:
        print("STATUS: ALL NEEDLES INJECTED — ready to test")
    else:
        print("STATUS: NEEDLES MISSING — run: python scripts/inject_needles.py --inject")
    print(f"{'='*60}")

    # ---- 3. Quick retrieval smoke test ----
    if chroma_found == total:
        print("\n[Retrieval Smoke Test]")
        test_query = "KARL_NEEDLE_001 retention bonus Senior Night Stockers"
        results = vs.search(query=test_query, n_results=3)
        print(f"  Query: \"{test_query}\"")
        print(f"  Top {len(results)} results:")
        for r in results:
            cid = r.get("chunk_id", "?")
            cit = r.get("citation", "?")
            sim = r.get("similarity", 0)
            print(f"    {cid:45s} ({cit}) sim={sim:.3f}")


# ========================================================================
# CLI
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inject/remove synthetic KARL_NEEDLE_xxx chunks for the needle test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inject_needles.py --inject            # Add needles
  python scripts/inject_needles.py --inject --dry-run  # Preview only
  python scripts/inject_needles.py --verify            # Check presence
  python scripts/inject_needles.py --remove            # Clean up after test
"""
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inject", action="store_true", help="Inject needle chunks into ChromaDB and BM25")
    group.add_argument("--remove", action="store_true", help="Remove needle chunks from ChromaDB and BM25")
    group.add_argument("--verify", action="store_true", help="Verify needle chunks are present")

    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing (inject/remove only)")

    args = parser.parse_args()

    if args.inject:
        inject_needles(dry_run=args.dry_run)
    elif args.remove:
        remove_needles(dry_run=args.dry_run)
    elif args.verify:
        verify_needles()


if __name__ == "__main__":
    main()


