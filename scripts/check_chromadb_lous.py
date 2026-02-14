"""Check if LOU chunks are in ChromaDB."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.vector_store import ContractVectorStore

print("=" * 80)
print("CHROMADB LOU CHUNK CHECK")
print("=" * 80)

store = ContractVectorStore()

# Search for LOU chunks by doc_type
print("\nSearching for chunks with doc_type='lou'...")
lou_results = store.search(
    query="minimum wage letter of understanding",
    n_results=20,
    doc_type="lou"  # Filter by doc_type
)

print(f"\nFound {len(lou_results)} LOU chunks in ChromaDB:")
for i, chunk in enumerate(lou_results[:10], 1):
    print(f"{i}. {chunk.get('chunk_id', 'N/A')}")
    print(f"   citation: {chunk.get('citation', 'N/A')}")
    print(f"   doc_type: {chunk.get('doc_type', 'N/A')}")
    print(f"   similarity: {chunk.get('similarity', 'N/A')}")
    print()

# Also check without filter
print("\n" + "-" * 80)
print("Searching WITHOUT doc_type filter (all chunks):")
print("-" * 80)
all_results = store.search(
    query="minimum wage letter of understanding",
    n_results=20
)

lou_in_all = [c for c in all_results if c.get('doc_type') == 'lou']
print(f"\nLOU chunks in top 20 (no filter): {len(lou_in_all)}")
for c in lou_in_all:
    print(f"  {c.get('chunk_id')}: {c.get('citation')} (similarity: {c.get('similarity', 'N/A')})")


