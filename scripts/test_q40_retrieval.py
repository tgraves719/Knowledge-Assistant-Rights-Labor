"""Test Q40 retrieval to understand why LOU chunks aren't being found."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.router import HybridRetriever

# Q40 question
query = "The Colorado minimum wage increases to $15.00/hour mid-contract. The 'Minimum Wage' letter of understanding requires starting rates '$0.40 above the operative minimum wage.' Must wage tables be immediately recalculated, or does this only apply at contract ratification?"

print("=" * 80)
print("Q40 RETRIEVAL TEST")
print("=" * 80)
print(f"\nQuery: {query}\n")

retriever = HybridRetriever()

# Test retrieval
result = retriever.retrieve(query, n_results=10)

print(f"Retrieved {len(result['chunks'])} chunks:\n")

for i, chunk in enumerate(result['chunks'], 1):
    print(f"{i}. {chunk.get('chunk_id', 'N/A')}")
    print(f"   doc_type: {chunk.get('doc_type', 'N/A')}")
    print(f"   citation: {chunk.get('citation', 'N/A')}")
    print(f"   similarity: {chunk.get('similarity', 'N/A')}")
    
    # Check if it's a LOU chunk
    if chunk.get('doc_type') == 'lou':
        print(f"   ✅ LOU CHUNK")
        content_preview = (chunk.get('content') or '')[:200].replace('\n', ' ')
        print(f"   Preview: {content_preview}...")
    else:
        # Check if it mentions minimum wage
        content_lower = (chunk.get('content') or '').lower()
        if 'minimum wage' in content_lower or '$0.40' in content_lower:
            print(f"   [WARNING] Mentions minimum wage but is CBA chunk")
    
    print()

# Count LOU vs CBA chunks
lou_count = sum(1 for c in result['chunks'] if c.get('doc_type') == 'lou')
cba_count = sum(1 for c in result['chunks'] if c.get('doc_type') == 'cba')
print(f"\nSummary: {lou_count} LOU chunks, {cba_count} CBA chunks in top 10")

