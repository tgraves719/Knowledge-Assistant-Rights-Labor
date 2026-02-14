"""Check LOU chunking status and investigate Q40 (Minimum Wage LOU) failure."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHUNKS_DIR

CHUNKS_FILE = CHUNKS_DIR / "contract_chunks_enriched.json"


def main():
    print("=" * 80)
    print("LOU CHUNKING INVESTIGATION")
    print("=" * 80)
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Count LOU chunks
    lou_chunks = [c for c in chunks if c.get('doc_type') == 'lou']
    print(f"\nTotal LOU chunks: {len(lou_chunks)}")
    
    # List all LOU chunks
    print("\nAll LOU chunks:")
    for c in lou_chunks:
        print(f"  {c['chunk_id']}: {c.get('citation', 'N/A')}")
        if c.get('content'):
            preview = c['content'][:100].replace('\n', ' ')
            print(f"    Preview: {preview}...")
    
    # Search for Minimum Wage LOU
    print("\n" + "-" * 80)
    print("Minimum Wage LOU Search:")
    print("-" * 80)
    
    min_wage_keywords = ['minimum wage', 'minimum', '$0.40 above', 'operative minimum']
    min_wage_chunks = []
    
    for c in chunks:
        content_lower = (c.get('content') or '').lower()
        citation_lower = (c.get('citation') or '').lower()
        
        for kw in min_wage_keywords:
            if kw in content_lower or kw in citation_lower:
                min_wage_chunks.append(c)
                break
    
    print(f"Found {len(min_wage_chunks)} chunks with minimum wage keywords:")
    for c in min_wage_chunks:
        print(f"\n  {c['chunk_id']}")
        print(f"    doc_type: {c.get('doc_type', 'N/A')}")
        print(f"    citation: {c.get('citation', 'N/A')}")
        print(f"    article_num: {c.get('article_num', 'N/A')}")
        if c.get('content'):
            # Find the line with minimum wage
            lines = c['content'].split('\n')
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in min_wage_keywords):
                    print(f"    Relevant line {i+1}: {line[:150]}")
                    break
    
    # Check Q40 expected answer
    print("\n" + "-" * 80)
    print("Q40 Expected: Letter of Understanding")
    print("-" * 80)
    
    lou_citations = [c.get('citation', '') for c in lou_chunks]
    print(f"LOU citations found: {set(lou_citations)}")
    
    # Check if any LOU has minimum wage
    min_wage_lou = [c for c in lou_chunks if any(kw in (c.get('content') or '').lower() for kw in min_wage_keywords)]
    print(f"\nLOU chunks with minimum wage: {len(min_wage_lou)}")
    for c in min_wage_lou:
        print(f"  {c['chunk_id']}: {c.get('citation', 'N/A')}")
    
    # Check recent benchmark results
    print("\n" + "-" * 80)
    print("Recent Benchmark Results (Q40):")
    print("-" * 80)
    
    results_file = PROJECT_ROOT / "data" / "test_set" / "comprehensive_results.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        q40_results = [r for r in results.get('results', []) if r.get('test_id') == 40]
        if q40_results:
            q40 = q40_results[0]
            print(f"  Question: {q40.get('question', 'N/A')[:80]}...")
            print(f"  Expected: {q40.get('expected', [])}")
            print(f"  Retrieved: {q40.get('retrieved', [])}")
            print(f"  Passed: {q40.get('passed', False)}")
        else:
            print("  Q40 not found in results")
    else:
        print("  Results file not found")


if __name__ == "__main__":
    main()


