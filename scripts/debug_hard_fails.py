"""
Debug Hard Fail Questions - Inspect chunks for questions that fail across all ablation modes.

Hard Fails:
- Q3: Probationary period (Article 26)
- Q40: Minimum wage Letter of Understanding
- Q11: Multi-hop promotion (Article 8, 9, Appendix A)
- Q14: Pharmacy Tech Sunday OT (Article 13, 12, Appendix A)
- Q24: Vacation hours for wage progression (Article 8, 42)
- Q37: Flood/store closure (Article 48, 58)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHUNKS_DIR

CHUNKS_FILE = CHUNKS_DIR / "contract_chunks_enriched.json"


def search_chunks(query_terms: list, article_num: int = None):
    """Search chunks by keywords and optionally article number."""
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    results = []
    for chunk in chunks:
        if article_num and chunk.get('article_num') != article_num:
            continue
        
        content_lower = chunk.get('content', '').lower()
        citation_lower = chunk.get('citation', '').lower()
        
        # Check if any query term appears
        matches = [term for term in query_terms if term.lower() in content_lower or term.lower() in citation_lower]
        if matches:
            results.append({
                'chunk_id': chunk.get('chunk_id'),
                'citation': chunk.get('citation'),
                'article_num': chunk.get('article_num'),
                'section_num': chunk.get('section_num'),
                'content_preview': chunk.get('content', '')[:150],
                'matched_terms': matches
            })
    
    return results


def main():
    print("=" * 80)
    print("HARD FAIL DEBUGGING - Chunk Inspection")
    print("=" * 80)
    
    # Q3: Probationary period (Article 26)
    print("\n[Q3] Probationary Period - Expected: Article 26")
    print("Question: 'How many hours constitute the probationary period for new employees?'")
    print("-" * 80)
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    art26_all = [c for c in chunks if c.get('article_num') == 26]
    print(f"Total Article 26 chunks: {len(art26_all)}")
    
    # Search for hours/520
    art26_hours = [c for c in art26_all if '520' in c.get('content', '') or ('hours' in c.get('content', '').lower() and 'probation' in c.get('content', '').lower())]
    print(f"Chunks with '520' or 'hours' + 'probation': {len(art26_hours)}")
    for c in art26_hours:
        print(f"  {c['chunk_id']}: {c.get('citation', 'N/A')}")
        print(f"    Content: {c.get('content', '')[:200]}...")
        print()
    
    # Show all Article 26 chunks
    print("All Article 26 chunks:")
    for c in art26_all:
        print(f"  {c['chunk_id']}: {c.get('citation', 'N/A')} - {c.get('content', '')[:100]}...")
    
    # Q40: Minimum wage Letter of Understanding
    print("\n[Q40] Minimum Wage Letter of Understanding")
    print("-" * 80)
    lou = search_chunks(['minimum wage', 'letter of understanding', 'lou', '$0.40 above'], article_num=None)
    if lou:
        print(f"Found {len(lou)} chunks with minimum wage LOU keywords:")
        for c in lou[:5]:
            print(f"  {c['chunk_id']}: {c['citation']} (Article {c['article_num']})")
            print(f"    Matched: {c['matched_terms']}")
            print(f"    Preview: {c['content_preview']}...")
            print()
    else:
        print("  ❌ NO CHUNKS FOUND with minimum wage LOU keywords!")
    
    # Q11: Multi-hop promotion (Article 8, 9, Appendix A)
    print("\n[Q11] Multi-hop Promotion - Expected: Article 8, 9, Appendix A")
    print("-" * 80)
    art8 = search_chunks(['basket hours', 'promoted', 'classification', 'wage progression'], article_num=8)
    art9 = search_chunks(['basket hours', 'promoted', 'classification'], article_num=9)
    
    print(f"Article 8 chunks: {len(art8)}")
    for c in art8[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    print(f"\nArticle 9 chunks: {len(art9)}")
    for c in art9[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    # Check for Appendix A
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    appendix = [c for c in chunks if 'appendix' in (c.get('citation') or '').lower() or 'appendix' in (c.get('article_title') or '').lower()]
    print(f"\nAppendix A chunks: {len(appendix)}")
    for c in appendix[:3]:
        print(f"  {c['chunk_id']}: {c.get('citation', 'N/A')} - {c.get('content', '')[:100]}...")
    
    # Q14: Pharmacy Tech Sunday OT
    print("\n[Q14] Pharmacy Tech Sunday OT - Expected: Article 13, 12, Appendix A")
    print("-" * 80)
    art13 = search_chunks(['pharmacy technician', 'pharmacy tech', '2.0x', 'double time', 'sunday'], article_num=13)
    art12_ot = search_chunks(['overtime', 'pharmacy'], article_num=12)
    
    print(f"Article 13 (Pharmacy Tech): {len(art13)} chunks")
    for c in art13[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    print(f"\nArticle 12 (Overtime + Pharmacy): {len(art12_ot)} chunks")
    for c in art12_ot[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    # Q24: Vacation hours for wage progression
    print("\n[Q24] Vacation Hours for Wage Progression - Expected: Article 8, 42")
    print("-" * 80)
    art8_vac = search_chunks(['vacation', 'hours worked', 'actually work', 'wage progression'], article_num=8)
    art42_vac = search_chunks(['vacation', 'hours worked', 'pension', 'article 44'], article_num=42)
    
    print(f"Article 8 (Vacation + Wage): {len(art8_vac)} chunks")
    for c in art8_vac[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    print(f"\nArticle 42 (Vacation + Pension): {len(art42_vac)} chunks")
    for c in art42_vac[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    # Q37: Flood/store closure
    print("\n[Q37] Flood/Store Closure - Expected: Article 48, 58")
    print("-" * 80)
    art48_flood = search_chunks(['flood', 'store closing', 'closure', 'act of god', 'natural disaster'], article_num=48)
    art58_flood = search_chunks(['flood', 'act of god', 'natural disaster', 'effects bargaining'], article_num=58)
    
    print(f"Article 48 (Flood/Closure): {len(art48_flood)} chunks")
    for c in art48_flood[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    print(f"\nArticle 58 (Act of God): {len(art58_flood)} chunks")
    for c in art58_flood[:3]:
        print(f"  {c['chunk_id']}: {c['citation']} - {c['content_preview'][:100]}...")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

