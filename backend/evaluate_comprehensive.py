"""
Comprehensive Evaluation Script - Tests RAG retrieval against 55-question test suite.

Evaluates:
- Retrieval accuracy (finding correct articles)
- Level-by-level performance
- Identification of limitations for Level 10 questions
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from backend.config import DATA_DIR
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore


@dataclass
class TestResult:
    test_id: int
    level: int
    category: str
    question: str
    expected_articles: list
    retrieved_articles: list
    retrieval_passed: bool
    notes: str


def check_retrieval(expected: list, chunks: list) -> tuple[bool, list]:
    """Check if expected articles were retrieved."""
    if not expected:
        return True, []  # No expectation = pass
    
    retrieved_articles = []
    for chunk in chunks:
        citation = chunk.get('citation', '').lower()
        article_title = chunk.get('article_title', '').lower()
        
        # Extract article references
        import re
        article_matches = re.findall(r'article\s*(\d+)', citation)
        for match in article_matches:
            ref = f"Article {match}"
            if ref not in retrieved_articles:
                retrieved_articles.append(ref)
        
        if 'letter of understanding' in citation.lower():
            if 'Letter of Understanding' not in retrieved_articles:
                retrieved_articles.append('Letter of Understanding')
        
        if 'appendix' in citation.lower():
            if 'Appendix A' not in retrieved_articles:
                retrieved_articles.append('Appendix A')
    
    # Check if any expected article was found
    found = False
    for expected_article in expected:
        expected_lower = expected_article.lower()
        for retrieved in retrieved_articles:
            if expected_lower in retrieved.lower() or retrieved.lower() in expected_lower:
                found = True
                break
        if found:
            break
    
    return found, retrieved_articles


def run_comprehensive_eval():
    """Run the comprehensive evaluation."""
    print("Loading comprehensive test set...")
    test_file = DATA_DIR / "test_set" / "comprehensive_test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    
    print("Initializing RAG system...")
    vs = ContractVectorStore()
    retriever = HybridRetriever(vs)
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print("=" * 80)
    
    results = []
    level_stats = defaultdict(lambda: {'passed': 0, 'total': 0})
    
    for i, tc in enumerate(test_cases, 1):
        question = tc['question']
        expected = tc.get('expected_articles', [])
        level = tc['level']
        category = tc['category']
        
        # Run retrieval
        intent = classify_intent(question)
        result = retriever.retrieve(question, intent, n_results=5)
        chunks = result['chunks']
        
        # Check if wage lookup is needed
        if tc.get('requires_wage_lookup') and 'Appendix A' in expected:
            # Wage queries should hit the wage tables
            if result.get('wage_info'):
                # Add Appendix A to retrieved if wage lookup succeeded
                chunks = chunks + [{'citation': 'Appendix A'}]
        
        # Evaluate
        passed, retrieved = check_retrieval(expected, chunks)
        
        # For Level 10 "impossible" questions, check limitation identification
        if tc.get('should_identify_limitation'):
            # For these, we check if the system would identify limitations
            # In retrieval-only mode, we check if results are sparse/irrelevant
            if not expected and len(retrieved) == 0:
                passed = True
                notes = "Correctly found no relevant content"
            elif not expected:
                passed = False
                notes = f"Should identify limitation, got: {retrieved[:2]}"
            else:
                notes = ""
        else:
            notes = "" if passed else f"Expected {expected}, got {retrieved[:3]}"
        
        test_result = TestResult(
            test_id=tc['id'],
            level=level,
            category=category,
            question=question,
            expected_articles=expected,
            retrieved_articles=retrieved,
            retrieval_passed=passed,
            notes=notes
        )
        results.append(test_result)
        
        # Track stats
        level_stats[level]['total'] += 1
        if passed:
            level_stats[level]['passed'] += 1
        
        # Print progress
        status = "PASS" if passed else "FAIL"
        status_emoji = "[OK]" if passed else "[XX]"
        print(f"  {status_emoji} L{level} Q{tc['id']}: {category}")
        if not passed and notes:
            print(f"       {notes[:70]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)
    
    total_passed = sum(1 for r in results if r.retrieval_passed)
    total = len(results)
    
    print(f"\nOverall Retrieval Accuracy: {total_passed}/{total} ({100*total_passed/total:.1f}%)")
    
    print("\nBy Level:")
    level_names = {
        1: "Direct Retrieval (Basic)",
        2: "Calculation (Intermediate)", 
        3: "Multi-Article (Advanced)",
        4: "Edge Cases (Expert)",
        5: "Ambiguity (Expert+)",
        6: "Temporal (Expert++)",
        7: "Procedural (Expert++)",
        8: "External Law (Expert+++)",
        9: "Systemic (Expert+++)",
        10: "Impossible/Boundary"
    }
    
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        rate = 100 * stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        bar_len = int(rate / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        name = level_names.get(level, f"Level {level}")
        print(f"  L{level:2d} [{bar}] {stats['passed']}/{stats['total']} ({rate:.0f}%) - {name}")
    
    # Show failures by level
    print("\n" + "-" * 80)
    print("Failed Tests by Level:")
    for level in sorted(level_stats.keys()):
        failures = [r for r in results if r.level == level and not r.retrieval_passed]
        if failures:
            print(f"\n  Level {level} ({len(failures)} failures):")
            for f in failures[:3]:  # Show first 3 per level
                print(f"    Q{f.test_id}: {f.question[:60]}...")
                print(f"         Expected: {f.expected_articles}")
                print(f"         Got: {f.retrieved_articles[:3]}")
    
    # Save results
    output_file = DATA_DIR / "test_set" / "comprehensive_results.json"
    results_data = {
        'summary': {
            'total': total,
            'passed': total_passed,
            'accuracy': total_passed / total,
            'by_level': {str(k): v for k, v in level_stats.items()}
        },
        'results': [
            {
                'test_id': r.test_id,
                'level': r.level,
                'category': r.category,
                'question': r.question,
                'expected': r.expected_articles,
                'retrieved': r.retrieved_articles,
                'passed': r.retrieval_passed,
                'notes': r.notes
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results_data


if __name__ == "__main__":
    run_comprehensive_eval()

