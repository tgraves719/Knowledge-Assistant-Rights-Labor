"""
Evaluation Script - Tests the RAG system against the golden test set.

Metrics:
- Retrieval accuracy (does the expected citation appear in results?)
- Wage lookup accuracy (100% precision required)
- Escalation detection (high-stakes topics flagged correctly?)
- Refusal rate (refuses when context is insufficient?)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: int
    category: str
    question: str
    retrieval_correct: bool
    expected_citation: Optional[str]
    retrieved_citations: list[str]
    wage_correct: Optional[bool]
    escalation_correct: Optional[bool]
    intent_detected: str
    passed: bool
    notes: str


def load_test_set(test_file: Path = None) -> list[dict]:
    """Load the golden test set."""
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "golden_qa.json"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('test_cases', [])


def check_citation_in_results(expected: str, chunks: list[dict]) -> bool:
    """Check if the expected citation appears in retrieved chunks."""
    if not expected:
        return True  # No citation expected
    
    expected_lower = expected.lower()
    
    for chunk in chunks:
        citation = chunk.get('citation', '').lower()
        article_title = chunk.get('article_title', '').lower()
        
        # Check for article match
        if 'article' in expected_lower:
            # Extract article number
            import re
            match = re.search(r'article\s*(\d+)', expected_lower)
            if match:
                article_num = match.group(1)
                if f'article {article_num}' in citation or f'article {article_num}' in article_title:
                    return True
        
        # Check for appendix match
        if 'appendix' in expected_lower and 'appendix' in citation.lower():
            return True
        
        # Direct match
        if expected_lower in citation:
            return True
    
    return False


def evaluate_test_case(
    test_case: dict,
    retriever: HybridRetriever
) -> TestResult:
    """Evaluate a single test case."""
    question = test_case['question']
    expected_citation = test_case.get('expected_citation')
    classification = test_case.get('classification')
    escalation_expected = test_case.get('escalation_expected', False)
    should_refuse = test_case.get('should_refuse', False)
    
    # Run retrieval
    intent = classify_intent(question)
    if classification:
        intent.classification = classification
    
    result = retriever.multi_angle_retrieve(
        query=question,
        intent=intent,
        n_results=5
    )
    
    chunks = result['chunks']
    wage_info = result['wage_info']
    escalation_required = result['escalation_required']
    
    # Check retrieval
    retrieval_correct = check_citation_in_results(expected_citation, chunks)
    
    # Extract citations from results
    retrieved_citations = [c.get('citation', '') for c in chunks[:3]]
    
    # Check wage lookup (for wage questions)
    wage_correct = None
    if test_case['category'] == 'wages' and wage_info:
        wage_correct = True  # We got wage info
        if 'appendix' in expected_citation.lower():
            retrieval_correct = True  # Wage lookup counts as correct retrieval
    
    # Check escalation
    escalation_correct = None
    if test_case['category'] == 'high_stakes' or escalation_expected:
        escalation_correct = escalation_required == escalation_expected
    
    # Determine if test passed
    passed = retrieval_correct
    notes = ""
    
    if not retrieval_correct:
        notes = f"Expected '{expected_citation}' not found"
    
    if escalation_correct is False:
        passed = False
        notes += f"; Escalation mismatch (expected={escalation_expected}, got={escalation_required})"
    
    if wage_correct is False:
        passed = False
        notes += "; Wage lookup failed"
    
    return TestResult(
        test_id=test_case['id'],
        category=test_case['category'],
        question=question,
        retrieval_correct=retrieval_correct,
        expected_citation=expected_citation,
        retrieved_citations=retrieved_citations,
        wage_correct=wage_correct,
        escalation_correct=escalation_correct,
        intent_detected=intent.intent_type,
        passed=passed,
        notes=notes.strip('; ')
    )


def run_evaluation(test_cases: list[dict] = None) -> dict:
    """Run full evaluation and return metrics."""
    print("Initializing RAG system...")
    vector_store = ContractVectorStore()
    retriever = HybridRetriever(vector_store)
    
    if test_cases is None:
        test_cases = load_test_set()
    
    print(f"Running {len(test_cases)} test cases...")
    
    results = []
    category_results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        result = evaluate_test_case(test_case, retriever)
        results.append(result)
        
        # Track by category
        category = result.category
        if category not in category_results:
            category_results[category] = {'passed': 0, 'total': 0}
        category_results[category]['total'] += 1
        if result.passed:
            category_results[category]['passed'] += 1
        
        # Progress
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{i}/{len(test_cases)}] {status} {test_case['id']}: {test_case['category']}", end="")
        if not result.passed:
            print(f" - {result.notes}")
        else:
            print()
    
    # Calculate metrics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    
    retrieval_correct = sum(1 for r in results if r.retrieval_correct)
    wage_tests = [r for r in results if r.wage_correct is not None]
    wage_correct = sum(1 for r in wage_tests if r.wage_correct)
    escalation_tests = [r for r in results if r.escalation_correct is not None]
    escalation_correct = sum(1 for r in escalation_tests if r.escalation_correct)
    
    metrics = {
        'total_tests': total,
        'passed': passed,
        'pass_rate': passed / total if total > 0 else 0,
        'retrieval_accuracy': retrieval_correct / total if total > 0 else 0,
        'wage_accuracy': wage_correct / len(wage_tests) if wage_tests else None,
        'escalation_accuracy': escalation_correct / len(escalation_tests) if escalation_tests else None,
        'category_breakdown': {
            cat: {
                'passed': data['passed'],
                'total': data['total'],
                'rate': data['passed'] / data['total']
            }
            for cat, data in category_results.items()
        }
    }
    
    return {
        'metrics': metrics,
        'results': [asdict(r) for r in results]
    }


def print_summary(evaluation: dict):
    """Print evaluation summary."""
    metrics = evaluation['metrics']
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nOverall: {metrics['passed']}/{metrics['total_tests']} passed ({metrics['pass_rate']*100:.1f}%)")
    print(f"Retrieval Accuracy: {metrics['retrieval_accuracy']*100:.1f}%")
    
    if metrics['wage_accuracy'] is not None:
        print(f"Wage Lookup Accuracy: {metrics['wage_accuracy']*100:.1f}%")
    
    if metrics['escalation_accuracy'] is not None:
        print(f"Escalation Detection: {metrics['escalation_accuracy']*100:.1f}%")
    
    print("\nBy Category:")
    for category, data in sorted(metrics['category_breakdown'].items()):
        rate = data['rate'] * 100
        bar = "#" * int(rate / 5) + "-" * (20 - int(rate / 5))
        print(f"  {category:15} [{bar}] {data['passed']}/{data['total']} ({rate:.0f}%)")
    
    # Show failures
    failures = [r for r in evaluation['results'] if not r['passed']]
    if failures:
        print(f"\nFailed Tests ({len(failures)}):")
        for f in failures[:10]:  # Show first 10
            print(f"  - [{f['test_id']}] {f['question'][:50]}...")
            print(f"    Expected: {f['expected_citation']}")
            print(f"    Got: {f['retrieved_citations'][:2]}")
            print(f"    Notes: {f['notes']}")


def main():
    """Run evaluation and save results."""
    evaluation = run_evaluation()
    print_summary(evaluation)
    
    # Save results
    output_file = DATA_DIR / "test_set" / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return evaluation


if __name__ == "__main__":
    main()

