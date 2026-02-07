"""
Comprehensive Generation Evaluation for Karl RAG System.

Evaluates:
- Answer accuracy (does the answer match expected?)
- Citation quality (are citations correct and present?)
- Escalation compliance (high-stakes queries get escalation?)
- Grounding (no hallucinations?)

Uses the comprehensive 55-question test set.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from backend.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore
from backend.generation.prompts import build_prompt
from backend.generation.verifier import (
    verify_response, 
    extract_citations,
    has_escalation_language,
    has_uncertainty_language
)


@dataclass
class GenerationResult:
    """Result of a generation evaluation."""
    test_id: int
    level: int
    category: str
    question: str
    answer: str
    citations: list
    expected_articles: list
    retrieval_passed: bool
    citation_correct: bool
    escalation_correct: bool
    has_hallucination: bool
    verification_confidence: float
    generation_time: float


def init_llm():
    """Initialize Gemini LLM."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("\nERROR: No GEMINI_API_KEY found!")
        print("Set it in .env or as environment variable.")
        return None

    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        print("ERROR: google-genai not installed")
        return None


def generate_response(client, question: str, system_prompt: str) -> str:
    """Generate LLM response with retry."""
    try:
        from google import genai
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        return response.text
    except Exception as e:
        return f"ERROR: {e}"


def check_citation_overlap(answer_citations: list, expected_articles: list) -> bool:
    """Check if any expected article appears in answer citations."""
    if not expected_articles:
        return True  # No expectation = pass
    
    for citation in answer_citations:
        citation_lower = citation.lower()
        for expected in expected_articles:
            if expected.lower().replace(" ", "") in citation_lower.replace(" ", ""):
                return True
    return False


def run_generation_eval(limit: int = None, levels: list = None):
    """Run full generation evaluation."""
    print("\n" + "=" * 70)
    print("KARL RAG - COMPREHENSIVE GENERATION EVALUATION")
    print("=" * 70)
    
    # Initialize LLM
    genai = init_llm()
    if not genai:
        return None
    print(f"LLM: {LLM_MODEL}")
    
    # Initialize RAG
    print("Initializing RAG system...")
    vs = ContractVectorStore()
    retriever = HybridRetriever(vs)
    print(f"Loaded {vs.count()} chunks")
    
    # Load test set
    test_file = DATA_DIR / "test_set" / "comprehensive_test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    
    # Filter by levels if specified
    if levels:
        test_cases = [tc for tc in test_cases if tc['level'] in levels]
    
    # Limit for quick testing
    if limit:
        test_cases = test_cases[:limit]
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print("-" * 70)
    
    results = []
    level_stats = defaultdict(lambda: {
        'total': 0, 
        'citation_correct': 0, 
        'escalation_correct': 0,
        'no_hallucination': 0,
        'high_confidence': 0
    })
    
    for i, tc in enumerate(test_cases, 1):
        question = tc['question']
        expected = tc.get('expected_articles', [])
        level = tc['level']
        category = tc['category']
        should_limit = tc.get('should_identify_limitation', False)
        
        print(f"\n[{i}/{len(test_cases)}] L{level} - {category}")
        print(f"Q: {question[:60]}...")
        
        # Retrieval
        intent = classify_intent(question)
        retrieval = retriever.retrieve(question, intent, n_results=5)
        chunks = retrieval["chunks"]
        wage_info = retrieval.get("wage_info")
        
        # Build prompt
        system_prompt = build_prompt(
            query=question,
            chunks=chunks,
            wage_info=wage_info,
            requires_escalation=intent.requires_escalation
        )
        
        # Generate
        start_time = time.time()
        answer = generate_response(genai, question, system_prompt)
        gen_time = time.time() - start_time
        
        if answer.startswith("ERROR:"):
            print(f"  ERROR: {answer}")
            # Add small delay to avoid rate limiting
            time.sleep(1)
            continue
        
        # Verify
        verification = verify_response(
            answer,
            chunks,
            requires_escalation=intent.requires_escalation
        )
        
        # Extract metrics
        answer_citations = extract_citations(answer)
        citation_correct = check_citation_overlap(answer_citations, expected)
        
        # For "impossible" questions, check if system correctly limits
        if should_limit:
            citation_correct = has_uncertainty_language(answer)
        
        # Check escalation
        needs_escalation = intent.requires_escalation or tc.get('requires_escalation', False)
        escalation_correct = not needs_escalation or has_escalation_language(answer)
        
        # Hallucination check (citations not in retrieved chunks)
        has_hallucination = len(verification.issues) > 0 and "Hallucinated" in str(verification.issues)
        
        result = GenerationResult(
            test_id=tc['id'],
            level=level,
            category=category,
            question=question,
            answer=answer[:300],
            citations=answer_citations,
            expected_articles=expected,
            retrieval_passed=citation_correct,
            citation_correct=citation_correct,
            escalation_correct=escalation_correct,
            has_hallucination=has_hallucination,
            verification_confidence=verification.confidence,
            generation_time=gen_time
        )
        results.append(result)
        
        # Update stats
        stats = level_stats[level]
        stats['total'] += 1
        if citation_correct:
            stats['citation_correct'] += 1
        if escalation_correct:
            stats['escalation_correct'] += 1
        if not has_hallucination:
            stats['no_hallucination'] += 1
        if verification.confidence >= 0.8:
            stats['high_confidence'] += 1
        
        # Print result
        status = "PASS" if citation_correct and escalation_correct else "FAIL"
        print(f"  {status} | Citations: {answer_citations[:2]} | Conf: {verification.confidence:.2f} | Time: {gen_time:.1f}s")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 70)
    
    total = len(results)
    if total == 0:
        print("No results to evaluate")
        return None
    
    overall_citation = sum(1 for r in results if r.citation_correct) / total
    overall_escalation = sum(1 for r in results if r.escalation_correct) / total
    overall_no_hallucination = sum(1 for r in results if not r.has_hallucination) / total
    avg_confidence = sum(r.verification_confidence for r in results) / total
    avg_time = sum(r.generation_time for r in results) / total
    
    print(f"\nOverall Metrics ({total} tests):")
    print(f"  Citation Accuracy:    {100*overall_citation:.1f}%")
    print(f"  Escalation Accuracy:  {100*overall_escalation:.1f}%")
    print(f"  No Hallucinations:    {100*overall_no_hallucination:.1f}%")
    print(f"  Avg Confidence:       {avg_confidence:.2f}")
    print(f"  Avg Gen Time:         {avg_time:.2f}s")
    
    print("\nBy Level:")
    level_names = {
        1: "Direct Retrieval",
        2: "Calculation", 
        3: "Multi-Article",
        4: "Edge Cases",
        5: "Ambiguity",
        6: "Temporal",
        7: "Procedural",
        8: "External Law",
        9: "Systemic",
        10: "Impossible"
    }
    
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        if stats['total'] == 0:
            continue
        cite_rate = 100 * stats['citation_correct'] / stats['total']
        esc_rate = 100 * stats['escalation_correct'] / stats['total']
        name = level_names.get(level, f"Level {level}")
        print(f"  L{level:2d} | Citation: {cite_rate:5.1f}% | Escalation: {esc_rate:5.1f}% | {name}")
    
    # Save results
    output_file = DATA_DIR / "test_set" / "generation_results.json"
    results_data = {
        'summary': {
            'total': total,
            'citation_accuracy': overall_citation,
            'escalation_accuracy': overall_escalation,
            'no_hallucination_rate': overall_no_hallucination,
            'avg_confidence': avg_confidence,
            'avg_generation_time': avg_time
        },
        'results': [
            {
                'test_id': r.test_id,
                'level': r.level,
                'category': r.category,
                'question': r.question,
                'answer_preview': r.answer[:200],
                'citations': r.citations,
                'expected': r.expected_articles,
                'citation_correct': r.citation_correct,
                'escalation_correct': r.escalation_correct,
                'has_hallucination': r.has_hallucination,
                'confidence': r.verification_confidence,
                'gen_time': r.generation_time
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    return results_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    parser.add_argument("--limit", type=int, help="Limit number of tests")
    parser.add_argument("--levels", type=int, nargs="+", help="Only test specific levels")
    
    args = parser.parse_args()
    
    run_generation_eval(limit=args.limit, levels=args.levels)

