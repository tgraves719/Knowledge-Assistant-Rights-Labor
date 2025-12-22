"""
End-to-End Generation Test for Karl RAG System.

Tests the full pipeline:
1. Query classification
2. Retrieval (hybrid search)
3. Prompt building
4. LLM generation
5. Citation verification

Requires: GEMINI_API_KEY environment variable
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from backend.config import GEMINI_API_KEY, LLM_MODEL
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore
from backend.generation.prompts import build_prompt
from backend.generation.verifier import verify_response, format_response_with_sources


def check_api_key():
    """Check if API key is configured."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("\n" + "=" * 70)
        print("ERROR: No Gemini API key found!")
        print("=" * 70)
        print("\nTo test LLM generation, you need to:")
        print("1. Get a free API key from: https://aistudio.google.com/app/apikey")
        print("2. Create a .env file in the project root with:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("\nOr set it as an environment variable:")
        print("   $env:GEMINI_API_KEY='your_api_key_here'  (PowerShell)")
        print("=" * 70 + "\n")
        return None
    return api_key


def init_llm(api_key):
    """Initialize the Gemini LLM."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
        return None


def generate_response(genai, question: str, system_prompt: str) -> str:
    """Generate LLM response."""
    try:
        model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=system_prompt
        )
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"ERROR: {e}"


def run_generation_test():
    """Run end-to-end generation tests."""
    print("\n" + "=" * 70)
    print("KARL RAG SYSTEM - End-to-End Generation Test")
    print("=" * 70)
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        return False
    
    # Initialize LLM
    print("\n[1] Initializing LLM...")
    genai = init_llm(api_key)
    if not genai:
        return False
    print(f"    Model: {LLM_MODEL}")
    
    # Initialize retrieval
    print("\n[2] Initializing RAG system...")
    vector_store = ContractVectorStore()
    retriever = HybridRetriever(vector_store)
    print(f"    Loaded {vector_store.count()} chunks")
    
    # Test questions - one from each level
    test_questions = [
        {
            "question": "What is the Sunday premium pay rate?",
            "level": "L1 - Direct",
            "expected_cite": "Article 13",
        },
        {
            "question": "How many hours do I need to work to get a raise?",
            "level": "L2 - Calculation",
            "expected_cite": "Article 8 or Appendix A",
        },
        {
            "question": "I was just called into a meeting with my manager about my performance. What are my rights?",
            "level": "High-Stakes",
            "expected_cite": "Article 45 (Weingarten)",
            "requires_escalation": True,
        },
        {
            "question": "What holidays do we get off?",
            "level": "L1 - Direct",
            "expected_cite": "Article 16",
        },
        {
            "question": "How does vacation scheduling work?",
            "level": "L3 - Multi-Article",
            "expected_cite": "Article 17",
        },
    ]
    
    print("\n[3] Running Generation Tests...")
    print("-" * 70)
    
    results = []
    
    for i, tc in enumerate(test_questions, 1):
        question = tc["question"]
        expected = tc.get("expected_cite", "")
        requires_esc = tc.get("requires_escalation", False)
        
        print(f"\n--- Test {i}: {tc['level']} ---")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        
        # Classify intent
        intent = classify_intent(question)
        print(f"Intent: {intent.intent_type} (escalation: {intent.requires_escalation})")
        
        # Retrieve context
        retrieval = retriever.retrieve(question, intent, n_results=5)
        chunks = retrieval["chunks"]
        wage_info = retrieval.get("wage_info")
        
        print(f"Retrieved {len(chunks)} chunks:")
        for c in chunks[:3]:
            print(f"  - {c['citation']}")
        
        # Build prompt
        query_expansions = retrieval.get("query_expansions", [])
        system_prompt = build_prompt(
            query=question,
            chunks=chunks,
            wage_info=wage_info,
            requires_escalation=requires_esc or intent.requires_escalation,
            query_expansions=query_expansions
        )
        
        # Generate response
        print("\nGenerating response...")
        answer = generate_response(genai, question, system_prompt)
        
        if answer.startswith("ERROR:"):
            print(f"  {answer}")
            results.append({"passed": False, "error": answer})
            continue
        
        # Verify response
        verification = verify_response(
            answer,
            chunks,
            requires_escalation=requires_esc or intent.requires_escalation
        )
        
        # Format result
        formatted = format_response_with_sources(answer, chunks, wage_info)
        
        print(f"\n--- Generated Answer ---")
        print(answer[:500] + ("..." if len(answer) > 500 else ""))
        print(f"\n--- Verification ---")
        print(f"Valid: {verification.is_valid}")
        print(f"Citations found: {verification.citations_found}")
        print(f"Escalation present: {verification.escalation_present}")
        print(f"Confidence: {verification.confidence:.2f}")
        if verification.issues:
            print(f"Issues: {verification.issues}")
        
        results.append({
            "passed": verification.is_valid,
            "citations": verification.citations_found,
            "confidence": verification.confidence
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Average Confidence: {sum(r.get('confidence', 0) for r in results)/total:.2f}")
    
    return passed == total


def run_single_query(question: str):
    """Run a single query for interactive testing."""
    api_key = check_api_key()
    if not api_key:
        return
    
    genai = init_llm(api_key)
    if not genai:
        return
    
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Initialize
    vector_store = ContractVectorStore()
    retriever = HybridRetriever(vector_store)
    
    # Process
    intent = classify_intent(question)
    retrieval = retriever.retrieve(question, intent, n_results=5)
    chunks = retrieval["chunks"]
    wage_info = retrieval.get("wage_info")
    
    query_expansions = retrieval.get("query_expansions", [])
    if query_expansions:
        print(f"Query expanded: {query_expansions}")
    
    system_prompt = build_prompt(
        query=question,
        chunks=chunks,
        wage_info=wage_info,
        requires_escalation=intent.requires_escalation,
        query_expansions=query_expansions
    )
    
    answer = generate_response(genai, question, system_prompt)
    
    print(f"\n{answer}")
    print("-" * 50)
    
    verification = verify_response(answer, chunks, intent.requires_escalation)
    print(f"\nCitations: {verification.citations_found}")
    print(f"Valid: {verification.is_valid}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        run_single_query(question)
    else:
        # Full test suite
        run_generation_test()

