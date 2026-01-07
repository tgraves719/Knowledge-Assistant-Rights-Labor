"""
Hypothesis Layer - Pre-retrieval reasoning using LLM to predict section titles.

This "Rosetta Stone" component bridges the vocabulary gap between user queries
and legal contract terminology by hypothesizing which section titles would
contain the answer.

Example:
    User: "When do I get a break?"
    Hypothesized Titles: ["Relief Periods", "Rest Intervals", "Meal Periods"]

These titles are used to:
1. Append to search query for better BM25/vector matching
2. Boost chunks whose article_title matches any hypothesis
"""

import time
from typing import List, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import (
    HYPOTHESIS_MODEL,
    HYPOTHESIS_MAX_TITLES,
    HYPOTHESIS_TIMEOUT_MS,
    TITLE_BOOST_SCORE,
    CAG_ENABLE_HYPOTHESIS_LAYER,
    CAG_ENABLE_TITLE_BOOSTING,
    GEMINI_API_KEY,
)


@dataclass
class HypothesisResult:
    """Result of hypothesis generation."""
    hypothesized_titles: List[str]
    query_expansion: str  # Original query + hypothesized terms
    latency_ms: float
    model_used: str
    success: bool
    error: Optional[str] = None


# System prompt for hypothesis generation
HYPOTHESIS_SYSTEM_PROMPT = """You are a labor law expert who specializes in union collective bargaining agreements.

Your task: Given a worker's question, predict which section TITLES in a union contract would contain the answer.

Union contracts use formal legal terminology. Workers often use informal language.

Examples of vocabulary mapping:
- "break" -> "Relief Periods", "Rest Periods", "Meal Periods"
- "fired" -> "Discharge", "Termination", "Just Cause"
- "pay raise" -> "Wage Progression", "Step Increases", "Wages"
- "schedule" -> "Hours of Work", "Weekly Schedule", "Scheduling"
- "laid off" -> "Layoff", "Reduction in Force", "Recall Rights"
- "overtime" -> "Overtime", "Premium Pay", "Hours of Work"
- "vacation" -> "Vacations", "Vacation Pay", "Time Off"
- "sick" -> "Sick Leave", "Health and Welfare", "Leaves of Absence"
- "union rep" -> "Stewards", "Union Representation", "Weingarten Rights"
- "grievance" -> "Grievance Procedure", "Dispute Resolution", "Arbitration"

Output ONLY the section titles, one per line, no numbers or bullets.
Output exactly {max_titles} titles, ordered by likelihood of containing the answer."""


HYPOTHESIS_USER_PROMPT = """Worker's question: "{query}"

List {max_titles} likely section titles that would contain this answer:"""


class HypothesisGenerator:
    """
    Generates hypothesized section titles using LLM reasoning.

    Uses Gemini 2.0 Flash for fast inference with good reasoning capability.
    This is the "brain" of the Rosetta Stone architecture.
    """

    def __init__(self):
        """Initialize the hypothesis generator."""
        self._genai = None
        self._model = None

    def _ensure_client(self):
        """Lazy-load the Gemini client."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                api_key = GEMINI_API_KEY
                if api_key:
                    genai.configure(api_key=api_key)
                    self._genai = genai
                    self._model = genai.GenerativeModel(
                        model_name=HYPOTHESIS_MODEL,
                        system_instruction=HYPOTHESIS_SYSTEM_PROMPT.format(
                            max_titles=HYPOTHESIS_MAX_TITLES
                        )
                    )
            except ImportError:
                pass

    def generate_sync(self, query: str) -> HypothesisResult:
        """
        Generate hypothesized section titles synchronously.

        Args:
            query: User's question

        Returns:
            HypothesisResult with titles and metadata
        """
        if not CAG_ENABLE_HYPOTHESIS_LAYER:
            return HypothesisResult(
                hypothesized_titles=[],
                query_expansion=query,
                latency_ms=0,
                model_used="disabled",
                success=False,
                error="Hypothesis layer disabled"
            )

        start_time = time.time()

        try:
            self._ensure_client()

            if self._model is None:
                return HypothesisResult(
                    hypothesized_titles=[],
                    query_expansion=query,
                    latency_ms=0,
                    model_used="none",
                    success=False,
                    error="Gemini client not available"
                )

            # Generate hypotheses
            prompt = HYPOTHESIS_USER_PROMPT.format(
                query=query,
                max_titles=HYPOTHESIS_MAX_TITLES
            )

            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temp for more focused output
                    "max_output_tokens": 100,
                },
                request_options={"timeout": 30}  # 30 second timeout for hypothesis
            )

            # Parse response: split by newlines, clean up
            raw_titles = response.text.strip().split('\n')
            titles = [
                t.strip().strip('-').strip('*').strip().strip('•').strip()
                for t in raw_titles
                if t.strip() and len(t.strip()) > 2
            ][:HYPOTHESIS_MAX_TITLES]

            # Build expanded query by appending hypothesized titles
            expansion_terms = " ".join(titles)
            query_expansion = f"{query} ({expansion_terms})"

            latency_ms = (time.time() - start_time) * 1000

            return HypothesisResult(
                hypothesized_titles=titles,
                query_expansion=query_expansion,
                latency_ms=latency_ms,
                model_used=HYPOTHESIS_MODEL,
                success=True
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HypothesisResult(
                hypothesized_titles=[],
                query_expansion=query,
                latency_ms=latency_ms,
                model_used=HYPOTHESIS_MODEL,
                success=False,
                error=str(e)
            )


def apply_title_boosting(
    chunks: List[dict],
    hypothesized_titles: List[str],
    boost_score: float = None
) -> List[dict]:
    """
    Apply score boost to chunks whose article_title matches hypothesized titles.

    This is the second key function of the Rosetta Stone architecture:
    once we know what titles to look for, we boost matching chunks.

    Args:
        chunks: List of retrieved chunks with 'similarity' or 'rrf_score' field
        hypothesized_titles: List of hypothesized section titles
        boost_score: Score to add for matches (default from config)

    Returns:
        Chunks with boosted scores, re-sorted by score
    """
    if not CAG_ENABLE_TITLE_BOOSTING or not hypothesized_titles:
        return chunks

    if boost_score is None:
        boost_score = TITLE_BOOST_SCORE

    if not chunks:
        return chunks

    # Normalize hypothesized titles for matching
    normalized_hypotheses = [h.lower().strip() for h in hypothesized_titles]

    boosted_chunks = []
    for chunk in chunks:
        chunk_copy = dict(chunk)

        # Get article title from chunk metadata
        article_title = chunk.get('article_title', '').lower()

        # Check for fuzzy match with any hypothesis
        matched = False
        for hypothesis in normalized_hypotheses:
            # Check if hypothesis words appear in article title
            hypothesis_words = [w for w in hypothesis.split() if len(w) > 2]
            if hypothesis_words and all(word in article_title for word in hypothesis_words):
                matched = True
                break
            # Also check if article title words appear in hypothesis
            title_words = [w for w in article_title.split() if len(w) > 2]
            if len(title_words) >= 2 and all(word in hypothesis for word in title_words[:2]):
                matched = True
                break
            # Direct substring match for common cases
            if hypothesis in article_title or article_title in hypothesis:
                matched = True
                break

        if matched:
            # Boost the similarity score
            score_field = 'similarity' if 'similarity' in chunk_copy else 'rrf_score'
            chunk_copy[score_field] = chunk_copy.get(score_field, 0) + boost_score
            chunk_copy['hypothesis_matched'] = True

        boosted_chunks.append(chunk_copy)

    # Re-sort by score
    score_field = 'similarity' if 'similarity' in boosted_chunks[0] else 'rrf_score'
    boosted_chunks.sort(key=lambda x: x.get(score_field, 0), reverse=True)

    return boosted_chunks


# Module-level singleton for reuse
_hypothesis_generator = None


def get_hypothesis_generator() -> HypothesisGenerator:
    """Get or create the hypothesis generator singleton."""
    global _hypothesis_generator
    if _hypothesis_generator is None:
        _hypothesis_generator = HypothesisGenerator()
    return _hypothesis_generator


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test hypothesis generation on key vocabulary-mismatch cases."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Testing Hypothesis Generator (Rosetta Stone Layer)")
    print("=" * 60)

    generator = HypothesisGenerator()

    test_queries = [
        "When do I get a break?",
        "I was just fired. What should I do?",
        "How much overtime pay do I get?",
        "Can I take a day off for my kid's graduation?",
        "What are my rights if I'm being harassed?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = generator.generate_sync(query)

        if result.success:
            print(f"  Hypothesized Titles:")
            for title in result.hypothesized_titles:
                print(f"    - {title}")
            print(f"  Latency: {result.latency_ms:.0f}ms")
            print(f"  Expanded Query: {result.query_expansion[:80]}...")
        else:
            print(f"  Failed: {result.error}")

        print("-" * 40)


if __name__ == "__main__":
    main()
