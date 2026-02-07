"""
Query Interpreter - Deep semantic analysis for multi-angle retrieval.

This module enhances the existing hypothesis layer by:
1. Extracting structured query understanding (intent, entities, concepts)
2. Generating hypothetical document snippets (HyDE) for embedding search
3. Creating multiple search queries from different angles
4. Identifying explicit article references for direct lookup

The interpreter runs BEFORE retrieval to maximize recall across
different phrasings and vocabulary gaps.

Example:
    User: "A vendor is doing a major reset of the snack aisle. How many per year?"

    Interpretation:
    - intent: "find_limit"
    - key_concepts: ["vendor", "reset", "limit", "per year"]
    - hypothetical_answer: "Vendors are permitted to perform X resets per store per year..."
    - search_queries: [
        "vendor reset limit per year",
        "vendor work restrictions",
        "outside work bargaining unit"
      ]
    - fallback_articles: [2]  # Recognition/Work Jurisdiction
"""

import time
import re
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import (
    GEMINI_API_KEY,
    INTERPRETER_MODEL,
    MANIFESTS_DIR,
)

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class QueryInterpretation:
    """Structured interpretation of a user query."""

    # Original query
    original_query: str

    # Extracted understanding
    intent: str  # e.g., "find_limit", "understand_process", "get_definition"
    key_concepts: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # e.g., {"role": "vendor", "action": "reset"}

    # Hypothetical document snippets (HyDE)
    hypothetical_answers: List[str] = field(default_factory=list)

    # Multiple search angles
    search_queries: List[str] = field(default_factory=list)

    # Explicit article references (for direct lookup)
    explicit_articles: List[int] = field(default_factory=list)

    # Suggested contract sections/topics
    likely_sections: List[str] = field(default_factory=list)

    # Metadata
    latency_ms: float = 0
    success: bool = True
    error: Optional[str] = None
    model_used: str = ""


# System prompt for deep query interpretation
INTERPRETER_SYSTEM_PROMPT = """You are a union contract expert who helps interpret worker questions.

Your task: Analyze a worker's question and extract structured information to help find the answer in a collective bargaining agreement.

You must output valid JSON with this exact structure:
{
  "intent": "brief description of what they want to know",
  "key_concepts": ["list", "of", "main", "concepts"],
  "entities": {"type": "value"},
  "hypothetical_answers": [
    "What the contract text might say if it answers this question. Write 1-2 sentences that SOUND like contract language."
  ],
  "search_queries": [
    "2-3 different ways to search for this information",
    "using different vocabulary and angles"
  ],
  "likely_sections": ["Section titles that might contain the answer"],
  "explicit_articles": [list of article numbers if mentioned, empty otherwise]
}

CRITICAL RULES:
1. hypothetical_answers should sound like LEGAL CONTRACT TEXT, not casual speech
2. search_queries should use BOTH worker slang AND formal contract terms
3. If the query mentions "Article X" explicitly, include X in explicit_articles
4. Think about what SECTION TITLES in a union contract would contain this info

VOCABULARY GUIDE (worker term -> contract term):
- vendor/vendor work -> recognition, work jurisdiction, bargaining unit work
- reset/major reset -> vendor work, merchandising, stocking
- fired/canned -> discharge, termination
- write up -> discipline, warning
- break -> rest period, relief period
- overtime/OT -> overtime, premium pay
- floater -> personal holiday
- steward/rep -> union representative

Example input: "Can a vendor stock the shelves?"
Example output:
{
  "intent": "understand vendor work restrictions",
  "key_concepts": ["vendor", "stocking", "work restrictions", "bargaining unit"],
  "entities": {"actor": "vendor", "action": "stocking shelves"},
  "hypothetical_answers": [
    "Vendors shall be permitted to perform stocking and merchandising work under the following conditions...",
    "Work performed by vendors shall not displace bargaining unit employees..."
  ],
  "search_queries": [
    "vendor stocking work permitted",
    "recognition bargaining unit work jurisdiction",
    "outside work vendor restrictions"
  ],
  "likely_sections": ["Recognition", "Work Jurisdiction", "Vendor Work"],
  "explicit_articles": []
}"""


INTERPRETER_USER_PROMPT = """Analyze this worker question and output JSON:

Question: "{query}"

JSON:"""


class QueryInterpreter:
    """
    Deep semantic interpreter for union contract queries.

    Uses LLM reasoning to extract structured understanding and generate
    multiple search angles for better retrieval.
    """

    def __init__(self):
        """Initialize the query interpreter."""
        self._client = None
        self._article_titles = None

    def _ensure_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            if genai is None:
                return
            api_key = GEMINI_API_KEY
            if api_key:
                self._client = genai.Client(api_key=api_key)

    def _load_article_titles(self) -> Dict[int, str]:
        """Load article titles for reference extraction."""
        if self._article_titles is None:
            try:
                manifest_file = MANIFESTS_DIR / "safeway_pueblo_clerks_2022.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                        self._article_titles = {
                            int(k): v for k, v in
                            manifest.get('article_titles', {}).items()
                        }
                else:
                    self._article_titles = {}
            except Exception:
                self._article_titles = {}
        return self._article_titles

    def _extract_explicit_articles(self, query: str) -> List[int]:
        """Extract explicit article references from query."""
        articles = []

        # Pattern: "Article X" or "article X" or "Art. X" or "Art X"
        patterns = [
            r'article\s+(\d+)',
            r'art\.?\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                try:
                    articles.append(int(match))
                except ValueError:
                    pass

        return list(set(articles))

    def interpret(self, query: str) -> QueryInterpretation:
        """
        Interpret a user query to extract structured understanding.

        Args:
            query: The user's question

        Returns:
            QueryInterpretation with extracted information
        """
        start_time = time.time()

        # Always extract explicit article references (fast, no LLM needed)
        explicit_articles = self._extract_explicit_articles(query)

        try:
            self._ensure_client()

            if self._client is None:
                # Fallback: return basic interpretation without LLM
                return QueryInterpretation(
                    original_query=query,
                    intent="unknown",
                    key_concepts=query.lower().split()[:5],
                    search_queries=[query],
                    explicit_articles=explicit_articles,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error="Gemini client not available"
                )

            # Generate interpretation
            prompt = INTERPRETER_USER_PROMPT.format(query=query)

            response = self._client.models.generate_content(
                model=INTERPRETER_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=INTERPRETER_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                )
            )

            # Parse JSON response
            try:
                # Clean the response text
                response_text = response.text.strip()
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
                    response_text = re.sub(r'\n?```$', '', response_text)

                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                return QueryInterpretation(
                    original_query=query,
                    intent="parse_error",
                    search_queries=[query],
                    explicit_articles=explicit_articles,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"JSON parse error: {e}"
                )

            # Merge explicit articles from regex with LLM-detected ones
            llm_articles = data.get("explicit_articles", [])
            all_articles = list(set(explicit_articles + [
                int(a) for a in llm_articles if isinstance(a, (int, str)) and str(a).isdigit()
            ]))

            latency_ms = (time.time() - start_time) * 1000

            return QueryInterpretation(
                original_query=query,
                intent=data.get("intent", "unknown"),
                key_concepts=data.get("key_concepts", []),
                entities=data.get("entities", {}),
                hypothetical_answers=data.get("hypothetical_answers", []),
                search_queries=data.get("search_queries", [query]),
                likely_sections=data.get("likely_sections", []),
                explicit_articles=all_articles,
                latency_ms=latency_ms,
                success=True,
                model_used=INTERPRETER_MODEL
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return QueryInterpretation(
                original_query=query,
                intent="error",
                search_queries=[query],
                explicit_articles=explicit_articles,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )

    def get_all_search_queries(self, interpretation: QueryInterpretation) -> List[str]:
        """
        Get all search queries to try, including hypothetical answers.

        Returns queries in priority order:
        1. Original query
        2. Hypothetical answers (for HyDE-style matching)
        3. Alternative search queries
        """
        queries = [interpretation.original_query]

        # Add hypothetical answers for HyDE matching
        for hypo in interpretation.hypothetical_answers:
            if hypo and hypo not in queries:
                queries.append(hypo)

        # Add alternative search queries
        for sq in interpretation.search_queries:
            if sq and sq not in queries:
                queries.append(sq)

        return queries


# Module-level singleton
_interpreter = None


def get_interpreter() -> QueryInterpreter:
    """Get or create the interpreter singleton."""
    global _interpreter
    if _interpreter is None:
        _interpreter = QueryInterpreter()
    return _interpreter


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test the query interpreter on challenging cases."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Testing Query Interpreter")
    print("=" * 60)

    interpreter = QueryInterpreter()

    # Test cases that currently fail with the existing system
    test_queries = [
        "A vendor is seen doing a 'major reset' of the snack aisle. How many of these are they allowed per year?",
        "Can the store make me work on Sunday without extra pay?",
        "What happens if I get skipped for a shift?",
        "I've been here 6 months, do I get health insurance yet?",
        "My manager wants me to work through my lunch - is that allowed?",
        "Check Article 2 - it talks about vendors",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 40)

        result = interpreter.interpret(query)

        if result.success:
            print(f"Intent: {result.intent}")
            print(f"Key Concepts: {result.key_concepts}")
            print(f"Entities: {result.entities}")
            print(f"\nHypothetical Answers:")
            for i, hypo in enumerate(result.hypothetical_answers, 1):
                print(f"  {i}. {hypo[:100]}...")
            print(f"\nSearch Queries:")
            for i, sq in enumerate(result.search_queries, 1):
                print(f"  {i}. {sq}")
            print(f"\nLikely Sections: {result.likely_sections}")
            print(f"Explicit Articles: {result.explicit_articles}")
            print(f"\nLatency: {result.latency_ms:.0f}ms")
        else:
            print(f"FAILED: {result.error}")

    print("\n" + "=" * 60)
    print("All Search Queries for first test case:")
    result = interpreter.interpret(test_queries[0])
    all_queries = interpreter.get_all_search_queries(result)
    for i, q in enumerate(all_queries, 1):
        print(f"  {i}. {q}")


if __name__ == "__main__":
    main()
