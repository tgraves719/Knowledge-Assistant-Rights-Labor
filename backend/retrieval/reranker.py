"""
LLM Reranker - Scores retrieved chunks by semantic relevance.

This module uses Gemini Flash to rerank retrieved chunks based on how well
they actually answer the user's question. It runs AFTER multi-angle retrieval
merges results and BEFORE full article expansion.

The reranker:
1. Takes retrieved chunks and the original query
2. Asks Gemini to score each chunk 1-10 for relevance
3. Combines LLM score with original similarity score
4. Returns chunks reordered by combined score

On any failure, returns original chunks unchanged (graceful degradation).
"""

import time
import re
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import (
    GEMINI_API_KEY,
    CAG_ENABLE_RERANKER,
    RERANKER_MODEL,
    RERANKER_TIMEOUT_MS,
    RERANKER_ORIGINAL_WEIGHT,
    RERANKER_LLM_WEIGHT,
    RERANKER_MAX_CHUNKS,
    RERANKER_CONTENT_TRUNCATE,
)

logger = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class RerankerResult:
    """Result of LLM reranking."""

    chunks: List[dict]
    latency_ms: float
    model_used: str
    success: bool
    error: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)  # chunk_id -> LLM score
    position_changes: int = 0  # How many chunks moved position


# System prompt for relevance scoring
RERANKER_SYSTEM_PROMPT = """You are a relevance scorer for union contract document retrieval.

Your task: Given a worker's question and contract excerpts, score each excerpt's relevance to answering the question.

SCORING SCALE (1-10):
- 10: Directly and completely answers the question
- 8-9: Highly relevant, contains key information needed
- 6-7: Partially relevant, provides useful context
- 4-5: Tangentially related, mentions related topics
- 1-3: Not relevant to this specific question

SCORING TIPS:
- A definition section is relevant if the question uses that term
- Procedural sections are relevant for "how do I" questions
- Exception clauses are relevant for eligibility/limit questions
- Look for SEMANTIC relevance, not just keyword matches
- Consider what the worker actually needs to know

Output valid JSON mapping chunk IDs to scores. Example:
{"0": 8, "1": 5, "2": 9}

Score EVERY chunk. Do not skip any."""


RERANKER_USER_PROMPT = """Worker's question: "{query}"
{interpretation_context}
Contract excerpts to score:

{formatted_chunks}

JSON scores (chunk ID -> relevance 1-10):"""


class LLMReranker:
    """
    Reranks retrieved chunks using LLM-based relevance scoring.

    Uses Gemini Flash to score how well each chunk answers the question,
    then combines with original similarity score for final ranking.
    """

    def __init__(self):
        """Initialize the reranker."""
        self._client = None

    def _ensure_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            if genai is None:
                logger.warning("google-genai not installed")
                return
            api_key = GEMINI_API_KEY
            if api_key:
                self._client = genai.Client(api_key=api_key)

    def _format_chunks(self, chunks: List[dict]) -> str:
        """Format chunks for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            # Truncate content to limit prompt size
            if len(content) > RERANKER_CONTENT_TRUNCATE:
                content = content[:RERANKER_CONTENT_TRUNCATE] + "..."

            citation = chunk.get('citation', f'Chunk {i}')
            formatted.append(f"---\nID: {i}\nCitation: {citation}\nContent: {content}\n---")

        return "\n".join(formatted)

    def _build_interpretation_context(self, interpretation) -> str:
        """Build context string from query interpretation if available."""
        if interpretation is None:
            return ""

        parts = []
        if hasattr(interpretation, 'intent') and interpretation.intent:
            parts.append(f"Intent: {interpretation.intent}")
        if hasattr(interpretation, 'key_concepts') and interpretation.key_concepts:
            parts.append(f"Key concepts: {', '.join(interpretation.key_concepts[:5])}")

        if parts:
            return "\n" + "\n".join(parts) + "\n"
        return ""

    def _parse_scores(self, response_text: str, num_chunks: int) -> Dict[str, int]:
        """Parse LLM response into chunk scores."""
        try:
            # Clean response
            text = response_text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            # Extract JSON object — 2.5 models may prepend thinking text
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                text = text[first_brace:last_brace + 1]

            scores = json.loads(text)

            # Normalize keys to strings and validate scores
            result = {}
            for key, value in scores.items():
                try:
                    idx = str(key)
                    score = int(value)
                    # Clamp to valid range
                    score = max(1, min(10, score))
                    result[idx] = score
                except (ValueError, TypeError):
                    continue

            # Fill in missing scores with default (5)
            for i in range(num_chunks):
                if str(i) not in result:
                    result[str(i)] = 5
                    logger.debug(f"Chunk {i} missing score, defaulting to 5")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reranker JSON: {e}")
            # Return default scores
            return {str(i): 5 for i in range(num_chunks)}

    def _compute_final_scores(
        self,
        chunks: List[dict],
        llm_scores: Dict[str, int]
    ) -> List[dict]:
        """Combine original scores with LLM scores and reorder."""
        # Track original positions for metrics
        original_order = [
            chunk.get('chunk_id', chunk.get('citation', str(i)))
            for i, chunk in enumerate(chunks)
        ]

        for i, chunk in enumerate(chunks):
            original_score = chunk.get('similarity', 0)
            llm_score = llm_scores.get(str(i), 5) / 10.0  # Normalize to 0-1

            # Store scores for debugging
            chunk['original_similarity'] = original_score
            chunk['rerank_score'] = llm_score

            # Compute weighted combination
            chunk['similarity'] = (
                RERANKER_ORIGINAL_WEIGHT * original_score +
                RERANKER_LLM_WEIGHT * llm_score
            )

        # Sort by new combined similarity
        chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # Count position changes
        new_order = [
            chunk.get('chunk_id', chunk.get('citation', str(i)))
            for i, chunk in enumerate(chunks)
        ]
        position_changes = sum(1 for i, cid in enumerate(new_order) if i < len(original_order) and original_order[i] != cid)

        return chunks, position_changes

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        interpretation=None
    ) -> RerankerResult:
        """
        Rerank chunks by semantic relevance using LLM.

        Args:
            query: The user's original question
            chunks: Retrieved chunks with similarity scores
            interpretation: Optional QueryInterpretation for context

        Returns:
            RerankerResult with reordered chunks
        """
        # Early exit if disabled
        if not CAG_ENABLE_RERANKER:
            return RerankerResult(
                chunks=chunks,
                latency_ms=0,
                model_used="disabled",
                success=False,
                error="Reranker disabled"
            )

        # Nothing to rerank
        if not chunks:
            return RerankerResult(
                chunks=[],
                latency_ms=0,
                model_used="none",
                success=True
            )

        start_time = time.time()

        try:
            self._ensure_client()

            if self._client is None:
                logger.warning("Gemini client not available for reranking")
                return RerankerResult(
                    chunks=chunks,
                    latency_ms=(time.time() - start_time) * 1000,
                    model_used="none",
                    success=False,
                    error="Gemini client not available"
                )

            # Limit chunks to prevent huge prompts
            chunks_to_rank = chunks[:RERANKER_MAX_CHUNKS]

            # Build prompt
            formatted_chunks = self._format_chunks(chunks_to_rank)
            interpretation_context = self._build_interpretation_context(interpretation)

            prompt = RERANKER_USER_PROMPT.format(
                query=query,
                interpretation_context=interpretation_context,
                formatted_chunks=formatted_chunks
            )

            # Call Gemini
            response = self._client.models.generate_content(
                model=RERANKER_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=RERANKER_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                )
            )

            # Parse scores
            llm_scores = self._parse_scores(response.text, len(chunks_to_rank))

            # Compute final scores and reorder
            reranked_chunks, position_changes = self._compute_final_scores(
                chunks_to_rank, llm_scores
            )

            # Add back any chunks that weren't ranked (beyond RERANKER_MAX_CHUNKS)
            if len(chunks) > RERANKER_MAX_CHUNKS:
                reranked_chunks.extend(chunks[RERANKER_MAX_CHUNKS:])

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Reranker completed: {len(chunks_to_rank)} chunks, "
                f"{latency_ms:.0f}ms, {position_changes} position changes"
            )

            return RerankerResult(
                chunks=reranked_chunks,
                latency_ms=latency_ms,
                model_used=RERANKER_MODEL,
                success=True,
                scores={str(k): v/10.0 for k, v in llm_scores.items()},
                position_changes=position_changes
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Reranker failed: {e}, returning original order")

            return RerankerResult(
                chunks=chunks,  # Return unchanged
                latency_ms=latency_ms,
                model_used=RERANKER_MODEL,
                success=False,
                error=str(e)
            )


# Module-level singleton
_reranker = None


def get_reranker() -> LLMReranker:
    """Get or create the reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker()
    return _reranker


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test the reranker on sample data."""
    print("Testing LLM Reranker")
    print("=" * 60)

    # Mock chunks
    test_chunks = [
        {
            "chunk_id": "art24_s1",
            "citation": "Article 24, Section 1",
            "content": "Employees shall be entitled to one (1) meal period of thirty (30) minutes for each shift of six (6) hours or more.",
            "similarity": 0.75
        },
        {
            "chunk_id": "art25_s1",
            "citation": "Article 25, Section 1",
            "content": "All employees shall receive a fifteen (15) minute relief period for each four (4) hours of work.",
            "similarity": 0.72
        },
        {
            "chunk_id": "art10_s3",
            "citation": "Article 10, Section 3",
            "content": "The Employer shall post work schedules by Wednesday for the following week.",
            "similarity": 0.68
        },
        {
            "chunk_id": "art43_s1",
            "citation": "Article 43, Section 1",
            "content": "No employee shall be discharged or disciplined without just cause.",
            "similarity": 0.65
        },
    ]

    reranker = LLMReranker()

    # Test case: breaks question
    query = "When do I get a break?"
    print(f"\nQuery: {query}")
    print("-" * 40)

    result = reranker.rerank(query=query, chunks=test_chunks)

    if result.success:
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"Position changes: {result.position_changes}")
        print(f"\nReranked order:")
        for i, chunk in enumerate(result.chunks):
            print(f"  {i+1}. [{chunk['citation']}]")
            print(f"      Original: {chunk.get('original_similarity', 'N/A'):.3f}")
            print(f"      Rerank:   {chunk.get('rerank_score', 'N/A'):.3f}")
            print(f"      Combined: {chunk.get('similarity', 'N/A'):.3f}")
    else:
        print(f"FAILED: {result.error}")


if __name__ == "__main__":
    main()
