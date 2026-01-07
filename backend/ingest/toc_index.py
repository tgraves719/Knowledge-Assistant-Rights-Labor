"""
Table of Contents Index Builder for Concept-Based Retrieval.

Phase 4 of the CAG Architecture: Pre-compute article-level concept mappings
to enable fast vocabulary bridging at query time without runtime LLM calls.

This module builds:
1. Article-level aggregation of worker_questions and alternative_names
2. Concept-to-article mapping for two-stage retrieval
3. JSON index for fast loading at query time
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

from backend.config import CHUNKS_DIR


# =============================================================================
# INDEX STRUCTURE
# =============================================================================

class ConceptIndex:
    """
    Pre-computed concept index for vocabulary bridging.

    Structure:
    {
        "articles": {
            "25": {
                "title": "RELIEF PERIODS",
                "all_worker_questions": ["When do I get a break?", ...],
                "all_alternative_names": ["break", "rest", ...],
                "chunk_ids": ["art25_sec61", ...]
            }
        },
        "concept_to_articles": {
            "break": [25, 24],
            "vacation": [17],
            ...
        },
        "question_to_articles": {
            "when do i get a break": [25, 24],
            ...
        }
    }
    """

    def __init__(self, index_path: Optional[Path] = None):
        """Initialize the concept index."""
        self.index_path = index_path or Path(CHUNKS_DIR) / "concept_index.json"
        self.articles = {}
        self.concept_to_articles = defaultdict(set)
        self.question_to_articles = defaultdict(set)
        self._loaded = False

    def load(self) -> bool:
        """Load index from JSON file."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.articles = data.get("articles", {})

            # Convert lists back to sets for efficient lookup
            self.concept_to_articles = defaultdict(set)
            for concept, articles in data.get("concept_to_articles", {}).items():
                self.concept_to_articles[concept] = set(articles)

            self.question_to_articles = defaultdict(set)
            for question, articles in data.get("question_to_articles", {}).items():
                self.question_to_articles[question] = set(articles)

            self._loaded = True
            return True

        except Exception as e:
            print(f"Error loading concept index: {e}")
            return False

    def save(self) -> None:
        """Save index to JSON file."""
        # Convert sets to lists for JSON serialization
        data = {
            "articles": self.articles,
            "concept_to_articles": {
                concept: sorted(list(articles))
                for concept, articles in self.concept_to_articles.items()
            },
            "question_to_articles": {
                question: sorted(list(articles))
                for question, articles in self.question_to_articles.items()
            }
        }

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Concept index saved to {self.index_path}")

    def find_articles_by_concept(self, query: str) -> list[int]:
        """
        Find articles that match query terms against alternative_names.

        Args:
            query: User's search query

        Returns:
            List of article numbers, sorted by match strength
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        query_words = set(query_lower.split())

        article_scores = defaultdict(int)

        # Check each concept for matches
        for concept, articles in self.concept_to_articles.items():
            # Exact match
            if concept in query_lower:
                for art in articles:
                    article_scores[art] += 3
            # Word match
            elif concept in query_words:
                for art in articles:
                    article_scores[art] += 2
            # Partial match
            elif any(concept in word or word in concept for word in query_words):
                for art in articles:
                    article_scores[art] += 1

        # Sort by score descending
        return sorted(article_scores.keys(), key=lambda x: article_scores[x], reverse=True)

    def find_articles_by_question(self, query: str) -> list[int]:
        """
        Find articles whose worker_questions are similar to the query.

        Uses simple word overlap scoring.

        Args:
            query: User's search query

        Returns:
            List of article numbers, sorted by match strength
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        query_words = set(query_lower.split())

        article_scores = defaultdict(float)

        for question, articles in self.question_to_articles.items():
            question_words = set(question.split())

            # Jaccard similarity
            intersection = len(query_words & question_words)
            union = len(query_words | question_words)

            if union > 0 and intersection > 0:
                similarity = intersection / union
                for art in articles:
                    article_scores[art] = max(article_scores[art], similarity)

        # Sort by score descending, filter out zero scores
        return sorted(
            [a for a, s in article_scores.items() if s > 0.1],
            key=lambda x: article_scores[x],
            reverse=True
        )

    def get_article_info(self, article_num: int) -> Optional[dict]:
        """Get information about a specific article."""
        if not self._loaded:
            self.load()
        return self.articles.get(str(article_num))


# =============================================================================
# INDEX BUILDER
# =============================================================================

def build_concept_index(
    chunks_path: Path = None,
    output_path: Path = None
) -> ConceptIndex:
    """
    Build concept index from enriched chunks.

    Args:
        chunks_path: Path to enriched chunks JSON
        output_path: Path to save concept index

    Returns:
        Populated ConceptIndex
    """
    if chunks_path is None:
        chunks_path = Path(CHUNKS_DIR) / "contract_chunks_enriched.json"

    if output_path is None:
        output_path = Path(CHUNKS_DIR) / "concept_index.json"

    print(f"Building concept index from {chunks_path}")

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    index = ConceptIndex(output_path)

    # Group chunks by article
    articles_data = defaultdict(lambda: {
        "title": "",
        "all_worker_questions": set(),
        "all_alternative_names": set(),
        "chunk_ids": []
    })

    for chunk in chunks:
        article_num = chunk.get("article_num")
        if article_num is None:
            continue  # Skip LOUs for now

        article_key = str(article_num)
        article_data = articles_data[article_key]

        # Capture article title
        if chunk.get("article_title") and not article_data["title"]:
            article_data["title"] = chunk["article_title"]

        # Aggregate chunk ID
        if chunk.get("chunk_id"):
            article_data["chunk_ids"].append(chunk["chunk_id"])

        # Aggregate worker questions
        for q in chunk.get("worker_questions", []):
            if q and isinstance(q, str):
                q_normalized = q.strip().lower()
                article_data["all_worker_questions"].add(q_normalized)
                index.question_to_articles[q_normalized].add(article_num)

        # Aggregate alternative names
        for name in chunk.get("alternative_names", []):
            if name and isinstance(name, str):
                name_normalized = name.strip().lower()
                article_data["all_alternative_names"].add(name_normalized)
                index.concept_to_articles[name_normalized].add(article_num)

    # Convert sets to lists for storage
    for article_key, data in articles_data.items():
        index.articles[article_key] = {
            "title": data["title"],
            "all_worker_questions": sorted(list(data["all_worker_questions"])),
            "all_alternative_names": sorted(list(data["all_alternative_names"])),
            "chunk_ids": data["chunk_ids"]
        }

    # Save the index
    index.save()

    # Print summary
    print(f"\nConcept Index Summary:")
    print(f"  Articles indexed: {len(index.articles)}")
    print(f"  Unique concepts: {len(index.concept_to_articles)}")
    print(f"  Unique questions: {len(index.question_to_articles)}")

    # Show some examples
    print(f"\n  Sample concepts:")
    for concept in list(index.concept_to_articles.keys())[:10]:
        articles = list(index.concept_to_articles[concept])
        print(f"    '{concept}' -> Articles {articles}")

    return index


# =============================================================================
# CLI
# =============================================================================

def main():
    """Build concept index from enriched chunks."""
    import argparse

    parser = argparse.ArgumentParser(description="Build concept index for vocabulary bridging")
    parser.add_argument("--input", type=str, default="data/chunks/contract_chunks_enriched.json")
    parser.add_argument("--output", type=str, default="data/chunks/concept_index.json")

    args = parser.parse_args()

    index = build_concept_index(
        chunks_path=Path(args.input),
        output_path=Path(args.output)
    )

    # Test some lookups
    print("\n" + "=" * 60)
    print("Testing Concept Lookups:")
    print("=" * 60)

    test_queries = [
        "When do I get a break?",
        "Can I get fired?",
        "How much vacation time do I get?",
        "What are my overtime rights?",
    ]

    for query in test_queries:
        concept_articles = index.find_articles_by_concept(query)
        question_articles = index.find_articles_by_question(query)
        print(f"\nQuery: '{query}'")
        print(f"  By concept: {concept_articles[:3]}")
        print(f"  By question: {question_articles[:3]}")


if __name__ == "__main__":
    main()
