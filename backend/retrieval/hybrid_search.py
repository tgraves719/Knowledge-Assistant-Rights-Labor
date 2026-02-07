"""
Hybrid Search Module - Combines semantic and keyword search with RRF fusion.

Components:
1. Vector/Semantic Search (via ChromaDB + embeddings)
2. BM25 Keyword Search (for exact term matching)
3. Reciprocal Rank Fusion (RRF) to combine results

Designed for scalability across 100+ union contracts.
"""

import re
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CHUNKS_DIR, TOP_K_RESULTS, BM25_K1, BM25_B
from backend.retrieval.query_expansion import expand_query, get_keyword_variants

# Phase 4: Concept index for vocabulary bridging
_concept_index = None

def get_concept_index():
    """Lazy-load the concept index."""
    global _concept_index
    if _concept_index is None:
        try:
            from backend.ingest.toc_index import ConceptIndex
            _concept_index = ConceptIndex()
            if _concept_index.load():
                print(f"Loaded concept index with {len(_concept_index.concept_to_articles)} concepts")
            else:
                print("Concept index not found, will skip concept-based retrieval")
                _concept_index = None
        except Exception as e:
            print(f"Could not load concept index: {e}")
            _concept_index = None
    return _concept_index


@dataclass
class SearchResult:
    """Container for a search result with score information."""
    chunk_id: str
    content: str
    citation: str
    metadata: dict
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    vector_rank: int = 0
    keyword_rank: int = 0


# =============================================================================
# BM25 IMPLEMENTATION
# =============================================================================

class BM25Index:
    """
    BM25 index for keyword-based search.
    
    Stores term frequencies and computes BM25 scores.
    Designed for quick rebuilding when new contracts are added.
    """
    
    def __init__(self, k1: float = None, b: float = None):
        """
        Initialize BM25 with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (default from config: 1.8 for legal docs)
            b: Document length normalization (default from config: 0.75)

        Note: k1=1.8 is higher than typical (1.2-1.5) because legal documents
        like union contracts use specific terms repeatedly. Higher k1 increases
        sensitivity to term frequency, rewarding documents that mention
        "Relief Period" multiple times.
        """
        self.k1 = k1 if k1 is not None else BM25_K1
        self.b = b if b is not None else BM25_B
        
        # Index structures
        self.documents: Dict[str, dict] = {}  # chunk_id -> chunk data
        self.doc_lengths: Dict[str, int] = {}  # chunk_id -> word count
        self.avg_doc_length: float = 0.0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)  # term -> num docs containing it
        self.term_freq: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {chunk_id: count}
        self.num_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Lowercase and extract words
        text = text.lower()
        # Keep alphanumeric and some punctuation
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Filter very short tokens
        return [t for t in tokens if len(t) >= 2]
    
    def build_index(self, chunks: List[dict]):
        """
        Build the BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'content'
        """
        self.documents = {}
        self.doc_lengths = {}
        self.term_doc_freq = defaultdict(int)
        self.term_freq = defaultdict(dict)
        
        total_length = 0
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            content = chunk.get('content', '')
            
            # Also include citation and title for matching
            searchable_text = f"{content} {chunk.get('citation', '')} {chunk.get('article_title', '')}"
            
            self.documents[chunk_id] = chunk
            tokens = self._tokenize(searchable_text)
            self.doc_lengths[chunk_id] = len(tokens)
            total_length += len(tokens)
            
            # Count term frequencies
            seen_terms = set()
            for token in tokens:
                if token not in self.term_freq[token]:
                    self.term_freq[token][chunk_id] = 0
                self.term_freq[token][chunk_id] += 1
                
                if token not in seen_terms:
                    self.term_doc_freq[token] += 1
                    seen_terms.add(token)
        
        self.num_docs = len(chunks)
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
    
    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term."""
        n = self.term_doc_freq.get(term, 0)
        if n == 0:
            return 0.0
        # Standard IDF formula with smoothing
        return math.log((self.num_docs - n + 0.5) / (n + 0.5) + 1.0)
    
    def score_document(self, chunk_id: str, query_terms: List[str]) -> float:
        """
        Calculate BM25 score for a document given query terms.
        """
        if chunk_id not in self.doc_lengths:
            return 0.0
        
        doc_len = self.doc_lengths[chunk_id]
        score = 0.0
        
        for term in query_terms:
            if term not in self.term_freq or chunk_id not in self.term_freq[term]:
                continue
            
            tf = self.term_freq[term][chunk_id]
            idf = self._idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, n_results: int = 10, expand: bool = True) -> List[Tuple[str, float]]:
        """
        Search the index with BM25 ranking.
        
        Args:
            query: Search query
            n_results: Number of results to return
            expand: Whether to use query expansion
        
        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        # Get query terms with optional expansion
        if expand:
            expanded = expand_query(query)
            # Include both original and expanded terms
            query_text = f"{query} {' '.join(expanded.expanded_terms)}"
        else:
            query_text = query
        
        query_terms = self._tokenize(query_text)
        
        if not query_terms:
            return []
        
        # Score all documents
        scores = []
        for chunk_id in self.documents:
            score = self.score_document(chunk_id, query_terms)
            if score > 0:
                scores.append((chunk_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:n_results]


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF is robust to outliers and doesn't require score normalization.
    
    Formula: RRF(d) = Σ (weight_i / (k + rank_i(d)))
    
    Args:
        rankings: List of ranked result lists, each containing (id, score) tuples
        k: Ranking constant (default 60, per original RRF paper)
        weights: Optional weights for each ranking list
    
    Returns:
        Combined ranked list with RRF scores
    """
    if not rankings:
        return []
    
    if weights is None:
        weights = [1.0] * len(rankings)
    
    # Build RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    
    for ranking_idx, ranking in enumerate(rankings):
        weight = weights[ranking_idx]
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] += weight / (k + rank)
    
    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


# =============================================================================
# HYBRID SEARCHER
# =============================================================================

class HybridSearcher:
    """
    Combines vector search and BM25 keyword search with RRF fusion.
    
    Usage:
        searcher = HybridSearcher(vector_store)
        results = searcher.search("What are my break periods?")
    """
    
    def __init__(self, vector_store=None, chunks_file: Path = None):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: ContractVectorStore instance for semantic search
            chunks_file: Path to chunks JSON file for BM25 index
        """
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        
        # Load chunks for BM25 index - prefer enriched chunks
        if chunks_file is None:
            # Try enriched first, then smart, then original
            chunks_file = CHUNKS_DIR / "contract_chunks_enriched.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks_smart.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks.json"
        
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            self.bm25_index.build_index(chunks)
            self.chunks_by_id = {c['chunk_id']: c for c in chunks}
        else:
            self.chunks_by_id = {}

        # Phase 4: Preload concept index for vocabulary bridging
        self.concept_index = get_concept_index()
    
    def _ensure_vector_store(self):
        """Lazy-load vector store if not provided."""
        if self.vector_store is None:
            from backend.retrieval.vector_store import ContractVectorStore
            self.vector_store = ContractVectorStore()

    def get_concept_boost_articles(self, query: str) -> List[int]:
        """
        Phase 4: Find articles to boost based on concept index matching.

        Uses pre-computed worker_questions and alternative_names to bridge
        vocabulary gaps without runtime LLM calls.

        Args:
            query: User's search query

        Returns:
            List of article numbers to boost, ordered by match strength
        """
        if self.concept_index is None:
            return []

        # Get articles matching by concept (alternative_names)
        concept_articles = self.concept_index.find_articles_by_concept(query)

        # Get articles matching by question similarity
        question_articles = self.concept_index.find_articles_by_question(query)

        # Combine with question matches taking priority (more precise)
        # then concept matches (broader substring matching)
        seen = set()
        combined = []
        for art in question_articles[:5]:
            if art not in seen:
                combined.append(art)
                seen.add(art)
        for art in concept_articles[:5]:
            if art not in seen:
                combined.append(art)
                seen.add(art)

        return combined[:5]  # Return top 5 articles to boost
    
    def search(
        self,
        query: str,
        n_results: int = None,
        use_expansion: bool = True,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        boost_articles: List[int] = None,
        concept_query: str = None,
        **vector_kwargs
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: User query (may include hypothesis expansions)
            n_results: Number of results to return
            use_expansion: Whether to use query expansion
            vector_weight: Weight for semantic search in RRF (default 1.0)
            keyword_weight: Weight for keyword search in RRF (default 1.0)
            boost_articles: List of article numbers to boost in results
            concept_query: Original user query for concept matching (defaults to query)
            **vector_kwargs: Additional args for vector search (classification, urgency_tier, etc.)

        Returns:
            List of SearchResult with combined ranking
        """
        if n_results is None:
            n_results = TOP_K_RESULTS

        self._ensure_vector_store()

        # Phase 4: Get concept-based article boosts (vocabulary bridging)
        # Use concept_query (original user question) for stable matching,
        # not the hypothesis-expanded query which varies per run
        concept_boost_articles = self.get_concept_boost_articles(concept_query or query)
        if concept_boost_articles:
            if boost_articles:
                # Combine with explicit boosts, concept-based first
                boost_articles = list(set(concept_boost_articles + list(boost_articles)))
            else:
                boost_articles = concept_boost_articles

        # 1. Expand query if enabled
        if use_expansion:
            expanded = expand_query(query)
            semantic_query = expanded.combined_query
        else:
            semantic_query = query
        
        # 2. Vector/Semantic Search
        # Get more results than needed for better fusion
        vector_results = self.vector_store.search(
            query=semantic_query,
            n_results=n_results * 2,
            boost_articles=boost_articles,
            **vector_kwargs
        )
        vector_ranking = [(r['chunk_id'], r.get('similarity', 0)) for r in vector_results]
        
        # 3. BM25 Keyword Search
        keyword_ranking = self.bm25_index.search(
            query=query,
            n_results=n_results * 2,
            expand=use_expansion
        )
        
        # 4. Reciprocal Rank Fusion
        rrf_ranking = reciprocal_rank_fusion(
            rankings=[vector_ranking, keyword_ranking],
            weights=[vector_weight, keyword_weight]
        )
        
        # 4.5. Apply topic-based article boosting to RRF scores
        if boost_articles:
            boosted_ranking = []
            for chunk_id, rrf_score in rrf_ranking:
                chunk = self.chunks_by_id.get(chunk_id, {})
                article_num = chunk.get('article_num', 0)
                if article_num in boost_articles:
                    # Significant boost (doubles typical RRF score) to ensure
                    # topic-relevant articles appear in results
                    rrf_score += 0.03
                boosted_ranking.append((chunk_id, rrf_score))
            rrf_ranking = sorted(boosted_ranking, key=lambda x: x[1], reverse=True)
        
        # 5. Build SearchResult objects
        results = []
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_ranking, 1)}
        keyword_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_ranking, 1)}
        vector_scores = {chunk_id: score for chunk_id, score in vector_ranking}
        keyword_scores = {chunk_id: score for chunk_id, score in keyword_ranking}
        
        for chunk_id, rrf_score in rrf_ranking[:n_results]:
            chunk = self.chunks_by_id.get(chunk_id, {})
            
            result = SearchResult(
                chunk_id=chunk_id,
                content=chunk.get('content', ''),
                citation=chunk.get('citation', ''),
                metadata={
                    'article_num': chunk.get('article_num'),
                    'article_title': chunk.get('article_title', ''),
                    'section_num': chunk.get('section_num'),
                    'subsection': chunk.get('subsection', ''),
                    'topics': chunk.get('topics', chunk.get('topic_tags', [])),
                    'applies_to': chunk.get('applies_to', ['all']),
                    'summary': chunk.get('summary', ''),
                    'is_definition': chunk.get('is_definition', False),
                    'is_exception': chunk.get('is_exception', False),
                    'hire_date_sensitive': chunk.get('hire_date_sensitive', False),
                    'is_high_stakes': chunk.get('is_high_stakes', False),
                    # Phase 4: Concept-indexed fields
                    'worker_questions': chunk.get('worker_questions', []),
                    'alternative_names': chunk.get('alternative_names', []),
                },
                vector_score=vector_scores.get(chunk_id, 0),
                keyword_score=keyword_scores.get(chunk_id, 0),
                rrf_score=rrf_score,
                vector_rank=vector_ranks.get(chunk_id, 999),
                keyword_rank=keyword_ranks.get(chunk_id, 999),
            )
            results.append(result)
        
        return results
    
    def search_to_chunks(
        self,
        query: str,
        n_results: int = None,
        boost_articles: List[int] = None,
        concept_query: str = None,
        **kwargs
    ) -> List[dict]:
        """
        Search and return results as chunk dictionaries (for compatibility).
        """
        results = self.search(query, n_results, boost_articles=boost_articles,
                              concept_query=concept_query, **kwargs)
        
        return [
            {
                'chunk_id': r.chunk_id,
                'content': r.content,
                'citation': r.citation,
                'similarity': r.rrf_score,  # Use RRF as similarity for compatibility
                **r.metadata
            }
            for r in results
        ]


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test hybrid search on failing queries."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Loading hybrid searcher...")
    searcher = HybridSearcher()
    
    test_queries = [
        ("How far back can I get retroactive pay?", "Article 46"),
        ("What are my break periods?", "Article 24"),
        ("I was just terminated. What should I do?", "Article 43"),
    ]
    
    print("\n" + "=" * 70)
    print("HYBRID SEARCH TEST - Previously Failing Queries")
    print("=" * 70)
    
    for query, expected in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        
        results = searcher.search(query, n_results=5)
        
        print(f"\nResults (RRF combined):")
        found = False
        for i, r in enumerate(results, 1):
            match = "<-- MATCH" if expected in r.citation else ""
            if match:
                found = True
            print(f"  [{i}] {r.citation}")
            print(f"      Vector rank: {r.vector_rank}, Keyword rank: {r.keyword_rank}")
            print(f"      RRF score: {r.rrf_score:.4f} {match}")
        
        status = "PASS" if found else "FAIL"
        print(f"\n  Status: {status}")
        print("-" * 50)


if __name__ == "__main__":
    main()

