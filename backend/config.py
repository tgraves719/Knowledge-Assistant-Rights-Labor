"""Configuration settings for Karl RAG system."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
WAGES_DIR = DATA_DIR / "wages"
TEST_SET_DIR = DATA_DIR / "test_set"

# Contract settings
CONTRACT_ID = "safeway_pueblo_clerks_2022"
CONTRACT_MD_FILE = PROJECT_ROOT / "SW+Pueblo+Clerks+2022.2025.md"
CONTRACT_JSON_FILE = PROJECT_ROOT / "SW+Pueblo+Clerks+2022.2025.json"

# Vector DB settings
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model, no API needed
COLLECTION_NAME = "union_contracts"

# LLM settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-2.0-flash-lite"

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.1  # Lower threshold to let semantic search do its job

# High-stakes topics that require escalation language
HIGH_STAKES_TOPICS = [
    "discharge", "termination", "fired", "discipline",
    "harassment", "discrimination", "retaliation",
    "safety", "injury", "immigration", "weingarten"
]

# =============================================================================
# CAG (Context-Aware Generation) Configuration - "Rosetta Stone" Architecture
# =============================================================================

# Feature flags for gradual rollout (start disabled, enable after testing)
CAG_ENABLE_HYPOTHESIS_LAYER = True       # Phase 2: LLM hypothesis generation
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True  # Phase 3: Fetch all chunks from winning article
CAG_ENABLE_TITLE_BOOSTING = True          # Phase 2: Boost chunks matching hypothesized titles

# Hybrid Search Tuning (Phase 1)
HYBRID_VECTOR_WEIGHT = 1.0   # Was 1.2 - equalize for better RRF fusion
HYBRID_KEYWORD_WEIGHT = 1.0  # Was 0.8 - equalize for better RRF fusion
BM25_K1 = 1.8                # Was 1.5 - higher saturation for legal docs with repeated terms
BM25_B = 0.75                # Document length normalization (keep default)

# Hypothesis Layer Configuration (Phase 2)
HYPOTHESIS_MODEL = "gemini-2.0-flash"  # Better reasoning than flash-lite
HYPOTHESIS_MAX_TITLES = 3              # Number of section titles to generate
HYPOTHESIS_TIMEOUT_MS = 2000           # Timeout for hypothesis LLM call
TITLE_BOOST_SCORE = 0.5                # Score boost when hypothesis matches article_title

# Full Article Expansion Configuration (Phase 3)
FULL_ARTICLE_MAX_CHUNKS = 15           # Maximum chunks to fetch from winning article
FULL_ARTICLE_MIN_TOP_K_MATCH = 2       # Minimum occurrences in top-5 to trigger expansion

# Query Interpreter Configuration (Phase 4 - Multi-angle retrieval)
CAG_ENABLE_QUERY_INTERPRETER = True    # Deep semantic analysis before retrieval
INTERPRETER_MODEL = "gemini-2.0-flash" # Model for query interpretation
INTERPRETER_TIMEOUT_MS = 15000         # Timeout for interpretation (15s)
MULTI_QUERY_MAX_SEARCHES = 3           # Max number of search angles to try
MULTI_QUERY_RESULTS_PER_SEARCH = 5     # Results per search angle
MULTI_QUERY_TOTAL_RESULTS = 10         # Total unique results after merging

# Manifests directory for article titles
MANIFESTS_DIR = DATA_DIR / "manifests"

# =============================================================================
# LLM Reranker Configuration (Phase 5)
# =============================================================================

CAG_ENABLE_RERANKER = True             # Enable LLM-based reranking
RERANKER_MODEL = "gemini-2.0-flash"    # Same model as interpreter
RERANKER_TIMEOUT_MS = 10000            # 10 second timeout
RERANKER_ORIGINAL_WEIGHT = 0.3         # Weight for original similarity score
RERANKER_LLM_WEIGHT = 0.7              # Weight for LLM relevance score
RERANKER_MAX_CHUNKS = 15               # Max chunks to rerank per call
RERANKER_CONTENT_TRUNCATE = 500        # Max chars per chunk in prompt

