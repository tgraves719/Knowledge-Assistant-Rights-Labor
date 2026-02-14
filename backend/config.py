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
TABLES_DIR = DATA_DIR / "tables"
MANIFESTS_DIR = DATA_DIR / "manifests"
ONTOLOGIES_DIR = DATA_DIR / "ontologies"


def _discover_default_contract_id() -> str:
    """
    Resolve default contract in this order:
    1) explicit env override
    2) legacy benchmark default when present
    3) first manifest found in data/manifests
    4) legacy fallback for bootstrap environments
    """
    for env_name in ("KARL_CONTRACT_ID", "CONTRACT_ID"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    legacy_default = "safeway_pueblo_clerks_2022"
    if (MANIFESTS_DIR / f"{legacy_default}.json").exists():
        return legacy_default

    manifests = sorted(MANIFESTS_DIR.glob("*.json"))
    if manifests:
        return manifests[0].stem

    return legacy_default

# Contract settings
CONTRACT_ID = _discover_default_contract_id()
CONTRACT_MD_FILE = PROJECT_ROOT / "SW+Pueblo+Clerks+2022.2025.md"
CONTRACT_JSON_FILE = PROJECT_ROOT / "SW+Pueblo+Clerks+2022.2025.json"

# Vector DB settings
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model, no API needed
COLLECTION_NAME = "union_contracts"

# LLM settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-2.5-pro"

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
# v1.5 LEAN CONFIG: Based on ablation analysis, these features were adding noise, not signal
CAG_ENABLE_HYPOTHESIS_LAYER = False      # DISABLED: Ablation showed +1.8% accuracy without it
CAG_ENABLE_FULL_ARTICLE_EXPANSION = False  # DISABLED: Ablation showed +5.9% on Multi-Hop without it
CAG_ENABLE_TITLE_BOOSTING = False          # DISABLED: Depends on hypothesis layer

# Hybrid Search Tuning (Phase 1)
# v1.5 LEAN: Pure vector search outperformed hybrid fusion
HYBRID_VECTOR_WEIGHT = float(os.getenv("KARL_HYBRID_VECTOR_WEIGHT", "1.0"))   # Vector search (semantic)
HYBRID_KEYWORD_WEIGHT = float(os.getenv("KARL_HYBRID_KEYWORD_WEIGHT", "0.0"))  # BM25 keyword search
BM25_K1 = 1.8                # Was 1.5 - higher saturation for legal docs with repeated terms
BM25_B = 0.75                # Document length normalization (keep default)

# Hypothesis Layer Configuration (Phase 2)
HYPOTHESIS_MODEL = "gemini-2.5-flash"  # Fast reasoning for pre-retrieval hypotheses
HYPOTHESIS_MAX_TITLES = 3              # Number of section titles to generate
HYPOTHESIS_TIMEOUT_MS = 2000           # Timeout for hypothesis LLM call
TITLE_BOOST_SCORE = 0.5                # Score boost when hypothesis matches article_title

# Full Article Expansion Configuration (Phase 3)
FULL_ARTICLE_MAX_CHUNKS = 15           # Maximum chunks to fetch from winning article
FULL_ARTICLE_MIN_TOP_K_MATCH = 2       # Minimum occurrences in top-5 to trigger expansion

# Query Interpreter Configuration (Phase 4 - Multi-angle retrieval)
CAG_ENABLE_QUERY_INTERPRETER = True    # Deep semantic analysis before retrieval
INTERPRETER_MODEL = "gemini-2.5-flash" # Model for query interpretation
INTERPRETER_TIMEOUT_MS = 15000         # Timeout for interpretation (15s)
MULTI_QUERY_MAX_SEARCHES = 3           # Max number of search angles to try
MULTI_QUERY_RESULTS_PER_SEARCH = 5     # Results per search angle
MULTI_QUERY_TOTAL_RESULTS = 10         # Total unique results after merging

# =============================================================================
# LLM Reranker Configuration (Phase 5)
# =============================================================================

CAG_ENABLE_RERANKER = True             # Enable LLM-based reranking
RERANKER_MODEL = "gemini-2.5-flash"    # Same model as interpreter
RERANKER_TIMEOUT_MS = 20000            # 20 second timeout (2.5 Flash uses thinking)
RERANKER_ORIGINAL_WEIGHT = 0.3         # Weight for original similarity score
RERANKER_LLM_WEIGHT = 0.7              # Weight for LLM relevance score
RERANKER_MAX_CHUNKS = 15               # Max chunks to rerank per call
RERANKER_CONTENT_TRUNCATE = 500        # Max chars per chunk in prompt

