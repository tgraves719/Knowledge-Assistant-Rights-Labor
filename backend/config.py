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

