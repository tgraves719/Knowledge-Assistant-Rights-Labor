"""
FastAPI Backend for Karl - Union Contract RAG System.

Provides endpoints for:
- /api/query - Main Q&A endpoint
- /api/wage - Direct wage lookup
- /api/health - Health check
"""

import os
import json
import time
import asyncio
import re
import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    MANIFESTS_DIR,
    HYBRID_VECTOR_WEIGHT,
)
from backend.contracts import (
    list_contract_catalog,
    get_contract_catalog_entry,
    resolve_default_contract_id,
)
from backend.retrieval.router import (
    HybridRetriever,
    classify_intent,
    ensure_contract_manifest,
    VACATION_ENTITLEMENT_QUERY_PATTERN,
)
from backend.retrieval.vector_store import ContractVectorStore
from backend.chunk_files import resolve_chunk_file
from backend.contract_outline import (
    resolve_contract_outline_file,
    load_contract_outline,
    article_titles_from_outline,
)
from backend.pdf_nav_files import resolve_pdf_nav_index_file
from backend.table_nav_files import resolve_table_nav_index_file
from backend.pdf_nav_index import (
    build_pdf_nav_index,
    load_pdf_nav_index,
    to_runtime_navigation_maps,
    resolve_contract_pdf_path as resolve_contract_pdf_source_path,
)
from backend.table_nav_index import (
    build_table_nav_index,
    load_table_nav_index,
    to_runtime_table_navigation_maps,
)
from backend.generation.prompts import build_prompt, SYSTEM_PROMPT
from backend.generation.verifier import (
    verify_response,
    add_escalation_if_missing,
    format_response_with_sources
)
from backend.generation.context import get_session_context
from backend.user.profile import (
    UserProfile,
    EmploymentType,
    get_user_profile,
    update_user_profile,
    estimate_hours_worked,
    get_classification_options,
    resolve_classification_display_name,
)

# Global instances
retriever = None
vector_store = None

# LLM client (lazy loaded)
try:
    from google import genai as _genai_sdk
except ImportError:
    _genai_sdk = None

_genai_client = None


_UNAVAILABLE_ANSWER_PATTERNS = (
    r"\bi\s+(?:cannot|can['’]t)\s+find\b",
    r"\bi\s+am\s+unable\s+to\s+find\b",
    r"\bi\s+do\s+not\s+have\b",
    r"\bi\s+cannot\s+determine\b",
    r"\bthat\s+(?:specific\s+)?information\s+is\s+not\s+available\b",
)

_QUERY_EVIDENCE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "we",
    "what", "when", "where", "which", "who", "why", "you", "your",
}

_TOPIC_RECOVERY_SIGNALS = {
    "vacation": ("vacation", "time off", "holiday", "personal holiday", "floater", "pto"),
    "overtime": ("overtime", "ot", "time and a half"),
    "grievance": ("grievance", "arbitration", "dispute", "complaint"),
    "layoff": ("layoff", "bump", "displacement", "reduction in hours", "reduction"),
    "term": ("term", "start", "end", "expiration", "effective", "agreement"),
    "breaks": ("break", "lunch", "meal", "rest", "between shifts"),
    "bereavement": ("bereavement", "funeral", "death"),
}

_FOREIGN_CONTRACT_UNAVAILABLE_ANSWER = (
    "I cannot find that specific information in your contract. "
    "Your question appears to reference a different agreement. "
    "Please switch contract context or contact your steward for the correct contract."
)


def _normalize_text_token_space(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _contract_aliases(contract_id: str) -> set[str]:
    """
    Build deterministic alias phrases for explicit contract mention detection.

    Aliases are intentionally multi-token to reduce accidental matches.
    """
    aliases: set[str] = set()
    cid = str(contract_id or "").strip().lower()
    if not cid:
        return aliases

    phrase = cid.replace("_", " ")
    phrase = re.sub(r"\blocal\s*\d+\b", " ", phrase)
    phrase = re.sub(r"\blocal\d+\b", " ", phrase)
    phrase = re.sub(r"\b(19|20)\d{2}\b", " ", phrase)
    phrase = _normalize_text_token_space(phrase)
    if phrase:
        aliases.add(phrase)

    # Expand common collapsed employer token.
    if "kingsoopers" in phrase:
        aliases.add(phrase.replace("kingsoopers", "king soopers"))

    entry = get_contract_catalog_entry(cid)
    if entry:
        employer = _normalize_text_token_space(entry.get("employer") or "")
        if employer:
            employer = re.sub(r"\b(inc|llc|ltd|co|companies|company|division|of|a|the)\b", " ", employer)
            employer = _normalize_text_token_space(employer)
            words = employer.split()
            if len(words) >= 2:
                aliases.add(" ".join(words[:2]))

    return {a for a in aliases if len(a.split()) >= 2}


@lru_cache(maxsize=16)
def _active_and_foreign_aliases(active_contract_id: str) -> tuple[set[str], set[str]]:
    active = _contract_aliases(active_contract_id)
    foreign: set[str] = set()
    for entry in list_contract_catalog():
        other = str(entry.get("contract_id") or "")
        if not other or other == active_contract_id:
            continue
        foreign.update(_contract_aliases(other))
    return active, foreign


def _mentions_foreign_contract_context(question: str, active_contract_id: str) -> bool:
    """
    Detect explicit references to a different known contract family.

    This prevents deterministic recovery from treating foreign-contract prompts
    as answerable in the active contract context.
    """
    query = _normalize_text_token_space(question)
    if not query:
        return False
    active_aliases, foreign_aliases = _active_and_foreign_aliases(active_contract_id)
    for alias in sorted(foreign_aliases, key=len, reverse=True):
        if alias in query and alias not in active_aliases:
            return True
    return False


def _classification_option_for_contract(contract_id: str, classification: Optional[str]) -> Optional[dict]:
    """Resolve a classification option record for contract-scoped guardrails."""
    value = str(classification or "").strip().lower()
    if not value:
        return None
    for opt in get_classification_options(contract_id=contract_id, include_unmapped=True):
        if str(opt.get("value") or "").strip().lower() == value:
            return opt
    return None


def get_genai_client():
    """Lazy load Google GenAI client."""
    global _genai_client
    if _genai_client is None and _genai_sdk is not None:
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if api_key:
            _genai_client = _genai_sdk.Client(api_key=api_key)
    return _genai_client


def _is_unavailable_answer(text: str) -> bool:
    """Detect uncertainty/unavailability phrasing in model output."""
    value = str(text or "").strip().lower()
    if not value:
        return False
    # Evaluate only opening span to avoid matching quoted contract text deep in
    # evidence-heavy responses.
    head = value[:320]
    return any(re.search(pattern, head) for pattern in _UNAVAILABLE_ANSWER_PATTERNS)


def _question_terms(question: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(question or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _QUERY_EVIDENCE_STOPWORDS}


def _has_strong_evidence_for_query(question: str, chunks: list[dict], min_token_hits: int = 3) -> bool:
    """
    Determine whether retrieved chunks likely contain actionable evidence.

    Uses deterministic lexical overlap + citation presence checks so we can
    avoid false "not available" responses when strong evidence exists.
    """
    if not chunks:
        return False

    query_terms = _question_terms(question)
    if not query_terms:
        return False

    for chunk in chunks[:8]:
        citation = str(chunk.get("citation") or "")
        if not re.search(r"article\s+\d+", citation, re.IGNORECASE):
            continue

        text = (
            f"{citation} "
            f"{chunk.get('article_title', '')} "
            f"{chunk.get('content_with_tables') or chunk.get('content') or ''}"
        ).lower()
        hit_count = sum(1 for term in query_terms if term in text)
        if hit_count >= min_token_hits:
            return True

    return False


def _merge_unique_chunks(primary: list[dict], secondary: list[dict], limit: int = 12) -> list[dict]:
    """Stable chunk merge preserving order and uniqueness by chunk_id/citation."""
    merged: list[dict] = []
    seen: set[str] = set()
    for collection in (primary or [], secondary or []):
        for chunk in collection:
            chunk_key = str(chunk.get("chunk_id") or chunk.get("citation") or "")
            if not chunk_key or chunk_key in seen:
                continue
            seen.add(chunk_key)
            merged.append(chunk)
            if len(merged) >= limit:
                return merged
    return merged


def _normalize_article_anchors(values: list) -> list[int]:
    anchors: list[int] = []
    seen: set[int] = set()
    for raw in values or []:
        try:
            num = int(raw)
        except (TypeError, ValueError):
            continue
        if num <= 0 or num in seen:
            continue
        seen.add(num)
        anchors.append(num)
    return anchors


def _has_article_anchor_evidence(article_numbers: list[int], chunks: list[dict], min_article_hits: int = 1) -> bool:
    if not article_numbers or not chunks or min_article_hits <= 0:
        return False
    article_set = set(article_numbers)
    hits = 0
    seen_articles: set[int] = set()
    for chunk in chunks[:10]:
        article_num = chunk.get("article_num")
        if not isinstance(article_num, int):
            continue
        if article_num not in article_set or article_num in seen_articles:
            continue
        seen_articles.add(article_num)
        hits += 1
        if hits >= min_article_hits:
            return True
    return False


def _query_supports_topic_recovery(topic: Optional[str], query: str) -> bool:
    topic_key = str(topic or "").strip().lower()
    if not topic_key:
        return False
    query_lower = str(query or "").lower()
    signals = _TOPIC_RECOVERY_SIGNALS.get(topic_key, ())
    if not signals:
        return bool(re.search(rf"\b{re.escape(topic_key)}\b", query_lower))
    for sig in signals:
        if re.search(rf"\b{re.escape(sig)}\b", query_lower):
            return True
    return False


def _is_vacation_entitlement_query(text: str) -> bool:
    normalized = _normalize_text_token_space(text)
    if not normalized:
        return False
    return bool(re.search(VACATION_ENTITLEMENT_QUERY_PATTERN, normalized))


def _format_iso_date(date_value: Optional[str]) -> str:
    value = str(date_value or "").strip()
    if not value:
        return ""
    try:
        return datetime.date.fromisoformat(value).strftime("%B %-d, %Y")
    except Exception:
        try:
            return datetime.date.fromisoformat(value).strftime("%B %d, %Y")
        except Exception:
            return value


def _format_vacation_tiers(tiers: list[dict]) -> str:
    parts = []
    for tier in tiers or []:
        years = int(tier.get("years_of_service") or 0)
        weeks = int(tier.get("weeks_per_year") or 0)
        parts.append(
            f"{weeks} week{'s' if weeks != 1 else ''} after {years} year{'s' if years != 1 else ''}"
        )
    return ", ".join(parts)


def _format_vacation_conditions(conditions: dict) -> str:
    if not isinstance(conditions, dict):
        return "general eligibility conditions"
    parts = []
    before = str(conditions.get("hire_date_on_or_before") or "").strip()
    after = str(conditions.get("hire_date_on_or_after") or "").strip()
    hours = conditions.get("anniversary_hours_min")
    if before:
        parts.append(f"hired on or before {_format_iso_date(before)}")
    if after:
        parts.append(f"hired on or after {_format_iso_date(after)}")
    if isinstance(hours, int) and hours > 0:
        parts.append(f"at least {hours} hours in the anniversary year")
    return "; ".join(parts) if parts else "general eligibility conditions"


def _build_vacation_entitlement_answer(entitlement_info: dict) -> str:
    selected = entitlement_info.get("selected_schedule") or {}
    considered = entitlement_info.get("schedules_considered") or []
    months = int(entitlement_info.get("months_employed") or 0)
    years = entitlement_info.get("years_completed")
    weeks = entitlement_info.get("estimated_weeks_per_year")
    citation = str(entitlement_info.get("citation") or selected.get("citation") or "Article 17")

    if selected and weeks is not None and years is not None:
        cond_text = _format_vacation_conditions(selected.get("conditions") or {})
        tier_text = _format_vacation_tiers(selected.get("tiers") or [])
        return (
            f"Based on {citation}, at {months} months of service ({years} completed years), "
            f"your vacation accrual is {weeks} week{'s' if weeks != 1 else ''} per year "
            f"under the schedule for employees with {cond_text}. "
            f"Vacation ladder: {tier_text}."
        )

    if considered:
        schedule_lines = []
        for s in considered[:3]:
            schedule_lines.append(
                f"- {s.get('citation') or 'Article 17'}: "
                f"{_format_vacation_conditions(s.get('conditions') or {})}; "
                f"{_format_vacation_tiers(s.get('tiers') or [])}"
            )
        return (
            "Your contract includes these vacation accrual schedules. "
            "I need your hire date (and anniversary-year hours if close to thresholds) to pick one schedule exactly:\n"
            + "\n".join(schedule_lines)
        )

    return (
        "I found vacation article coverage, but I could not deterministically resolve "
        "an accrual schedule from the entitlement artifact."
    )


def _entitlement_sources(entitlement_info: dict) -> list[dict]:
    sources = []
    seen = set()
    for ev in (entitlement_info.get("entitlement_evidence") or []):
        if not isinstance(ev, dict):
            continue
        citation = str(ev.get("citation") or "").strip()
        if not citation or citation in seen:
            continue
        seen.add(citation)
        sources.append(
            {
                "citation": citation,
                "article_num": ev.get("article_num"),
                "section_num": ev.get("section_num"),
                "content": str(ev.get("source_excerpt") or "").strip(),
                "source_method": "entitlement_artifact",
            }
        )
    return sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global retriever, vector_store
    print("Initializing Karl RAG system...")

    vector_store = None
    if HYBRID_VECTOR_WEIGHT > 0:
        try:
            vector_store = ContractVectorStore()
            print(f"Loaded {vector_store.count()} contract chunks")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            vector_store = None
    else:
        print("Vector retrieval disabled (KARL_HYBRID_VECTOR_WEIGHT=0).")

    retriever = HybridRetriever(vector_store)
    
    yield
    
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Karl - Union Contract RAG",
    description="AI-powered contract Q&A for union agreements",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class QueryRequest(BaseModel):
    """Request model for Q&A queries."""
    question: str = Field(..., description="User's question about the contract")
    union_local_id: str = Field(..., description="Union local identifier")
    contract_id: str = Field(..., description="Contract identifier")
    contract_version: str = Field(..., description="Contract version identifier")
    user_classification: Optional[str] = Field(None, description="User's job classification")
    hours_worked: int = Field(0, description="Total hours worked (for wage calculations)")
    months_employed: int = Field(0, description="Months employed (for courtesy clerk wages)")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")


class WageLookupRequest(BaseModel):
    """Request model for direct wage lookups."""
    classification: str = Field(..., description="Job classification")
    contract_id: Optional[str] = Field(None, description="Contract identifier")
    hours_worked: int = Field(0, description="Total hours worked")
    months_employed: int = Field(0, description="Months employed")
    effective_date: Optional[str] = Field(None, description="Contract effective date")


class Citation(BaseModel):
    """A contract citation."""
    article: Optional[int] = None
    section: Optional[int] = None
    citation: str
    quote: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for Q&A queries."""
    answer: str
    citations: list[str]
    sources: list[dict]
    intent_type: str
    escalation_required: bool
    union_local_id: Optional[str] = None
    contract_id: str
    contract_version: Optional[str] = None
    high_stakes_topic: bool = False
    active_urgent_context: bool = False
    escalation_policy: Optional[str] = None
    wage_info: Optional[dict] = None
    entitlement_info: Optional[dict] = None
    confidence: float
    verification_passed: bool
    # CAG (Context-Aware Generation) metrics
    hypothesis_titles: Optional[list[str]] = Field(None, description="Hypothesized section titles from Rosetta Stone layer")
    hypothesis_latency_ms: Optional[float] = Field(None, description="Latency of hypothesis generation in ms")
    full_article_expanded: Optional[bool] = Field(None, description="Whether full article expansion was triggered")
    winning_article: Optional[int] = Field(None, description="Article number that triggered full expansion")
    # Reranker metrics (Phase 5)
    reranker_latency_ms: Optional[float] = Field(None, description="Latency of LLM reranking in ms")
    reranker_position_changes: Optional[int] = Field(None, description="Number of chunks that changed position after reranking")
    # Interpreter metrics (Phase 4)
    interpretation_latency_ms: Optional[float] = Field(None, description="Latency of query interpretation in ms")
    search_angles_used: Optional[int] = Field(None, description="Number of search angles tried")


class WageResponse(BaseModel):
    """Response model for wage lookups."""
    contract_id: Optional[str] = None
    classification: str
    step: str
    rate: float
    effective_date: str
    citation: str
    table_evidence: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    chunks_loaded: int
    contract_chunks: int = 0
    active_contract_id: Optional[str] = None
    llm_available: bool


# Onboarding Models

class ContractOption(BaseModel):
    """Runtime-selectable contract metadata."""
    contract_id: str
    union_local_id: str
    region_id: Optional[str] = None
    contract_version: str
    employer: str
    term_start: Optional[str] = None
    term_end: Optional[str] = None


class ContractsResponse(BaseModel):
    """Catalog of available contracts for frontend selection."""
    default_contract_id: Optional[str] = None
    contracts: list[ContractOption]

class OnboardingOptionsResponse(BaseModel):
    """Available options for onboarding form."""
    classifications: list[dict]
    employment_types: list[dict]
    employers: list[str]
    default_contract_id: Optional[str] = None
    contracts: list[ContractOption] = Field(default_factory=list)


class ClassificationOption(BaseModel):
    """Classification option for a specific contract context."""
    value: str
    label: str
    wage_available: bool = True
    wage_key: Optional[str] = None
    source: Optional[str] = None


class ClassificationsResponse(BaseModel):
    """Contract-scoped classification options."""
    contract_id: str
    classifications: list[ClassificationOption]


class ProfileUpdateRequest(BaseModel):
    """Request to update user profile."""
    contract_id: Optional[str] = None
    classification: Optional[str] = None
    employment_type: Optional[str] = None  # "full_time" or "part_time"
    hire_date: Optional[str] = None  # ISO format: "2023-03-15"
    exact_hours: Optional[int] = None  # If user knows their exact hours


class ProfileResponse(BaseModel):
    """User profile with calculated fields."""
    session_id: str
    contract_id: str
    union_local_id: Optional[str] = None
    classification: Optional[str] = None
    classification_display: Optional[str] = None
    employment_type: Optional[str] = None
    hire_date: Optional[str] = None
    months_employed: Optional[int] = None
    estimated_hours: Optional[int] = None
    is_grandfathered: Optional[bool] = None
    is_complete: bool = False
    employer: str = ""


class WageEstimateResponse(BaseModel):
    """Wage lookup with estimate transparency."""
    classification: str
    classification_display: str
    current_rate: float
    current_step: str
    effective_date: str
    citation: str
    # Estimate transparency
    is_estimate: bool
    confidence: str  # "exact", "high", "medium", "low"
    basis: str  # How we calculated this
    disclaimer: str
    # Next step info
    next_step: Optional[str] = None
    next_rate: Optional[float] = None
    hours_to_next_step: Optional[int] = None
    table_evidence: list[dict] = Field(default_factory=list)
    # Verification guidance
    verification_message: str


# Endpoints

def _count_contract_chunks(contract_id: Optional[str]) -> int:
    """Count chunks available for a contract from chunk artifacts."""
    if not contract_id:
        return 0
    chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    if not chunks_file or not chunks_file.exists():
        return 0
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception:
        return 0

    allow_unscoped = len(list(MANIFESTS_DIR.glob("*.json"))) == 1
    count = 0
    for chunk in all_chunks:
        chunk_contract_id = chunk.get("contract_id")
        if chunk_contract_id == contract_id:
            count += 1
        elif allow_unscoped and chunk_contract_id in (None, ""):
            count += 1
    return count


@app.get("/api/health", response_model=HealthResponse)
async def health_check(contract_id: Optional[str] = None):
    """Check system health."""
    active_contract_id = contract_id or resolve_default_contract_id()
    contract_chunks = _count_contract_chunks(active_contract_id)
    chunks_loaded = vector_store.count() if vector_store else contract_chunks
    llm_available = get_genai_client() is not None

    return HealthResponse(
        status="healthy" if contract_chunks > 0 else "degraded",
        chunks_loaded=chunks_loaded,
        contract_chunks=contract_chunks,
        active_contract_id=active_contract_id,
        llm_available=llm_available
    )


@app.get("/api/contracts", response_model=ContractsResponse)
async def get_contracts():
    """List available contract manifests for frontend/runtime selection."""
    contracts = [ContractOption(**c) for c in list_contract_catalog()]
    return ContractsResponse(
        default_contract_id=resolve_default_contract_id(),
        contracts=contracts,
    )


# =============================================================================
# ONBOARDING & PROFILE ENDPOINTS
# =============================================================================

@app.get("/api/onboard/options", response_model=OnboardingOptionsResponse)
async def get_onboarding_options():
    """Get available options for the onboarding form."""
    contract_catalog = list_contract_catalog()
    employers = sorted({c.get("employer", "") for c in contract_catalog if c.get("employer")})
    return OnboardingOptionsResponse(
        classifications=[],
        employment_types=[
            {"value": "full_time", "label": "Full-Time (32+ hrs/week)"},
            {"value": "part_time", "label": "Part-Time (under 32 hrs/week)"},
        ],
        employers=employers,
        default_contract_id=None,
        contracts=[ContractOption(**c) for c in contract_catalog],
    )


@app.get("/api/classifications", response_model=ClassificationsResponse)
async def get_classifications(contract_id: Optional[str] = None):
    """Get contract-scoped job classifications for onboarding/settings UI."""
    effective_contract_id = contract_id or resolve_default_contract_id()
    if not effective_contract_id:
        raise HTTPException(status_code=404, detail="No contract manifests found")
    try:
        ensure_contract_manifest(effective_contract_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    options = get_classification_options(contract_id=effective_contract_id)
    return ClassificationsResponse(
        contract_id=effective_contract_id,
        classifications=[ClassificationOption(**o) for o in options],
    )


@app.get("/api/profile/{session_id}", response_model=ProfileResponse)
async def get_profile(session_id: str):
    """Get user profile for a session."""
    profile = get_user_profile(session_id)

    return ProfileResponse(
        session_id=session_id,
        contract_id=profile.contract_id,
        union_local_id=profile.union_local,
        classification=profile.classification,
        classification_display=resolve_classification_display_name(
            profile.classification,
            contract_id=profile.contract_id,
        ),
        employment_type=profile.employment_type.value if profile.employment_type else None,
        hire_date=profile.hire_date.isoformat() if profile.hire_date else None,
        months_employed=profile.months_employed,
        estimated_hours=profile.estimated_hours,
        is_grandfathered=profile.is_grandfathered,
        is_complete=profile.is_complete,
        employer=profile.employer,
    )


@app.put("/api/profile/{session_id}", response_model=ProfileResponse)
async def update_profile(session_id: str, request: ProfileUpdateRequest):
    """Update user profile."""
    updates = request.model_dump(exclude_none=True)
    if "contract_id" in updates:
        contract_id = updates["contract_id"]
        try:
            ensure_contract_manifest(contract_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    profile = update_user_profile(session_id, updates)

    return ProfileResponse(
        session_id=session_id,
        contract_id=profile.contract_id,
        union_local_id=profile.union_local,
        classification=profile.classification,
        classification_display=resolve_classification_display_name(
            profile.classification,
            contract_id=profile.contract_id,
        ),
        employment_type=profile.employment_type.value if profile.employment_type else None,
        hire_date=profile.hire_date.isoformat() if profile.hire_date else None,
        months_employed=profile.months_employed,
        estimated_hours=profile.estimated_hours,
        is_grandfathered=profile.is_grandfathered,
        is_complete=profile.is_complete,
        employer=profile.employer,
    )


@app.get("/api/wage/estimate/{session_id}", response_model=WageEstimateResponse)
async def get_wage_estimate(session_id: str):
    """
    Get wage estimate based on user profile.

    Uses hire date and employment type to estimate hours worked,
    then looks up the corresponding wage step.

    Always includes transparency about estimate confidence.
    """
    profile = get_user_profile(session_id)

    if not profile.classification:
        raise HTTPException(
            status_code=400,
            detail="Please set your job classification first. Use PUT /api/profile/{session_id}"
        )

    # Get hours estimate
    hours_estimate = estimate_hours_worked(profile)

    # Look up wage
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    estimated_hours = hours_estimate.estimated_hours if hours_estimate else 0
    months = profile.months_employed or 0

    class_opt = _classification_option_for_contract(
        contract_id=profile.contract_id,
        classification=profile.classification,
    )
    if class_opt and class_opt.get("wage_available") is False:
        role_label = str(class_opt.get("label") or profile.classification).strip()
        raise HTTPException(
            status_code=404,
            detail=(
                f"No Appendix A wage row is available for '{role_label}' "
                f"in contract '{profile.contract_id}'."
            ),
        )

    wage_info = retriever.lookup_wage(
        classification=profile.classification,
        hours_worked=estimated_hours,
        months_employed=months,
        contract_id=profile.contract_id,
    )

    if not wage_info:
        raise HTTPException(
            status_code=404,
            detail=f"Wage information not found for classification: {profile.classification}"
        )

    # Determine confidence and disclaimer
    if hours_estimate:
        is_estimate = hours_estimate.confidence != "exact"
        confidence = hours_estimate.confidence
        basis = hours_estimate.basis
        disclaimer = hours_estimate.disclaimer
    else:
        is_estimate = True
        confidence = "low"
        basis = "No hire date provided - showing starting rate."
        disclaimer = "Please provide your hire date for a more accurate estimate."

    # Verification message
    verification_message = (
        "To verify your exact wage rate:\n"
        "- Check your most recent pay stub\n"
        "- Visit the Company HR Portal\n"
        "- Ask your store manager or HR representative"
    )

    return WageEstimateResponse(
        classification=profile.classification,
        classification_display=resolve_classification_display_name(
            profile.classification,
            contract_id=profile.contract_id,
        ) or profile.classification,
        current_rate=wage_info["rate"],
        current_step=wage_info["step"],
        effective_date=wage_info["effective_date"],
        citation=wage_info["citation"],
        is_estimate=is_estimate,
        confidence=confidence,
        basis=basis,
        disclaimer=disclaimer,
        next_step=wage_info.get("next_step"),
        next_rate=wage_info.get("next_rate"),
        hours_to_next_step=wage_info.get("hours_to_next"),
        table_evidence=wage_info.get("table_evidence", []),
        verification_message=verification_message,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_contract(request: QueryRequest):
    """
    Answer a question about the union contract.

    Uses RAG to retrieve relevant contract sections and generate
    a grounded, citation-focused response.

    If session_id is provided, uses stored profile for personalization.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    # Contract context is required for strict tenant isolation.
    effective_contract_id = request.contract_id
    catalog_entry = get_contract_catalog_entry(effective_contract_id)
    if not catalog_entry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown contract_id '{request.contract_id}'.",
        )
    effective_contract_id = catalog_entry["contract_id"]
    try:
        ensure_contract_manifest(effective_contract_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Validate local/version context against manifest for auditability.
    manifest_path = MANIFESTS_DIR / f"{effective_contract_id}.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_union_local = manifest.get("union_local")
    if expected_union_local and request.union_local_id != expected_union_local:
        raise HTTPException(
            status_code=400,
            detail=(
                f"union_local_id mismatch. Expected '{expected_union_local}', "
                f"got '{request.union_local_id}'."
            ),
        )

    expected_contract_version = manifest.get("contract_version")
    if not expected_contract_version:
        term_start = manifest.get("term_start", "")
        term_end = manifest.get("term_end", "")
        expected_contract_version = f"{term_start}__{term_end}"

    if request.contract_version != expected_contract_version:
        raise HTTPException(
            status_code=400,
            detail=(
                f"contract_version mismatch. Expected '{expected_contract_version}', "
                f"got '{request.contract_version}'."
            ),
        )

    # Get user profile and conversation context for this session
    conversation_context = ""
    detected_entities = {}
    user_profile = None
    is_wage_estimate = False

    if request.session_id:
        # Get user profile
        profile = get_user_profile(request.session_id)
        if profile.has_basic_info:
            user_profile = profile.to_dict()
            # Add display name
            user_profile["classification_display"] = resolve_classification_display_name(
                profile.classification,
                contract_id=profile.contract_id,
            ) if profile.classification else None

        # Get conversation context
        ctx = get_session_context(request.session_id)
        conversation_context = ctx.get_full_context()

        # If this seems like a follow-up question, use context from previous turns
        followup_indicators = ["them", "that", "it", "they", "this", "those", "the same"]
        is_followup = any(ind in request.question.lower() for ind in followup_indicators)

        if is_followup and ctx.get_last_topic():
            detected_entities["topic"] = ctx.get_last_topic()

    # Use profile classification if not explicitly provided
    effective_classification = request.user_classification
    if not effective_classification and user_profile:
        effective_classification = user_profile.get("classification")

    # Use profile hours/months if not explicitly provided
    hours_worked = request.hours_worked
    months_employed = request.months_employed

    if user_profile and hours_worked == 0:
        hours_worked = user_profile.get("estimated_hours") or 0
        is_wage_estimate = True  # Mark as estimate if using profile data

    if user_profile and months_employed == 0:
        months_employed = user_profile.get("months_employed") or 0

    # Classify intent with user's classification for role-based boosting
    intent = classify_intent(
        request.question,
        user_classification=effective_classification,
        contract_id=effective_contract_id,
    )
    foreign_contract_reference = _mentions_foreign_contract_context(
        request.question,
        effective_contract_id,
    )

    # Deterministic tenant guard: explicit foreign-contract references must not
    # flow into retrieval/generation for the active contract context.
    if foreign_contract_reference:
        answer = _FOREIGN_CONTRACT_UNAVAILABLE_ANSWER
        verification = verify_response(
            response=answer,
            chunks=[],
            requires_escalation=False,
        )
        formatted = format_response_with_sources(answer, [], None)
        return QueryResponse(
            answer=formatted["response"],
            citations=formatted["citations"],
            sources=formatted["sources"],
            intent_type=intent.intent_type,
            escalation_required=False,
            union_local_id=request.union_local_id,
            contract_id=effective_contract_id,
            contract_version=request.contract_version,
            high_stakes_topic=intent.high_stakes_topic,
            active_urgent_context=intent.active_urgent_context,
            escalation_policy=intent.escalation_policy,
            wage_info=None,
            confidence=verification.confidence,
            verification_passed=verification.is_valid,
            hypothesis_titles=None,
            hypothesis_latency_ms=None,
            full_article_expanded=False,
            winning_article=None,
            reranker_latency_ms=None,
            reranker_position_changes=None,
            interpretation_latency_ms=None,
            search_angles_used=None,
        )

    # Deterministic guardrail: when a role is selected but has no Appendix A wage
    # mapping for this contract, return explicit unavailability for wage intents.
    if intent.intent_type == "wage" and effective_classification:
        class_opt = _classification_option_for_contract(
            contract_id=effective_contract_id,
            classification=effective_classification,
        )
        if class_opt and class_opt.get("wage_available") is False:
            role_label = str(class_opt.get("label") or effective_classification).strip()
            answer = (
                f"I cannot find a contract wage-table rate for '{role_label}' in this contract. "
                "Your selected role appears in contract references, but no contract-scoped wage row is available. "
                "Please verify your classification with your steward."
            )
            verification = verify_response(
                response=answer,
                chunks=[],
                requires_escalation=False,
            )
            formatted = format_response_with_sources(answer, [], None)
            return QueryResponse(
                answer=formatted["response"],
                citations=formatted["citations"],
                sources=formatted["sources"],
                intent_type=intent.intent_type,
                escalation_required=False,
                union_local_id=request.union_local_id,
                contract_id=effective_contract_id,
                contract_version=request.contract_version,
                high_stakes_topic=intent.high_stakes_topic,
                active_urgent_context=intent.active_urgent_context,
                escalation_policy=intent.escalation_policy,
                wage_info=None,
                confidence=verification.confidence,
                verification_passed=verification.is_valid,
                hypothesis_titles=None,
                hypothesis_latency_ms=None,
                full_article_expanded=False,
                winning_article=None,
                reranker_latency_ms=None,
                reranker_position_changes=None,
                interpretation_latency_ms=None,
                search_angles_used=None,
            )

    # Deterministic guardrail: wage estimates require a classification context.
    if intent.intent_type == "wage" and not effective_classification:
        retrieval_result = await asyncio.to_thread(
            retriever.multi_angle_retrieve,
            query=request.question,
            intent=intent,
            n_results=5,
            hours_worked=hours_worked,
            months_employed=months_employed,
            contract_id=effective_contract_id,
        )
        chunks = retrieval_result.get("chunks", [])
        answer = (
            "I can estimate your pay rate once your job classification is set. "
            "Please select your role in onboarding/settings, then ask again. "
            "I will use Appendix A for your selected contract."
        )
        verification = verify_response(
            response=answer,
            chunks=chunks,
            requires_escalation=False,
        )
        formatted = format_response_with_sources(answer, chunks, None)
        return QueryResponse(
            answer=formatted["response"],
            citations=formatted["citations"],
            sources=formatted["sources"],
            intent_type=intent.intent_type,
            escalation_required=False,
            union_local_id=request.union_local_id,
            contract_id=effective_contract_id,
            contract_version=request.contract_version,
            high_stakes_topic=intent.high_stakes_topic,
            active_urgent_context=intent.active_urgent_context,
            escalation_policy=intent.escalation_policy,
            wage_info=None,
            confidence=verification.confidence,
            verification_passed=verification.is_valid,
            hypothesis_titles=None,
            hypothesis_latency_ms=None,
            full_article_expanded=False,
            winning_article=None,
            reranker_latency_ms=None,
            reranker_position_changes=None,
            interpretation_latency_ms=None,
            search_angles_used=None,
        )

    # Retrieve relevant chunks and wage info using multi-angle retrieval
    # This uses deep query interpretation for better semantic matching
    retrieval_result = await asyncio.to_thread(
        retriever.multi_angle_retrieve,
        query=request.question,
        intent=intent,
        n_results=5,
        hours_worked=hours_worked,
        months_employed=months_employed,
        contract_id=effective_contract_id,
    )
    
    chunks = retrieval_result["chunks"]
    wage_info = retrieval_result["wage_info"]
    entitlement_info = retrieval_result.get("entitlement_info")
    escalation_required = retrieval_result["escalation_required"]
    query_expansions = retrieval_result.get("query_expansions", [])

    # Extract CAG (Rosetta Stone) metrics
    hypothesis_result = retrieval_result.get("hypothesis_result")
    hypothesis_titles = None
    hypothesis_latency_ms = None
    if hypothesis_result and hypothesis_result.success:
        hypothesis_titles = hypothesis_result.hypothesized_titles
        hypothesis_latency_ms = hypothesis_result.latency_ms

    # Extract Phase 4: Query Interpretation metrics
    interpretation = retrieval_result.get("interpretation")
    interpretation_latency_ms = None
    search_angles_used = retrieval_result.get("search_angles_used", 1)
    explicit_articles = retrieval_result.get("explicit_articles_fetched", [])
    anchor_articles = _normalize_article_anchors(
        list(getattr(intent, "relevant_articles", []) or []) + list(explicit_articles or [])
    )
    if interpretation and interpretation.success:
        interpretation_latency_ms = interpretation.latency_ms

    # Extract Phase 5: Reranker metrics
    reranker_result = retrieval_result.get("reranker_result")
    reranker_latency_ms = None
    reranker_position_changes = None
    if reranker_result and reranker_result.success:
        reranker_latency_ms = reranker_result.latency_ms
        reranker_position_changes = reranker_result.position_changes

    # Check if full article expansion was triggered
    full_article_expanded = any(c.get('is_full_article_context') for c in chunks)
    winning_article = next(
        (c.get('winning_article') for c in chunks if c.get('is_full_article_context')),
        None
    )

    # Deterministic vacation entitlement path:
    # answer accrual-amount questions from ingestion-owned entitlement artifacts.
    if (
        str(intent.topic or "").strip().lower() == "vacation"
        and _is_vacation_entitlement_query(request.question)
        and not escalation_required
    ):
        hire_date_hint = str((user_profile or {}).get("hire_date") or "").strip() or None
        enriched_entitlement = await asyncio.to_thread(
            retriever.lookup_vacation_entitlement,
            months_employed=months_employed,
            hours_worked=hours_worked,
            hire_date=hire_date_hint,
            contract_id=effective_contract_id,
        )
        if enriched_entitlement:
            entitlement_info = enriched_entitlement
        if entitlement_info:
            answer = _build_vacation_entitlement_answer(entitlement_info)
            verification = verify_response(
                response=answer,
                chunks=chunks,
                requires_escalation=False,
            )
            sources = _entitlement_sources(entitlement_info)
            citations = []
            for source in sources:
                citation = str(source.get("citation") or "").strip()
                if citation and citation not in citations:
                    citations.append(citation)
            if not citations:
                citation_text = str(entitlement_info.get("citation") or "").strip()
                if citation_text:
                    citations = [c.strip() for c in citation_text.split(";") if c.strip()]

            if request.session_id:
                ctx = get_session_context(request.session_id)
                ctx.add_turn(
                    question=request.question,
                    answer=answer,
                    citations=citations,
                    detected_entities={
                        "topic": intent.topic,
                        "classification": request.user_classification or intent.classification,
                    }
                )

            return QueryResponse(
                answer=answer,
                citations=citations,
                sources=sources,
                intent_type=intent.intent_type,
                escalation_required=False,
                union_local_id=request.union_local_id,
                contract_id=effective_contract_id,
                contract_version=request.contract_version,
                high_stakes_topic=intent.high_stakes_topic,
                active_urgent_context=intent.active_urgent_context,
                escalation_policy=intent.escalation_policy,
                wage_info=None,
                entitlement_info=entitlement_info,
                confidence=verification.confidence,
                verification_passed=verification.is_valid,
                hypothesis_titles=hypothesis_titles,
                hypothesis_latency_ms=hypothesis_latency_ms,
                full_article_expanded=full_article_expanded,
                winning_article=winning_article,
                reranker_latency_ms=reranker_latency_ms,
                reranker_position_changes=reranker_position_changes,
                interpretation_latency_ms=interpretation_latency_ms,
                search_angles_used=search_angles_used,
            )
    
    # Only include wage info in prompt if this is actually a wage query
    # This prevents erroneous wage artifacts appearing on unrelated questions
    question_lower = request.question.lower()

    # Specific wage-related phrases (not just "pay" which matches "vacation pay")
    wage_phrases = [
        "my wage", "my pay", "my rate", "my salary", "my hourly",
        "how much do i make", "how much will i make", "how much should i make",
        "how much should i be making", "what should i be making", "what should i make",
        "what do i make", "what am i making", "what's my pay",
        "pay rate", "wage rate", "hourly rate", "starting pay",
        "get paid", "being paid", "am i paid", "appendix a"
    ]

    # Exclude phrases that contain "pay" but aren't about wages
    exclude_phrases = ["vacation pay", "holiday pay", "sick pay", "pay stub", "pay period"]
    has_exclude = any(ep in question_lower for ep in exclude_phrases)

    is_wage_query = (
        intent.intent_type == "wage" or
        (any(wp in question_lower for wp in wage_phrases) and not has_exclude)
    )
    
    # Build prompt with context, user profile, and verification guidance
    system_prompt = build_prompt(
        query=request.question,
        chunks=chunks,
        wage_info=wage_info if is_wage_query else None,
        contract_context={
            "contract_id": effective_contract_id,
            "union_local_id": request.union_local_id,
            "contract_version": request.contract_version,
            "employer": manifest.get("employer", ""),
        },
        requires_escalation=escalation_required,
        query_expansions=query_expansions,
        user_classification=effective_classification,
        conversation_context=conversation_context,
        user_profile=user_profile,
        is_wage_estimate=is_wage_estimate and is_wage_query
    )
    
    # Generate response (pass chunks for fallback)
    answer = await generate_response(request.question, system_prompt, chunks)

    # Add escalation if missing but required
    answer = add_escalation_if_missing(answer, escalation_required)

    # Deterministic evidence-gap recovery:
    # if the model says "not available/cannot find" despite strong evidence,
    # run one broader retrieval pass and regenerate once before returning.
    anchor_evidence_present = (
        _has_article_anchor_evidence(anchor_articles, chunks, min_article_hits=2)
        and _query_supports_topic_recovery(intent.topic, request.question)
    )
    recovery_evidence_present = (not foreign_contract_reference) and (
        _has_strong_evidence_for_query(request.question, chunks) or anchor_evidence_present
    )
    if _is_unavailable_answer(answer) and not escalation_required and recovery_evidence_present:
        retry_result = await asyncio.to_thread(
            retriever.multi_angle_retrieve,
            query=request.question,
            intent=intent,
            n_results=8,
            hours_worked=hours_worked,
            months_employed=months_employed,
            contract_id=effective_contract_id,
        )

        retry_chunks = retry_result.get("chunks", [])
        merged_chunks = _merge_unique_chunks(chunks, retry_chunks, limit=12)
        if merged_chunks:
            chunks = merged_chunks
            full_article_expanded = any(c.get('is_full_article_context') for c in chunks)
            winning_article = next(
                (c.get('winning_article') for c in chunks if c.get('is_full_article_context')),
                None
            )

        if anchor_articles and hasattr(retriever, "_ensure_topic_article_coverage"):
            chunks = retriever._ensure_topic_article_coverage(
                chunks=chunks,
                article_numbers=anchor_articles,
                contract_id=effective_contract_id,
                max_additional=3,
                query_text=request.question,
            )
        if anchor_articles and hasattr(retriever, "_prioritize_topic_articles"):
            chunks = retriever._prioritize_topic_articles(
                chunks=chunks,
                article_numbers=anchor_articles,
                topic=intent.topic,
                query_text=request.question,
            )
        if anchor_articles:
            seed_chunks = [c for c in chunks if c.get("is_topic_seed")]
            non_seed_chunks = [c for c in chunks if not c.get("is_topic_seed")]
            if seed_chunks:
                chunks = seed_chunks + non_seed_chunks

        retry_wage_info = retry_result.get("wage_info")
        if not wage_info and retry_wage_info:
            wage_info = retry_wage_info
        retry_entitlement_info = retry_result.get("entitlement_info")
        if not entitlement_info and retry_entitlement_info:
            entitlement_info = retry_entitlement_info

        retry_expansions = retry_result.get("query_expansions", [])
        if retry_expansions:
            merged_expansions = list(dict.fromkeys(list(query_expansions) + list(retry_expansions)))
            query_expansions = merged_expansions

        search_angles_used = max(search_angles_used, int(retry_result.get("search_angles_used", 1) or 1))

        retry_prompt = build_prompt(
            query=request.question,
            chunks=chunks,
            wage_info=wage_info if is_wage_query else None,
            contract_context={
                "contract_id": effective_contract_id,
                "union_local_id": request.union_local_id,
                "contract_version": request.contract_version,
                "employer": manifest.get("employer", ""),
            },
            requires_escalation=escalation_required,
            query_expansions=query_expansions,
            user_classification=effective_classification,
            conversation_context=conversation_context,
            user_profile=user_profile,
            is_wage_estimate=is_wage_estimate and is_wage_query
        )
        answer = await generate_response(request.question, retry_prompt, chunks)
        answer = add_escalation_if_missing(answer, escalation_required)

        # Final deterministic guard: if model still refuses but evidence is strong,
        # return chunk-grounded fallback rather than claiming information is absent.
        anchor_evidence_present = (
            _has_article_anchor_evidence(anchor_articles, chunks, min_article_hits=2)
            and _query_supports_topic_recovery(intent.topic, request.question)
        )
        recovery_evidence_present = (not foreign_contract_reference) and (
            _has_strong_evidence_for_query(request.question, chunks) or anchor_evidence_present
        )
        if _is_unavailable_answer(answer) and recovery_evidence_present:
            answer = generate_fallback_response(chunks)
    
    # Verify response
    verification = verify_response(
        response=answer,
        chunks=chunks,
        requires_escalation=escalation_required
    )
    
    # Format response with sources (only include wage_info for wage queries)
    formatted = format_response_with_sources(answer, chunks, wage_info if is_wage_query else None)
    
    # Save turn to conversation context
    if request.session_id:
        ctx = get_session_context(request.session_id)
        ctx.add_turn(
            question=request.question,
            answer=formatted["response"],
            citations=formatted["citations"],
            detected_entities={
                "topic": intent.topic,
                "classification": request.user_classification or intent.classification,
            }
        )
    
    return QueryResponse(
        answer=formatted["response"],
        citations=formatted["citations"],
        sources=formatted["sources"],
        intent_type=intent.intent_type,
        escalation_required=escalation_required,
        union_local_id=request.union_local_id,
        contract_id=effective_contract_id,
        contract_version=request.contract_version,
        high_stakes_topic=intent.high_stakes_topic,
        active_urgent_context=intent.active_urgent_context,
        escalation_policy=intent.escalation_policy,
        wage_info=wage_info if is_wage_query else None,
        entitlement_info=entitlement_info,
        confidence=verification.confidence,
        verification_passed=verification.is_valid,
        # CAG metrics
        hypothesis_titles=hypothesis_titles,
        hypothesis_latency_ms=hypothesis_latency_ms,
        full_article_expanded=full_article_expanded,
        winning_article=winning_article,
        # Reranker metrics (Phase 5)
        reranker_latency_ms=reranker_latency_ms,
        reranker_position_changes=reranker_position_changes,
        # Interpreter metrics (Phase 4)
        interpretation_latency_ms=interpretation_latency_ms,
        search_angles_used=search_angles_used
    )


@app.post("/api/wage", response_model=WageResponse)
async def lookup_wage(request: WageLookupRequest):
    """
    Look up wage rate for a specific classification and experience level.
    
    Returns deterministic wage data from Appendix A.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    effective_contract_id = _resolve_contract_id_for_viewer(request.contract_id)
    class_opt = _classification_option_for_contract(
        contract_id=effective_contract_id,
        classification=request.classification,
    )
    if class_opt and class_opt.get("wage_available") is False:
        role_label = str(class_opt.get("label") or request.classification).strip()
        raise HTTPException(
            status_code=404,
            detail=(
                f"No Appendix A wage row is available for '{role_label}' "
                f"in contract '{effective_contract_id}'."
            ),
        )

    wage_info = retriever.lookup_wage(
        classification=request.classification,
        hours_worked=request.hours_worked,
        months_employed=request.months_employed,
        effective_date=request.effective_date,
        contract_id=effective_contract_id,
    )
    
    if not wage_info:
        raise HTTPException(
            status_code=404, 
            detail=f"Wage information not found for classification: {request.classification}"
        )
    
    return WageResponse(
        contract_id=effective_contract_id,
        classification=wage_info["classification"],
        step=wage_info["step"],
        rate=wage_info["rate"],
        effective_date=wage_info["effective_date"],
        citation=wage_info["citation"],
        table_evidence=wage_info.get("table_evidence", []),
    )


# =============================================================================
# CONTRACT VIEWER ENDPOINTS
# =============================================================================

class ManifestResponse(BaseModel):
    """Response model for contract manifest/TOC."""
    contract_id: str
    article_titles: dict[str, str]
    total_articles: int


class ArticleSectionResponse(BaseModel):
    """Response model for a single section."""
    section_num: int
    subsection: Optional[str] = None
    citation: str
    content: str
    summary: Optional[str] = None


class ArticleResponse(BaseModel):
    """Response model for full article."""
    article_num: int
    article_title: str
    sections: list[ArticleSectionResponse]


class SectionResponse(BaseModel):
    """Response model for a single section lookup."""
    article_num: int
    article_title: str
    section_num: int
    subsection: Optional[str] = None
    citation: str
    content: str
    summary: Optional[str] = None


class PdfLocationResponse(BaseModel):
    """Response model for PDF deep-link location."""
    contract_id: str
    pdf_url: Optional[str] = None
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    matched_by: str = "none"


def _resolve_contract_id_for_viewer(contract_id: Optional[str]) -> str:
    """Resolve contract_id for viewer endpoints with single-manifest fallback."""
    if contract_id:
        entry = get_contract_catalog_entry(contract_id)
        if not entry:
            try:
                ensure_contract_manifest(contract_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            raise HTTPException(
                status_code=400,
                detail=f"Contract '{contract_id}' is not active in the runtime catalog.",
            )
        return entry["contract_id"]

    manifests = sorted(MANIFESTS_DIR.glob("*.json"))
    if len(manifests) == 1:
        return manifests[0].stem

    raise HTTPException(
        status_code=400,
        detail="contract_id is required when multiple manifests are present",
    )


def _viewer_allow_legacy_unscoped_chunks() -> bool:
    """Allow unscoped chunk fallback only in single-manifest mode."""
    return len(list(MANIFESTS_DIR.glob("*.json"))) == 1


def _viewer_article_titles_cache_key(contract_id: str) -> tuple[str, str, int, int, str, int, bool]:
    manifest_file = MANIFESTS_DIR / f"{contract_id}.json"
    manifest_mtime_ns = manifest_file.stat().st_mtime_ns if manifest_file.exists() else -1
    outline_file = resolve_contract_outline_file(contract_id=contract_id, allow_shared_fallback=True)
    outline_path = str(outline_file.resolve()) if outline_file and outline_file.exists() else ""
    outline_mtime_ns = outline_file.stat().st_mtime_ns if outline_file and outline_file.exists() else -1
    chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    chunks_path = str(chunks_file.resolve()) if chunks_file and chunks_file.exists() else ""
    chunks_mtime_ns = chunks_file.stat().st_mtime_ns if chunks_file and chunks_file.exists() else -1
    allow_unscoped = _viewer_allow_legacy_unscoped_chunks()
    return (
        contract_id,
        outline_path,
        outline_mtime_ns,
        manifest_mtime_ns,
        chunks_path,
        chunks_mtime_ns,
        allow_unscoped,
    )


@lru_cache(maxsize=64)
def _build_viewer_article_titles_cached(
    contract_id: str,
    outline_path: str,
    outline_mtime_ns: int,
    manifest_mtime_ns: int,
    chunks_path: str,
    chunks_mtime_ns: int,
    allow_unscoped: bool,
) -> dict[str, str]:
    """
    Build manifest article titles with chunk-backed fallback for missing keys.

    Some manifests can have incomplete `article_titles` even when chunks include
    additional article numbers. This keeps Contract-tab TOC complete.
    """
    _ = outline_mtime_ns, manifest_mtime_ns, chunks_mtime_ns  # cache busting sentinels
    if outline_path:
        outline = load_contract_outline(Path(outline_path))
        outline_titles = article_titles_from_outline(outline)
        if outline_titles:
            return dict(sorted(outline_titles.items(), key=lambda kv: int(kv[0])))

    manifest_file = MANIFESTS_DIR / f"{contract_id}.json"
    manifest_titles: dict[str, str] = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        raw_titles = manifest.get("article_titles") or {}
        if isinstance(raw_titles, dict):
            for raw_key, raw_title in raw_titles.items():
                key_str = str(raw_key).strip()
                if not key_str.isdigit():
                    continue
                article_num = int(key_str)
                if article_num <= 0:
                    continue
                title = str(raw_title or "").strip() or f"Article {article_num}"
                manifest_titles[str(article_num)] = title

    if not chunks_path:
        return dict(sorted(manifest_titles.items(), key=lambda kv: int(kv[0])))

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception:
        return dict(sorted(manifest_titles.items(), key=lambda kv: int(kv[0])))

    for chunk in all_chunks:
        chunk_contract_id = chunk.get("contract_id")
        if not (
            chunk_contract_id == contract_id
            or (allow_unscoped and chunk_contract_id in (None, ""))
        ):
            continue
        raw_article = chunk.get("article_num")
        try:
            article_num = int(raw_article)
        except Exception:
            continue
        if article_num <= 0:
            continue
        key = str(article_num)
        existing = str(manifest_titles.get(key) or "").strip()
        chunk_title = str(chunk.get("article_title") or "").strip()
        if not existing:
            manifest_titles[key] = chunk_title or f"Article {article_num}"

    return dict(sorted(manifest_titles.items(), key=lambda kv: int(kv[0])))


def _build_viewer_article_titles(contract_id: str) -> dict[str, str]:
    return _build_viewer_article_titles_cached(*_viewer_article_titles_cache_key(contract_id))


@lru_cache(maxsize=32)
def _resolve_contract_pdf_path(contract_id: str) -> Optional[Path]:
    return resolve_contract_pdf_source_path(contract_id)


@lru_cache(maxsize=16)
def _build_pdf_navigation_index(contract_id: str) -> dict:
    nav_path = resolve_pdf_nav_index_file(
        contract_id=contract_id,
        allow_shared_fallback=True,
    )
    if nav_path and nav_path.exists():
        loaded = load_pdf_nav_index(nav_path)
        runtime_maps = to_runtime_navigation_maps(loaded)
        if runtime_maps["article_pages"] or runtime_maps["section_pages"]:
            return runtime_maps

    generated = build_pdf_nav_index(contract_id=contract_id)
    return to_runtime_navigation_maps(generated)


@lru_cache(maxsize=16)
def _build_table_navigation_index(contract_id: str) -> dict:
    nav_path = resolve_table_nav_index_file(
        contract_id=contract_id,
        allow_shared_fallback=True,
    )
    if nav_path and nav_path.exists():
        loaded = load_table_nav_index(nav_path)
        runtime_maps = to_runtime_table_navigation_maps(loaded)
        if runtime_maps["table_pages"]:
            return runtime_maps

    generated = build_table_nav_index(contract_id=contract_id)
    return to_runtime_table_navigation_maps(generated)


@app.get("/api/contract-pdf")
async def get_contract_pdf(contract_id: Optional[str] = None):
    """Serve the active contract PDF for inline viewing."""
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    pdf_path = _resolve_contract_pdf_path(effective_contract_id)
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Contract PDF not found")

    safe_name = pdf_path.name.replace('"', '')
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.get("/api/pdf-location", response_model=PdfLocationResponse)
async def get_pdf_location(
    article_num: Optional[int] = None,
    section_num: Optional[int] = None,
    subsection: Optional[str] = None,
    table_id: Optional[str] = None,
    row_index: Optional[int] = None,
    contract_id: Optional[str] = None,
):
    """
    Resolve best-effort PDF page for a citation target.

    Supports table-backed citations first, then section/article fallback.
    Subsection and row_index are accepted for API compatibility.
    """
    _ = subsection, row_index  # reserved for future fine-grained mapping
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    pdf_path = _resolve_contract_pdf_path(effective_contract_id)
    if not pdf_path or not pdf_path.exists():
        return PdfLocationResponse(contract_id=effective_contract_id)

    normalized_article_num: Optional[int] = None
    try:
        if article_num is not None:
            parsed_article = int(article_num)
            if parsed_article > 0:
                normalized_article_num = parsed_article
    except Exception:
        normalized_article_num = None

    normalized_section_num: Optional[int] = None
    try:
        if section_num is not None:
            parsed_section = int(section_num)
            if parsed_section > 0:
                normalized_section_num = parsed_section
    except Exception:
        normalized_section_num = None

    normalized_table_id = str(table_id or "").strip()
    page_number: Optional[int] = None
    matched_by = "none"

    table_index = _build_table_navigation_index(effective_contract_id)
    table_pages = table_index.get("table_pages") or {}
    table_meta = table_index.get("table_meta") or {}
    if normalized_table_id:
        page_number = table_pages.get(normalized_table_id)
        if page_number is None:
            lowered = normalized_table_id.lower()
            for key, value in table_pages.items():
                if str(key).lower() != lowered:
                    continue
                normalized_table_id = str(key)
                page_number = value
                break

        if normalized_article_num is None:
            table_ref = table_meta.get(normalized_table_id) or {}
            candidate_article = table_ref.get("article_num")
            if isinstance(candidate_article, int) and candidate_article > 0:
                normalized_article_num = candidate_article
        if normalized_section_num is None:
            table_ref = table_meta.get(normalized_table_id) or {}
            candidate_section = table_ref.get("section_num")
            if isinstance(candidate_section, int) and candidate_section > 0:
                normalized_section_num = candidate_section

        if page_number is not None:
            matched_by = "table_row" if row_index is not None else "table"

    index = _build_pdf_navigation_index(effective_contract_id)

    if page_number is None and normalized_article_num is not None and normalized_section_num is not None:
        key = f"{normalized_article_num}:{normalized_section_num}"
        page_number = (index.get("section_pages") or {}).get(key)
        if page_number is not None:
            matched_by = "section"

    if page_number is None and normalized_article_num is not None:
        page_number = (index.get("article_pages") or {}).get(normalized_article_num)
        if page_number is not None:
            matched_by = "article"

    return PdfLocationResponse(
        contract_id=effective_contract_id,
        pdf_url=f"/api/contract-pdf?contract_id={effective_contract_id}",
        page_number=page_number,
        total_pages=index.get("total_pages") or table_index.get("total_pages"),
        matched_by=matched_by,
    )


@app.get("/api/manifest", response_model=ManifestResponse)
async def get_manifest(contract_id: Optional[str] = None):
    """
    Get contract table of contents.

    Returns article numbers mapped to their titles for navigation.
    """
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    manifest_file = MANIFESTS_DIR / f"{effective_contract_id}.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Contract manifest not found")
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    article_titles = _build_viewer_article_titles(effective_contract_id)

    return ManifestResponse(
        contract_id=manifest.get("contract_id", "unknown"),
        article_titles=article_titles,
        total_articles=len(article_titles)
    )


@app.get("/api/article/{article_num}", response_model=ArticleResponse)
async def get_article(article_num: int, contract_id: Optional[str] = None):
    """
    Get all sections for a specific article.

    Returns the article title and all sections with their content.
    """
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    chunks_file = resolve_chunk_file(contract_id=effective_contract_id, allow_shared_fallback=True)

    if not chunks_file or not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Contract chunks not found")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Filter to this article
    allow_unscoped = _viewer_allow_legacy_unscoped_chunks()
    article_chunks = [
        c for c in all_chunks
        if c.get('article_num') == article_num
        and (
            c.get("contract_id") == effective_contract_id
            or (allow_unscoped and c.get("contract_id") in (None, ""))
        )
    ]

    if not article_chunks:
        raise HTTPException(status_code=404, detail=f"Article {article_num} not found")

    # Sort by section number, then subsection
    article_chunks.sort(key=lambda x: (x.get('section_num') or 0, x.get('subsection') or ''))

    # Get article title from first chunk
    article_title = article_chunks[0].get('article_title', f'Article {article_num}')

    # Build sections list
    sections = []
    for chunk in article_chunks:
        sections.append(ArticleSectionResponse(
            section_num=chunk.get('section_num') or 0,
            subsection=chunk.get('subsection'),
            citation=chunk.get('citation', ''),
            content=chunk.get('content_with_tables') or chunk.get('content', ''),
            summary=chunk.get('summary')
        ))

    return ArticleResponse(
        article_num=article_num,
        article_title=article_title,
        sections=sections
    )


@app.get("/api/section/{article_num}/{section_num}", response_model=SectionResponse)
async def get_section(
    article_num: int,
    section_num: int,
    subsection: str = None,
    contract_id: Optional[str] = None,
):
    """
    Get a specific section from an article.

    Used for citation popover previews.

    Args:
        article_num: The article number
        section_num: The section number
        subsection: Optional subsection (e.g., 'a', 'b', 'c') for filtering
    """
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    chunks_file = resolve_chunk_file(contract_id=effective_contract_id, allow_shared_fallback=True)

    if not chunks_file or not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Contract chunks not found")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Find the specific section
    allow_unscoped = _viewer_allow_legacy_unscoped_chunks()
    matching_chunks = [
        c for c in all_chunks
        if c.get('article_num') == article_num and c.get('section_num') == section_num
        and (
            c.get("contract_id") == effective_contract_id
            or (allow_unscoped and c.get("contract_id") in (None, ""))
        )
    ]

    if not matching_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Article {article_num}, Section {section_num} not found"
        )

    # If subsection specified, filter to just that subsection
    if subsection:
        subsection_chunks = [
            c for c in matching_chunks
            if (c.get('subsection') or '').lower() == subsection.lower()
        ]
        if subsection_chunks:
            matching_chunks = subsection_chunks
        # If subsection not found, fall back to showing all subsections

    # Use the first matching chunk (there may be subsections)
    chunk = matching_chunks[0]

    # If there are multiple subsections, combine content
    if len(matching_chunks) > 1:
        combined_content = "\n\n".join([
            f"**{c.get('subsection', '')}**: {c.get('content_with_tables') or c.get('content', '')}"
            if c.get('subsection') else (c.get('content_with_tables') or c.get('content', ''))
            for c in matching_chunks
        ])
    else:
        combined_content = chunk.get('content_with_tables') or chunk.get('content', '')

    return SectionResponse(
        article_num=article_num,
        article_title=chunk.get('article_title', f'Article {article_num}'),
        section_num=section_num,
        subsection=chunk.get('subsection'),
        citation=chunk.get('citation', f'Article {article_num}, Section {section_num}'),
        content=combined_content,
        summary=chunk.get('summary')
    )


# Track rate limit state
_rate_limit_until = 0


async def generate_response(question: str, system_prompt: str, chunks: list = None) -> str:
    """
    Generate LLM response using Gemini with retry logic.
    
    Implements exponential backoff for rate limits (429 errors).
    Falls back to showing raw chunks if all retries fail.
    """
    global _rate_limit_until
    
    client = get_genai_client()

    if not client:
        return generate_fallback_response(chunks)
    
    # Check if we're in a rate limit cooldown
    if time.time() < _rate_limit_until:
        wait_time = int(_rate_limit_until - time.time())
        print(f"Rate limited, waiting {wait_time}s before retry")
        return generate_fallback_response(chunks)
    
    # Retry with exponential backoff
    max_retries = 3
    base_delay = 1  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=LLM_MODEL,
                contents=question,
                config=_genai_sdk.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return response.text
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit (429) errors
            if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
                print(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    # Set a cooldown period after exhausting retries
                    _rate_limit_until = time.time() + 30  # 30 second cooldown
                    print("Rate limit retries exhausted, entering 30s cooldown")
            else:
                print(f"LLM error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay)
    
    # All retries failed, use fallback
    return generate_fallback_response(chunks)


def generate_fallback_response(chunks: list = None) -> str:
    """
    Generate a helpful fallback response showing retrieved chunks.
    
    Workers still get useful information even when the LLM is unavailable.
    """
    if not chunks:
        return (
            "I'm currently unable to search the contract. "
            "Please try again in a moment, or contact your steward directly."
        )
    
    # Build a response from the raw chunks
    response_parts = [
        "I found the following relevant sections from your contract:\n"
    ]
    
    for i, chunk in enumerate(chunks[:4], 1):  # Show top 4 chunks
        citation = chunk.get('citation', 'Unknown section')
        content = chunk.get('content', '')[:500]  # Limit content length
        
        # Clean up the content
        content = content.strip()
        if len(chunk.get('content', '')) > 500:
            content += "..."
        
        response_parts.append(f"**{citation}**:\n{content}\n")
    
    response_parts.append(
        "\n*Note: I couldn't generate a synthesized answer right now. "
        "Please review the sections above or contact your steward for help interpreting this information.*"
    )
    
    return "\n".join(response_parts)


# =============================================================================
# STATIC FILE SERVING (Frontend)
# =============================================================================

# Get the frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_frontend():
    """Serve the modular frontend HTML page."""
    index_path = FRONTEND_DIR / "modular" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. API is running at /api/"}


@app.get("/modular")
async def serve_modular_frontend():
    """Serve the modularized frontend app."""
    index_path = FRONTEND_DIR / "modular" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Modular frontend not found")


# Mount static files for any additional frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# Run with: uvicorn backend.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

