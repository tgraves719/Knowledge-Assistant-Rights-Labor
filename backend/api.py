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
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    CHUNKS_DIR,
    DATA_DIR,
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
    resolve_contract_source_json_path,
    to_runtime_navigation_maps,
)
from backend.table_nav_index import (
    build_table_nav_index,
    load_table_nav_index,
    to_runtime_table_navigation_maps,
)
from backend.effective_contracts import (
    load_effective_contract,
    resolve_contract_source_pdf_path,
    resolve_latest_effective_content_hash,
    resolve_latest_effective_version_id,
)
from backend.source_docs import resolve_source_doc_pdf_name, source_doc_applies_to_contract
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
    "premiums": ("premium", "premiums", "night premium", "sunday premium", "holiday premium"),
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


def _is_definition_locator_query(question: str) -> bool:
    q = str(question or "").lower()
    return bool(
        re.search(r"\b(where|which)\b", q)
        and re.search(r"\b(defined|definition|rules?)\b", q)
    )


def _primary_topic_signal(question: str) -> tuple[Optional[str], tuple[str, ...]]:
    q = str(question or "").lower()
    best_topic: Optional[str] = None
    best_signals: tuple[str, ...] = ()
    best_score = 0
    for topic, signals in _TOPIC_RECOVERY_SIGNALS.items():
        hits = sum(1 for signal in signals if signal and signal in q)
        if hits > best_score:
            best_score = hits
            best_topic = topic
            best_signals = tuple(signals)
    return best_topic, best_signals


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

    for chunk in chunks[:12]:
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


def _is_clause_phrase_query(question: str) -> bool:
    q = _normalize_text_token_space(question)
    if not q:
        return False
    tokens = q.split()
    if len(tokens) < 8:
        return False
    return (" shall " in f" {q} ") or (" will " in f" {q} ")


def _is_clause_presence_probe_query(question: str) -> bool:
    q = str(question or "").lower()
    if not q:
        return False
    return bool(
        re.search(r"\bis it true\b", q)
        or re.search(r"\bis that in (?:the )?(?:current )?(?:effective )?agreement\b", q)
        or re.search(r"\bis (?:this|that) in (?:the )?contract\b", q)
        or re.search(r"\bin the current effective agreement\b", q)
    )


def _has_contiguous_clause_phrase_evidence(question: str, chunks: list[dict]) -> bool:
    """
    Require a contiguous query phrase for clause-like prompts.

    This prevents false-unavailable recovery from synthesizing an answer from
    merely adjacent-topic chunks when the user is effectively testing whether a
    specific clause sentence exists in the current agreement.
    """
    if not chunks:
        return False
    q = _normalize_text_token_space(question)
    q_tokens = q.split()
    if len(q_tokens) < 4:
        return False

    # Try larger windows first. Ignore windows that are mostly stopwords.
    for window_size in (6, 5):
        if len(q_tokens) < window_size:
            continue
        for i in range(0, len(q_tokens) - window_size + 1):
            window = q_tokens[i:i + window_size]
            content_terms = [t for t in window if t not in _QUERY_EVIDENCE_STOPWORDS]
            # Require a denser phrase to avoid generic legal-language overlaps
            # (e.g. "the employee straight time") from triggering recovery.
            if len(content_terms) < 4:
                continue
            phrase = " ".join(window)
            for chunk in chunks[:12]:
                citation = str(chunk.get("citation") or "")
                if not re.search(r"article\s+\d+", citation, re.IGNORECASE):
                    continue
                text = _normalize_text_token_space(
                    f"{citation} {chunk.get('article_title', '')} {chunk.get('content_with_tables') or chunk.get('content') or ''}"
                )
                if phrase and phrase in text:
                    return True
    return False


def _has_strict_clause_phrase_evidence(question: str, chunks: list[dict]) -> bool:
    """
    Stronger near-quote requirement for explicit clause-presence probes.

    This reduces false fallback synthesis when the user is effectively asking
    whether a specific sentence still exists after an MOA.
    """
    if not chunks:
        return False
    q = _normalize_text_token_space(question)
    q_tokens = q.split()
    if len(q_tokens) < 6:
        return False

    for window_size in (7, 6):
        if len(q_tokens) < window_size:
            continue
        for i in range(0, len(q_tokens) - window_size + 1):
            window = q_tokens[i:i + window_size]
            content_terms = [t for t in window if t not in _QUERY_EVIDENCE_STOPWORDS]
            if len(content_terms) < 5:
                continue
            phrase = " ".join(window)
            for chunk in chunks[:12]:
                citation = str(chunk.get("citation") or "")
                if not re.search(r"article\s+\d+", citation, re.IGNORECASE):
                    continue
                text = _normalize_text_token_space(
                    f"{citation} {chunk.get('article_title', '')} {chunk.get('content_with_tables') or chunk.get('content') or ''}"
                )
                if phrase and phrase in text:
                    return True
    return False


def _has_recovery_evidence_for_query(
    *,
    question: str,
    chunks: list[dict],
    anchor_articles: list[int],
    topic: Optional[str],
    foreign_contract_reference: bool,
) -> bool:
    if foreign_contract_reference:
        return False
    clause_query = _is_clause_phrase_query(question)
    clause_presence_probe = _is_clause_presence_probe_query(question)
    anchor_evidence_present = (
        _has_article_anchor_evidence(anchor_articles, chunks, min_article_hits=2)
        and _query_supports_topic_recovery(topic, question)
    )
    lexical_evidence_present = _has_strong_evidence_for_query(question, chunks)
    if clause_query:
        lexical_evidence_present = lexical_evidence_present and _has_contiguous_clause_phrase_evidence(question, chunks)
    if clause_query and clause_presence_probe:
        lexical_evidence_present = lexical_evidence_present and _has_strict_clause_phrase_evidence(question, chunks)
        # For explicit clause-presence probes, article/topic anchors alone are
        # too noisy; require near-quote lexical evidence to avoid false claims.
        anchor_evidence_present = False
    return lexical_evidence_present or anchor_evidence_present


def _has_retry_candidate_evidence_for_query(
    *,
    question: str,
    chunks: list[dict],
    anchor_articles: list[int],
    topic: Optional[str],
    foreign_contract_reference: bool,
) -> bool:
    if foreign_contract_reference:
        return False
    anchor_evidence_present = (
        _has_article_anchor_evidence(anchor_articles, chunks, min_article_hits=2)
        and _query_supports_topic_recovery(topic, question)
    )
    return _has_strong_evidence_for_query(question, chunks) or anchor_evidence_present


def _fallback_relevant_chunks(
    question: str,
    chunks: list[dict],
    min_token_hits: int = 2,
    preferred_articles: Optional[list[int]] = None,
) -> list[dict]:
    """
    Keep fallback snippets scoped to query-relevant chunks.

    Prevents confusing fallback dumps when retrieval is noisy or the query is
    genuinely absent from the contract.
    """
    if not chunks:
        return []
    query_terms = _question_terms(question)
    if not query_terms:
        return []
    locator_query = _is_definition_locator_query(question)
    _topic, topic_signals = _primary_topic_signal(question)
    preferred_article_set: set[int] = set()
    for raw in preferred_articles or []:
        try:
            num = int(raw)
        except (TypeError, ValueError):
            continue
        if num > 0:
            preferred_article_set.add(num)

    scored: list[tuple[float, int, dict]] = []
    for chunk in chunks[:10]:
        citation = str(chunk.get("citation") or "")
        if not re.search(r"article\s+\d+", citation, re.IGNORECASE):
            continue
        section_num = int(chunk.get("section_num") or 0)
        title = str(chunk.get("article_title") or "").lower()
        text = (
            f"{citation} "
            f"{title} "
            f"{chunk.get('content_with_tables') or chunk.get('content') or ''}"
        ).lower()
        hits = sum(1 for term in query_terms if term in text)
        title_signal_hits = sum(1 for signal in topic_signals if signal and signal in title)
        article_num = chunk.get("article_num")
        in_preferred = isinstance(article_num, int) and article_num in preferred_article_set
        include = hits >= min_token_hits
        if not include and locator_query and title_signal_hits > 0 and hits >= 1:
            include = True
        if not include and in_preferred and hits >= 1:
            include = True
        if not include:
            continue

        score = float(hits) + (0.75 * float(title_signal_hits))
        if locator_query and title_signal_hits > 0:
            score += 0.5
        if in_preferred:
            score += 0.85
        scored.append((score, section_num, chunk))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored[:4]]


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
    effective_version_id: Optional[str] = None
    effective_content_hash: Optional[str] = None
    amendments_applied: list[str] = Field(default_factory=list)
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


@lru_cache(maxsize=64)
def _effective_runtime_metadata(contract_id: str) -> tuple[Optional[str], list[str]]:
    version_id = resolve_latest_effective_version_id(contract_id)
    if not version_id:
        return None, []
    payload = load_effective_contract(contract_id=contract_id, effective_version_id=version_id) or {}
    amendments = payload.get("amendments_applied") if isinstance(payload, dict) else []
    if not isinstance(amendments, list):
        amendments = []
    normalized = []
    for patch_id in amendments:
        value = str(patch_id or "").strip()
        if value and value not in normalized:
            normalized.append(value)
    return version_id, normalized


@lru_cache(maxsize=64)
def _effective_runtime_content_hash(contract_id: str) -> Optional[str]:
    return resolve_latest_effective_content_hash(contract_id)


def _response_effective_metadata(
    contract_id: str,
    chunks: Optional[list[dict]] = None,
    wage_info: Optional[dict] = None,
) -> tuple[Optional[str], list[str]]:
    version_id, amendments = _effective_runtime_metadata(contract_id)

    for chunk in chunks or []:
        candidate_version = str(chunk.get("effective_version_id") or "").strip()
        if candidate_version:
            version_id = candidate_version
            break

    from_chunks: list[str] = []
    for chunk in chunks or []:
        for patch_id in (chunk.get("amendments_applied") or []):
            value = str(patch_id or "").strip()
            if value and value not in from_chunks:
                from_chunks.append(value)

    from_wage: list[str] = []
    if isinstance(wage_info, dict):
        for patch_id in (wage_info.get("amendments_applied") or []):
            value = str(patch_id or "").strip()
            if value and value not in from_wage:
                from_wage.append(value)
        if not version_id:
            candidate_version = str(wage_info.get("effective_version_id") or "").strip()
            if candidate_version:
                version_id = candidate_version

    merged = []
    for collection in (amendments, from_chunks, from_wage):
        for patch_id in collection:
            if patch_id not in merged:
                merged.append(patch_id)
    return version_id, merged


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
    response_effective_version_id, response_amendments = _response_effective_metadata(
        effective_contract_id
    )

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

    response_effective_version_id, response_amendments = _response_effective_metadata(
        effective_contract_id
    )
    response_effective_content_hash = _effective_runtime_content_hash(
        effective_contract_id
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
            effective_version_id=response_effective_version_id,
            effective_content_hash=response_effective_content_hash,
            amendments_applied=response_amendments,
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
                effective_version_id=response_effective_version_id,
                effective_content_hash=response_effective_content_hash,
                amendments_applied=response_amendments,
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
            effective_version_id=response_effective_version_id,
            effective_content_hash=response_effective_content_hash,
            amendments_applied=response_amendments,
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
    response_effective_version_id, response_amendments = _response_effective_metadata(
        effective_contract_id,
        chunks=chunks,
        wage_info=wage_info,
    )
    response_effective_content_hash = _effective_runtime_content_hash(
        effective_contract_id
    )

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
                effective_version_id=response_effective_version_id,
                effective_content_hash=response_effective_content_hash,
                amendments_applied=response_amendments,
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
    retry_candidate_evidence_present = _has_retry_candidate_evidence_for_query(
        question=request.question,
        chunks=chunks,
        anchor_articles=anchor_articles,
        topic=intent.topic,
        foreign_contract_reference=foreign_contract_reference,
    )
    if _is_unavailable_answer(answer) and not escalation_required and retry_candidate_evidence_present:
        # Recovery should be deterministic and stable. Use single-angle retrieval
        # here to avoid stochastic query-interpretation drift in second-pass rescue.
        retry_result = await asyncio.to_thread(
            retriever.retrieve,
            query=request.question,
            intent=intent,
            n_results=8,
            hours_worked=hours_worked,
            months_employed=months_employed,
            use_hybrid=True,
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
        recovery_evidence_present = _has_recovery_evidence_for_query(
            question=request.question,
            chunks=chunks,
            anchor_articles=anchor_articles,
            topic=intent.topic,
            foreign_contract_reference=foreign_contract_reference,
        )
        if _is_unavailable_answer(answer) and recovery_evidence_present:
            answer = generate_fallback_response(
                chunks,
                question=request.question,
                preferred_articles=anchor_articles,
                topic=intent.topic,
            )
    
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
        effective_version_id=response_effective_version_id,
        effective_content_hash=response_effective_content_hash,
        amendments_applied=response_amendments,
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
    source_candidates: list[dict] = Field(default_factory=list)
    selected_source_key: Optional[str] = None


class ContractHistoryPatchResponse(BaseModel):
    """Single applied amendment in the effective chain."""
    patch_id: str
    effective_date: Optional[str] = None
    ratified_date: Optional[str] = None
    source_pdf: Optional[str] = None
    source_doc_id: Optional[str] = None
    operation_count: int = 0
    approved_operation_count: int = 0
    patch_file_sha256: Optional[str] = None
    patch_payload_sha256: Optional[str] = None


class ContractHistoryResponse(BaseModel):
    """Contract lineage + source-mode metadata for Contract tab."""
    contract_id: str
    effective_version_id: Optional[str] = None
    effective_content_hash: Optional[str] = None
    has_effective_snapshot: bool = False
    source_modes: list[str] = Field(default_factory=list)
    base_pdf: Optional[str] = None
    amendment_pdfs: list[str] = Field(default_factory=list)
    amendment_source_doc_ids: list[str] = Field(default_factory=list)
    applied_patch_ids: list[str] = Field(default_factory=list)
    base_chunk_total: int = 0
    effective_chunk_total: int = 0
    base_doc_type_counts: dict[str, int] = Field(default_factory=dict)
    effective_doc_type_counts: dict[str, int] = Field(default_factory=dict)
    patch_count: int = 0
    patches: list[ContractHistoryPatchResponse] = Field(default_factory=list)


class ContractBrowseItemSummaryResponse(BaseModel):
    key: str
    kind: str
    label: str
    title: Optional[str] = None
    article_num: Optional[int] = None
    doc_type: Optional[str] = None
    item_count: int = 0


class ContractBrowseGroupResponse(BaseModel):
    key: str
    label: str
    items: list[ContractBrowseItemSummaryResponse] = Field(default_factory=list)


class ContractBrowseResponse(BaseModel):
    contract_id: str
    groups: list[ContractBrowseGroupResponse] = Field(default_factory=list)


class ContractBrowseItemSectionResponse(BaseModel):
    citation: str
    content: str
    summary: Optional[str] = None


class ContractBrowseItemResponse(BaseModel):
    contract_id: str
    key: str
    kind: str
    label: str
    title: Optional[str] = None
    doc_type: Optional[str] = None
    sections: list[ContractBrowseItemSectionResponse] = Field(default_factory=list)


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


def _is_moa_pdf_name(name: str) -> bool:
    return "moa" in str(name or "").lower()


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for raw in values or []:
        value = str(raw or "").strip()
        if not value or value in deduped:
            continue
        deduped.append(value)
    return deduped


def _sorted_unique_strings(values: list[str]) -> list[str]:
    deduped = _dedupe_strings(values)
    return sorted(deduped, key=lambda v: v.lower())


def _load_patch_chain_payload(contract_id: str, effective_version_id: Optional[str]) -> dict:
    version_id = str(effective_version_id or "").strip()
    if not version_id:
        return {}
    path = DATA_DIR / "contracts" / contract_id / "effective" / version_id / "patch_chain.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _discover_base_chunks_path_for_history(contract_id: str) -> Optional[Path]:
    contract_root = DATA_DIR / "contracts" / contract_id
    candidates = [
        contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
        contract_root / "chunks" / "contract_chunks_enriched.json",
        contract_root / "chunks" / f"contract_chunks_smart_{contract_id}.json",
        contract_root / "chunks" / "contract_chunks_smart.json",
        DATA_DIR / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
        contract_root / "base" / "contract_chunks_enriched.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_chunk_doc_type_counts(path: Optional[Path]) -> tuple[int, dict[str, int]]:
    if path is None or not path.exists():
        return 0, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return 0, {}
    if not isinstance(payload, list):
        return 0, {}
    counts: dict[str, int] = {}
    total = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        total += 1
        doc_type = str(row.get("doc_type") or "").strip().lower() or "unknown"
        counts[doc_type] = int(counts.get(doc_type, 0)) + 1
    return total, dict(sorted(counts.items()))


def _build_contract_history_payload(contract_id: str) -> dict:
    effective_version_id = resolve_latest_effective_version_id(contract_id)
    effective_payload = load_effective_contract(contract_id=contract_id, effective_version_id=effective_version_id)
    effective_payload = effective_payload if isinstance(effective_payload, dict) else {}
    patch_chain = _load_patch_chain_payload(contract_id=contract_id, effective_version_id=effective_version_id)

    source_documents = effective_payload.get("source_documents")
    source_documents = source_documents if isinstance(source_documents, dict) else {}
    source_dir = DATA_DIR / "contracts" / contract_id / "source"
    source_pdf_names = sorted(
        [p.name for p in source_dir.glob("*.pdf")] if source_dir.exists() else [],
        key=lambda name: name.lower(),
    )

    base_pdf = str(source_documents.get("base_pdf") or "").strip() or None
    if not base_pdf:
        for name in source_pdf_names:
            if not _is_moa_pdf_name(name):
                base_pdf = name
                break
    if not base_pdf and source_pdf_names:
        base_pdf = source_pdf_names[0]

    amendment_candidates: list[str] = []
    amendment_source_doc_ids: list[str] = []
    for value in (source_documents.get("amendment_pdfs") or []):
        amendment_candidates.append(str(value))
    for value in (source_documents.get("amendment_source_doc_ids") or []):
        candidate_id = str(value)
        if candidate_id and not source_doc_applies_to_contract(candidate_id, contract_id):
            continue
        amendment_source_doc_ids.append(candidate_id)
    for row in (patch_chain.get("patches") or []):
        if not isinstance(row, dict):
            continue
        amendment_candidates.append(str(row.get("source_pdf") or ""))
        patch_source_doc_id = str(row.get("source_doc_id") or "")
        if patch_source_doc_id and not source_doc_applies_to_contract(patch_source_doc_id, contract_id):
            continue
        amendment_source_doc_ids.append(patch_source_doc_id)
    for name in source_pdf_names:
        if _is_moa_pdf_name(name):
            amendment_candidates.append(name)
    amendment_source_doc_ids = _sorted_unique_strings(amendment_source_doc_ids)
    for source_doc_id in amendment_source_doc_ids:
        pdf_name = str(resolve_source_doc_pdf_name(source_doc_id) or "").strip()
        if pdf_name:
            amendment_candidates.append(pdf_name)
    amendment_pdfs = _sorted_unique_strings(amendment_candidates)

    applied_patch_ids = _dedupe_strings(
        [str(v) for v in (patch_chain.get("applied_patch_ids") or [])]
        or [str(v) for v in (effective_payload.get("amendments_applied") or [])]
    )

    patches: list[dict] = []
    patch_rows = patch_chain.get("patches")
    if isinstance(patch_rows, list):
        for row in patch_rows:
            if not isinstance(row, dict):
                continue
            patch_source_doc_id = str(row.get("source_doc_id") or "") or None
            if patch_source_doc_id and not source_doc_applies_to_contract(patch_source_doc_id, contract_id):
                continue
            patches.append(
                {
                    "patch_id": str(row.get("patch_id") or ""),
                    "effective_date": str(row.get("effective_date") or "") or None,
                    "ratified_date": str(row.get("ratified_date") or "") or None,
                    "source_pdf": str(row.get("source_pdf") or "") or None,
                    "source_doc_id": patch_source_doc_id,
                    "operation_count": int(row.get("operation_count") or 0),
                    "approved_operation_count": int(row.get("approved_operation_count") or 0),
                    "patch_file_sha256": str(row.get("patch_file_sha256") or "") or None,
                    "patch_payload_sha256": str(row.get("patch_payload_sha256") or "") or None,
                }
            )
    if not patches:
        for patch_id in applied_patch_ids:
            patches.append(
                {
                    "patch_id": patch_id,
                    "effective_date": None,
                    "ratified_date": None,
                    "source_pdf": None,
                    "source_doc_id": None,
                    "operation_count": 0,
                    "approved_operation_count": 0,
                    "patch_file_sha256": None,
                    "patch_payload_sha256": None,
                }
            )

    modes = ["effective", "base"]
    if amendment_pdfs or amendment_source_doc_ids:
        modes.append("moa")

    effective_content_hash = (
        str(resolve_latest_effective_content_hash(contract_id) or "").strip()
        or str(patch_chain.get("effective_content_hash") or "").strip()
        or str(effective_payload.get("effective_content_hash") or "").strip()
        or None
    )
    base_chunks_path = _discover_base_chunks_path_for_history(contract_id)
    effective_chunks_path = (
        DATA_DIR
        / "contracts"
        / contract_id
        / "effective"
        / str(effective_version_id or "")
        / "index_inputs"
        / f"contract_chunks_enriched_{contract_id}.json"
    )
    base_chunk_total, base_doc_type_counts = _load_chunk_doc_type_counts(base_chunks_path)
    effective_chunk_total, effective_doc_type_counts = _load_chunk_doc_type_counts(
        effective_chunks_path if effective_version_id else None
    )

    return {
        "contract_id": contract_id,
        "effective_version_id": str(effective_version_id or "").strip() or None,
        "effective_content_hash": effective_content_hash,
        "has_effective_snapshot": bool(effective_version_id),
        "source_modes": modes,
        "base_pdf": base_pdf,
        "amendment_pdfs": amendment_pdfs,
        "amendment_source_doc_ids": amendment_source_doc_ids,
        "applied_patch_ids": applied_patch_ids,
        "base_chunk_total": base_chunk_total,
        "effective_chunk_total": effective_chunk_total,
        "base_doc_type_counts": base_doc_type_counts,
        "effective_doc_type_counts": effective_doc_type_counts,
        "patch_count": len(patches),
        "patches": patches,
    }


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


def _normalize_text_source_view(source_view: Optional[str]) -> str:
    value = str(source_view or "").strip().lower()
    if value in {"base", "previous", "prev"}:
        return "base"
    return "effective"


def _resolve_base_chunk_file(contract_id: str) -> Optional[Path]:
    candidates: list[Path] = []
    contract_chunk_dir = DATA_DIR / "contracts" / str(contract_id or "") / "chunks"
    if contract_chunk_dir.exists():
        candidates.extend(
            contract_chunk_dir / name for name in (
                "contract_chunks_enriched.json",
                "contract_chunks_smart.json",
                "contract_chunks.json",
            )
        )
        candidates.extend(
            contract_chunk_dir / name for name in (
                f"contract_chunks_enriched_{contract_id}.json",
                f"contract_chunks_smart_{contract_id}.json",
                f"contract_chunks_{contract_id}.json",
            )
        )
    # Fallback to per-contract chunk artifacts under data/chunks (but not effective index inputs).
    candidates.extend(
        CHUNKS_DIR / name for name in (
            f"contract_chunks_enriched_{contract_id}.json",
            f"contract_chunks_smart_{contract_id}.json",
            f"contract_chunks_{contract_id}.json",
            f"{contract_id}_contract_chunks_enriched.json",
            f"{contract_id}_contract_chunks_smart.json",
            f"{contract_id}_contract_chunks.json",
        )
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_viewer_chunk_file_for_view(contract_id: str, source_view: Optional[str] = None) -> Optional[Path]:
    normalized = _normalize_text_source_view(source_view)
    if normalized == "base":
        base_chunks = _resolve_base_chunk_file(contract_id)
        if base_chunks:
            return base_chunks
    return resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)


def _load_viewer_chunks_for_contract(contract_id: str, source_view: Optional[str] = None) -> list[dict]:
    chunks_file = _resolve_viewer_chunk_file_for_view(contract_id=contract_id, source_view=source_view)
    if not chunks_file or not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Contract chunks not found")
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load contract chunks")
    allow_unscoped = _viewer_allow_legacy_unscoped_chunks()
    scoped_chunks = [
        c for c in all_chunks
        if isinstance(c, dict)
        and (
            c.get("contract_id") == contract_id
            or (allow_unscoped and c.get("contract_id") in (None, ""))
        )
    ]
    return scoped_chunks


def _normalize_browse_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_side_letter_citation(citation: str) -> Optional[dict]:
    text = _normalize_browse_text(citation)
    if not text:
        return None
    match = re.match(
        r"^Letter of (Understanding|Agreement)\s+(\d+)\s*:\s*(.+?)(?:,\s*Part\s+([A-Za-z0-9\-]+))?$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    letter_kind = str(match.group(1) or "").strip().lower()
    number = int(match.group(2))
    title = str(match.group(3) or "").strip()
    part = str(match.group(4) or "").strip() or None
    kind = "lou" if letter_kind == "understanding" else "loa"
    short_prefix = "LOU" if kind == "lou" else "LOA"
    return {
        "kind": kind,
        "number": number,
        "title": title,
        "part": part,
        "key": f"{kind}:{number}",
        "label": f"{short_prefix} {number}",
    }


def _parse_appendix_group(chunk: dict) -> Optional[dict]:
    citation = _normalize_browse_text(chunk.get("citation") or "")
    if not citation:
        return None
    table_id = _normalize_browse_text(chunk.get("table_id") or "")
    chunk_id = _normalize_browse_text(chunk.get("chunk_id") or "")
    key_seed = table_id or chunk_id or citation
    slug = re.sub(r"[^a-z0-9]+", "_", key_seed.lower()).strip("_") or "appendix"
    return {
        "kind": "appendix",
        "key": f"appendix:{slug}",
        "label": "Appendix",
        "title": citation,
        "part": None,
    }


def _chunk_content_for_browse(chunk: dict) -> str:
    return str(chunk.get("content_with_tables") or chunk.get("content") or "").strip()


def _sort_browse_chunks(chunks: list[dict]) -> list[dict]:
    def _part_value(chunk: dict) -> tuple[int, str]:
        citation = _normalize_browse_text(chunk.get("citation") or "")
        match = re.search(r",\s*Part\s+([A-Za-z0-9\-]+)\s*$", citation, re.IGNORECASE)
        if match:
            token = str(match.group(1) or "").strip()
            if token.isdigit():
                return (int(token), "")
            return (999999, token.lower())
        chunk_id = _normalize_browse_text(chunk.get("chunk_id") or "")
        match_chunk = re.search(r"_part(\d+)$", chunk_id, re.IGNORECASE)
        if match_chunk:
            return (int(match_chunk.group(1)), "")
        return (999999, chunk_id.lower())

    return sorted(chunks, key=_part_value)


def _viewer_contract_browse_cache_key(contract_id: str) -> tuple[str, str, int, bool]:
    chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    chunks_path = str(chunks_file.resolve()) if chunks_file and chunks_file.exists() else ""
    chunks_mtime_ns = chunks_file.stat().st_mtime_ns if chunks_file and chunks_file.exists() else -1
    allow_unscoped = _viewer_allow_legacy_unscoped_chunks()
    return (contract_id, chunks_path, chunks_mtime_ns, allow_unscoped)


@lru_cache(maxsize=32)
def _build_contract_browse_payload_cached(
    contract_id: str,
    chunks_path: str,
    chunks_mtime_ns: int,
    allow_unscoped: bool,
) -> dict:
    _ = chunks_mtime_ns
    article_titles = _build_viewer_article_titles(contract_id)
    groups: list[dict] = []

    article_items = [
        {
            "key": f"article:{article_num}",
            "kind": "article",
            "label": f"Article {article_num}",
            "title": str(title or "").split("\n")[0].strip() or f"Article {article_num}",
            "article_num": int(article_num),
            "doc_type": "cba",
            "item_count": 1,
        }
        for article_num, title in sorted(
            ((int(k), v) for k, v in article_titles.items() if str(k).isdigit()),
            key=lambda kv: kv[0]
        )
    ]
    if article_items:
        groups.append({"key": "articles", "label": "Articles", "items": article_items})

    if not chunks_path:
        return {"contract_id": contract_id, "groups": groups}

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception:
        return {"contract_id": contract_id, "groups": groups}

    scoped_chunks = [
        c for c in all_chunks
        if isinstance(c, dict)
        and (
            c.get("contract_id") == contract_id
            or (allow_unscoped and c.get("contract_id") in (None, ""))
        )
    ]

    side_letter_groups: dict[str, dict] = {}
    appendix_groups: dict[str, dict] = {}
    for chunk in scoped_chunks:
        doc_type = str(chunk.get("doc_type") or "").strip().lower()
        if doc_type in {"lou", "loa"}:
            parsed = _parse_side_letter_citation(str(chunk.get("citation") or ""))
            if not parsed:
                continue
            key = str(parsed["key"])
            bucket = side_letter_groups.setdefault(
                key,
                {
                    "key": key,
                    "kind": parsed["kind"],
                    "label": str(parsed["label"]),
                    "title": str(parsed["title"]),
                    "article_num": None,
                    "doc_type": doc_type,
                    "item_count": 0,
                },
            )
            bucket["item_count"] = int(bucket.get("item_count") or 0) + 1
        elif doc_type == "appendix":
            parsed = _parse_appendix_group(chunk)
            if not parsed:
                continue
            key = str(parsed["key"])
            bucket = appendix_groups.setdefault(
                key,
                {
                    "key": key,
                    "kind": "appendix",
                    "label": "Appendix",
                    "title": str(parsed["title"]),
                    "article_num": None,
                    "doc_type": "appendix",
                    "item_count": 0,
                },
            )
            bucket["item_count"] = int(bucket.get("item_count") or 0) + 1

    if side_letter_groups:
        def _side_letter_sort_key(item: dict) -> tuple[int, str, int, str]:
            kind = str(item.get("kind") or "")
            key = str(item.get("key") or "")
            number = 999999
            m = re.match(r"^(?:lou|loa):(\d+)$", key, re.IGNORECASE)
            if m:
                number = int(m.group(1))
            kind_rank = 0 if kind == "lou" else 1
            return (kind_rank, key, number, str(item.get("title") or "").lower())

        items = sorted(side_letter_groups.values(), key=_side_letter_sort_key)
        groups.append({"key": "side_letters", "label": "Letters of Understanding / Agreement", "items": items})

    if appendix_groups:
        items = sorted(appendix_groups.values(), key=lambda item: str(item.get("title") or "").lower())
        groups.append({"key": "appendices", "label": "Appendices", "items": items})

    return {"contract_id": contract_id, "groups": groups}


def _build_contract_browse_payload(contract_id: str) -> dict:
    return _build_contract_browse_payload_cached(*_viewer_contract_browse_cache_key(contract_id))


def _build_contract_browse_item_payload(contract_id: str, kind: str, key: str, source_view: Optional[str] = None) -> dict:
    normalized_kind = str(kind or "").strip().lower()
    normalized_key = str(key or "").strip()
    if normalized_kind not in {"lou", "loa", "appendix"}:
        raise HTTPException(status_code=400, detail="Unsupported browse item kind")
    if not normalized_key:
        raise HTTPException(status_code=400, detail="Browse item key is required")

    chunks = _load_viewer_chunks_for_contract(contract_id, source_view=source_view)
    matched_chunks: list[dict] = []
    label = ""
    title = ""
    doc_type = normalized_kind

    for chunk in chunks:
        chunk_doc_type = str(chunk.get("doc_type") or "").strip().lower()
        if chunk_doc_type != normalized_kind and not (normalized_kind == "appendix" and chunk_doc_type == "appendix"):
            continue
        if normalized_kind in {"lou", "loa"}:
            parsed = _parse_side_letter_citation(str(chunk.get("citation") or ""))
            if not parsed:
                continue
            if str(parsed["key"]) != normalized_key:
                continue
            label = str(parsed["label"])
            title = str(parsed["title"])
            doc_type = chunk_doc_type or normalized_kind
            matched_chunks.append(chunk)
        elif normalized_kind == "appendix":
            parsed = _parse_appendix_group(chunk)
            if not parsed or str(parsed["key"]) != normalized_key:
                continue
            label = "Appendix"
            title = str(parsed["title"])
            doc_type = "appendix"
            matched_chunks.append(chunk)

    if not matched_chunks:
        raise HTTPException(status_code=404, detail="Browse item not found")

    matched_chunks = _sort_browse_chunks(matched_chunks)
    sections = []
    for chunk in matched_chunks:
        citation = _normalize_browse_text(chunk.get("citation") or "") or title or label or "Contract Item"
        content = _chunk_content_for_browse(chunk)
        if not content:
            continue
        sections.append(
            {
                "citation": citation,
                "content": content,
                "summary": str(chunk.get("summary") or "").strip() or None,
            }
        )
    if not sections:
        raise HTTPException(status_code=404, detail="Browse item content not found")

    return {
        "contract_id": contract_id,
        "key": normalized_key,
        "kind": normalized_kind,
        "label": label or normalized_kind.upper(),
        "title": title or None,
        "doc_type": doc_type or normalized_kind,
        "sections": sections,
    }


@lru_cache(maxsize=128)
def _resolve_contract_pdf_path(
    contract_id: str,
    source_pdf: Optional[str] = None,
    source_type: Optional[str] = None,
    source_doc_id: Optional[str] = None,
) -> Optional[Path]:
    source_type_normalized = str(source_type or "").strip().lower()
    prefer_moa = source_type_normalized in {"moa", "amendment", "amended"}
    return resolve_contract_source_pdf_path(
        contract_id=contract_id,
        source_pdf=source_pdf,
        prefer_moa=prefer_moa,
        source_doc_id=source_doc_id,
    )


def _contract_pdf_url(
    contract_id: str,
    source_pdf: Optional[str] = None,
    source_type: Optional[str] = None,
    source_doc_id: Optional[str] = None,
) -> str:
    params = {"contract_id": contract_id}
    source_pdf_value = str(source_pdf or "").strip()
    if source_pdf_value:
        params["source_pdf"] = source_pdf_value
    source_type_value = str(source_type or "").strip()
    if source_type_value:
        params["source_type"] = source_type_value
    source_doc_id_value = str(source_doc_id or "").strip()
    if source_doc_id_value:
        params["source_doc_id"] = source_doc_id_value
    return f"/api/contract-pdf?{urlencode(params)}"


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


def _normalize_search_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _viewer_source_pages_cache_key(contract_id: str) -> tuple[str, str, int]:
    source_json_path = resolve_contract_source_json_path(contract_id)
    path_str = str(source_json_path.resolve()) if source_json_path and source_json_path.exists() else ""
    mtime_ns = source_json_path.stat().st_mtime_ns if source_json_path and source_json_path.exists() else -1
    return (contract_id, path_str, mtime_ns)


@lru_cache(maxsize=16)
def _load_viewer_source_pages_cached(contract_id: str, source_json_path_str: str, source_json_mtime_ns: int) -> list[dict]:
    _ = source_json_mtime_ns
    if not source_json_path_str:
        return []
    path = Path(source_json_path_str)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list):
        return []

    out: list[dict] = []
    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_number = idx + 1
        raw_page_number = page.get("page_number")
        try:
            parsed_page = int(raw_page_number)
            if parsed_page > 0:
                page_number = parsed_page
        except Exception:
            pass
        items = page.get("items") or []
        raw_parts: list[str] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw = str(item.get("value") or item.get("md") or "")
                if raw:
                    raw_parts.append(raw)
        raw_text = re.sub(r"\s+", " ", " ".join(raw_parts)).strip()
        norm_text = _normalize_search_text(raw_text)
        out.append({
            "page_number": page_number,
            "raw_text": raw_text,
            "norm_text": norm_text,
        })
    return out


def _load_viewer_source_pages(contract_id: str) -> list[dict]:
    return _load_viewer_source_pages_cached(*_viewer_source_pages_cache_key(contract_id))


def _lookup_non_article_pdf_page(contract_id: str, browse_kind: str, browse_key: str) -> Optional[int]:
    kind = str(browse_kind or "").strip().lower()
    key = str(browse_key or "").strip().lower()
    if kind not in {"lou", "loa", "appendix"} or not key:
        return None

    try:
        item = _build_contract_browse_item_payload(contract_id=contract_id, kind=kind, key=key)
    except HTTPException:
        return None

    pages = _load_viewer_source_pages(contract_id)
    if not pages:
        return None

    title = _normalize_search_text(item.get("title") or "")
    label = _normalize_search_text(item.get("label") or "")
    number = None
    if kind in {"lou", "loa"}:
        m = re.match(r"^(?:lou|loa):(\d+)$", key)
        if m:
            number = int(m.group(1))

    best_page: Optional[int] = None
    best_score = -1
    for page in pages:
        text = str(page.get("norm_text") or "")
        raw_text = str(page.get("raw_text") or "")
        if not text:
            continue
        score = 0

        # Penalize list pages so body content wins when available.
        if "carry forward the specific letters of understanding as listed below" in text:
            score -= 40

        if kind in {"lou", "loa"} and number is not None:
            if f" {number} " in f" {text} ":
                score += 5
            if f"{number}" in text and "dress requirements" in title and "dress requirements" in text:
                score += 35
            if "letters of understanding" in text or "letters of agreement" in text:
                score += 8
            if title and title in text:
                score += 50
            # Stronger anchor for numbered side-letter heading on page.
            title_words = [w for w in title.split() if len(w) >= 3][:5]
            if title_words:
                anchored = str(number) in text and all(w in text for w in title_words[: min(3, len(title_words))])
                if anchored:
                    score += 40
        elif kind == "appendix":
            if title and title in text:
                score += 60
            if "appendix" in text:
                score += 15
            # Common appendix labels like "appendix a" should anchor strongly.
            title_match = re.search(r"\bappendix\s+([a-z])\b", title)
            if title_match and f"appendix {title_match.group(1)}" in text:
                score += 30
            if label and label in text:
                score += 10

        if score > best_score and score > 0:
            best_score = score
            best_page = int(page.get("page_number") or 0) or None

    return best_page


def _lookup_effective_provenance_page(
    contract_id: str,
    article_num: Optional[int],
    section_num: Optional[int],
    source_type: Optional[str],
    source_pdf: Optional[str],
    source_doc_id: Optional[str] = None,
) -> Optional[int]:
    if article_num is None and section_num is None:
        return None
    payload = load_effective_contract(contract_id=contract_id)
    if not isinstance(payload, dict):
        return None
    source_type_norm = str(source_type or "").strip().lower()
    source_pdf_norm = str(source_pdf or "").strip().lower()
    source_doc_id_norm = str(source_doc_id or "").strip().lower()
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if article_num is not None and section.get("article_num") != article_num:
            continue
        if section_num is not None and section.get("section_num") != section_num:
            continue
        for ref in section.get("provenance") or []:
            if not isinstance(ref, dict):
                continue
            ref_source_type = str(ref.get("source_type") or "").strip().lower()
            ref_pdf = str(ref.get("pdf") or "").strip().lower()
            ref_source_doc_id = str(ref.get("source_doc_id") or "").strip().lower()
            if source_type_norm and source_type_norm not in ref_source_type:
                continue
            if source_pdf_norm and source_pdf_norm not in ref_pdf:
                continue
            if source_doc_id_norm and source_doc_id_norm != ref_source_doc_id:
                continue
            page = ref.get("pdf_page")
            try:
                parsed_page = int(page)
            except (TypeError, ValueError):
                continue
            if parsed_page > 0:
                return parsed_page
    return None


def _lookup_effective_provenance_refs(
    contract_id: str,
    article_num: Optional[int],
    section_num: Optional[int],
) -> list[dict]:
    """
    Return provenance refs for the best matching effective section.

    Preference order:
    1. exact article+section match
    2. first section in article (lowest section_num) when section is omitted
    """
    if article_num is None:
        return []
    payload = load_effective_contract(contract_id=contract_id)
    if not isinstance(payload, dict):
        return []

    candidates: list[dict] = []
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if section.get("article_num") != article_num:
            continue
        if section_num is not None and section.get("section_num") != section_num:
            continue
        candidates.append(section)

    if not candidates and section_num is None:
        # Fallback to article-only lookup: all sections in the article.
        for section in payload.get("sections") or []:
            if not isinstance(section, dict):
                continue
            if section.get("article_num") == article_num:
                candidates.append(section)

    if not candidates:
        return []

    def _section_sort_key(section: dict) -> tuple[int, str]:
        raw_section = section.get("section_num")
        try:
            sec = int(raw_section) if raw_section is not None else 0
        except (TypeError, ValueError):
            sec = 0
        subsection = str(section.get("subsection") or "")
        return (sec if sec > 0 else 999999, subsection)

    chosen = sorted(candidates, key=_section_sort_key)[0]
    refs = chosen.get("provenance") or []
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _build_pdf_source_candidates_from_provenance(
    *,
    contract_id: str,
    provenance_refs: list[dict],
) -> tuple[list[dict], Optional[dict]]:
    """
    Build deterministic source candidates for UI navigation.

    Returns (candidates, preferred_candidate_for_auto_mode).
    """
    candidates: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()

    def _append_candidate(*, key: str, label: str, source_type: Optional[str], source_pdf: Optional[str],
                          source_doc_id: Optional[str], page_number: Optional[int], is_previous: bool = False) -> None:
        page_num = None
        try:
            if page_number is not None:
                parsed = int(page_number)
                if parsed > 0:
                    page_num = parsed
        except Exception:
            page_num = None
        dedupe_key = (
            str(source_type or "").strip().lower(),
            str(source_pdf or "").strip().lower(),
            str(source_doc_id or "").strip().lower(),
            int(page_num or 0),
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        candidates.append(
            {
                "key": key,
                "label": label,
                "source_type": str(source_type or "").strip() or None,
                "source_pdf": str(source_pdf or "").strip() or None,
                "source_doc_id": str(source_doc_id or "").strip() or None,
                "page_number": page_num,
                "is_previous": bool(is_previous),
            }
        )

    # Effective auto mode is always available.
    _append_candidate(
        key="effective_auto",
        label="Current Effective (Auto)",
        source_type=None,
        source_pdf=None,
        source_doc_id=None,
        page_number=None,
        is_previous=False,
    )

    for idx, ref in enumerate(provenance_refs or []):
        source_type = str(ref.get("source_type") or "").strip().lower() or None
        source_pdf = str(ref.get("pdf") or "").strip() or None
        source_doc_id = str(ref.get("source_doc_id") or "").strip() or None
        page_num = ref.get("pdf_page")
        label_kind = "Source"
        if source_type and ("moa" in source_type or "amend" in source_type):
            label_kind = "MOA"
            normalized_type = "moa"
        elif source_type and ("base" in source_type or "cba" in source_type or "contract" in source_type):
            label_kind = "Base"
            normalized_type = "base"
        elif source_pdf and _is_moa_pdf_name(source_pdf):
            label_kind = "MOA"
            normalized_type = "moa"
        else:
            label_kind = "Base" if source_pdf else "Source"
            normalized_type = "base" if source_pdf else (source_type or None)

        page_suffix = ""
        try:
            if page_num is not None and int(page_num) > 0:
                page_suffix = f" p.{int(page_num)}"
        except Exception:
            page_suffix = ""

        pdf_suffix = f" ({source_pdf})" if source_pdf else ""
        label = f"{label_kind}{page_suffix}{pdf_suffix}"
        key = f"prov_{idx}_{normalized_type or 'src'}"
        _append_candidate(
            key=key,
            label=label,
            source_type=normalized_type or source_type,
            source_pdf=source_pdf,
            source_doc_id=source_doc_id,
            page_number=page_num if isinstance(page_num, int) else page_num,
            is_previous=(normalized_type == "base"),
        )

    # Add a stable alias for "previous" when we have a base provenance source.
    base_candidate = next((c for c in candidates if str(c.get("source_type") or "").lower() == "base"), None)
    if base_candidate:
        _append_candidate(
            key="previous_base",
            label="Previous (Base)",
            source_type=base_candidate.get("source_type"),
            source_pdf=base_candidate.get("source_pdf"),
            source_doc_id=base_candidate.get("source_doc_id"),
            page_number=base_candidate.get("page_number"),
            is_previous=True,
        )

    preferred = next((c for c in candidates if str(c.get("source_type") or "").lower() == "moa"), None)
    if preferred is None:
        preferred = next((c for c in candidates if str(c.get("source_type") or "").lower() == "base"), None)
    return candidates, preferred


@app.get("/api/contract-pdf")
async def get_contract_pdf(
    contract_id: Optional[str] = None,
    source_pdf: Optional[str] = None,
    source_type: Optional[str] = None,
    source_doc_id: Optional[str] = None,
):
    """Serve the active contract PDF for inline viewing."""
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    pdf_path = _resolve_contract_pdf_path(
        effective_contract_id,
        source_pdf=source_pdf,
        source_type=source_type,
        source_doc_id=source_doc_id,
    )
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
    browse_kind: Optional[str] = None,
    browse_key: Optional[str] = None,
    contract_id: Optional[str] = None,
    source_pdf: Optional[str] = None,
    source_page: Optional[int] = None,
    source_type: Optional[str] = None,
    source_doc_id: Optional[str] = None,
):
    """
    Resolve best-effort PDF page for a citation target.

    Supports table-backed citations first, then section/article fallback.
    Subsection and row_index are accepted for API compatibility.
    """
    _ = subsection, row_index  # reserved for future fine-grained mapping
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    pdf_path = _resolve_contract_pdf_path(
        effective_contract_id,
        source_pdf=source_pdf,
        source_type=source_type,
        source_doc_id=source_doc_id,
    )
    if not pdf_path or not pdf_path.exists():
        return PdfLocationResponse(contract_id=effective_contract_id)
    explicit_source_pdf = str(source_pdf or "").strip() or None
    url_source_pdf = explicit_source_pdf or (None if source_doc_id else pdf_path.name)

    source_page_value: Optional[int] = None
    try:
        if source_page is not None:
            parsed_source_page = int(source_page)
            if parsed_source_page > 0:
                source_page_value = parsed_source_page
    except Exception:
        source_page_value = None

    if source_page_value is not None:
        return PdfLocationResponse(
            contract_id=effective_contract_id,
            pdf_url=_contract_pdf_url(
                effective_contract_id,
                source_pdf=url_source_pdf,
                source_type=source_type,
                source_doc_id=source_doc_id,
            ),
            page_number=source_page_value,
            total_pages=None,
            matched_by="provenance",
        )

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

    provenance_refs = _lookup_effective_provenance_refs(
        contract_id=effective_contract_id,
        article_num=normalized_article_num,
        section_num=normalized_section_num,
    )
    source_candidates, auto_preferred_candidate = _build_pdf_source_candidates_from_provenance(
        contract_id=effective_contract_id,
        provenance_refs=provenance_refs,
    )
    selected_source_key: Optional[str] = None
    if source_candidates:
        if source_type or source_pdf or source_doc_id:
            requested_type = str(source_type or "").strip().lower()
            requested_pdf = str(explicit_source_pdf or "").strip().lower()
            requested_doc_id = str(source_doc_id or "").strip().lower()
            for candidate in source_candidates:
                c_type = str(candidate.get("source_type") or "").strip().lower()
                c_pdf = str(candidate.get("source_pdf") or "").strip().lower()
                c_doc = str(candidate.get("source_doc_id") or "").strip().lower()
                type_ok = (not requested_type) or (requested_type == c_type)
                pdf_ok = (not requested_pdf) or (requested_pdf == c_pdf)
                doc_ok = (not requested_doc_id) or (requested_doc_id == c_doc)
                if type_ok and pdf_ok and doc_ok:
                    selected_source_key = str(candidate.get("key") or "") or None
                    break
        elif auto_preferred_candidate:
            selected_source_key = str(auto_preferred_candidate.get("key") or "") or None

    provenance_page = _lookup_effective_provenance_page(
        contract_id=effective_contract_id,
        article_num=normalized_article_num,
        section_num=normalized_section_num,
        source_type=source_type,
        source_pdf=explicit_source_pdf,
        source_doc_id=source_doc_id,
    )
    if provenance_page is not None and (source_type or source_pdf or source_doc_id):
        return PdfLocationResponse(
            contract_id=effective_contract_id,
            pdf_url=_contract_pdf_url(
                effective_contract_id,
                source_pdf=url_source_pdf,
                source_type=source_type,
                source_doc_id=source_doc_id,
            ),
            page_number=provenance_page,
            total_pages=None,
            matched_by="provenance_section",
            source_candidates=source_candidates,
            selected_source_key=selected_source_key,
        )

    # Auto-select effective provenance source (MOA/base) when no explicit source
    # is requested. This improves TOC/citation navigation for amended sections.
    if provenance_refs and not (source_type or source_pdf or source_doc_id or source_page_value):
        chosen = auto_preferred_candidate
        chosen_page = None
        chosen_type = None
        chosen_pdf = None
        chosen_doc_id = None
        if chosen and chosen.get("page_number"):
            try:
                chosen_page = int(chosen.get("page_number"))
            except Exception:
                chosen_page = None
            chosen_type = str(chosen.get("source_type") or "").strip() or None
            chosen_pdf = str(chosen.get("source_pdf") or "").strip() or None
            chosen_doc_id = str(chosen.get("source_doc_id") or "").strip() or None
        if chosen_page and chosen_page > 0:
            return PdfLocationResponse(
                contract_id=effective_contract_id,
                pdf_url=_contract_pdf_url(
                    effective_contract_id,
                    source_pdf=chosen_pdf,
                    source_type=chosen_type,
                    source_doc_id=chosen_doc_id,
                ),
                page_number=chosen_page,
                total_pages=None,
                matched_by="effective_provenance_auto",
                source_candidates=source_candidates,
                selected_source_key=selected_source_key,
            )

    source_type_norm = str(source_type or "").strip().lower()
    resolved_source_doc_pdf_name = str(resolve_source_doc_pdf_name(str(source_doc_id or "").strip()) or "").strip()
    source_pdf_name = str(explicit_source_pdf or resolved_source_doc_pdf_name or pdf_path.name or "").strip()
    targeting_moa_source = (
        source_type_norm in {"moa", "amendment", "amended"}
        or _is_moa_pdf_name(source_pdf_name)
        or _is_moa_pdf_name(resolved_source_doc_pdf_name)
    )
    if targeting_moa_source and (source_type or source_pdf or source_doc_id):
        # Avoid base-index fallback pages when explicitly targeting MOA docs but
        # no MOA page provenance exists for this citation.
        return PdfLocationResponse(
            contract_id=effective_contract_id,
            pdf_url=_contract_pdf_url(
                effective_contract_id,
                source_pdf=url_source_pdf,
                source_type=source_type,
                source_doc_id=source_doc_id,
            ),
            page_number=None,
            total_pages=None,
            matched_by="provenance_missing",
            source_candidates=source_candidates,
            selected_source_key=selected_source_key,
        )

    normalized_table_id = str(table_id or "").strip()
    normalized_browse_kind = str(browse_kind or "").strip().lower() or None
    normalized_browse_key = str(browse_key or "").strip() or None
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

    if (
        page_number is None
        and normalized_browse_kind in {"lou", "loa", "appendix"}
        and normalized_browse_key
    ):
        page_number = _lookup_non_article_pdf_page(
            effective_contract_id,
            normalized_browse_kind,
            normalized_browse_key,
        )
        if page_number is not None:
            matched_by = "browse_item"

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
        pdf_url=_contract_pdf_url(
            effective_contract_id,
            source_pdf=url_source_pdf,
            source_type=source_type,
            source_doc_id=source_doc_id,
        ),
        page_number=page_number,
        total_pages=index.get("total_pages") or table_index.get("total_pages"),
        matched_by=matched_by,
        source_candidates=source_candidates,
        selected_source_key=selected_source_key,
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


@app.get("/api/contract-history", response_model=ContractHistoryResponse)
async def get_contract_history(contract_id: Optional[str] = None):
    """Return effective lineage + source-mode metadata for Contract tab."""
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    payload = _build_contract_history_payload(effective_contract_id)
    return ContractHistoryResponse(**payload)


@app.get("/api/contract-browse", response_model=ContractBrowseResponse)
async def get_contract_browse(contract_id: Optional[str] = None):
    """Return grouped Contract-tab browse items (articles, LOU/LOA, appendices)."""
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    payload = _build_contract_browse_payload(effective_contract_id)
    return ContractBrowseResponse(**payload)


@app.get("/api/contract-browse-item", response_model=ContractBrowseItemResponse)
async def get_contract_browse_item(
    kind: str,
    key: str,
    source_view: Optional[str] = None,
    contract_id: Optional[str] = None,
):
    """Return aggregated content for a non-article Contract-tab browse item."""
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    payload = _build_contract_browse_item_payload(
        contract_id=effective_contract_id,
        kind=kind,
        key=key,
        source_view=source_view,
    )
    return ContractBrowseItemResponse(**payload)


@app.get("/api/article/{article_num}", response_model=ArticleResponse)
async def get_article(
    article_num: int,
    contract_id: Optional[str] = None,
    source_view: Optional[str] = None,
):
    """
    Get all sections for a specific article.

    Returns the article title and all sections with their content.
    """
    effective_contract_id = _resolve_contract_id_for_viewer(contract_id)
    chunks_file = _resolve_viewer_chunk_file_for_view(
        contract_id=effective_contract_id,
        source_view=source_view,
    )

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
    source_view: Optional[str] = None,
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
    chunks_file = _resolve_viewer_chunk_file_for_view(
        contract_id=effective_contract_id,
        source_view=source_view,
    )

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
        return generate_fallback_response(chunks, question=question)
    
    # Check if we're in a rate limit cooldown
    if time.time() < _rate_limit_until:
        wait_time = int(_rate_limit_until - time.time())
        print(f"Rate limited, waiting {wait_time}s before retry")
        return generate_fallback_response(chunks, question=question)
    
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
    return generate_fallback_response(chunks, question=question)


def generate_fallback_response(
    chunks: list = None,
    question: str = "",
    preferred_articles: Optional[list[int]] = None,
    topic: Optional[str] = None,
) -> str:
    """
    Generate a helpful fallback response showing retrieved chunks.
    
    Workers still get useful information even when the LLM is unavailable.
    """
    if not chunks:
        return (
            "I'm currently unable to search the contract. "
            "Please try again in a moment, or contact your steward directly."
        )
    
    relevant_chunks = _fallback_relevant_chunks(
        question=question,
        chunks=chunks,
        min_token_hits=2,
        preferred_articles=preferred_articles,
    )
    if not relevant_chunks:
        return (
            "I can't find matching language for that in the current effective agreement. "
            "If this was changed or removed by an MOA, ask your steward to confirm the exact amendment text."
        )

    # Build a response from query-relevant chunks only.
    response_parts = [
        "I found the following relevant sections from your contract:\n"
    ]
    
    for i, chunk in enumerate(relevant_chunks, 1):
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

