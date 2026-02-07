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

from backend.config import GEMINI_API_KEY, LLM_MODEL
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore
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
    CLASSIFICATION_DISPLAY_NAMES,
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


def get_genai_client():
    """Lazy load Google GenAI client."""
    global _genai_client
    if _genai_client is None and _genai_sdk is not None:
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if api_key:
            _genai_client = _genai_sdk.Client(api_key=api_key)
    return _genai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global retriever, vector_store
    print("Initializing Karl RAG system...")
    
    try:
        vector_store = ContractVectorStore()
        retriever = HybridRetriever(vector_store)
        print(f"Loaded {vector_store.count()} contract chunks")
    except Exception as e:
        print(f"Warning: Could not initialize vector store: {e}")
        retriever = HybridRetriever()
    
    yield
    
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Karl - Union Contract RAG",
    description="AI-powered contract Q&A for UFCW Local 7",
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
    user_classification: Optional[str] = Field(None, description="User's job classification")
    hours_worked: int = Field(0, description="Total hours worked (for wage calculations)")
    months_employed: int = Field(0, description="Months employed (for courtesy clerk wages)")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")


class WageLookupRequest(BaseModel):
    """Request model for direct wage lookups."""
    classification: str = Field(..., description="Job classification")
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
    wage_info: Optional[dict] = None
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
    classification: str
    step: str
    rate: float
    effective_date: str
    citation: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    chunks_loaded: int
    llm_available: bool


# Onboarding Models

class OnboardingOptionsResponse(BaseModel):
    """Available options for onboarding form."""
    classifications: list[dict]
    employment_types: list[dict]
    employers: list[str]


class ProfileUpdateRequest(BaseModel):
    """Request to update user profile."""
    classification: Optional[str] = None
    employment_type: Optional[str] = None  # "full_time" or "part_time"
    hire_date: Optional[str] = None  # ISO format: "2023-03-15"
    exact_hours: Optional[int] = None  # If user knows their exact hours


class ProfileResponse(BaseModel):
    """User profile with calculated fields."""
    session_id: str
    classification: Optional[str] = None
    classification_display: Optional[str] = None
    employment_type: Optional[str] = None
    hire_date: Optional[str] = None
    months_employed: Optional[int] = None
    estimated_hours: Optional[int] = None
    is_grandfathered: Optional[bool] = None
    is_complete: bool = False
    employer: str = "Safeway"


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
    # Verification guidance
    verification_message: str


# Endpoints

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    chunks_loaded = vector_store.count() if vector_store else 0
    llm_available = get_genai_client() is not None

    return HealthResponse(
        status="healthy" if chunks_loaded > 0 else "degraded",
        chunks_loaded=chunks_loaded,
        llm_available=llm_available
    )


# =============================================================================
# ONBOARDING & PROFILE ENDPOINTS
# =============================================================================

@app.get("/api/onboard/options", response_model=OnboardingOptionsResponse)
async def get_onboarding_options():
    """Get available options for the onboarding form."""
    return OnboardingOptionsResponse(
        classifications=get_classification_options(),
        employment_types=[
            {"value": "full_time", "label": "Full-Time (32+ hrs/week)"},
            {"value": "part_time", "label": "Part-Time (under 32 hrs/week)"},
        ],
        employers=["Safeway"]  # Expand when multi-contract support added
    )


@app.get("/api/profile/{session_id}", response_model=ProfileResponse)
async def get_profile(session_id: str):
    """Get user profile for a session."""
    profile = get_user_profile(session_id)

    return ProfileResponse(
        session_id=session_id,
        classification=profile.classification,
        classification_display=CLASSIFICATION_DISPLAY_NAMES.get(profile.classification) if profile.classification else None,
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
    profile = update_user_profile(session_id, updates)

    return ProfileResponse(
        session_id=session_id,
        classification=profile.classification,
        classification_display=CLASSIFICATION_DISPLAY_NAMES.get(profile.classification) if profile.classification else None,
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

    wage_info = retriever.lookup_wage(
        classification=profile.classification,
        hours_worked=estimated_hours,
        months_employed=months,
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
        classification_display=CLASSIFICATION_DISPLAY_NAMES.get(profile.classification, profile.classification),
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
            user_profile["classification_display"] = CLASSIFICATION_DISPLAY_NAMES.get(
                profile.classification
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
    intent = classify_intent(request.question, user_classification=effective_classification)

    # Retrieve relevant chunks and wage info using multi-angle retrieval
    # This uses deep query interpretation for better semantic matching
    retrieval_result = retriever.multi_angle_retrieve(
        query=request.question,
        intent=intent,
        n_results=5,
        hours_worked=hours_worked,
        months_employed=months_employed
    )
    
    chunks = retrieval_result["chunks"]
    wage_info = retrieval_result["wage_info"]
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
        wage_info=wage_info if is_wage_query else None,
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
    
    wage_info = retriever.lookup_wage(
        classification=request.classification,
        hours_worked=request.hours_worked,
        months_employed=request.months_employed,
        effective_date=request.effective_date
    )
    
    if not wage_info:
        raise HTTPException(
            status_code=404, 
            detail=f"Wage information not found for classification: {request.classification}"
        )
    
    return WageResponse(
        classification=wage_info["classification"],
        step=wage_info["step"],
        rate=wage_info["rate"],
        effective_date=wage_info["effective_date"],
        citation=wage_info["citation"]
    )


# =============================================================================
# CONTRACT VIEWER ENDPOINTS
# =============================================================================

# Data directory for contract files
DATA_DIR = Path(__file__).parent.parent / "data"


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


@app.get("/api/manifest", response_model=ManifestResponse)
async def get_manifest():
    """
    Get contract table of contents.

    Returns article numbers mapped to their titles for navigation.
    """
    manifest_file = DATA_DIR / "manifests" / "safeway_pueblo_clerks_2022.json"

    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Contract manifest not found")

    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return ManifestResponse(
        contract_id=manifest.get("contract_id", "unknown"),
        article_titles=manifest.get("article_titles", {}),
        total_articles=len(manifest.get("article_titles", {}))
    )


@app.get("/api/article/{article_num}", response_model=ArticleResponse)
async def get_article(article_num: int):
    """
    Get all sections for a specific article.

    Returns the article title and all sections with their content.
    """
    chunks_file = DATA_DIR / "chunks" / "contract_chunks.json"

    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Contract chunks not found")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Filter to this article
    article_chunks = [c for c in all_chunks if c.get('article_num') == article_num]

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
            content=chunk.get('content', ''),
            summary=chunk.get('summary')
        ))

    return ArticleResponse(
        article_num=article_num,
        article_title=article_title,
        sections=sections
    )


@app.get("/api/section/{article_num}/{section_num}", response_model=SectionResponse)
async def get_section(article_num: int, section_num: int, subsection: str = None):
    """
    Get a specific section from an article.

    Used for citation popover previews.

    Args:
        article_num: The article number
        section_num: The section number
        subsection: Optional subsection (e.g., 'a', 'b', 'c') for filtering
    """
    chunks_file = DATA_DIR / "chunks" / "contract_chunks.json"

    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Contract chunks not found")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Find the specific section
    matching_chunks = [
        c for c in all_chunks
        if c.get('article_num') == article_num and c.get('section_num') == section_num
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
            f"**{c.get('subsection', '')}**: {c.get('content', '')}"
            if c.get('subsection') else c.get('content', '')
            for c in matching_chunks
        ])
    else:
        combined_content = chunk.get('content', '')

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
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=question,
                config=_genai_sdk.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                )
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
    """Serve the frontend HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. API is running at /api/"}


# Mount static files for any additional frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# Run with: uvicorn backend.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

