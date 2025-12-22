"""
FastAPI Backend for Karl - Union Contract RAG System.

Provides endpoints for:
- /api/query - Main Q&A endpoint
- /api/wage - Direct wage lookup
- /api/health - Health check
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Global instances
retriever = None
vector_store = None

# LLM client (lazy loaded)
genai = None


def get_genai():
    """Lazy load Google GenAI client."""
    global genai
    if genai is None:
        try:
            import google.generativeai as _genai
            api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if api_key:
                _genai.configure(api_key=api_key)
            genai = _genai
        except ImportError:
            pass
    return genai


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


# Endpoints

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    chunks_loaded = vector_store.count() if vector_store else 0
    llm_available = get_genai() is not None and bool(GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    
    return HealthResponse(
        status="healthy" if chunks_loaded > 0 else "degraded",
        chunks_loaded=chunks_loaded,
        llm_available=llm_available
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_contract(request: QueryRequest):
    """
    Answer a question about the union contract.
    
    Uses RAG to retrieve relevant contract sections and generate
    a grounded, citation-focused response.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Get conversation context for this session
    conversation_context = ""
    detected_entities = {}
    
    if request.session_id:
        ctx = get_session_context(request.session_id)
        conversation_context = ctx.get_full_context()
        
        # If this seems like a follow-up question, use context from previous turns
        followup_indicators = ["them", "that", "it", "they", "this", "those", "the same"]
        is_followup = any(ind in request.question.lower() for ind in followup_indicators)
        
        if is_followup and ctx.get_last_topic():
            detected_entities["topic"] = ctx.get_last_topic()
    
    # Classify intent with user's classification for role-based boosting
    intent = classify_intent(request.question, user_classification=request.user_classification)
    
    # Retrieve relevant chunks and wage info
    retrieval_result = retriever.retrieve(
        query=request.question,
        intent=intent,
        n_results=5,
        hours_worked=request.hours_worked,
        months_employed=request.months_employed
    )
    
    chunks = retrieval_result["chunks"]
    wage_info = retrieval_result["wage_info"]
    escalation_required = retrieval_result["escalation_required"]
    query_expansions = retrieval_result.get("query_expansions", [])
    
    # Only include wage info in prompt if this is actually a wage query
    # This prevents erroneous wage artifacts appearing on unrelated questions
    is_wage_query = intent.intent_type == "wage" or any(
        kw in request.question.lower() 
        for kw in ["pay", "wage", "rate", "salary", "how much", "hourly"]
    )
    
    # Build prompt with context and user classification
    system_prompt = build_prompt(
        query=request.question,
        chunks=chunks,
        wage_info=wage_info if is_wage_query else None,
        requires_escalation=escalation_required,
        query_expansions=query_expansions,
        user_classification=request.user_classification,
        conversation_context=conversation_context
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
    
    # Format response with sources
    formatted = format_response_with_sources(answer, chunks, wage_info)
    
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
        verification_passed=verification.is_valid
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


# Track rate limit state
_rate_limit_until = 0


async def generate_response(question: str, system_prompt: str, chunks: list = None) -> str:
    """
    Generate LLM response using Gemini with retry logic.
    
    Implements exponential backoff for rate limits (429 errors).
    Falls back to showing raw chunks if all retries fail.
    """
    global _rate_limit_until
    
    genai_client = get_genai()
    
    if not genai_client:
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
            model = genai_client.GenerativeModel(
                model_name=LLM_MODEL,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(question)
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


# Run with: uvicorn backend.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

