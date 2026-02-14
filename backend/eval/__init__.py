"""
Evaluation module for Karl RAG system.

Contains:
- entailment: Citation entailment checking (NLI + LLM fallback)
- precedence: "Specific Overrides General" detection
- grader: LLM-as-judge grading with structured output
"""

from .entailment import (
    EntailmentResult,
    EntailmentSummary,
    EntailmentChecker,
    check_entailment,
    extract_claim_citation_pairs,
)

from .precedence import (
    PrecedenceCheck,
    PrecedenceResult,
    PrecedenceException,
)

from .grader import (
    EvaluationResult,
    LLMGrader,
)

__all__ = [
    # Entailment
    "EntailmentResult",
    "EntailmentSummary",
    "EntailmentChecker",
    "check_entailment",
    "extract_claim_citation_pairs",
    # Precedence
    "PrecedenceCheck",
    "PrecedenceResult",
    "PrecedenceException",
    # Grader
    "EvaluationResult",
    "LLMGrader",
]

