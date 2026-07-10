"""
LLM-as-Judge Grader for Karl RAG System.

Implements structured 4-dimension grading with hard fail conditions:
1. Factual Accuracy (0-3)
2. Citation Entailment (0.0-1.0)
3. Completeness (0-3)
4. Appropriate Uncertainty (bool)

Hard fail conditions override all scores to 0:
- precedence_failure: Applied general rule where exception exists
- cross_contamination_detected: Retrieved chunks from wrong contract
- citation_fabrication: Cited non-existent Article/Section
- citation_entailment_score < 0.5: Cap at 2
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field


# ============================================================================
# EVALUATION RESULT SCHEMA (Pydantic for structured output)
# ============================================================================

class EntailmentResultSchema(BaseModel):
    """Single claim-citation entailment result."""
    claim: str
    citation: str
    cited_text: str = ""
    verdict: Literal["SUPPORTS", "CONTRADICTS", "IRRELEVANT"] = "IRRELEVANT"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Full evaluation result with 4 dimensions and hard fail conditions."""
    question_id: str = ""
    contract_id: str = ""

    # The 4 Dimensions
    factual_accuracy: int = Field(default=0, ge=0, le=3,
        description="0=hallucinated, 1=wrong but honest, 2=partially correct, 3=correct+grounded")
    citation_entailment_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Percentage of claims with SUPPORTS verdict")
    completeness: int = Field(default=0, ge=0, le=3,
        description="0=missing most parts, 1=partial, 2=good but missing nuance, 3=complete")
    uncertainty_calibrated: bool = Field(default=False,
        description="Did it refuse/hedge correctly when appropriate?")

    # Sub-scores with justifications
    factual_accuracy_justification: str = ""
    completeness_justification: str = ""
    uncertainty_justification: str = ""

    # Hard Fail Conditions
    precedence_failure: bool = Field(default=False,
        description="Applied general rule where specific exception exists")
    cross_contamination_detected: bool = Field(default=False,
        description="Retrieved/cited chunks from wrong contract")
    citation_fabrication: bool = Field(default=False,
        description="Cited non-existent Article/Section")

    # Entailment Details
    entailment_results: List[EntailmentResultSchema] = Field(default_factory=list)

    # Question metadata
    bucket: str = ""
    category: str = ""
    difficulty: str = ""

    @property
    def final_score(self) -> int:
        """Final score with hard fail overrides."""
        if self.precedence_failure:
            return 0
        if self.cross_contamination_detected:
            return 0
        if self.citation_fabrication:
            return 0
        if self.citation_entailment_score < 0.5:
            return min(2, self.factual_accuracy)
        return self.factual_accuracy

    @property
    def composite_score(self) -> float:
        """Weighted composite score (0.0-3.0) across all dimensions."""
        if self.final_score == 0:
            return 0.0

        weights = {
            "factual_accuracy": 0.4,
            "completeness": 0.3,
            "entailment": 0.2,
            "uncertainty": 0.1,
        }

        score = (
            self.final_score * weights["factual_accuracy"]
            + self.completeness * weights["completeness"]
            + (self.citation_entailment_score * 3) * weights["entailment"]
            + (3 if self.uncertainty_calibrated else 0) * weights["uncertainty"]
        )
        return round(score, 2)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = self.model_dump()
        d["final_score"] = self.final_score
        d["composite_score"] = self.composite_score
        return d


# ============================================================================
# GRADER PROMPT
# ============================================================================

GRADER_PROMPT = """You are evaluating a union contract Q&A system called Karl.

QUESTION: {question}
USER CONTEXT: {user_context}
EXPECTED ANSWER: {ground_truth}
SYSTEM ANSWER: {system_answer}
RETRIEVED CHUNKS (abbreviated):
{chunks_text}
ENTAILMENT RESULTS: {entailment_text}
APPLICABLE EXCEPTIONS: {exceptions_text}

Score on 4 dimensions (be strict - this is high-stakes labor law):

1. **Factual Accuracy** (0-3):
   - 3: Semantically correct, all facts match contract text, no errors
   - 2: Core correct but missing nuance OR 1+ minor factual errors
   - 1: Wrong answer but hedged/uncertain, OR refuses when should answer
   - 0: Wrong + confident, OR hallucinated facts, OR fabricated citation

2. **Completeness** (0-3):
   - 3: Addresses all parts of question with relevant detail
   - 2: Addresses main point but misses secondary aspects
   - 1: Addresses only part of the question
   - 0: Fails to address the question

3. **Appropriate Uncertainty** (true/false):
   - true: System correctly refused/hedged when information is missing or ambiguous,
           OR confidently answered when evidence was clear
   - false: System was overconfident when wrong, OR unnecessarily hedged when evidence was clear

CRITICAL CHECK - Precedence Failure:
If an exception in APPLICABLE EXCEPTIONS applies to this user but the answer used a general rule instead, note this as a PRECEDENCE FAILURE.

CRITICAL CHECK - Citation Fabrication:
If the answer cites an Article/Section that does NOT appear in the retrieved chunks and is not a real part of the contract, note this as CITATION FABRICATION.

Respond in EXACTLY this format (one line per field):
FACTUAL_ACCURACY: [0-3]
FACTUAL_JUSTIFICATION: [one sentence]
COMPLETENESS: [0-3]
COMPLETENESS_JUSTIFICATION: [one sentence]
UNCERTAINTY_CALIBRATED: [true/false]
UNCERTAINTY_JUSTIFICATION: [one sentence]
PRECEDENCE_FAILURE: [true/false]
CITATION_FABRICATION: [true/false]"""


# ============================================================================
# GRADER CLASS
# ============================================================================

class LLMGrader:
    """
    LLM-as-judge grader with structured 4-dimension evaluation.

    Uses a different LLM than the system to avoid self-evaluation bias.
    Integrates entailment results from EntailmentChecker.
    Integrates precedence results from PrecedenceCheck.

    Usage:
        grader = LLMGrader()
        result = grader.grade(
            question="What is overtime?",
            system_answer="...",
            ground_truth="...",
            chunks=[...],
            entailment_summary=summary,
            precedence_result=prec_result
        )
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        llm_client=None
    ):
        """
        Initialize the grader.

        Args:
            model: LLM model for grading. Should be different from system LLM.
            llm_client: Optional pre-initialized LLM client.
        """
        self.model = model
        self._llm_client = llm_client

    def _init_llm(self):
        """Lazy-initialize LLM client."""
        if self._llm_client is not None:
            return
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                self._llm_client = genai.Client(api_key=api_key)
        except ImportError:
            pass

    def _format_chunks(self, chunks: List[dict], max_chunks: int = 8) -> str:
        """Format chunks for the grader prompt."""
        if not chunks:
            return "No chunks retrieved."

        parts = []
        for c in chunks[:max_chunks]:
            citation = c.get("citation", "unknown")
            content = c.get("content", "")[:200]
            parts.append(f"[{citation}]: {content}...")

        return "\n".join(parts)

    def _format_entailment(self, entailment_summary) -> str:
        """Format entailment results for the grader prompt."""
        if entailment_summary is None:
            return "No entailment results available."

        if not hasattr(entailment_summary, "results") or not entailment_summary.results:
            return "No claim-citation pairs found in answer."

        parts = [
            f"Support rate: {entailment_summary.support_rate:.0%} "
            f"({entailment_summary.supported_count}/{entailment_summary.total_claims} claims supported)"
        ]

        if entailment_summary.has_contradiction:
            parts.append(f"WARNING: {entailment_summary.contradicted_count} contradiction(s) detected!")

        for r in entailment_summary.results[:5]:
            parts.append(f"  - [{r.verdict}] \"{r.claim[:60]}...\" (cited: {r.citation})")

        return "\n".join(parts)

    def _format_exceptions(self, precedence_result) -> str:
        """Format precedence exceptions for the grader prompt."""
        if precedence_result is None:
            return "No precedence check performed."

        if not hasattr(precedence_result, "applicable_exceptions"):
            return "No applicable exceptions."

        if not precedence_result.applicable_exceptions:
            return "No specific exceptions apply to this user context."

        parts = []
        for exc in precedence_result.applicable_exceptions:
            parts.append(
                f"- {exc.description}\n"
                f"  General ({exc.general_article}): {exc.general_rule_summary}\n"
                f"  Exception ({exc.specific_article}): {exc.specific_rule_summary}"
            )

        return "\n".join(parts)

    def _detect_citation_fabrication(self, answer: str, chunks: List[dict]) -> bool:
        """
        Detect if the answer fabricates citations not in retrieved chunks.

        A fabricated citation is one that references an Article/Section
        not present in ANY retrieved chunk.
        """
        # Extract citations from the answer
        citation_pattern = r"Article\s+(\d+)(?:,?\s*Section\s+(\d+))?"
        answer_citations = re.findall(citation_pattern, answer, re.IGNORECASE)

        if not answer_citations:
            return False

        # Get all article numbers from chunks
        chunk_articles = set()
        for chunk in chunks:
            article_num = chunk.get("article_num")
            if article_num:
                chunk_articles.add(int(article_num))

            # Also parse from citation string
            citation = chunk.get("citation", "")
            for match in re.finditer(r"Article\s+(\d+)", citation, re.IGNORECASE):
                chunk_articles.add(int(match.group(1)))

        # Check each answer citation
        for article_str, section_str in answer_citations:
            article_num = int(article_str)
            if article_num not in chunk_articles:
                # This article was cited but never retrieved
                # Could be fabricated, or could be from general knowledge
                # Only flag if not a well-known article
                return True

        return False

    def grade(
        self,
        question: str,
        system_answer: str,
        ground_truth: str = "",
        user_context: Dict[str, str] = None,
        chunks: List[dict] = None,
        entailment_summary=None,
        precedence_result=None,
        question_id: str = "",
        contract_id: str = "",
        bucket: str = "",
        category: str = "",
        difficulty: str = "",
    ) -> EvaluationResult:
        """
        Grade a system answer using LLM-as-judge.

        Args:
            question: The user's question
            system_answer: Karl's answer
            ground_truth: Expected correct answer (from gold set)
            user_context: User profile info (classification, hire_date, etc.)
            chunks: Retrieved chunks
            entailment_summary: EntailmentSummary from entailment checker
            precedence_result: PrecedenceResult from precedence checker
            question_id: Question identifier
            contract_id: Contract identifier
            bucket: Question bucket (world_knowledge, contract_only, etc.)
            category: Question category
            difficulty: Question difficulty

        Returns:
            EvaluationResult with all scores and hard fail conditions
        """
        self._init_llm()
        chunks = chunks or []
        user_context = user_context or {}

        # Pre-compute hard fail conditions
        prec_failure = (
            precedence_result is not None
            and hasattr(precedence_result, "precedence_failure")
            and precedence_result.precedence_failure
        )
        citation_fab = self._detect_citation_fabrication(system_answer, chunks)

        # Compute entailment score
        entailment_score = 1.0
        entailment_results_schema = []
        if entailment_summary and hasattr(entailment_summary, "support_rate"):
            entailment_score = entailment_summary.support_rate
            if hasattr(entailment_summary, "results"):
                for r in entailment_summary.results:
                    entailment_results_schema.append(EntailmentResultSchema(
                        claim=r.claim,
                        citation=r.citation,
                        cited_text=getattr(r, "cited_text", ""),
                        verdict=r.verdict,
                        confidence=r.confidence,
                    ))

        # Build the grader prompt
        prompt = GRADER_PROMPT.format(
            question=question,
            user_context=json.dumps(user_context) if user_context else "No context",
            ground_truth=ground_truth or "Not provided",
            system_answer=system_answer,
            chunks_text=self._format_chunks(chunks),
            entailment_text=self._format_entailment(entailment_summary),
            exceptions_text=self._format_exceptions(precedence_result),
        )

        # Default result (used if LLM fails)
        result = EvaluationResult(
            question_id=question_id,
            contract_id=contract_id,
            factual_accuracy=0,
            citation_entailment_score=entailment_score,
            completeness=0,
            uncertainty_calibrated=False,
            precedence_failure=prec_failure,
            cross_contamination_detected=False,
            citation_fabrication=citation_fab,
            entailment_results=entailment_results_schema,
            bucket=bucket,
            category=category,
            difficulty=difficulty,
        )

        if not self._llm_client:
            result.factual_accuracy_justification = "LLM grader unavailable"
            return result

        try:
            from google import genai
            response = self._llm_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                )
            )

            text = response.text.strip()
            parsed = self._parse_grader_response(text)

            result.factual_accuracy = parsed.get("factual_accuracy", 0)
            result.factual_accuracy_justification = parsed.get("factual_justification", "")
            result.completeness = parsed.get("completeness", 0)
            result.completeness_justification = parsed.get("completeness_justification", "")
            result.uncertainty_calibrated = parsed.get("uncertainty_calibrated", False)
            result.uncertainty_justification = parsed.get("uncertainty_justification", "")

            # LLM can also flag these
            if parsed.get("precedence_failure", False):
                result.precedence_failure = True
            if parsed.get("citation_fabrication", False):
                result.citation_fabrication = True

        except Exception as e:
            result.factual_accuracy_justification = f"LLM grader error: {e}"

        return result

    def _parse_grader_response(self, text: str) -> dict:
        """Parse the structured grader response."""
        result = {}

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()

            if key == "FACTUAL_ACCURACY":
                try:
                    result["factual_accuracy"] = max(0, min(3, int(value.strip())))
                except ValueError:
                    result["factual_accuracy"] = 0

            elif key == "FACTUAL_JUSTIFICATION":
                result["factual_justification"] = value

            elif key == "COMPLETENESS":
                try:
                    result["completeness"] = max(0, min(3, int(value.strip())))
                except ValueError:
                    result["completeness"] = 0

            elif key == "COMPLETENESS_JUSTIFICATION":
                result["completeness_justification"] = value

            elif key == "UNCERTAINTY_CALIBRATED":
                result["uncertainty_calibrated"] = value.lower() == "true"

            elif key == "UNCERTAINTY_JUSTIFICATION":
                result["uncertainty_justification"] = value

            elif key == "PRECEDENCE_FAILURE":
                result["precedence_failure"] = value.lower() == "true"

            elif key == "CITATION_FABRICATION":
                result["citation_fabrication"] = value.lower() == "true"

        return result

    def grade_batch(
        self,
        test_cases: List[dict],
        system_answers: List[str],
        chunks_list: List[List[dict]],
        entailment_summaries: List = None,
        precedence_results: List = None,
    ) -> List[EvaluationResult]:
        """
        Grade a batch of test cases.

        Args:
            test_cases: List of test case dicts with question, ground_truth, etc.
            system_answers: List of system answers
            chunks_list: List of retrieved chunk lists
            entailment_summaries: Optional list of EntailmentSummary objects
            precedence_results: Optional list of PrecedenceResult objects

        Returns:
            List of EvaluationResult objects
        """
        results = []
        n = len(test_cases)

        entailment_summaries = entailment_summaries or [None] * n
        precedence_results = precedence_results or [None] * n

        for i, tc in enumerate(test_cases):
            result = self.grade(
                question=tc.get("question", ""),
                system_answer=system_answers[i] if i < len(system_answers) else "",
                ground_truth=tc.get("ground_truth", tc.get("expected_answer", "")),
                user_context=tc.get("user_context", {}),
                chunks=chunks_list[i] if i < len(chunks_list) else [],
                entailment_summary=entailment_summaries[i],
                precedence_result=precedence_results[i],
                question_id=str(tc.get("id", i)),
                contract_id=tc.get("contract_id", ""),
                bucket=tc.get("bucket", ""),
                category=tc.get("category", ""),
                difficulty=tc.get("difficulty", ""),
            )
            results.append(result)

        return results


# ============================================================================
# AGGREGATION UTILITIES
# ============================================================================

def aggregate_results(
    results: List[EvaluationResult],
    group_by: str = None
) -> dict:
    """
    Aggregate evaluation results into summary statistics.

    Args:
        results: List of EvaluationResult objects
        group_by: Optional field to group by ('bucket', 'category', 'difficulty')

    Returns:
        Dict with overall and per-group statistics
    """
    if not results:
        return {"total": 0}

    def compute_stats(result_list: List[EvaluationResult]) -> dict:
        n = len(result_list)
        if n == 0:
            return {}

        scores = [r.final_score for r in result_list]
        composites = [r.composite_score for r in result_list]
        entailments = [r.citation_entailment_score for r in result_list]

        return {
            "total": n,
            "mean_final_score": round(sum(scores) / n, 2),
            "mean_composite_score": round(sum(composites) / n, 2),
            "mean_entailment": round(sum(entailments) / n, 2),
            "score_distribution": {
                0: sum(1 for s in scores if s == 0),
                1: sum(1 for s in scores if s == 1),
                2: sum(1 for s in scores if s == 2),
                3: sum(1 for s in scores if s == 3),
            },
            "hard_fail_counts": {
                "precedence_failure": sum(1 for r in result_list if r.precedence_failure),
                "cross_contamination": sum(1 for r in result_list if r.cross_contamination_detected),
                "citation_fabrication": sum(1 for r in result_list if r.citation_fabrication),
            },
            "uncertainty_calibrated_rate": round(
                sum(1 for r in result_list if r.uncertainty_calibrated) / n, 2
            ),
        }

    summary = {"overall": compute_stats(results)}

    if group_by:
        groups = {}
        for r in results:
            key = getattr(r, group_by, "unknown")
            if not key:
                key = "unknown"
            groups.setdefault(key, []).append(r)

        summary["by_" + group_by] = {
            k: compute_stats(v) for k, v in sorted(groups.items())
        }

    return summary


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_grader_instance: Optional[LLMGrader] = None


def get_grader(model: str = "gemini-2.0-flash-exp") -> LLMGrader:
    """Get or create the singleton LLMGrader instance."""
    global _grader_instance
    if _grader_instance is None:
        _grader_instance = LLMGrader(model=model)
    return _grader_instance


# ============================================================================
# TESTING
# ============================================================================

def main():
    """Test the grader with sample data."""
    print("=" * 60)
    print("Testing LLM Grader")
    print("=" * 60)

    # Test the EvaluationResult model
    result = EvaluationResult(
        question_id="1",
        contract_id="safeway_pueblo_clerks_2022",
        factual_accuracy=3,
        citation_entailment_score=0.95,
        completeness=3,
        uncertainty_calibrated=True,
        factual_accuracy_justification="Answer matches contract text exactly.",
        completeness_justification="All parts addressed.",
        uncertainty_justification="Confident when evidence is clear.",
    )

    print(f"\nTest 1 - Perfect score:")
    print(f"  Final score: {result.final_score}")
    print(f"  Composite: {result.composite_score}")

    # Test hard fail
    result2 = EvaluationResult(
        question_id="2",
        factual_accuracy=3,
        citation_entailment_score=0.9,
        completeness=3,
        uncertainty_calibrated=True,
        precedence_failure=True,
    )

    print(f"\nTest 2 - Precedence failure:")
    print(f"  Final score: {result2.final_score} (should be 0)")
    print(f"  Composite: {result2.composite_score} (should be 0)")

    # Test entailment cap
    result3 = EvaluationResult(
        question_id="3",
        factual_accuracy=3,
        citation_entailment_score=0.3,
        completeness=3,
        uncertainty_calibrated=True,
    )

    print(f"\nTest 3 - Low entailment (should cap at 2):")
    print(f"  Final score: {result3.final_score} (should be 2)")

    # Test aggregation
    results = [result, result2, result3]
    summary = aggregate_results(results)
    print(f"\nAggregation:")
    print(f"  Mean final score: {summary['overall']['mean_final_score']}")
    print(f"  Hard fails: {summary['overall']['hard_fail_counts']}")

    print("\nSerializer test:")
    print(json.dumps(result.to_dict(), indent=2)[:300] + "...")


if __name__ == "__main__":
    main()


