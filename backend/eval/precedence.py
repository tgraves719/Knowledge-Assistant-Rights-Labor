"""
"Specific Overrides General" Precedence Check for Karl RAG System.

Detects the most dangerous failure mode: answering with a general rule
when a specific exception applies to the user's context.

Example:
- General: "All employees get 1.25x Sunday premium" (Article 13)
- Specific: Pharmacy Techs can waive overtime (Article 56, Section 165)
- If user is a Pharmacy Tech and asks about overtime, the answer MUST
  reference Article 56, not just Article 12.

This is a HARD FAIL condition: precedence_failure = True → score = 0.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal


@dataclass
class PrecedenceException:
    """A specific exception that overrides a general rule."""
    description: str
    general_article: str        # e.g., "Article 12" (general overtime)
    specific_article: str       # e.g., "Article 56" (pharmacy tech overtime)
    trigger_field: str          # e.g., "classification", "hire_date"
    trigger_value: str          # e.g., "pharmacy_tech", "before_march_27_2005"
    general_rule_summary: str   # What the general rule says
    specific_rule_summary: str  # What the exception says


@dataclass
class PrecedenceResult:
    """Result of a precedence check."""
    precedence_failure: bool = False
    applicable_exceptions: List[PrecedenceException] = field(default_factory=list)
    exception_retrieved: bool = False
    answer_uses_exception: bool = False
    details: str = ""
    confidence: float = 0.0
    method: Literal["rule_based", "llm", "none"] = "none"


# ============================================================================
# KNOWN EXCEPTION PATTERNS (for this contract)
# These are loaded statically but could be extracted from a manifest.
# ============================================================================

KNOWN_EXCEPTIONS: List[PrecedenceException] = [
    # --- Hire date-based exceptions ---
    PrecedenceException(
        description="Holiday pay differs by hire date (pre/post March 27, 2005)",
        general_article="Article 16",
        specific_article="Article 16",
        trigger_field="hire_date",
        trigger_value="before_march_27_2005",
        general_rule_summary="Employees hired on/after March 27, 2005: Memorial Day, Thanksgiving, Christmas (+ Labor Day from 2017). Personal holidays: 1 after 2 years, 2 after 3, 3 after 4.",
        specific_rule_summary="Employees hired on/before March 26, 2005: New Year's, Memorial Day, July 4th, Labor Day, Thanksgiving, Christmas. Plus 2 personal holidays immediately."
    ),
    PrecedenceException(
        description="Holiday premium pay differs by hire date",
        general_article="Article 16",
        specific_article="Article 16",
        trigger_field="hire_date",
        trigger_value="before_march_27_2005",
        general_rule_summary="Employees hired on/after March 27, 2005: $1.00/hour premium for holiday work (except Christmas at 1.5x).",
        specific_rule_summary="Employees hired on/before March 26, 2005: 1.5x for all holiday work plus holiday pay."
    ),
    PrecedenceException(
        description="Vacation accrual differs by hire date",
        general_article="Article 17",
        specific_article="Article 17",
        trigger_field="hire_date",
        trigger_value="before_march_27_2005",
        general_rule_summary="Post-March 27, 2005: 1 week after 1 year, 2 after 3, 3 after 8, 4 after 12. Requires 1040 hours.",
        specific_rule_summary="Pre-March 27, 2005: 1 week after 1 year, 2 after 2, 3 after 5, 4 after 12, 5 after 20. Requires 832 hours."
    ),
    # --- Classification-based exceptions ---
    PrecedenceException(
        description="Pharmacy Technicians can waive overtime for 6th day",
        general_article="Article 12",
        specific_article="Article 56, Section 165",
        trigger_field="classification",
        trigger_value="pharmacy_tech",
        general_rule_summary="All overtime is at 1.5x the base rate for hours over 8/day or 40/week.",
        specific_rule_summary="Pharmacy Techs may voluntarily waive overtime and work a 6th day at straight time as long as weekly total doesn't exceed 40 hours."
    ),
    PrecedenceException(
        description="Pharmacy Technicians have specific split shift exemption",
        general_article="Article 21",
        specific_article="Article 56, Section 166",
        trigger_field="classification",
        trigger_value="pharmacy_tech",
        general_rule_summary="Split shift rules apply generally.",
        specific_rule_summary="Split shift section does not apply where pharmacy technicians voluntarily work two or more stores in one day."
    ),
    PrecedenceException(
        description="Pharmacy Technicians have specific seniority protections (pre-Sept 4, 1994)",
        general_article="Article 27",
        specific_article="Article 56, Section 167",
        trigger_field="classification",
        trigger_value="pharmacy_tech",
        general_rule_summary="General seniority rules apply.",
        specific_rule_summary="No current Pharmacy Tech hired before September 4, 1994 shall lose seniority for vacations and service awards."
    ),
    PrecedenceException(
        description="Pharmacy Technicians cannot be bumped by other classifications",
        general_article="Article 29",
        specific_article="Article 56, Section 170",
        trigger_field="classification",
        trigger_value="pharmacy_tech",
        general_rule_summary="During layoff, more senior employees can displace less senior in same or lower classification.",
        specific_rule_summary="Pharmacy Technician positions cannot be bumped by persons in other classifications."
    ),
    PrecedenceException(
        description="Cake Decorator layoff protection - requires 6 months specially trained",
        general_article="Article 29, Section 77",
        specific_article="Article 29, Section 78",
        trigger_field="classification",
        trigger_value="cake_decorator",
        general_rule_summary="During layoff, a more senior employee can displace a less senior one in same classification.",
        specific_rule_summary="'Specially trained' Cake Decorators may only be replaced by a more senior employee who has worked at least 6 months as a Cake Decorator."
    ),
    PrecedenceException(
        description="Night premium excludes Courtesy Clerks from $2.00 rate",
        general_article="Article 15, Section 34",
        specific_article="Article 15, Section 34",
        trigger_field="classification",
        trigger_value="courtesy_clerk",
        general_rule_summary="$2.00/hour premium for all work between midnight and 6am.",
        specific_rule_summary="Courtesy Clerks receive only $0.25/hour for work between midnight and 6am (not $2.00)."
    ),
    PrecedenceException(
        description="Health plan eligibility differs for Courtesy Clerks",
        general_article="Article 40",
        specific_article="Article 40",
        trigger_field="classification",
        trigger_value="courtesy_clerk",
        general_rule_summary="Part-time employees eligible first of month after 12 months, start at Plan C.",
        specific_rule_summary="Courtesy Clerks eligible first of month after 36 months or age 19 (whichever later), employee-only Plan C, can progress to Plan B but NOT Plan A."
    ),
    # --- Sanitation Clerk grandfathering ---
    PrecedenceException(
        description="Sanitation Clerks from May 11, 1996 have job security protections",
        general_article="Article 2",
        specific_article="Article 2, Section 4",
        trigger_field="classification",
        trigger_value="sanitation_clerk",
        general_rule_summary="Standard classification and work jurisdiction rules.",
        specific_rule_summary="Sanitation Clerks employed as of May 11, 1996 have specific job security protections under Article 2, Section 4."
    ),
    # --- Prior experience credit for Pharmacy Techs ---
    PrecedenceException(
        description="Pharmacy Technicians get 960 hours credit for certificate",
        general_article="Article 9",
        specific_article="Article 56, Section 163",
        trigger_field="classification",
        trigger_value="pharmacy_tech",
        general_rule_summary="Prior experience credit from previous grocery employers per Article 9.",
        specific_rule_summary="Pharmacy Techs with approved certificate get 960 hours credit in progression, regardless of prior employer."
    ),
]


class PrecedenceCheck:
    """
    Check if a general rule was applied where a specific exception exists.

    Uses a two-stage approach:
    1. Rule-based: Check known exception patterns against user context
    2. LLM-based: For complex cases, ask LLM to identify applicable exceptions

    Usage:
        checker = PrecedenceCheck()
        result = checker.check(
            query="What is my overtime rate?",
            user_context={"classification": "pharmacy_tech"},
            answer="Overtime is 1.5x per Article 12.",
            chunks=[...]
        )
        if result.precedence_failure:
            # HARD FAIL - answer used general rule, ignored exception
    """

    def __init__(
        self,
        known_exceptions: List[PrecedenceException] = None,
        use_llm_fallback: bool = True,
        llm_model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize precedence checker.

        Args:
            known_exceptions: List of known exception patterns. Defaults to KNOWN_EXCEPTIONS.
            use_llm_fallback: Whether to use LLM for complex cases.
            llm_model: LLM model for fallback analysis.
        """
        self.exceptions = known_exceptions or KNOWN_EXCEPTIONS
        self.use_llm_fallback = use_llm_fallback
        self.llm_model = llm_model
        self._llm_client = None

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

    def find_applicable_exceptions(
        self,
        user_context: Dict[str, str],
        query: str = ""
    ) -> List[PrecedenceException]:
        """
        Find exceptions that apply to this user based on their context.

        Args:
            user_context: Dict with keys like 'classification', 'hire_date', etc.
            query: The user's query (used for topic matching).

        Returns:
            List of applicable PrecedenceException objects.
        """
        applicable = []
        user_classification = user_context.get("classification", "").lower()
        user_hire_date = user_context.get("hire_date", "").lower()

        for exc in self.exceptions:
            # Check if this exception's trigger matches user context
            if exc.trigger_field == "classification":
                if user_classification and exc.trigger_value.lower() in user_classification:
                    applicable.append(exc)

            elif exc.trigger_field == "hire_date":
                if user_hire_date:
                    # Check if user's hire date matches the trigger
                    if self._matches_hire_date_trigger(user_hire_date, exc.trigger_value):
                        applicable.append(exc)

        return applicable

    def _matches_hire_date_trigger(self, hire_date: str, trigger: str) -> bool:
        """Check if a hire date matches a trigger condition."""
        hire_date_lower = hire_date.lower()

        if trigger == "before_march_27_2005":
            # Check if explicitly stated
            if "before" in hire_date_lower and "2005" in hire_date_lower:
                return True
            if "march 26, 2005" in hire_date_lower or "march 26 2005" in hire_date_lower:
                return True

            # Try to parse and compare
            try:
                from datetime import datetime
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"]:
                    try:
                        dt = datetime.strptime(hire_date.strip(), fmt)
                        cutoff = datetime(2005, 3, 27)
                        return dt < cutoff
                    except ValueError:
                        continue
            except Exception:
                pass

        return False

    def _check_article_in_chunks(self, article_ref: str, chunks: List[dict]) -> bool:
        """Check if a specific article/section appears in retrieved chunks."""
        # Extract article number from reference
        article_match = re.search(r"Article\s+(\d+)", article_ref, re.IGNORECASE)
        section_match = re.search(r"Section\s+(\d+)", article_ref, re.IGNORECASE)

        if not article_match:
            return False

        target_article = int(article_match.group(1))
        target_section = int(section_match.group(1)) if section_match else None

        for chunk in chunks:
            chunk_article = chunk.get("article_num")
            chunk_section = chunk.get("section_num")

            if chunk_article and int(chunk_article) == target_article:
                if target_section is None:
                    return True
                if chunk_section and int(chunk_section) == target_section:
                    return True

        return False

    def _answer_references_article(self, answer: str, article_ref: str) -> bool:
        """Check if the answer references a specific article/section."""
        article_match = re.search(r"Article\s+(\d+)", article_ref, re.IGNORECASE)
        if not article_match:
            return False

        article_num = article_match.group(1)
        section_match = re.search(r"Section\s+(\d+)", article_ref, re.IGNORECASE)

        # Check if answer mentions this article
        answer_pattern = rf"Article\s+{article_num}"
        if not re.search(answer_pattern, answer, re.IGNORECASE):
            return False

        # If we need a specific section, check that too
        if section_match:
            section_num = section_match.group(1)
            section_pattern = rf"Section\s+{section_num}"
            return bool(re.search(section_pattern, answer, re.IGNORECASE))

        return True

    def _topic_matches_exception(self, query: str, exception: PrecedenceException) -> bool:
        """Check if the query topic is relevant to a given exception."""
        query_lower = query.lower()

        # Extract keywords from the exception description and summaries
        keywords = []

        # Topic keywords based on article content
        topic_keywords = {
            "Article 12": ["overtime", "ot", "time and a half", "1.5x"],
            "Article 13": ["sunday", "premium", "sunday pay"],
            "Article 15": ["night", "midnight", "premium", "night pay"],
            "Article 16": ["holiday", "personal holiday", "float", "christmas", "thanksgiving"],
            "Article 17": ["vacation", "time off", "paid vacation", "accrual"],
            "Article 21": ["split shift"],
            "Article 27": ["seniority"],
            "Article 29": ["layoff", "lay off", "bump", "displacement", "reduction"],
            "Article 40": ["health", "insurance", "medical", "dental", "plan", "eligibility"],
            "Article 2": ["job security", "sanitation", "work jurisdiction"],
            "Article 9": ["prior experience", "credit", "hours"],
            "Article 56": ["pharmacy", "pharm tech"],
        }

        for article_ref in [exception.general_article, exception.specific_article]:
            # Match against first article reference found
            for art_key, kws in topic_keywords.items():
                if art_key in article_ref:
                    keywords.extend(kws)

        # Also check exception description
        desc_lower = exception.description.lower()
        if "overtime" in desc_lower:
            keywords.extend(["overtime", "ot"])
        if "holiday" in desc_lower:
            keywords.extend(["holiday"])
        if "vacation" in desc_lower:
            keywords.extend(["vacation"])
        if "layoff" in desc_lower or "bump" in desc_lower:
            keywords.extend(["layoff", "bump", "laid off"])
        if "night" in desc_lower:
            keywords.extend(["night"])
        if "health" in desc_lower:
            keywords.extend(["health", "insurance", "medical"])
        if "split shift" in desc_lower:
            keywords.extend(["split shift"])
        if "seniority" in desc_lower:
            keywords.extend(["seniority"])

        return any(kw in query_lower for kw in keywords)

    def check(
        self,
        query: str,
        user_context: Dict[str, str],
        answer: str,
        chunks: List[dict]
    ) -> PrecedenceResult:
        """
        Check if a precedence failure occurred.

        A precedence failure means:
        - An exception applies to this user (based on context)
        - The answer used a general rule instead of the specific exception

        Args:
            query: The user's question
            user_context: Dict with classification, hire_date, etc.
            answer: The system's answer
            chunks: Retrieved chunks

        Returns:
            PrecedenceResult with failure status and details
        """
        if not user_context:
            return PrecedenceResult(
                precedence_failure=False,
                details="No user context provided, cannot check precedence.",
                method="none"
            )

        # Stage 1: Rule-based check using known exceptions
        applicable = self.find_applicable_exceptions(user_context, query)

        # Filter to exceptions relevant to the query topic
        relevant = [exc for exc in applicable if self._topic_matches_exception(query, exc)]

        if not relevant:
            return PrecedenceResult(
                precedence_failure=False,
                applicable_exceptions=[],
                details="No applicable exceptions found for this query/context combination.",
                method="rule_based"
            )

        # For each relevant exception, check if it was properly handled
        failures = []
        for exc in relevant:
            exception_retrieved = self._check_article_in_chunks(
                exc.specific_article, chunks
            )
            answer_uses_exception = self._answer_references_article(
                answer, exc.specific_article
            )

            # Also check if the answer text discusses the exception concept
            # (even without citing the specific article number)
            answer_lower = answer.lower()
            exc_keywords = self._get_exception_keywords(exc)
            answer_mentions_concept = any(kw in answer_lower for kw in exc_keywords)

            # Failure conditions:
            # 1. Exception exists but answer doesn't reference specific article
            #    AND doesn't mention the exception concept
            if not answer_uses_exception and not answer_mentions_concept:
                # Check if it at least retrieved the exception chunk
                if exception_retrieved:
                    failures.append((exc, "Exception was retrieved but not used in answer"))
                else:
                    failures.append((exc, "Exception was neither retrieved nor used"))

        if failures:
            exc, reason = failures[0]
            return PrecedenceResult(
                precedence_failure=True,
                applicable_exceptions=relevant,
                exception_retrieved=self._check_article_in_chunks(
                    exc.specific_article, chunks
                ),
                answer_uses_exception=False,
                details=f"PRECEDENCE FAILURE: {exc.description}. {reason}. "
                        f"General rule ({exc.general_article}): {exc.general_rule_summary} "
                        f"Exception ({exc.specific_article}): {exc.specific_rule_summary}",
                confidence=0.85,
                method="rule_based"
            )

        # If we have LLM fallback enabled and rule-based passed, do LLM check
        if self.use_llm_fallback and relevant:
            return self._check_with_llm(query, user_context, answer, chunks, relevant)

        return PrecedenceResult(
            precedence_failure=False,
            applicable_exceptions=relevant,
            exception_retrieved=True,
            answer_uses_exception=True,
            details="Answer correctly references applicable exceptions.",
            confidence=0.85,
            method="rule_based"
        )

    def _get_exception_keywords(self, exc: PrecedenceException) -> List[str]:
        """Extract keywords from an exception that indicate it was addressed."""
        keywords = []
        specific_lower = exc.specific_rule_summary.lower()

        # Extract key numbers, rates, durations
        # Look for dollar amounts
        for match in re.finditer(r"\$[\d.]+", specific_lower):
            keywords.append(match.group())

        # Look for multipliers like "2.0x", "1.5x"
        for match in re.finditer(r"[\d.]+x", specific_lower):
            keywords.append(match.group())

        # Look for specific terms that differentiate the exception
        differentiators = [
            "waive", "voluntarily", "6 months", "six months",
            "960 hours", "specially trained", "cannot be bumped",
            "$0.25", "0.25", "plan c", "36 months",
            "832 hours", "1040 hours",
        ]
        for term in differentiators:
            if term in specific_lower:
                keywords.append(term)

        return keywords

    def _check_with_llm(
        self,
        query: str,
        user_context: Dict[str, str],
        answer: str,
        chunks: List[dict],
        known_exceptions: List[PrecedenceException]
    ) -> PrecedenceResult:
        """
        Use LLM to verify precedence handling for complex cases.
        """
        self._init_llm()
        if not self._llm_client:
            # Can't check without LLM, trust rule-based result
            return PrecedenceResult(
                precedence_failure=False,
                applicable_exceptions=known_exceptions,
                details="LLM fallback unavailable; rule-based check passed.",
                confidence=0.6,
                method="rule_based"
            )

        exceptions_text = "\n".join([
            f"- {exc.description}\n"
            f"  General rule ({exc.general_article}): {exc.general_rule_summary}\n"
            f"  Exception ({exc.specific_article}): {exc.specific_rule_summary}"
            for exc in known_exceptions
        ])

        chunks_text = "\n".join([
            f"[{c.get('citation', 'unknown')}]: {c.get('content', '')[:300]}..."
            for c in chunks[:10]
        ])

        prompt = f"""You are checking if a union contract Q&A system correctly handled "Specific Overrides General" precedence.

USER QUESTION: {query}
USER CONTEXT: {user_context}

APPLICABLE EXCEPTIONS (specific rules that should override general rules for this user):
{exceptions_text}

SYSTEM ANSWER:
{answer}

RETRIEVED CHUNKS:
{chunks_text}

ANALYSIS TASK:
1. Does the answer correctly apply the specific exception(s) listed above?
2. Or does it incorrectly apply the general rule, ignoring the exception?

RESPOND WITH EXACTLY ONE OF:
- PASS: The answer correctly handles the specific exception
- FAIL: The answer uses the general rule and ignores the applicable exception
- UNCLEAR: Cannot determine from the answer

Then provide a one-sentence explanation.

Format:
VERDICT: [PASS/FAIL/UNCLEAR]
EXPLANATION: [one sentence]"""

        try:
            from google import genai
            response = self._llm_client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                )
            )

            text = response.text.strip()
            verdict = "UNCLEAR"
            explanation = ""

            if "VERDICT:" in text:
                verdict_line = [l for l in text.split("\n") if "VERDICT:" in l][0]
                if "FAIL" in verdict_line.upper():
                    verdict = "FAIL"
                elif "PASS" in verdict_line.upper():
                    verdict = "PASS"

            if "EXPLANATION:" in text:
                exp_line = [l for l in text.split("\n") if "EXPLANATION:" in l][0]
                explanation = exp_line.split("EXPLANATION:", 1)[1].strip()

            if verdict == "FAIL":
                return PrecedenceResult(
                    precedence_failure=True,
                    applicable_exceptions=known_exceptions,
                    exception_retrieved=any(
                        self._check_article_in_chunks(exc.specific_article, chunks)
                        for exc in known_exceptions
                    ),
                    answer_uses_exception=False,
                    details=f"LLM PRECEDENCE FAILURE: {explanation}",
                    confidence=0.9,
                    method="llm"
                )

            return PrecedenceResult(
                precedence_failure=False,
                applicable_exceptions=known_exceptions,
                exception_retrieved=True,
                answer_uses_exception=True,
                details=f"LLM confirmed correct precedence handling: {explanation}",
                confidence=0.9,
                method="llm"
            )

        except Exception as e:
            return PrecedenceResult(
                precedence_failure=False,
                applicable_exceptions=known_exceptions,
                details=f"LLM check failed: {e}. Rule-based check passed.",
                confidence=0.5,
                method="rule_based"
            )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_checker_instance: Optional[PrecedenceCheck] = None


def get_precedence_checker() -> PrecedenceCheck:
    """Get or create the singleton PrecedenceCheck instance."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = PrecedenceCheck()
    return _checker_instance


def check_precedence(
    query: str,
    user_context: Dict[str, str],
    answer: str,
    chunks: List[dict]
) -> PrecedenceResult:
    """
    Convenience function to check precedence.

    Args:
        query: User's question
        user_context: Dict with classification, hire_date, etc.
        answer: System's answer
        chunks: Retrieved chunks

    Returns:
        PrecedenceResult with failure status
    """
    checker = get_precedence_checker()
    return checker.check(query, user_context, answer, chunks)


# ============================================================================
# TESTING
# ============================================================================

def main():
    """Test the precedence checker."""
    print("=" * 60)
    print("Testing Precedence Checker")
    print("=" * 60)

    checker = PrecedenceCheck(use_llm_fallback=False)

    # Test Case 1: Pharmacy Tech overtime - SHOULD FAIL
    print("\n--- Test 1: Pharmacy Tech overtime (should detect failure) ---")
    result = checker.check(
        query="What is my overtime rate?",
        user_context={"classification": "pharmacy_tech"},
        answer="According to Article 12, Section 28, overtime is paid at time and one-half (1.5x) your base hourly rate.",
        chunks=[
            {"article_num": 12, "section_num": 28, "citation": "Article 12, Section 28",
             "content": "Overtime at 1.5x rate..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")

    # Test Case 2: Pharmacy Tech overtime - SHOULD PASS
    print("\n--- Test 2: Pharmacy Tech overtime (should pass) ---")
    result = checker.check(
        query="What is my overtime rate?",
        user_context={"classification": "pharmacy_tech"},
        answer="Per Article 12, overtime is generally 1.5x. However, as a Pharmacy Technician, "
               "Article 56, Section 165 allows you to voluntarily waive overtime and work a 6th day "
               "at straight time as long as your weekly total doesn't exceed 40 hours.",
        chunks=[
            {"article_num": 12, "section_num": 28, "citation": "Article 12, Section 28",
             "content": "Overtime at 1.5x rate..."},
            {"article_num": 56, "section_num": 165, "citation": "Article 56, Section 165",
             "content": "Pharmacy tech overtime waiver..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")

    # Test Case 3: Cake Decorator layoff bumping - SHOULD FAIL
    print("\n--- Test 3: Cake Decorator bumping (should detect failure) ---")
    result = checker.check(
        query="Can a more senior clerk bump me from my position during a layoff?",
        user_context={"classification": "cake_decorator"},
        answer="According to Article 29, Section 77, during layoffs, more senior employees can "
               "displace less senior employees in the same classification.",
        chunks=[
            {"article_num": 29, "section_num": 77, "citation": "Article 29, Section 77",
             "content": "Layoff procedures..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")

    # Test Case 4: Regular clerk overtime - SHOULD PASS (no exception applies)
    print("\n--- Test 4: Regular clerk overtime (should pass - no exception) ---")
    result = checker.check(
        query="What is my overtime rate?",
        user_context={"classification": "all_purpose_clerk"},
        answer="Per Article 12, Section 28, overtime is 1.5x your base rate.",
        chunks=[
            {"article_num": 12, "section_num": 28, "citation": "Article 12, Section 28",
             "content": "Overtime at 1.5x..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")

    # Test Case 5: Pre-March 2005 employee holiday - SHOULD FAIL
    print("\n--- Test 5: Pre-March 2005 holiday (should detect failure) ---")
    result = checker.check(
        query="How many holidays do I get?",
        user_context={"classification": "all_purpose_clerk", "hire_date": "2004-01-15"},
        answer="Employees hired after March 27, 2005 receive Memorial Day, Thanksgiving, "
               "and Christmas as paid holidays per Article 16.",
        chunks=[
            {"article_num": 16, "section_num": 35, "citation": "Article 16, Section 35",
             "content": "Holiday provisions..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")

    # Test Case 6: Courtesy clerk night premium - SHOULD FAIL
    print("\n--- Test 6: Courtesy clerk night premium (should detect failure) ---")
    result = checker.check(
        query="What is the night premium?",
        user_context={"classification": "courtesy_clerk"},
        answer="Per Article 15, Section 34, a premium of $2.00 per hour is paid for all "
               "work between midnight and 6am.",
        chunks=[
            {"article_num": 15, "section_num": 34, "citation": "Article 15, Section 34",
             "content": "Night premium provisions..."}
        ]
    )
    print(f"  Failure detected: {result.precedence_failure}")
    print(f"  Details: {result.details[:100]}...")


if __name__ == "__main__":
    main()


