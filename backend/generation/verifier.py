"""
Citation Verifier - Validates that LLM responses are properly grounded.

Checks:
- Every substantive claim has a citation
- Citations match retrieved context
- High-stakes topics include escalation language
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of response verification."""
    is_valid: bool
    citations_found: list[str]
    citations_missing: bool
    escalation_present: bool
    issues: list[str]
    confidence: float


# Pattern to match Article X, Section Y citations
CITATION_PATTERN = r'\*?\*?Article\s+(\d+)(?:,?\s*Section\s+(\d+))?\*?\*?'

# Escalation phrases
ESCALATION_PHRASES = [
    "contact your steward",
    "speak with your steward",
    "reach out to your steward",
    "serious matter",
    "strongly recommend",
    "speak with a union representative",
]

# Phrases that indicate uncertainty/refusal (good!)
UNCERTAINTY_PHRASES = [
    "cannot find",
    "not in your contract",
    "unable to find",
    "please contact",
    "check with your steward",
    "not specified",
]


def extract_citations(text: str) -> list[str]:
    """Extract all Article/Section citations from text."""
    citations = []
    matches = re.finditer(CITATION_PATTERN, text, re.IGNORECASE)
    
    for match in matches:
        article_num = match.group(1)
        section_num = match.group(2)
        
        if section_num:
            citations.append(f"Article {article_num}, Section {section_num}")
        else:
            citations.append(f"Article {article_num}")
    
    return list(set(citations))  # Deduplicate


def has_escalation_language(text: str) -> bool:
    """Check if text contains escalation language."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in ESCALATION_PHRASES)


def has_uncertainty_language(text: str) -> bool:
    """Check if text contains appropriate uncertainty language."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in UNCERTAINTY_PHRASES)


def verify_citations_against_context(
    citations: list[str],
    chunks: list[dict]
) -> tuple[list[str], list[str]]:
    """
    Verify that citations match the retrieved context.
    
    Returns:
        Tuple of (valid_citations, invalid_citations)
    """
    # Extract article numbers from chunks
    context_articles = set()
    for chunk in chunks:
        article_num = chunk.get('article_num')
        if article_num:
            context_articles.add(int(article_num))
        
        # Also check citation field
        citation = chunk.get('citation', '')
        match = re.search(r'Article\s+(\d+)', citation)
        if match:
            context_articles.add(int(match.group(1)))
    
    valid = []
    invalid = []
    
    for citation in citations:
        match = re.search(r'Article\s+(\d+)', citation)
        if match:
            article_num = int(match.group(1))
            if article_num in context_articles:
                valid.append(citation)
            else:
                invalid.append(citation)
    
    return valid, invalid


def check_answer_grounding(response: str, chunks: list[dict]) -> dict:
    """
    Check if the answer content is grounded in the retrieved chunks.
    
    Looks for:
    - Dollar amounts that should be in context
    - Specific numbers (hours, days, percentages) that should be in context
    - Key terms from the answer that should appear in context
    
    Returns:
        dict with grounding_score (0-1) and potential_issues
    """
    issues = []
    grounding_score = 1.0
    
    # Combine all chunk content for checking
    context_text = " ".join(c.get('content', '') for c in chunks).lower()
    response_lower = response.lower()
    
    # Check dollar amounts
    dollar_pattern = r'\$(\d+(?:\.\d{2})?)'
    response_dollars = set(re.findall(dollar_pattern, response))
    context_dollars = set(re.findall(dollar_pattern, context_text))
    
    ungrounded_dollars = response_dollars - context_dollars
    if ungrounded_dollars:
        # Allow small tolerance for calculated values
        for dollar in ungrounded_dollars:
            issues.append(f"Dollar amount ${dollar} not found in context")
            grounding_score -= 0.15
    
    # Check hour/day numbers (only specific ones, not generic)
    time_patterns = [
        (r'(\d+)\s*hours?\s*(?:per|a|each)\s*(?:week|day)', 'hours'),
        (r'(\d+)\s*(?:consecutive\s+)?days?', 'days'),
        (r'(\d+)\s*weeks?', 'weeks'),
        (r'(\d+)\s*months?', 'months'),
    ]
    
    for pattern, unit in time_patterns:
        response_matches = re.findall(pattern, response_lower)
        context_matches = re.findall(pattern, context_text)
        
        for num in response_matches:
            if num not in context_matches and int(num) > 2:  # Ignore small numbers
                issues.append(f"Time value '{num} {unit}' may not be grounded")
                grounding_score -= 0.1
    
    # Check for percentage values
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    response_percents = set(re.findall(percent_pattern, response))
    context_percents = set(re.findall(percent_pattern, context_text))
    
    ungrounded_percents = response_percents - context_percents
    if ungrounded_percents:
        for pct in ungrounded_percents:
            issues.append(f"Percentage {pct}% not found in context")
            grounding_score -= 0.15
    
    # Clamp score
    grounding_score = max(0.0, min(1.0, grounding_score))
    
    return {
        'grounding_score': grounding_score,
        'potential_issues': issues[:5],  # Limit to top 5 issues
        'is_well_grounded': grounding_score >= 0.7
    }


def verify_response(
    response: str,
    chunks: list[dict],
    requires_escalation: bool = False,
    is_refusal: bool = False
) -> VerificationResult:
    """
    Verify that a response is properly grounded and cited.
    
    Args:
        response: The LLM's response text
        chunks: Retrieved context chunks
        requires_escalation: Whether escalation language is required
        is_refusal: Whether this should be a refusal (no info found)
    
    Returns:
        VerificationResult with validation details
    """
    issues = []
    
    # Extract citations
    citations = extract_citations(response)
    
    # Check escalation language
    has_escalation = has_escalation_language(response)
    has_uncertainty = has_uncertainty_language(response)
    
    # Verify citations against context
    if citations and chunks:
        valid_citations, invalid_citations = verify_citations_against_context(citations, chunks)
        
        if invalid_citations:
            issues.append(f"Citations not in context: {invalid_citations}")
    else:
        valid_citations = citations
        invalid_citations = []
    
    # Check grounding of specific values (dollars, hours, etc.)
    grounding_result = check_answer_grounding(response, chunks)
    if not grounding_result['is_well_grounded']:
        for issue in grounding_result['potential_issues'][:2]:  # Add top 2 issues
            issues.append(f"Grounding issue: {issue}")
    
    # Check for issues
    
    # Issue: No citations but making claims
    substantive_length = len(response) > 100
    if not citations and substantive_length and not has_uncertainty:
        issues.append("No citations found for substantive response")
    
    # Issue: Escalation required but not present
    if requires_escalation and not has_escalation:
        issues.append("High-stakes query missing escalation language")
    
    # Issue: Citations outside context
    if invalid_citations:
        issues.append(f"Hallucinated citations: {invalid_citations}")
    
    # Calculate confidence (combine citation confidence and grounding score)
    confidence = 1.0
    if issues:
        confidence -= 0.15 * len(issues)
    if citations:
        confidence += 0.1  # Bonus for having citations
    if invalid_citations:
        confidence -= 0.3  # Penalty for bad citations
    
    # Factor in grounding score
    confidence = (confidence + grounding_result['grounding_score']) / 2
    
    confidence = max(0.0, min(1.0, confidence))
    
    is_valid = len(issues) == 0 or (has_uncertainty and not invalid_citations)
    
    return VerificationResult(
        is_valid=is_valid,
        citations_found=citations,
        citations_missing=not citations and substantive_length and not has_uncertainty,
        escalation_present=has_escalation,
        issues=issues,
        confidence=confidence
    )


def add_escalation_if_missing(response: str, requires_escalation: bool) -> str:
    """Add escalation language if required but missing."""
    if not requires_escalation:
        return response
    
    if has_escalation_language(response):
        return response
    
    escalation = "\n\n**This is a serious matter. I strongly recommend contacting your steward immediately.**"
    return response + escalation


def format_response_with_sources(
    response: str,
    chunks: list[dict],
    wage_info: dict = None
) -> dict:
    """
    Format the final response with source information.
    
    Returns:
        dict with response, sources, and metadata
    """
    citations = extract_citations(response)
    
    sources = []
    for citation in citations:
        match = re.search(r'Article\s+(\d+)', citation)
        if match:
            article_num = int(match.group(1))
            # Find matching chunk
            for chunk in chunks:
                if chunk.get('article_num') == article_num:
                    sources.append({
                        'citation': citation,
                        'article_title': chunk.get('article_title', ''),
                        'doc_type': chunk.get('doc_type', 'cba')
                    })
                    break
    
    # Add wage source if applicable
    if wage_info:
        sources.append({
            'citation': 'Appendix A',
            'article_title': 'Wage Tables',
            'doc_type': 'appendix'
        })
    
    return {
        'response': response,
        'citations': citations,
        'sources': sources,
        'has_wage_info': wage_info is not None
    }


def main():
    """Test the verifier."""
    print("Testing Citation Verifier...")
    
    # Test cases
    test_cases = [
        {
            "response": "According to **Article 12, Section 28**, overtime is paid at time and one half. The contract states that this applies to hours over 8 in a day.",
            "chunks": [{"article_num": 12, "citation": "Article 12, Section 28"}],
            "requires_escalation": False
        },
        {
            "response": "Your manager cannot do that. You should file a grievance.",
            "chunks": [{"article_num": 46, "citation": "Article 46, Section 135"}],
            "requires_escalation": False
        },
        {
            "response": "This is a serious situation involving potential discipline. According to **Article 45, Section 132**, you have the right to union representation. I strongly recommend contacting your steward immediately.",
            "chunks": [{"article_num": 45, "citation": "Article 45, Section 132"}],
            "requires_escalation": True
        },
        {
            "response": "I cannot find that specific information in your contract. Please contact your steward for clarification.",
            "chunks": [],
            "requires_escalation": False
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = verify_response(
            case["response"],
            case["chunks"],
            case["requires_escalation"]
        )
        
        print(f"\n--- Test Case {i} ---")
        print(f"Valid: {result.is_valid}")
        print(f"Citations: {result.citations_found}")
        print(f"Escalation Present: {result.escalation_present}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.issues:
            print(f"Issues: {result.issues}")


if __name__ == "__main__":
    main()

