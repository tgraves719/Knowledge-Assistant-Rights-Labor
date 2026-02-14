"""
Citation Entailment Checker for Karl RAG System.

Uses a two-stage approach:
1. Lightweight NLI model for fast, cheap first-pass screening
2. LLM fallback for low-confidence or CONTRADICTS cases

This catches the sneakiest failure mode: answers that "look grounded" but aren't.
The model can cite the right article and still hallucinate the rule.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional
from functools import lru_cache

# Attempt to import transformers for NLI model
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using LLM-only entailment.")


@dataclass
class EntailmentResult:
    """Result of a single claim-citation entailment check."""
    claim: str
    citation: str  # e.g., "Article 12, Section 28"
    cited_text: str  # The actual text from the cited chunk
    verdict: Literal["SUPPORTS", "CONTRADICTS", "IRRELEVANT"]
    confidence: float  # 0.0 to 1.0
    method: Literal["nli", "llm"]  # Which method was used
    
    def __repr__(self):
        return f"EntailmentResult({self.verdict}, conf={self.confidence:.2f}, claim='{self.claim[:50]}...')"


@dataclass  
class EntailmentSummary:
    """Summary of all entailment checks for an answer."""
    results: List[EntailmentResult] = field(default_factory=list)
    
    @property
    def total_claims(self) -> int:
        return len(self.results)
    
    @property
    def supported_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "SUPPORTS")
    
    @property
    def contradicted_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "CONTRADICTS")
    
    @property
    def irrelevant_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "IRRELEVANT")
    
    @property
    def support_rate(self) -> float:
        """Percentage of claims that are supported by citations."""
        if not self.results:
            return 1.0  # No claims = vacuously true
        return self.supported_count / len(self.results)
    
    @property
    def has_contradiction(self) -> bool:
        return self.contradicted_count > 0
    
    @property
    def all_supported(self) -> bool:
        return all(r.verdict == "SUPPORTS" for r in self.results)


class EntailmentChecker:
    """
    Two-stage entailment checker: NLI first, LLM fallback.
    
    Usage:
        checker = EntailmentChecker()
        result = checker.check("Overtime is 1.5x", "Article 12", chunk_text)
    """
    
    # NLI model options (in order of preference)
    NLI_MODELS = [
        "microsoft/deberta-v3-large-mnli",  # Best accuracy
        "roberta-large-mnli",                # Good balance
        "facebook/bart-large-mnli",          # Fast
    ]
    
    # Confidence threshold for NLI - below this, escalate to LLM
    NLI_CONFIDENCE_THRESHOLD = 0.75
    
    def __init__(
        self, 
        use_nli: bool = True,
        nli_model: str = None,
        llm_client = None,
        llm_model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the entailment checker.
        
        Args:
            use_nli: Whether to use NLI model for first-pass (faster, cheaper)
            nli_model: Specific NLI model to use (default: deberta-v3-large-mnli)
            llm_client: Optional pre-initialized LLM client for fallback
            llm_model: LLM model name for fallback
        """
        self.use_nli = use_nli and HAS_TRANSFORMERS
        self.nli_pipeline = None
        self.llm_client = llm_client
        self.llm_model = llm_model
        
        if self.use_nli:
            self._init_nli(nli_model)
    
    def _init_nli(self, model_name: str = None):
        """Initialize the NLI pipeline."""
        if not HAS_TRANSFORMERS:
            self.use_nli = False
            return
            
        model_name = model_name or self.NLI_MODELS[0]
        
        try:
            self.nli_pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1  # CPU; use 0 for GPU
            )
            print(f"Loaded NLI model: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load NLI model {model_name}: {e}")
            self.use_nli = False
    
    def _init_llm(self):
        """Lazy-initialize LLM client if needed."""
        if self.llm_client is not None:
            return
            
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                self.llm_client = genai.Client(api_key=api_key)
        except ImportError:
            print("Warning: google-genai not installed for LLM fallback")
    
    def check_nli(self, claim: str, cited_text: str) -> Tuple[str, float]:
        """
        Check entailment using NLI model.
        
        Returns:
            Tuple of (verdict, confidence)
        """
        if not self.nli_pipeline:
            raise RuntimeError("NLI pipeline not initialized")
        
        # NLI models work with premise (cited_text) and hypothesis (claim)
        # We use zero-shot classification with entailment labels
        result = self.nli_pipeline(
            sequences=claim,
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="This text supports the statement: {}",
            multi_label=False
        )
        
        # Map NLI labels to our verdicts
        label_map = {
            "entailment": "SUPPORTS",
            "contradiction": "CONTRADICTS", 
            "neutral": "IRRELEVANT"
        }
        
        # Get top label and score
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        verdict = label_map.get(top_label, "IRRELEVANT")
        
        return verdict, top_score
    
    def check_llm(self, claim: str, cited_text: str) -> Tuple[str, float]:
        """
        Check entailment using LLM (more accurate but slower/expensive).
        
        Returns:
            Tuple of (verdict, confidence)
        """
        self._init_llm()
        
        if not self.llm_client:
            # Fallback to conservative "IRRELEVANT" if no LLM
            return "IRRELEVANT", 0.5
        
        prompt = f"""You are checking if a cited text supports a claim.

CLAIM: {claim}

CITED TEXT: {cited_text}

Does the cited text support the claim?

Respond with exactly one of:
- SUPPORTS: The cited text directly supports or implies the claim
- CONTRADICTS: The cited text states something incompatible with the claim
- IRRELEVANT: The cited text doesn't address the claim (neither supports nor contradicts)

Also provide a confidence score from 0.0 to 1.0.

Format your response as:
VERDICT: [SUPPORTS/CONTRADICTS/IRRELEVANT]
CONFIDENCE: [0.0-1.0]
REASON: [One sentence explanation]"""

        try:
            from google import genai
            response = self.llm_client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic
                    max_output_tokens=150,
                )
            )
            
            text = response.text.strip()
            
            # Parse response
            verdict = "IRRELEVANT"
            confidence = 0.5
            
            if "VERDICT:" in text:
                verdict_line = [l for l in text.split("\n") if "VERDICT:" in l][0]
                if "SUPPORTS" in verdict_line.upper():
                    verdict = "SUPPORTS"
                elif "CONTRADICTS" in verdict_line.upper():
                    verdict = "CONTRADICTS"
                else:
                    verdict = "IRRELEVANT"
            
            if "CONFIDENCE:" in text:
                conf_line = [l for l in text.split("\n") if "CONFIDENCE:" in l][0]
                conf_match = re.search(r"(\d+\.?\d*)", conf_line)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    confidence = min(1.0, max(0.0, confidence))
            
            return verdict, confidence
            
        except Exception as e:
            print(f"LLM entailment check failed: {e}")
            return "IRRELEVANT", 0.5
    
    def check(
        self, 
        claim: str, 
        citation: str,
        cited_text: str,
        force_llm: bool = False
    ) -> EntailmentResult:
        """
        Check if cited text entails the claim.
        
        Uses NLI first, escalates to LLM if:
        - NLI confidence is low
        - NLI verdict is CONTRADICTS (want to be sure)
        - force_llm is True
        
        Args:
            claim: The factual claim made in the answer
            citation: The citation reference (e.g., "Article 12, Section 28")
            cited_text: The actual text from the cited chunk
            force_llm: Skip NLI and go straight to LLM
            
        Returns:
            EntailmentResult with verdict and confidence
        """
        # Truncate long texts to avoid model limits
        cited_text_truncated = cited_text[:2000] if len(cited_text) > 2000 else cited_text
        claim_truncated = claim[:500] if len(claim) > 500 else claim
        
        method = "llm"
        
        if self.use_nli and not force_llm:
            try:
                verdict, confidence = self.check_nli(claim_truncated, cited_text_truncated)
                method = "nli"
                
                # Escalate to LLM if low confidence or contradiction
                should_escalate = (
                    confidence < self.NLI_CONFIDENCE_THRESHOLD or
                    verdict == "CONTRADICTS"
                )
                
                if should_escalate:
                    llm_verdict, llm_confidence = self.check_llm(claim_truncated, cited_text_truncated)
                    # Trust LLM if it's more confident
                    if llm_confidence > confidence:
                        verdict = llm_verdict
                        confidence = llm_confidence
                        method = "llm"
                        
            except Exception as e:
                print(f"NLI check failed, falling back to LLM: {e}")
                verdict, confidence = self.check_llm(claim_truncated, cited_text_truncated)
                method = "llm"
        else:
            verdict, confidence = self.check_llm(claim_truncated, cited_text_truncated)
        
        return EntailmentResult(
            claim=claim,
            citation=citation,
            cited_text=cited_text_truncated,
            verdict=verdict,
            confidence=confidence,
            method=method
        )
    
    def check_all(
        self,
        claim_citation_pairs: List[Tuple[str, str, str]],
    ) -> EntailmentSummary:
        """
        Check entailment for multiple claim-citation pairs.
        
        Args:
            claim_citation_pairs: List of (claim, citation, cited_text) tuples
            
        Returns:
            EntailmentSummary with all results
        """
        results = []
        for claim, citation, cited_text in claim_citation_pairs:
            result = self.check(claim, citation, cited_text)
            results.append(result)
        
        return EntailmentSummary(results=results)


def extract_claim_citation_pairs(
    answer: str,
    chunks: List[dict]
) -> List[Tuple[str, str, str]]:
    """
    Extract claim-citation pairs from an answer.
    
    Looks for patterns like:
    - "According to Article X, [claim]"
    - "[claim] (Article X, Section Y)"
    - "Article X states that [claim]"
    
    Args:
        answer: The LLM's answer text
        chunks: Retrieved chunks with content and citation info
        
    Returns:
        List of (claim, citation, cited_text) tuples
    """
    pairs = []
    
    # Build a map of citation -> chunk content
    citation_to_content = {}
    for chunk in chunks:
        article_num = chunk.get('article_num')
        section_num = chunk.get('section_num')
        content = chunk.get('content', '')
        
        if article_num:
            # Create multiple lookup keys
            citation_to_content[f"Article {article_num}"] = content
            if section_num:
                citation_to_content[f"Article {article_num}, Section {section_num}"] = content
    
    # Pattern 1: "According to Article X, [claim]"
    pattern1 = r"(?:According to|Per|Under|As stated in)\s+\*?\*?(Article\s+\d+(?:,?\s*Section\s+\d+)?)\*?\*?,?\s+(.+?)(?:\.|$)"
    for match in re.finditer(pattern1, answer, re.IGNORECASE):
        citation = match.group(1)
        claim = match.group(2).strip()
        cited_text = _find_cited_text(citation, citation_to_content)
        if cited_text and claim:
            pairs.append((claim, citation, cited_text))
    
    # Pattern 2: "[claim] (Article X, Section Y)" or "[claim] per Article X"
    pattern2 = r"(.+?)\s*(?:\(|\bper\b|\bunder\b)\s*\*?\*?(Article\s+\d+(?:,?\s*Section\s+\d+)?)\*?\*?\)?"
    for match in re.finditer(pattern2, answer, re.IGNORECASE):
        claim = match.group(1).strip()
        citation = match.group(2)
        # Skip if claim is too short or starts with common non-claim words
        if len(claim) < 10 or claim.lower().startswith(('this', 'see', 'refer')):
            continue
        cited_text = _find_cited_text(citation, citation_to_content)
        if cited_text and claim:
            pairs.append((claim, citation, cited_text))
    
    # Pattern 3: "Article X states that [claim]"
    pattern3 = r"\*?\*?(Article\s+\d+(?:,?\s*Section\s+\d+)?)\*?\*?\s+(?:states?|provides?|requires?|specifies?)\s+(?:that\s+)?(.+?)(?:\.|$)"
    for match in re.finditer(pattern3, answer, re.IGNORECASE):
        citation = match.group(1)
        claim = match.group(2).strip()
        cited_text = _find_cited_text(citation, citation_to_content)
        if cited_text and claim:
            pairs.append((claim, citation, cited_text))
    
    # Deduplicate by claim (keep first occurrence)
    seen_claims = set()
    unique_pairs = []
    for claim, citation, cited_text in pairs:
        claim_key = claim.lower()[:50]  # Normalize for comparison
        if claim_key not in seen_claims:
            seen_claims.add(claim_key)
            unique_pairs.append((claim, citation, cited_text))
    
    return unique_pairs


def _find_cited_text(citation: str, citation_to_content: dict) -> str:
    """Find the content for a citation, with fuzzy matching."""
    # Direct lookup
    if citation in citation_to_content:
        return citation_to_content[citation]
    
    # Try without section
    article_match = re.search(r"Article\s+(\d+)", citation)
    if article_match:
        article_only = f"Article {article_match.group(1)}"
        if article_only in citation_to_content:
            return citation_to_content[article_only]
    
    return ""


def check_entailment(
    answer: str,
    chunks: List[dict],
    checker: EntailmentChecker = None
) -> EntailmentSummary:
    """
    Convenience function to check all entailments in an answer.
    
    Args:
        answer: The LLM's answer text
        chunks: Retrieved chunks
        checker: Optional pre-initialized EntailmentChecker
        
    Returns:
        EntailmentSummary with all results
    """
    if checker is None:
        checker = EntailmentChecker(use_nli=HAS_TRANSFORMERS)
    
    pairs = extract_claim_citation_pairs(answer, chunks)
    
    if not pairs:
        # No claims with citations found
        return EntailmentSummary(results=[])
    
    return checker.check_all(pairs)


# === Testing ===

def main():
    """Test the entailment checker."""
    print("=" * 60)
    print("Testing Entailment Checker")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "claim": "Overtime is paid at time and one-half (1.5x) the base rate",
            "citation": "Article 12, Section 28",
            "cited_text": "All hours worked over eight (8) in one day shall be paid at the rate of time and one-half (1.5x) the employee's base hourly rate.",
            "expected": "SUPPORTS"
        },
        {
            "claim": "Overtime is paid at double time (2.0x)",
            "citation": "Article 12, Section 28", 
            "cited_text": "All hours worked over eight (8) in one day shall be paid at the rate of time and one-half (1.5x) the employee's base hourly rate.",
            "expected": "CONTRADICTS"
        },
        {
            "claim": "Employees must wear blue uniforms",
            "citation": "Article 12, Section 28",
            "cited_text": "All hours worked over eight (8) in one day shall be paid at the rate of time and one-half (1.5x) the employee's base hourly rate.",
            "expected": "IRRELEVANT"
        },
        {
            "claim": "The probationary period is sixty (60) calendar days",
            "citation": "Article 26, Section 62",
            "cited_text": "The first sixty (60) calendar days of employment shall be the probationary/trial period.",
            "expected": "SUPPORTS"
        },
    ]
    
    # Initialize checker (LLM-only for testing without transformers)
    checker = EntailmentChecker(use_nli=False)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Claim: {case['claim'][:60]}...")
        print(f"Expected: {case['expected']}")
        
        result = checker.check(
            claim=case['claim'],
            citation=case['citation'],
            cited_text=case['cited_text']
        )
        
        print(f"Verdict: {result.verdict}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.method}")
        print(f"Match: {'✓' if result.verdict == case['expected'] else '✗'}")
    
    # Test extraction
    print("\n" + "=" * 60)
    print("Testing Claim-Citation Extraction")
    print("=" * 60)
    
    test_answer = """
    According to **Article 12, Section 28**, overtime is paid at time and one-half. 
    The contract also specifies that Sunday work receives premium pay (Article 13).
    Article 26 states that the probationary period is sixty calendar days.
    """
    
    test_chunks = [
        {"article_num": 12, "section_num": 28, "content": "Overtime at 1.5x rate..."},
        {"article_num": 13, "content": "Sunday premium pay..."},
        {"article_num": 26, "content": "Probationary period..."},
    ]
    
    pairs = extract_claim_citation_pairs(test_answer, test_chunks)
    print(f"\nExtracted {len(pairs)} claim-citation pairs:")
    for claim, citation, _ in pairs:
        print(f"  - [{citation}] {claim[:50]}...")


if __name__ == "__main__":
    main()


