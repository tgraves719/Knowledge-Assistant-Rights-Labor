"""
Contract Manifest Extractor

Automatically extracts metadata from a contract for auto-configuration:
- Job classifications mentioned
- Article structure and titles
- Key dates (hire date cutoffs, effective dates)
- Employer and Union info

This enables zero-config onboarding of new contracts.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import date


@dataclass
class ContractManifest:
    """
    Metadata about a contract for multi-tenant configuration.
    
    This is auto-extracted from the contract content, not manually configured.
    """
    # Identity
    contract_id: str
    employer: str = ""
    union_local: str = ""
    bargaining_unit: str = ""
    
    # Term
    term_start: Optional[str] = None
    term_end: Optional[str] = None
    
    # Structure
    article_titles: dict = field(default_factory=dict)  # {1: "Recognition", 7: "Definitions"}
    total_articles: int = 0
    total_sections: int = 0
    has_appendix_a: bool = False
    has_lous: bool = False
    
    # Classifications (auto-detected)
    classifications: list[str] = field(default_factory=list)
    
    # Key dates (hire date cutoffs, grandfathering)
    key_dates: list[str] = field(default_factory=list)
    
    # Topics covered (for UI display)
    topics_covered: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ContractManifest":
        data = json.loads(json_str)
        return cls(**data)


class ManifestExtractor:
    """
    Extracts contract manifest from markdown content.
    
    Uses regex patterns to identify key information.
    For production, could use LLM for more accurate extraction.
    """
    
    # Classification patterns (common across UFCW contracts)
    CLASSIFICATION_PATTERNS = [
        r"(?:all\s*purpose\s*clerk|general\s*clerk)",
        r"courtesy\s*clerk",
        r"head\s*clerk",
        r"produce\s*(?:department\s*)?manager",
        r"bakery\s*(?:department\s*)?manager",
        r"cake\s*decorator",
        r"pharmacy\s*tech(?:nician)?",
        r"non[- ]?foods?\s*clerk",
        r"floral\s*(?:department\s*)?manager",
        r"meat\s*(?:cutter|clerk)",
        r"deli\s*clerk",
        r"sanitation\s*clerk",
        r"dug\s*shopper",
        r"drive\s*up\s*(?:and\s*)?go",
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"\d{4}-\d{2}-\d{2}",
    ]
    
    def __init__(self):
        self.content = ""
    
    def extract(self, content: str, contract_id: str) -> ContractManifest:
        """
        Extract manifest from contract markdown content.
        
        Args:
            content: Full markdown content of contract
            contract_id: Identifier for this contract
            
        Returns:
            ContractManifest with extracted metadata
        """
        self.content = content
        content_lower = content.lower()
        
        manifest = ContractManifest(contract_id=contract_id)
        
        # Extract employer/union
        manifest.employer = self._extract_employer()
        manifest.union_local = self._extract_union()
        manifest.bargaining_unit = self._extract_bargaining_unit()
        
        # Extract term dates
        term_dates = self._extract_term_dates()
        if term_dates:
            manifest.term_start = term_dates[0]
            manifest.term_end = term_dates[1] if len(term_dates) > 1 else None
        
        # Extract article structure
        manifest.article_titles = self._extract_article_titles()
        manifest.total_articles = len(manifest.article_titles)
        manifest.total_sections = self._count_sections()
        
        # Check for appendix and LOUs
        manifest.has_appendix_a = "appendix a" in content_lower or "appendix" in content_lower
        manifest.has_lous = "letter of understanding" in content_lower
        
        # Extract classifications
        manifest.classifications = self._extract_classifications()
        
        # Extract key dates
        manifest.key_dates = self._extract_key_dates()
        
        # Infer topics from article titles
        manifest.topics_covered = self._infer_topics(manifest.article_titles)
        
        return manifest
    
    def _extract_employer(self) -> str:
        """Extract employer name."""
        patterns = [
            r"between\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|LLC|Corporation|Company))",
            r"employer[:\s]+([A-Z][A-Za-z\s,\.]+(?:Inc\.|LLC))",
            r"(Safeway\s+Inc\.|Albertsons\s+LLC|King\s+Soopers|Kroger)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Employer"
    
    def _extract_union(self) -> str:
        """Extract union local."""
        patterns = [
            r"(UFCW\s*Local\s*\d+)",
            r"(United\s+Food\s+(?:and|&)\s+Commercial\s+Workers\s+Local\s*\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Union"
    
    def _extract_bargaining_unit(self) -> str:
        """Extract bargaining unit description."""
        patterns = [
            r"(Pueblo\s+Clerks?)",
            r"(Denver\s+(?:Metro\s+)?Clerks?)",
            r"(Colorado\s+Springs\s+Clerks?)",
            r"bargaining\s+unit[:\s]+([A-Za-z\s]+clerks?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Clerks"
    
    def _extract_term_dates(self) -> list[str]:
        """Extract contract term start and end dates."""
        dates = []
        
        # Look for term section
        term_pattern = r"(?:term|effective|in\s+force)[^\n]*(\d{4})[^\n]*(\d{4})"
        match = re.search(term_pattern, self.content, re.IGNORECASE)
        if match:
            # Find actual dates near these years
            for date_pattern in self.DATE_PATTERNS:
                for date_match in re.finditer(date_pattern, self.content, re.IGNORECASE):
                    dates.append(date_match.group(0))
                    if len(dates) >= 2:
                        break
                if len(dates) >= 2:
                    break
        
        return dates[:2]
    
    def _extract_article_titles(self) -> dict:
        """Extract article numbers and titles."""
        titles = {}
        
        # Pattern: ARTICLE N or ARTICLE N TITLE
        patterns = [
            r"#{1,2}\s*ARTICLE\s+(\d+)\s*\n#{1,2}\s*([A-Z][A-Z\s&,]+)",
            r"#{1,2}\s*ARTICLE\s+(\d+)\s+([A-Z][A-Z\s&,]+)",
            r"ARTICLE\s+(\d+)[:\s]+([A-Z][A-Z\s&,]+)",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, self.content):
                article_num = int(match.group(1))
                title = match.group(2).strip().title()
                if article_num not in titles:
                    titles[article_num] = title
        
        return dict(sorted(titles.items()))
    
    def _count_sections(self) -> int:
        """Count total sections in the contract."""
        section_pattern = r"Section\s+\*{0,2}(\d+)\*{0,2}"
        matches = re.findall(section_pattern, self.content, re.IGNORECASE)
        return len(set(matches))
    
    def _extract_classifications(self) -> list[str]:
        """Extract job classifications mentioned in the contract."""
        classifications = set()
        content_lower = self.content.lower()
        
        for pattern in self.CLASSIFICATION_PATTERNS:
            if re.search(pattern, content_lower):
                # Normalize the classification name
                match = re.search(pattern, content_lower)
                if match:
                    name = match.group(0)
                    # Convert to title case and normalize
                    name = re.sub(r'\s+', ' ', name).strip().title()
                    classifications.add(name)
        
        return sorted(list(classifications))
    
    def _extract_key_dates(self) -> list[str]:
        """Extract key dates (hire cutoffs, grandfathering dates)."""
        key_dates = set()
        
        # Look for dates with context suggesting they're important cutoffs
        context_patterns = [
            r"(?:hired|employed)\s+(?:on\s+or\s+)?(?:before|after|prior\s+to)\s+(" + "|".join(self.DATE_PATTERNS) + ")",
            r"(?:effective|as\s+of)\s+(" + "|".join(self.DATE_PATTERNS) + ")",
        ]
        
        for pattern in context_patterns:
            for match in re.finditer(pattern, self.content, re.IGNORECASE):
                key_dates.add(match.group(1))
        
        return sorted(list(key_dates))
    
    def _infer_topics(self, article_titles: dict) -> list[str]:
        """Infer topics from article titles."""
        topics = []
        
        topic_keywords = {
            "wages": ["wage", "pay", "compensation"],
            "scheduling": ["schedule", "hours", "assignment"],
            "vacation": ["vacation", "holiday", "time off"],
            "health_benefits": ["health", "benefit", "trust", "insurance"],
            "seniority": ["seniority", "layoff"],
            "grievance": ["grievance", "arbitration", "dispute"],
            "discipline": ["discharge", "discipline", "no discrimination"],
            "safety": ["safety", "protective"],
            "pension": ["pension", "retirement"],
            "leaves": ["leave", "absence", "family"],
        }
        
        titles_lower = " ".join(str(t).lower() for t in article_titles.values())
        
        for topic, keywords in topic_keywords.items():
            if any(kw in titles_lower for kw in keywords):
                topics.append(topic)
        
        return topics


def extract_manifest(markdown_path: Path, contract_id: str) -> ContractManifest:
    """
    Extract manifest from a markdown file.
    
    Args:
        markdown_path: Path to contract markdown file
        contract_id: Identifier for this contract
        
    Returns:
        ContractManifest with extracted metadata
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    extractor = ManifestExtractor()
    return extractor.extract(content, contract_id)


def save_manifest(manifest: ContractManifest, output_path: Path):
    """Save manifest to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(manifest.to_json())


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from backend.config import CONTRACT_MD_FILE, DATA_DIR
    
    print("Contract Manifest Extractor")
    print("=" * 60)
    
    # Extract manifest from Pueblo contract
    manifest = extract_manifest(
        CONTRACT_MD_FILE,
        contract_id="safeway_pueblo_clerks_2022"
    )
    
    print(f"\nContract: {manifest.contract_id}")
    print(f"Employer: {manifest.employer}")
    print(f"Union: {manifest.union_local}")
    print(f"Bargaining Unit: {manifest.bargaining_unit}")
    print(f"Term: {manifest.term_start} to {manifest.term_end}")
    
    print(f"\nStructure:")
    print(f"  Articles: {manifest.total_articles}")
    print(f"  Sections: {manifest.total_sections}")
    print(f"  Has Appendix A: {manifest.has_appendix_a}")
    print(f"  Has LOUs: {manifest.has_lous}")
    
    print(f"\nClassifications Detected ({len(manifest.classifications)}):")
    for cls in manifest.classifications:
        print(f"  - {cls}")
    
    print(f"\nKey Dates ({len(manifest.key_dates)}):")
    for date in manifest.key_dates[:5]:
        print(f"  - {date}")
    
    print(f"\nTopics Covered:")
    for topic in manifest.topics_covered:
        print(f"  - {topic}")
    
    print(f"\nArticle Titles:")
    for num, title in list(manifest.article_titles.items())[:10]:
        print(f"  {num}: {title}")
    if len(manifest.article_titles) > 10:
        print(f"  ... and {len(manifest.article_titles) - 10} more")
    
    # Save manifest
    output_path = DATA_DIR / "manifests" / f"{manifest.contract_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, output_path)
    print(f"\nSaved manifest to {output_path}")



