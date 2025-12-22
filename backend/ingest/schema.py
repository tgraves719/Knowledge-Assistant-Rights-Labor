"""
Schema definitions for enriched contract chunks.

Defines the metadata structure for the gold-standard Pueblo contract dataset.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# =============================================================================
# TAXONOMY DEFINITIONS
# =============================================================================

TOPICS = [
    "job_definitions",      # Article 7
    "wages",                # Article 8, Appendix A
    "promotions",           # Article 9
    "scheduling",           # Article 10
    "overtime",             # Article 12
    "sunday_premium",       # Article 13
    "travel_pay",           # Article 14
    "night_premium",        # Article 15
    "holidays",             # Article 16
    "personal_holidays",    # Article 16 Section 38
    "vacation",             # Article 17
    "health_benefits",      # Article 18, 40
    "bereavement",          # Article 20
    "jury_duty",            # Article 22
    "military_leave",       # Article 23
    "lunch_breaks",         # Article 24
    "rest_periods",         # Article 25
    "seniority",            # Article 27
    "layoff",               # Article 29
    "leaves",               # Articles 30-34
    "sick_leave",           # Article 35
    "safety",               # Article 36
    "pension",              # Article 42
    "discipline",           # Article 43
    "union_rights",         # Article 45
    "grievance",            # Article 46
    "store_closing",        # Article 48
    "no_strike",            # Article 49
    "lie_detector",         # Article 51
    "work_jurisdiction",    # Article 55
    "dress_code",           # LOU 8
    "dug",                  # Drive Up & Go - Article 7 Section 14h
    "management_rights",    # Article 6
    "union_security",       # Article 3
    "definitions",          # General definitional content
]

CLASSIFICATIONS = [
    "all",                  # Applies to everyone
    "all_purpose_clerk",
    "courtesy_clerk",
    "head_clerk",
    "produce_manager",
    "bakery_manager",
    "cake_decorator",
    "non_foods_clerk",
    "floral_manager",
    "pharmacy_tech",
    "dug_shopper",
    "sanitation_clerk",     # Legacy classification (May 11, 1996)
    "meat_cutter",          # If referenced
    "baker",
    "apprentice_meat_cutter",
]


# =============================================================================
# CHUNK SCHEMA
# =============================================================================

class EnrichedChunk(BaseModel):
    """
    Schema for a fully enriched contract chunk.
    
    Contains both structural metadata (from parsing) and semantic metadata
    (from LLM enrichment).
    """
    
    # ==========================================================================
    # IDENTIFICATION
    # ==========================================================================
    chunk_id: str = Field(
        ..., 
        description="Unique identifier: art{N}_sec{M}_{subsection} or lou{N}_{part}"
    )
    contract_id: str = Field(
        default="safeway_pueblo_clerks_2022",
        description="Contract identifier for multi-tenant filtering"
    )
    
    # ==========================================================================
    # HIERARCHY
    # ==========================================================================
    article_num: Optional[int] = Field(
        None,
        description="Article number (None for LOUs)"
    )
    article_title: Optional[str] = Field(
        None,
        description="Article title (e.g., 'DEFINITIONS', 'LAYOFFS')"
    )
    section_num: Optional[int] = Field(
        None,
        description="Section number within article"
    )
    subsection: Optional[str] = Field(
        None,
        description="Subsection letter (a, b, c...) or number (1, 2, 3...)"
    )
    subsection_title: Optional[str] = Field(
        None,
        description="Subsection title (e.g., 'DRIVE UP AND GO')"
    )
    citation: str = Field(
        ...,
        description="Human-readable citation: 'Article 7, Section 14, Subsection h'"
    )
    parent_context: Optional[str] = Field(
        None,
        description="Full hierarchy breadcrumb for context injection"
    )
    
    # ==========================================================================
    # CONTENT
    # ==========================================================================
    content: str = Field(
        ...,
        description="The actual text content of the chunk"
    )
    char_count: int = Field(
        ...,
        description="Character count for size monitoring"
    )
    
    # ==========================================================================
    # LLM-ENRICHED METADATA
    # ==========================================================================
    applies_to: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Job classifications this applies to"
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topic tags from TOPICS taxonomy"
    )
    cross_references: list[str] = Field(
        default_factory=list,
        description="References to other articles/sections (e.g., 'art40_sec116')"
    )
    summary: Optional[str] = Field(
        None,
        description="One-sentence summary of the provision"
    )
    
    # ==========================================================================
    # FLAGS
    # ==========================================================================
    is_definition: bool = Field(
        default=False,
        description="True if this chunk defines a term or classification"
    )
    is_exception: bool = Field(
        default=False,
        description="True if contains 'except', 'notwithstanding', override language"
    )
    hire_date_sensitive: bool = Field(
        default=False,
        description="True if different rules for pre/post March 27, 2005 hires"
    )
    is_high_stakes: bool = Field(
        default=False,
        description="True if involves discipline, termination, safety, harassment"
    )
    
    # ==========================================================================
    # VALIDATORS
    # ==========================================================================
    @field_validator('applies_to')
    @classmethod
    def validate_applies_to(cls, v):
        """Ensure all classifications are from the allowed list."""
        for classification in v:
            if classification not in CLASSIFICATIONS:
                raise ValueError(f"Invalid classification: {classification}. Must be one of {CLASSIFICATIONS}")
        return v
    
    @field_validator('topics')
    @classmethod
    def validate_topics(cls, v):
        """Ensure all topics are from the allowed list."""
        for topic in v:
            if topic not in TOPICS:
                raise ValueError(f"Invalid topic: {topic}. Must be one of {TOPICS}")
        return v
    
    @field_validator('char_count')
    @classmethod
    def compute_char_count(cls, v, info):
        """Auto-compute char_count from content if not provided."""
        if v == 0 and 'content' in info.data:
            return len(info.data['content'])
        return v
    
    def to_vector_metadata(self) -> dict:
        """
        Convert to metadata dict for ChromaDB storage.
        
        ChromaDB requires flat metadata values (no nested lists as-is),
        so we serialize lists to JSON strings.
        """
        import json
        return {
            "chunk_id": self.chunk_id,
            "contract_id": self.contract_id,
            "article_num": self.article_num or 0,
            "article_title": self.article_title or "",
            "section_num": self.section_num or 0,
            "subsection": self.subsection or "",
            "citation": self.citation,
            "parent_context": self.parent_context or "",
            "char_count": self.char_count,
            "applies_to": json.dumps(self.applies_to),
            "topics": json.dumps(self.topics),
            "cross_references": json.dumps(self.cross_references),
            "summary": self.summary or "",
            "is_definition": self.is_definition,
            "is_exception": self.is_exception,
            "hire_date_sensitive": self.hire_date_sensitive,
            "is_high_stakes": self.is_high_stakes,
        }
    
    @classmethod
    def from_vector_metadata(cls, metadata: dict, content: str) -> "EnrichedChunk":
        """
        Reconstruct from ChromaDB metadata.
        """
        import json
        return cls(
            chunk_id=metadata.get("chunk_id", ""),
            contract_id=metadata.get("contract_id", "safeway_pueblo_clerks_2022"),
            article_num=metadata.get("article_num") or None,
            article_title=metadata.get("article_title") or None,
            section_num=metadata.get("section_num") or None,
            subsection=metadata.get("subsection") or None,
            citation=metadata.get("citation", ""),
            parent_context=metadata.get("parent_context") or None,
            content=content,
            char_count=len(content),
            applies_to=json.loads(metadata.get("applies_to", '["all"]')),
            topics=json.loads(metadata.get("topics", "[]")),
            cross_references=json.loads(metadata.get("cross_references", "[]")),
            summary=metadata.get("summary") or None,
            is_definition=metadata.get("is_definition", False),
            is_exception=metadata.get("is_exception", False),
            hire_date_sensitive=metadata.get("hire_date_sensitive", False),
            is_high_stakes=metadata.get("is_high_stakes", False),
        )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_chunk(chunk_dict: dict) -> tuple[bool, list[str]]:
    """
    Validate a chunk dictionary against the schema.
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    try:
        EnrichedChunk(**chunk_dict)
        return True, []
    except Exception as e:
        errors.append(str(e))
        return False, errors


def validate_chunks(chunks: list[dict]) -> dict:
    """
    Validate a list of chunks and return summary.
    
    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "errors": [{"chunk_id": str, "errors": [str]}]
        }
    """
    results = {
        "total": len(chunks),
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    for chunk in chunks:
        is_valid, errors = validate_chunk(chunk)
        if is_valid:
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["errors"].append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "citation": chunk.get("citation", "unknown"),
                "errors": errors
            })
    
    return results


if __name__ == "__main__":
    # Test the schema
    test_chunk = EnrichedChunk(
        chunk_id="art7_sec14_h",
        article_num=7,
        article_title="DEFINITIONS",
        section_num=14,
        subsection="h",
        subsection_title="DRIVE UP AND GO",
        citation="Article 7, Section 14, Subsection h",
        parent_context="Article 7 (Definitions) > Section 14 (Job Classifications) > Subsection h",
        content="DUG Shoppers will select and pack customer-ready products...",
        char_count=890,
        applies_to=["courtesy_clerk", "dug_shopper"],
        topics=["job_definitions", "dug"],
        cross_references=["art55_sec162"],
        summary="Defines DUG Shopper classification, duties, and wage rates.",
        is_definition=True,
        is_exception=False,
        hire_date_sensitive=False,
        is_high_stakes=False,
    )
    
    print("Schema validation passed!")
    print(f"Chunk: {test_chunk.citation}")
    print(f"Topics: {test_chunk.topics}")
    print(f"Applies to: {test_chunk.applies_to}")
    print(f"\nVector metadata:")
    for k, v in test_chunk.to_vector_metadata().items():
        print(f"  {k}: {v}")

