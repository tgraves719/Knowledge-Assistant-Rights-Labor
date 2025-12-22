"""
Rule-Based Metadata Enricher - Fast tagging without LLM.

Uses regex patterns and article mappings to automatically tag chunks
with topics, applies_to, and flags based on content analysis.

This runs instantly and provides a baseline. LLM enrichment can
enhance these tags later.
"""

import re
import json
from pathlib import Path
from typing import List, Set


# =============================================================================
# ARTICLE TO TOPIC MAPPING
# =============================================================================

ARTICLE_TOPICS = {
    1: ["definitions"],
    2: ["work_jurisdiction"],
    3: ["union_security"],
    4: ["union_security"],
    5: ["management_rights"],
    6: ["management_rights"],
    7: ["job_definitions", "definitions"],
    8: ["wages"],
    9: ["promotions"],
    10: ["scheduling"],
    11: ["scheduling"],
    12: ["overtime"],
    13: ["sunday_premium", "premiums"],
    14: ["travel_pay"],
    15: ["night_premium", "premiums"],
    16: ["holidays", "personal_holidays"],
    17: ["vacation"],
    18: ["health_benefits"],
    19: ["holidays"],
    20: ["bereavement", "leaves"],
    21: ["scheduling"],
    22: ["jury_duty", "leaves"],
    23: ["military_leave", "leaves"],
    24: ["lunch_breaks"],
    25: ["rest_periods"],
    26: ["leaves"],
    27: ["seniority"],
    28: ["seniority"],
    29: ["layoff"],
    30: ["leaves"],
    31: ["leaves"],
    32: ["leaves"],
    33: ["leaves"],
    34: ["leaves"],
    35: ["sick_leave"],
    36: ["safety"],
    37: ["safety"],
    38: ["safety"],
    39: ["safety"],
    40: ["health_benefits"],
    41: ["health_benefits"],
    42: ["pension"],
    43: ["discipline"],
    44: ["pension"],
    45: ["union_rights"],
    46: ["grievance"],
    47: ["grievance"],
    48: ["store_closing"],
    49: ["no_strike"],
    50: ["no_strike"],
    51: ["lie_detector"],
    52: [],
    53: [],
    54: ["work_jurisdiction"],
    55: ["work_jurisdiction", "dug"],
    56: [],
    57: [],
    58: [],
}

# =============================================================================
# CONTENT PATTERNS FOR TOPIC DETECTION
# =============================================================================

TOPIC_PATTERNS = {
    "wages": [r"\$\d+\.\d+", r"hourly rate", r"wage rate", r"pay rate", r"per hour", r"minimum wage"],
    "overtime": [r"overtime", r"time and a half", r"1\s*[½1/2]", r"over forty"],
    "scheduling": [r"schedule", r"shift", r"work hours", r"posted", r"workweek"],
    "vacation": [r"vacation", r"annual leave", r"paid time off"],
    "sick_leave": [r"sick leave", r"illness", r"sick day"],
    "health_benefits": [r"health trust", r"health benefit", r"medical", r"insurance"],
    "discipline": [r"discharge", r"terminat", r"disciplin", r"just cause", r"warning"],
    "grievance": [r"grievance", r"arbitrat", r"dispute"],
    "layoff": [r"layoff", r"lay off", r"bumping", r"recall", r"displacement"],
    "seniority": [r"seniority", r"years of service", r"length of service"],
    "dug": [r"drive up", r"dug", r"personal shopper", r"clicklist"],
    "union_rights": [r"union representative", r"steward", r"weingarten"],
    "premiums": [r"premium", r"sunday pay", r"night pay", r"holiday pay", r"sunday premium"],
    "leaves": [r"leave of absence", r"loa", r"family leave", r"medical leave"],
    "safety": [r"safety", r"hazard", r"protective equipment", r"ppe", r"injury", r"labor-management"],
    "breaks": [r"rest period", r"break", r"meal period", r"lunch"],
    "dress_code": [r"dress code", r"uniform", r"appearance", r"grooming"],
    "probation": [r"probation", r"trial period", r"probationary"],
    "term": [r"term of agreement", r"effective date", r"expiration", r"january 18, 2025"],
}

# =============================================================================
# CLASSIFICATION PATTERNS
# =============================================================================

CLASSIFICATION_PATTERNS = {
    "courtesy_clerk": [r"courtesy clerk", r"bagger"],
    "head_clerk": [r"head clerk"],
    "all_purpose_clerk": [r"all purpose clerk", r"general clerk"],
    "cake_decorator": [r"cake decorator"],
    "produce_manager": [r"produce.*manager"],
    "bakery_manager": [r"bakery.*manager"],
    "pharmacy_tech": [r"pharmacy technician", r"pharmacy tech"],
    "non_foods_clerk": [r"non.?food", r"general merchandise"],
    "dug_shopper": [r"drive up and go", r"dug shopper", r"personal shopper"],
    "sanitation_clerk": [r"sanitation"],
}

# =============================================================================
# FLAG PATTERNS
# =============================================================================

EXCEPTION_PATTERNS = [
    r"\bexcept\b", r"\bunless\b", r"\bnotwithstanding\b",
    r"shall not apply", r"does not apply", r"excluded from"
]

DEFINITION_PATTERNS = [
    r"shall (mean|be defined|have the meaning)",
    r"is defined as", r"means", r"refers to",
    r"the term .* shall", r"for purposes of this"
]

HIRE_DATE_PATTERNS = [
    r"march 27,? 2005", r"march 26,? 2005",
    r"hired (on or )?before", r"hired (on or )?after",
    r"employees hired prior", r"employees hired after"
]

HIGH_STAKES_PATTERNS = [
    r"discharge", r"terminat", r"fired",
    r"disciplin", r"suspend", r"warning",
    r"harass", r"discriminat",
    r"safety", r"injury", r"hazard",
    r"grievance", r"arbitration"
]


def detect_topics(content: str, article_num: int) -> List[str]:
    """Detect topics from content and article number."""
    topics = set()
    
    # Add topics based on article number
    if article_num in ARTICLE_TOPICS:
        topics.update(ARTICLE_TOPICS[article_num])
    
    # Add topics based on content patterns
    content_lower = content.lower()
    for topic, patterns in TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                topics.add(topic)
                break
    
    return list(topics)


def detect_classifications(content: str) -> List[str]:
    """Detect job classifications mentioned in content."""
    classifications = []
    content_lower = content.lower()
    
    for classification, patterns in CLASSIFICATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                classifications.append(classification)
                break
    
    # If no specific classification found, it applies to all
    if not classifications:
        classifications = ["all"]
    
    return classifications


def detect_flags(content: str) -> dict:
    """Detect boolean flags from content."""
    content_lower = content.lower()
    
    return {
        "is_exception": any(re.search(p, content_lower) for p in EXCEPTION_PATTERNS),
        "is_definition": any(re.search(p, content_lower) for p in DEFINITION_PATTERNS),
        "hire_date_sensitive": any(re.search(p, content_lower) for p in HIRE_DATE_PATTERNS),
        "is_high_stakes": any(re.search(p, content_lower) for p in HIGH_STAKES_PATTERNS),
    }


def enrich_chunk(chunk: dict) -> dict:
    """Apply rule-based enrichment to a single chunk."""
    content = chunk.get("content", "")
    article_num = chunk.get("article_num", 0) or 0
    
    # Detect metadata
    topics = detect_topics(content, article_num)
    applies_to = detect_classifications(content)
    flags = detect_flags(content)
    
    # Build enriched chunk
    enriched = chunk.copy()
    enriched["topics"] = topics
    enriched["applies_to"] = applies_to
    enriched.update(flags)
    
    # Add cross-references (basic detection)
    cross_refs = []
    article_refs = re.findall(r"article\s+(\d+)", content.lower())
    for ref in article_refs:
        ref_num = int(ref)
        if ref_num != article_num:  # Don't self-reference
            cross_refs.append(f"art{ref_num}")
    enriched["cross_references"] = list(set(cross_refs))
    
    # Generate summary (just use first 100 chars as placeholder)
    first_sentence = content.split('.')[0][:100].strip()
    if first_sentence:
        enriched["summary"] = first_sentence + "..."
    else:
        enriched["summary"] = None
    
    return enriched


def enrich_all_chunks(input_path: Path, output_path: Path):
    """Enrich all chunks in a file."""
    print(f"Loading chunks from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Enriching {len(chunks)} chunks with rule-based patterns...")
    
    enriched_chunks = []
    topic_counts = {}
    classification_counts = {}
    
    for chunk in chunks:
        enriched = enrich_chunk(chunk)
        enriched_chunks.append(enriched)
        
        # Count topics
        for topic in enriched.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Count classifications
        for cls in enriched.get("applies_to", []):
            classification_counts[cls] = classification_counts.get(cls, 0) + 1
    
    # Save enriched chunks
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(enriched_chunks)} enriched chunks to {output_path}")
    
    # Print stats
    print("\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {topic}: {count}")
    
    print("\nClassification distribution:")
    for cls, count in sorted(classification_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    
    # Count flags
    flags = ["is_exception", "is_definition", "hire_date_sensitive", "is_high_stakes"]
    print("\nFlag counts:")
    for flag in flags:
        count = sum(1 for c in enriched_chunks if c.get(flag))
        print(f"  {flag}: {count}")


if __name__ == "__main__":
    from backend.config import CHUNKS_DIR
    
    input_path = CHUNKS_DIR / "contract_chunks_smart.json"
    output_path = CHUNKS_DIR / "contract_chunks_enriched.json"
    
    enrich_all_chunks(input_path, output_path)

