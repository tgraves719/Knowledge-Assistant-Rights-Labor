"""
Query Expansion Module - Scalable synonym and term expansion for union contracts.

Handles:
- Domain-specific synonyms (terminate/discharge/fire)
- Acronym expansion (OT → overtime)
- Concept mapping (pay → wage, rate, salary, compensation)
- Question reformulation for better semantic matching

Designed for scale across 100+ contracts with similar terminology.
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """Container for original and expanded query terms."""
    original: str
    expanded_terms: List[str]
    synonyms_used: Dict[str, List[str]]
    combined_query: str


# =============================================================================
# UNION CONTRACT DOMAIN SYNONYMS
# =============================================================================
# These mappings work across CBA contracts with standard labor terminology

SYNONYM_GROUPS = {
    # Employment Status
    "termination": [
        "terminate", "terminated", "termination",
        "discharge", "discharged", "dismissal", "dismissed",
        "fire", "fired", "firing", "let go",
        "separation", "separated", "end employment"
    ],
    
    # Compensation
    "wages": [
        "wage", "wages", "pay", "paid", "payment",
        "rate", "rates", "salary", "compensation",
        "hourly", "earnings", "remuneration"
    ],
    
    # Time Off - Breaks
    "breaks": [
        "break", "breaks", "rest period", "relief period",
        "lunch", "lunch break", "meal period", "meal break",
        "rest", "pause", "downtime"
    ],
    
    # Time Off - Leave
    "leave": [
        "leave", "time off", "absence", "vacation",
        "sick leave", "sick day", "personal day",
        "pto", "paid time off", "leave of absence"
    ],
    
    # Discipline
    "discipline": [
        "discipline", "disciplinary", "disciplined",
        "warning", "written warning", "verbal warning",
        "write up", "written up", "corrective action",
        "suspension", "suspended"
    ],
    
    # Grievance
    "grievance": [
        "grievance", "grieve", "complaint", "dispute",
        "arbitration", "arbitrate", "appeal",
        "file", "filing", "protest"
    ],
    
    # Seniority
    "seniority": [
        "seniority", "senior", "tenure", "length of service",
        "years of service", "time in position", "hire date"
    ],
    
    # Scheduling
    "schedule": [
        "schedule", "scheduling", "shift", "shifts",
        "hours", "work hours", "roster", "workweek"
    ],
    
    # Overtime
    "overtime": [
        "overtime", "ot", "over time", "extra hours",
        "time and a half", "1.5x", "double time"
    ],
    
    # Union Rights
    "representation": [
        "representation", "representative", "rep",
        "steward", "union steward", "union rep",
        "weingarten", "union representation"
    ],
    
    # Layoff
    "layoff": [
        "layoff", "lay off", "laid off", "reduction in force",
        "rif", "downsizing", "bumping", "displacement",
        "furlough", "furloughed"
    ],
    
    # Benefits
    "benefits": [
        "benefits", "insurance", "health insurance",
        "medical", "healthcare", "health care",
        "dental", "vision", "coverage"
    ],
    
    # Retroactive
    "retroactive": [
        "retroactive", "retro", "back pay", "backpay",
        "retroactively", "past due", "owed"
    ],
    
    # Just Cause
    "just_cause": [
        "just cause", "good cause", "proper cause",
        "sufficient cause", "for cause", "without cause"
    ],
}

# Reverse lookup: term → group name
TERM_TO_GROUP: Dict[str, str] = {}
for group_name, terms in SYNONYM_GROUPS.items():
    for term in terms:
        TERM_TO_GROUP[term.lower()] = group_name


# =============================================================================
# CONCEPT EXPANSIONS (for semantic search improvement)
# =============================================================================
# Map user intent phrases to contract language

CONCEPT_EXPANSIONS = {
    # Questions about rights
    "what are my rights": [
        "employee rights", "entitled to", "shall have the right"
    ],
    "what should i do": [
        "procedure", "steps", "process", "contact steward"
    ],
    "how long": [
        "days", "period", "time limit", "deadline", "within"
    ],
    "how much": [
        "rate", "amount", "dollar", "per hour", "compensation"
    ],
    "can i": [
        "may", "shall", "entitled", "permitted", "allowed"
    ],
    
    # Specific topics
    "retroactive pay": [
        "retroactive", "back pay", "payment error", "corrected"
    ],
    "break periods": [
        "relief period", "lunch period", "meal period", "rest period"
    ],
    "terminated": [
        "discharge", "dismissal", "separation", "end of employment"
    ],
}


# =============================================================================
# QUERY EXPANSION FUNCTIONS
# =============================================================================

def find_synonyms(term: str) -> List[str]:
    """Find all synonyms for a given term."""
    term_lower = term.lower()
    
    # Check if term is in any synonym group
    if term_lower in TERM_TO_GROUP:
        group_name = TERM_TO_GROUP[term_lower]
        return [t for t in SYNONYM_GROUPS[group_name] if t.lower() != term_lower]
    
    # Check for partial matches
    for group_terms in SYNONYM_GROUPS.values():
        for group_term in group_terms:
            if term_lower in group_term.lower() or group_term.lower() in term_lower:
                return [t for t in group_terms if t.lower() != term_lower]
    
    return []


def expand_query(query: str) -> ExpandedQuery:
    """
    Expand a query with synonyms and related terms.
    
    Returns ExpandedQuery with original, expansions, and combined query.
    """
    query_lower = query.lower()
    expanded_terms = []
    synonyms_used = {}
    
    # 1. Check concept expansions first (phrase-level)
    for phrase, expansions in CONCEPT_EXPANSIONS.items():
        if phrase in query_lower:
            expanded_terms.extend(expansions)
            synonyms_used[phrase] = expansions
    
    # 2. Extract individual words and check synonym groups
    # Skip common words that aren't union-contract-specific
    skip_words = {
        'what', 'where', 'when', 'how', 'why', 'who', 'which',
        'the', 'are', 'is', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'get', 'got', 'for', 'from', 'with', 'about',
        'that', 'this', 'these', 'those', 'there', 'here',
        'and', 'but', 'not', 'you', 'your', 'my', 'me', 'i'
    }
    
    words = re.findall(r'\b\w+\b', query_lower)
    for word in words:
        if len(word) < 3 or word in skip_words:
            continue
        
        synonyms = find_synonyms(word)
        if synonyms:
            # Add top 3 most relevant synonyms
            top_synonyms = synonyms[:3]
            expanded_terms.extend(top_synonyms)
            synonyms_used[word] = top_synonyms
    
    # 3. Check multi-word phrases in query
    for group_name, terms in SYNONYM_GROUPS.items():
        for term in terms:
            if len(term.split()) > 1 and term.lower() in query_lower:
                # Found a multi-word phrase, add other terms from group
                related = [t for t in terms if t.lower() != term.lower()][:3]
                expanded_terms.extend(related)
                synonyms_used[term] = related
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expansions = []
    for term in expanded_terms:
        if term.lower() not in seen and term.lower() not in query_lower:
            seen.add(term.lower())
            unique_expansions.append(term)
    
    # Create combined query for semantic search
    # Add top expansions to the original query
    if unique_expansions:
        expansion_str = " ".join(unique_expansions[:5])
        combined = f"{query} ({expansion_str})"
    else:
        combined = query
    
    return ExpandedQuery(
        original=query,
        expanded_terms=unique_expansions,
        synonyms_used=synonyms_used,
        combined_query=combined
    )


def get_keyword_variants(query: str) -> List[str]:
    """
    Get keyword variants for BM25 search.
    
    Returns list of individual keywords and their synonyms
    for keyword-based search.
    """
    expanded = expand_query(query)
    
    # Start with original query words
    words = re.findall(r'\b\w{3,}\b', query.lower())
    
    # Add expanded terms
    keywords = list(words) + expanded.expanded_terms
    
    # Remove duplicates
    return list(dict.fromkeys(keywords))


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test query expansion."""
    test_queries = [
        "How far back can I get retroactive pay?",
        "What are my break periods?",
        "I was just terminated. What should I do?",
        "What are my Weingarten rights?",
        "How long do I have to file a grievance?",
        "Can I be fired without cause?",
        "What is the overtime rate?",
    ]
    
    print("=" * 70)
    print("QUERY EXPANSION TEST")
    print("=" * 70)
    
    for query in test_queries:
        expanded = expand_query(query)
        print(f"\nOriginal: {query}")
        print(f"Synonyms found: {expanded.synonyms_used}")
        print(f"Expanded terms: {expanded.expanded_terms[:5]}")
        print(f"Combined: {expanded.combined_query[:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()

