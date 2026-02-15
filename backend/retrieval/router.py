"""
Intent Router - Classifies queries and routes to appropriate retrieval strategy.

Routes:
- Wage queries -> Structured JSON lookup
- Contract queries -> Vector search
- High-stakes queries -> Vector search + escalation flag
"""

import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    HIGH_STAKES_TOPICS,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT,
    CAG_ENABLE_HYPOTHESIS_LAYER, CAG_ENABLE_FULL_ARTICLE_EXPANSION,
    CAG_ENABLE_TITLE_BOOSTING, FULL_ARTICLE_MAX_CHUNKS,
    FULL_ARTICLE_MIN_TOP_K_MATCH,
    CAG_ENABLE_QUERY_INTERPRETER, MULTI_QUERY_MAX_SEARCHES,
    MULTI_QUERY_RESULTS_PER_SEARCH, MULTI_QUERY_TOTAL_RESULTS,
    CAG_ENABLE_RERANKER, MANIFESTS_DIR, CONTRACT_ID
)
from backend.retrieval.vector_store import ContractVectorStore
from backend.retrieval.hypothesis import (
    get_hypothesis_generator,
    apply_title_boosting,
    HypothesisResult
)
from backend.retrieval.query_interpreter import (
    get_interpreter,
    QueryInterpretation
)
from backend.chunk_files import resolve_chunk_file
from backend.wage_files import resolve_wage_file
from backend.user.profile import get_classification_options
from backend.language_lexicon_files import resolve_language_lexicon_file
from backend.contracts import resolve_contract_region_id


# ============================================================================
# QUERY EXPANSION - Maps worker slang to contract language
# ============================================================================

# Universal labor language mappings (work for any CBA)
UNIVERSAL_SLANG = {
    # Abbreviations
    "ot": "overtime",
    "pto": "vacation personal holiday time off",
    "fmla": "family medical leave",
    "loa": "leave of absence",

    # Float days / floating holidays -> Personal Holidays
    "float": "personal holiday",
    "float day": "personal holiday",
    "float days": "personal holidays",
    "floater": "personal holiday",
    "floaters": "personal holidays",
    "floating holiday": "personal holiday",

    # Common worker terms
    "fired": "discharge termination",
    "canned": "discharge termination",
    "let go": "discharge termination layoff",
    "pink slip": "discharge termination layoff",
    "written up": "discipline warning",
    "write up": "discipline warning",
    "writeup": "discipline warning",

    # Scheduling
    "my schedule": "work schedule hours",
    "when do i work": "schedule hours",
    "shift change": "schedule change",
    "called in": "call in reporting pay",
    "call out": "call in sick absence",
    "no call no show": "absence discipline",

    # Benefits
    "health insurance": "health benefits health trust",
    "medical": "health benefits",
    "dental": "health benefits",
    "vision": "health benefits",
    "retirement": "pension",

    # Pay-related
    "raise": "wage increase progression",
    "bump": "wage increase step",
    "time and a half": "overtime premium",
    "double time": "overtime premium",
    "sunday pay": "sunday premium",
    "night pay": "night premium",
    "holiday pay": "holiday premium",

    # Leave types
    "bereavement": "funeral leave",
    "maternity": "family care leave",
    "paternity": "family care leave",
    "jury duty": "jury service",
    "sick time": "sick leave",
    "sick days": "sick leave",

    # Roles (universal)
    "cashier": "all purpose clerk",

    # Union stuff
    "steward": "union steward union representative",
    "rep": "union representative steward",
    "dues": "union dues",
    "union meeting": "union business leave",

    # Misc
    "dress code": "uniform appearance dress",
    "uniform": "dress code appearance",
    "break": "rest period",
    "lunch": "meal period",
    "tardiness": "attendance discipline",
    "late": "attendance tardiness",
    "late policy": "attendance discipline",
}


# ============================================================================
# MANIFEST-BASED ROUTING - Per-contract config loaded from JSON
# ============================================================================

@lru_cache(maxsize=16)
def load_manifest_routing(contract_id: str = CONTRACT_ID) -> dict:
    """Load contract-specific routing config from manifest. Cached per contract_id."""
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        return manifest.get("query_routing", {})
    return {}


def ensure_contract_manifest(contract_id: str) -> None:
    """Raise if contract manifest is missing."""
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if not manifest_path.exists():
        raise ValueError(
            f"Unknown contract_id '{contract_id}'. Expected manifest at '{manifest_path}'."
        )


def get_slang_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get merged slang map: universal defaults + contract-specific overrides."""
    routing = load_manifest_routing(contract_id)
    return {
        **UNIVERSAL_SLANG,
        **_load_frozen_language_aliases(contract_id),
        **routing.get("slang_to_contract", {}),
    }


@lru_cache(maxsize=16)
def _load_frozen_language_aliases(contract_id: str = CONTRACT_ID) -> dict:
    """
    Load deterministic alias graph generated during ingestion.

    Returns alias -> canonical phrase mapping for query expansion.
    """
    lex_path = resolve_language_lexicon_file(contract_id=contract_id, allow_shared_fallback=True)
    if not lex_path:
        return {}
    try:
        with open(lex_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    aliases = payload.get("alias_to_canonical", {}) if isinstance(payload, dict) else {}
    if not isinstance(aliases, dict):
        return {}
    normalized: dict[str, str] = {}
    for alias, canonical in aliases.items():
        a = _normalize_query_text(alias)
        c = _normalize_query_text(canonical).replace("_", " ")
        if a and c:
            normalized[a] = c
    return normalized


def get_topic_article_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get topic-to-articles mapping from manifest, with inferred fallback."""
    routing = load_manifest_routing(contract_id)
    explicit = routing.get("topic_to_articles", {}) or {}
    inferred = infer_topic_article_map(contract_id)

    # Merge inferred defaults with explicit manifest mappings.
    merged: dict[str, list[int]] = dict(inferred)
    for topic, article_nums in explicit.items():
        current = merged.get(topic, [])
        merged[topic] = _normalize_article_list(current + list(article_nums or []))
    return merged


def get_topic_patterns(contract_id: str = CONTRACT_ID) -> dict:
    """Get merged topic patterns with additive contract extensions."""
    routing = load_manifest_routing(contract_id)
    contract_patterns = routing.get("topic_patterns", {}) or {}

    merged: dict[str, str] = {}
    for topic, base_pattern in _UNIVERSAL_TOPIC_PATTERNS.items():
        contract_pattern = contract_patterns.get(topic)
        if isinstance(contract_pattern, str) and contract_pattern.strip():
            merged[topic] = f"(?:{base_pattern})|(?:{contract_pattern})"
        else:
            merged[topic] = base_pattern

    for topic, pattern in contract_patterns.items():
        if topic in merged:
            continue
        if isinstance(pattern, str) and pattern.strip():
            merged[topic] = pattern

    return merged


def get_classification_article_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get classification-to-articles mapping from manifest."""
    return load_manifest_routing(contract_id).get("classification_to_articles", {})


def _normalize_article_list(values: list) -> list[int]:
    """Normalize article numbers to sorted unique ints."""
    normalized: list[int] = []
    seen = set()
    for value in values or []:
        try:
            article = int(value)
        except (TypeError, ValueError):
            continue
        if article in seen:
            continue
        seen.add(article)
        normalized.append(article)
    normalized.sort()
    return normalized


_TOPIC_ARTICLE_TITLE_HINTS: dict[str, tuple[str, ...]] = {
    "wages": ("rates of pay", "rate of pay", "wage"),
    "promotion": (
        "rates of pay",
        "prior experience",
        "new employees, transferred employees, promoted or demoted",
        "definitions of classifications",
    ),
    "probation": ("probationary period", "probation"),
    "vacation": ("vacation",),
    "bereavement": ("bereavement leave", "bereavement"),
    "personal_holiday": (
        "personal holiday",
        "holidays",
        "holiday pay",
    ),
    "sick_leave": ("sick leave",),
    "overtime": ("overtime",),
    "scheduling": ("schedule", "workweek", "available hours", "minimum weekly"),
    "seniority": ("seniority",),
    "layoff": ("layoff", "reduction in hours", "reduction of hours"),
    "grievance": ("grievance", "dispute procedure", "arbitration"),
    "discipline": ("discharge", "discrimination", "discipline"),
    "breaks": ("lunch break", "relief period", "meal period", "break"),
    "health_benefits": ("health and welfare", "health benefits", "health benefits plan"),
    "premiums": ("premium", "holiday pay", "night premium", "sunday premium"),
    "term": ("term of agreement",),
}

_TOPIC_LEXICAL_SIGNALS: dict[str, tuple[str, ...]] = {
    # Helps short paraphrases like "float days" surface personal-holiday rules.
    "personal_holiday": ("personal holiday", "personal holidays", "float day", "float days", "floater"),
    "breaks": (
        "rest period",
        "relief period",
        "lunch break",
        "between shifts",
        "minimum hours between",
    ),
    "vacation": (
        "vacation",
        "paid vacation after",
        "years of service",
        "anniversary year",
        "continuous service",
    ),
    "bereavement": ("bereavement leave", "funeral leave", "death in family", "funeral"),
    "sick_leave": ("sick leave", "sick pay"),
    "term": ("term of agreement", "effective date", "expiration", "start and end"),
}


@lru_cache(maxsize=16)
def infer_topic_article_map(contract_id: str = CONTRACT_ID) -> dict:
    """
    Infer topic-to-articles map from manifest article titles.

    This provides a contract-agnostic fallback when manifest query_routing
    has not yet been curated.
    """
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    article_titles = manifest.get("article_titles", {}) or {}
    inferred: dict[str, list[int]] = {topic: [] for topic in _TOPIC_ARTICLE_TITLE_HINTS}

    for article_key, raw_title in article_titles.items():
        try:
            article_num = int(article_key)
        except (TypeError, ValueError):
            continue

        title = str(raw_title or "").lower()
        if not title:
            continue

        for topic, hints in _TOPIC_ARTICLE_TITLE_HINTS.items():
            if any(hint in title for hint in hints):
                inferred[topic].append(article_num)

    return {
        topic: _normalize_article_list(article_nums)
        for topic, article_nums in inferred.items()
        if article_nums
    }

def expand_query(query: str, contract_id: str = CONTRACT_ID) -> Tuple[str, list]:
    """
    Expand query by replacing worker slang with contract terminology.

    Args:
        query: The user's question
        contract_id: Contract ID for loading contract-specific slang

    Returns:
        Tuple of (expanded_query, list of expansions applied)
    """
    slang_map = get_slang_map(contract_id)
    query_lower = query.lower()
    expanded = query
    expansions_applied = []

    # Sort by length (longest first) to avoid partial replacements
    sorted_slang = sorted(slang_map.items(), key=lambda x: len(x[0]), reverse=True)

    for slang, contract_term in sorted_slang:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(slang) + r'\b'
        if re.search(pattern, query_lower):
            # Append contract terms to the query rather than replacing
            # This preserves the original query while adding searchable terms
            if contract_term not in expanded.lower():
                expanded = f"{expanded} ({contract_term})"
                expansions_applied.append(f"{slang} -> {contract_term}")

    # Deterministic phrase detectors for common worker phrasing that may not
    # appear verbatim in contract text.
    pattern_expansions = [
        (
            r"(?:\bcontract\b|\bagreement\b|\bcba\b).*(?:\bstart\b|\bbegin\b|\beffective\b).*(?:\bend\b|\bexpir)"
            r"|(?:\bstart\b|\bbegin\b|\beffective\b).*(?:\bend\b|\bexpir).*(?:\bcontract\b|\bagreement\b|\bcba\b)"
            r"|\bterm\s*of\s*(?:agreement|contract)\b|\beffective\s*date\b|\bexpiration\s*date\b",
            "term of agreement effective date expiration date start date end date",
            "contract term pattern",
        ),
        (
            r"\bclose\b.*\bopen\b|\bopen\b.*\bclose\b",
            "minimum rest period between shifts relief periods lunch breaks",
            "close/open pattern",
        ),
        (
            r"\bminimum\b.*\bhours?\b.*\bbetween\b.*\bshifts?\b"
            r"|\bhours?\b.*\bbetween\b.*\bshifts?\b"
            r"|\bbetween\b.*\bend\b.*\bshift\b.*\bstart\b.*\bnext\b",
            "minimum rest period between shifts relief periods lunch breaks",
            "inter-shift rest pattern",
        ),
        (
            r"\b(funeral|died|death)\b",
            "bereavement leave funeral leave paid days",
            "bereavement pattern",
        ),
        (
            r"\b(store|location)\b.*\b(shut|closing|close)\b",
            "store closing severance pay",
            "store closing pattern",
        ),
    ]
    for pattern, contract_term, label in pattern_expansions:
        if re.search(pattern, query_lower):
            if contract_term not in expanded.lower():
                expanded = f"{expanded} ({contract_term})"
                expansions_applied.append(f"{label} -> {contract_term}")

    return expanded, expansions_applied


@lru_cache(maxsize=16)
def _contract_classification_aliases(contract_id: str = CONTRACT_ID) -> dict[str, str]:
    """Map normalized labels/phrases to canonical contract classification values."""
    options = get_classification_options(contract_id=contract_id) or []
    alias_to_value: dict[str, str] = {}

    for opt in options:
        value = str(opt.get("value") or "").strip().lower()
        label = str(opt.get("label") or "").strip().lower()
        if not value:
            continue

        value_phrase = value.replace("_", " ")
        alias_to_value[value] = value
        alias_to_value[value_phrase] = value
        if label:
            alias_to_value[label] = value

        # Normalize common '&' variants for stable matching.
        if " & " in value_phrase:
            alias_to_value[value_phrase.replace(" & ", " and ")] = value
        if " and " in value_phrase:
            alias_to_value[value_phrase.replace(" and ", " & ")] = value
        if label:
            if " & " in label:
                alias_to_value[label.replace(" & ", " and ")] = value
            if " and " in label:
                alias_to_value[label.replace(" and ", " & ")] = value

    return alias_to_value


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def normalize_classification_for_contract(
    classification: Optional[str],
    contract_id: str = CONTRACT_ID,
) -> Optional[str]:
    """Normalize user/query classification into canonical contract value."""
    if not classification:
        return None
    raw = _normalize_query_text(classification)
    if not raw:
        return None

    aliases = _contract_classification_aliases(contract_id)
    if raw in aliases:
        return aliases[raw]

    snake = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if snake in aliases:
        return aliases[snake]

    if raw.replace(" ", "_") in aliases:
        return aliases[raw.replace(" ", "_")]
    return snake or raw


@dataclass
class QueryIntent:
    """Classified intent of a user query."""
    intent_type: str  # 'wage', 'contract', 'high_stakes'
    confidence: float
    classification: Optional[str]  # For wage queries
    topic: Optional[str]
    requires_escalation: bool
    keywords_matched: list
    high_stakes_topic: bool = False
    active_urgent_context: bool = False
    escalation_policy: str = "deterministic_v1"
    relevant_articles: list = None  # Articles relevant to detected topic
    
    def __post_init__(self):
        if self.relevant_articles is None:
            self.relevant_articles = []


# Wage-related keywords (specific phrases to avoid false positives)
WAGE_KEYWORDS = [
    "my pay", "my wage", "my rate", "my salary", "my hourly",
    "wage rate", "pay rate", "hourly rate", "rate of pay",
    "what do i make", "what's my pay", "what am i making",
    "how much do i make", "how much should i make", "how much will i make",
    "how much should i be making", "what should i be making", "what should i make",
    "compensation", "starting pay", "experience pay", "step", "progression",
    "appendix a"
]

# Exclude these from wage detection (contain "pay" but aren't wage queries)
WAGE_EXCLUDE_PATTERNS = [
    "vacation", "holiday", "sick", "time off", "pto", "personal day",
    "pay stub", "pay period", "pay check"
]

# Classification extraction patterns
CLASSIFICATION_PATTERNS = {
    "courtesy_clerk": r"courtesy\s*clerk|bagger",
    "head_clerk": r"head\s*clerk",
    "all_purpose_clerk": r"all\s*purpose\s*clerk|clerk",
    "produce_manager": r"produce\s*(department)?\s*manager",
    "bakery_manager": r"bakery\s*manager",
    "cake_decorator": r"cake\s*decorator",
    "pharmacy_tech": r"pharmacy\s*tech",
    "non_foods_clerk": r"non.?food|gm\s*clerk|general\s*merchandise",
}

CONTRACT_TERM_CUE_PATTERN = (
    r"term\s*of|contract\s*term|agreement\s*term|term\s*of\s*(agreement|contract)|"
    r"expir|effective\s*date|"
    r"(?:contract|agreement|cba).*(?:start|begin|effective).*(?:end|expir)|"
    r"(?:start|begin|effective).*(?:end|expir).*(?:contract|agreement|cba)"
)

INTER_SHIFT_REST_CUE_PATTERN = (
    r"minimum\s*hours?\s*between\s*.*shifts?|"
    r"hours?\s*between\s*.*shifts?|"
    r"between\s*the\s*end\s*of\s*.*shift\s*and\s*the\s*start\s*of\s*.*next"
)

# Universal topic patterns for routing (language patterns, not article numbers)
_UNIVERSAL_TOPIC_PATTERNS = {
    "wages": r"wage|wages|rate\s*of\s*pay|rates\s*of\s*pay|pay\s*rate|hourly\s*rate|how much.*make|what do i make",
    "overtime": r"overtime|over\s*time|ot|time\s*and\s*a\s*half",
    "scheduling": r"schedul|shift|hours|when do i work",
    "seniority": r"seniority|senior|how long|years of service",
    "layoff": r"layoff|lay\s*off|bumping|displacement|reduction",
    "personal_holiday": r"personal\s*holiday|float\s*(day|days)?|floater|pto",
    "vacation": r"vacation|time\s*off|holiday|personal day",
    "bereavement": r"bereavement|funeral|death\s*in\s*the\s*family|died",
    "sick_leave": r"sick\s*leave|sick\s*day|illness|call\s*in\s*sick",
    "discipline": r"disciplin|warning|write\s*up|written up|tardiness|tardy|late|attendance",
    "grievance": r"grievance|arbitration|file\s*a\s*complaint",
    "breaks": rf"break|lunch|meal\s*period|relief|rest\s*period|{INTER_SHIFT_REST_CUE_PATTERN}",
    "premiums": r"premium|night\s*pay|sunday\s*pay|sunday\s*premium",
    "weingarten": r"weingarten|right\s*to\s*representation|union\s*rep",
    "health_benefits": r"health\s*(benefit|insurance|coverage|care)|medical\s*benefit|eligible.*(health|benefit)|benefit.*eligible",
    "promotion": r"promot|advance|move up|basket.*hours|credit.*hours",
    "drive_up_go": r"drive\s*up|dug|personal\s*shopper|clicklist",
    "probation": r"probation|probationary|trial\s*period|new\s*employee.*hours",
    "term": CONTRACT_TERM_CUE_PATTERN,
    "minimum_wage": r"minimum\s*wage|colorado.*wage|\$15",
    "joint_committee": r"joint.*committee|labor.*management\s*committee",
}

VACATION_ENTITLEMENT_QUERY_PATTERN = (
    r"(how\s+much|how\s+many).*\bvacation\b"
    r"|\bvacation\b.*(per\s+year|entitlement|accrual|do\s+i\s+get)"
    r"|\bweeks?\s+of\s+vacation\b"
)


# NOTE: TOPIC_ARTICLE_MAP and CLASSIFICATION_ARTICLE_MAP are now loaded from
# the contract manifest via get_topic_article_map() and get_classification_article_map().
# Article numbers are always contract-specific and should not be hardcoded.


def extract_classification(query: str) -> Optional[str]:
    """Extract job classification from query."""
    query_lower = _normalize_query_text(query)
    for class_name, pattern in CLASSIFICATION_PATTERNS.items():
        if re.search(pattern, query_lower):
            return class_name
    return None


def extract_classification_for_contract(query: str, contract_id: str = CONTRACT_ID) -> Optional[str]:
    """
    Extract contract-scoped classification from query.

    Preference order:
    1) Contract classification aliases (values + labels)
    2) Directional "to <classification>" target on promotion-style wording
    3) Longest phrase match
    4) Legacy static regex fallback
    """
    query_lower = _normalize_query_text(query)
    aliases = _contract_classification_aliases(contract_id)
    matches: list[tuple[str, str, int]] = []  # (phrase, value, start_idx)

    for phrase, value in aliases.items():
        if not phrase or len(phrase) < 3:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        m = re.search(pattern, query_lower)
        if not m:
            continue
        matches.append((phrase, value, m.start()))

    if matches:
        # Promotion/demotion style: prefer destination classification after "to".
        if re.search(r"\b(promot|promotion|transfer|move)\b", query_lower):
            directional = sorted(
                (
                    t for t in matches
                    if re.search(
                        rf"\bto\s+{re.escape(t[0])}\b",
                        query_lower,
                    )
                ),
                key=lambda t: len(t[0]),
                reverse=True,
            )
            if directional:
                return directional[0][1]

        # Default: longest match first, then earliest mention.
        matches.sort(key=lambda t: (-len(t[0]), t[2]))
        return matches[0][1]

    legacy = extract_classification(query_lower)
    if legacy:
        return normalize_classification_for_contract(legacy, contract_id=contract_id)
    return None


def extract_topic(query: str, contract_id: str = CONTRACT_ID) -> Optional[str]:
    """
    Extract main topic from query.

    Uses priority ordering to prefer more specific topics over generic ones.
    Merges universal patterns with contract-specific patterns from manifest.
    """
    topic_patterns = get_topic_patterns(contract_id)
    query_lower = _normalize_query_text(query)

    # Priority order: specific topics first, generic topics last
    TOPIC_PRIORITY = [
        "retirement_savings",
        "weingarten",
        "health_benefits",
        "drive_up_go",
        "probation",
        "promotion",
        "term",
        "personal_holiday",  # Check before vacation since it's more specific
        "bereavement",
        "layoff",
        "sick_leave",
        "vacation",
        "overtime",
        "grievance",
        "discipline",
        "seniority",
        "premiums",
        "breaks",
        "scheduling",  # Generic - matches "hours" so put last
    ]

    # First pass: check topics in priority order
    for topic in TOPIC_PRIORITY:
        if topic in topic_patterns:
            pattern = topic_patterns[topic]
            if re.search(pattern, query_lower):
                return topic

    # Second pass: check any remaining topics
    for topic, pattern in topic_patterns.items():
        if topic not in TOPIC_PRIORITY:
            if re.search(pattern, query_lower):
                return topic

    return None


def is_wage_query(query: str) -> tuple[bool, list]:
    """Check if query is asking about wages/pay."""
    query_lower = query.lower()

    # First check if this is actually about time off/benefits (not wages)
    for exclude in WAGE_EXCLUDE_PATTERNS:
        if exclude in query_lower:
            return False, []

    matched = []
    for keyword in WAGE_KEYWORDS:
        if keyword in query_lower:
            matched.append(keyword)

    # Also check for specific patterns
    wage_patterns = [
        r"how much (do|does|will|would|should) .+ (make|earn|get paid|be making|be earning)",
        r"what (is|are|should) (my|the) (pay|wage|rate)",
        r"what (is|are|'s) the .+ rate of pay",
        r"what should i (make|be making|earn|be earning)",
        r"\$\d+.*hour",  # Dollar amounts with hour
    ]

    for pattern in wage_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"pattern:{pattern}")

    return len(matched) > 0, matched


def is_contextual_wage_query(
    query: str,
    classification: Optional[str],
    topic: Optional[str],
) -> tuple[bool, list]:
    """
    Deterministic wage-intent fallback for progression/promotions.

    Captures queries that omit first-person wage phrases but clearly ask for
    rate/step outcomes (for example: promotion basket-hours calculations).
    """
    if not classification:
        return False, []

    q = (query or "").lower()
    if any(ex in q for ex in WAGE_EXCLUDE_PATTERNS):
        return False, []

    wage_signal = bool(
        re.search(r"\b(rate|wage|pay|paid|progression|step|increase|hourly)\b", q)
    )
    progression_signal = bool(
        re.search(r"\b\d[\d,]*\s*hours?\b", q)
        or "basket hours" in q
        or topic in {"promotion", "wages"}
    )
    if wage_signal and progression_signal:
        return True, ["contextual_wage_progression"]
    return False, []


CONDITIONAL_SUPPRESSOR_PATTERNS = [
    r"\bwhat are my rights if\b",
    r"\bhypothetical(ly)?\b",
    r"\bin case\b",
    r"\bif i were\b",
    r"\bif this happens\b",
    r"\bwhat if\b",
    r"\bif someone\b",
    r"\bcould i\b",
    r"\bwould i\b",
]


def classify_high_stakes_context(query: str) -> tuple[bool, bool, list]:
    """
    Deterministic two-stage high-stakes classifier.

    Returns:
        tuple: (high_stakes_topic, active_urgent_context, matched_patterns)
        - high_stakes_topic: informational/legal-risk topic detected
        - active_urgent_context: current live incident detected (escalation trigger)
        - matched_patterns: audit trail of matched rules
    """
    query_lower = query.lower()
    matched = []
    high_stakes_topic = False
    active_urgent_context = False

    # Stage 1: high-stakes topic detection (informational or active)
    topic_patterns = [
        r"\bdisciplin(e|ary|ed|ing)?\b",
        r"\bterminat(e|ed|ion|ing)\b",
        r"\bdischarg(e|ed)\b",
        r"\bfired\b",
        r"\bharass(ed|ment|ing)?\b",
        r"\bdiscriminat(ed|e|ion|ing)\b",
        r"\bretaliat(es|ed|e|ion|ing)\b",
        r"\bweingarten\b",
        r"\binvestigation(s)?\b",
        r"\bsuspend(ed|s|ing)?\b|\bsuspension\b",
        r"\bwritten up\b|\bwrite up\b",
        r"\bunsafe\b|\bdanger(ous)?\b|\binjur(y|ed)\b",
    ]

    for pattern in topic_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"topic:{pattern}")
            high_stakes_topic = True

    for topic in HIGH_STAKES_TOPICS:
        topic_norm = str(topic or "").strip().lower()
        if not topic_norm:
            continue
        # Use token-aware matching to avoid false positives like "hired" -> "fired".
        patt = rf"(?<![a-z0-9]){re.escape(topic_norm)}(?![a-z0-9])"
        if re.search(patt, query_lower):
            matched.append(f"topic_keyword:{topic_norm}")
            high_stakes_topic = True

    # Stage 2: active/urgent context detection (actual escalation trigger)
    active_patterns = [
        r"(i'?m|i am|was|been|being|getting) (just\s+)?(fired|terminated|discharged)",
        r"(i'?m|i am|was|been|being|getting) (disciplined|written up|suspended)",
        r"(i|i am|i was|i got|i have been|i've been).*(written up|suspended)",
        r"(my\s+)?(manager|boss|supervisor).*(wrote me up|disciplined me|suspended me)",
        r"(i'?m|i am) being (harass|discriminat)",
        r"(i'?m|i am|i was) (harassed|discriminated against|retaliated against)",
        r"(harassing|discriminating\s+against|retaliating\s+against)\s+me",
        r"(my\s+)?(manager|boss|supervisor|coworker).*(harass|discriminat|retaliat)",
        r"(i'?m|i am|i was).*(called|summoned).*(disciplinary\s+)?(meeting|office)",
        r"(called|summoned) (into|to) (a\s+)?(meeting|office)",
        r"just (got|been|was) (terminated|fired|discharged|written up|suspended)",
        r"(manager|boss|supervisor).*(wants|asked|told|called).*(meeting|office|talk)",
        r"(i'?m|i am|i got|i was) injured",
        r"(right now|today|yesterday|just happened).*(fired|terminated|discharged|written up|suspended|harass|discriminat|retaliat|injur|unsafe)",
        r"(fired|terminated|discharged|written up|suspended|harass|discriminat|retaliat|injur|unsafe).*(right now|today|yesterday|just happened)",
    ]
    for pattern in active_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"active:{pattern}")
            active_urgent_context = True

    if active_urgent_context:
        # Active incidents are always a subset of high-stakes topics.
        high_stakes_topic = True

    # Suppress escalation for conditional/hypothetical language
    suppressor_hit = False
    for pattern in CONDITIONAL_SUPPRESSOR_PATTERNS:
        if re.search(pattern, query_lower):
            matched.append(f"suppressor:{pattern}")
            suppressor_hit = True

    if suppressor_hit and active_urgent_context:
        # Explicit hypothetical framing should not trigger active escalation.
        active_urgent_context = False
        matched.append("suppressor:active_context_reset")

    return high_stakes_topic, active_urgent_context, matched


def is_high_stakes(query: str) -> tuple[bool, list, bool]:
    """Backward-compatible wrapper for legacy callers."""
    high_stakes_topic, active_urgent_context, matched = classify_high_stakes_context(query)
    return high_stakes_topic, matched, active_urgent_context


def classify_intent(query: str, user_classification: str = None, contract_id: str = CONTRACT_ID) -> QueryIntent:
    """
    Classify the intent of a user query.

    Args:
        query: The user's question
        user_classification: Optional classification from user profile (e.g., from dropdown)
        contract_id: Contract ID for loading contract-specific routing config

    Returns:
        QueryIntent with type, confidence, and metadata
    """
    ensure_contract_manifest(contract_id)

    # Use provided classification or try to extract from query
    classification = normalize_classification_for_contract(user_classification, contract_id=contract_id)
    if not classification:
        classification = extract_classification_for_contract(query, contract_id=contract_id)
    topic = extract_topic(query, contract_id)

    # Get relevant articles from manifest (contract-specific)
    topic_article_map = get_topic_article_map(contract_id)
    relevant_articles = topic_article_map.get(topic, []) if topic else []

    # Deterministic explicit article anchoring (works even without query interpreter).
    explicit_article_refs = []
    for match in re.findall(r"\b(?:article|art\.?)\s*(\d+)\b", query.lower()):
        try:
            explicit_article_refs.append(int(match))
        except ValueError:
            continue
    if explicit_article_refs:
        relevant_articles = list(set(relevant_articles + explicit_article_refs))

    # Add classification-specific articles for role-based boosting
    if classification:
        classification_article_map = get_classification_article_map(contract_id)
        classification_articles = classification_article_map.get(classification, [])
        relevant_articles = list(set(relevant_articles + classification_articles))
    
    is_wage, wage_matches = is_wage_query(query)
    if not is_wage:
        contextual_wage, contextual_matches = is_contextual_wage_query(
            query=query,
            classification=classification,
            topic=topic,
        )
        if contextual_wage:
            is_wage = True
            wage_matches = wage_matches + contextual_matches
    high_stakes_topic, active_urgent_context, hs_matches = classify_high_stakes_context(query)

    # Determine primary intent
    if high_stakes_topic:
        # Add discipline/grievance articles for high-stakes
        relevant_articles = list(set(relevant_articles + [43, 45, 46]))
        return QueryIntent(
            intent_type="high_stakes",
            confidence=0.9 if len(hs_matches) > 1 else 0.7,
            classification=classification,
            topic=topic,
            # Only escalate (show steward contact UI) for ACTIVE situations
            requires_escalation=active_urgent_context,
            keywords_matched=hs_matches,
            high_stakes_topic=high_stakes_topic,
            active_urgent_context=active_urgent_context,
            escalation_policy="deterministic_v2",
            relevant_articles=relevant_articles
        )
    
    if is_wage:
        # Check if we have enough info for a wage lookup
        confidence = 0.8 if classification else 0.6
        relevant_articles = list(set(relevant_articles + [8, 9]))  # Wages articles
        return QueryIntent(
            intent_type="wage",
            confidence=confidence,
            classification=classification,
            topic="wages",
            requires_escalation=False,
            keywords_matched=wage_matches,
            high_stakes_topic=False,
            active_urgent_context=False,
            escalation_policy="deterministic_v2",
            relevant_articles=relevant_articles
        )
    
    # Default to contract query
    return QueryIntent(
        intent_type="contract",
        confidence=0.7,
        classification=classification,
        topic=topic,
        requires_escalation=False,
        keywords_matched=[],
        high_stakes_topic=False,
        active_urgent_context=False,
        escalation_policy="deterministic_v2",
        relevant_articles=relevant_articles
    )


class HybridRetriever:
    """
    Combines hybrid search (vector + BM25) with structured wage lookups.
    
    Uses Reciprocal Rank Fusion to combine semantic and keyword search
    for better retrieval across union contract terminology.
    """
    
    def __init__(self, vector_store: ContractVectorStore = None):
        """Initialize the hybrid retriever."""
        self.vector_store = vector_store
        self.hybrid_searcher = None
        self.wages_data = None
        self._wages_by_contract = {}
        self._all_chunks_by_contract = {}
        self._load_wages(CONTRACT_ID)
    
    def _load_wages(self, contract_id: str = CONTRACT_ID):
        """Load wage data for a contract from JSON."""
        if contract_id in self._wages_by_contract:
            self.wages_data = self._wages_by_contract[contract_id]
            return

        wages_file = resolve_wage_file(contract_id=contract_id, allow_shared_fallback=True)
        if wages_file and wages_file.exists():
            with open(wages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._wages_by_contract[contract_id] = data
            self.wages_data = data
            return

        self._wages_by_contract[contract_id] = None
        self.wages_data = None

    def _allow_legacy_unscoped_chunks(self) -> bool:
        """
        Allow chunks missing contract_id only in single-manifest mode.

        This preserves backward compatibility for older corpora while preventing
        cross-tenant leakage once multiple manifests are present.
        """
        return len(list(MANIFESTS_DIR.glob("*.json"))) == 1
    
    def _ensure_hybrid_searcher(self):
        """Lazy-initialize the hybrid searcher."""
        if self.hybrid_searcher is None:
            # Create vector store only when vector search is enabled.
            if self.vector_store is None and HYBRID_VECTOR_WEIGHT > 0:
                self.vector_store = ContractVectorStore()
            from backend.retrieval.hybrid_search import HybridSearcher
            self.hybrid_searcher = HybridSearcher(vector_store=self.vector_store)
    
    def lookup_wage(
        self, 
        classification: str,
        hours_worked: int = 0,
        months_employed: int = 0,
        effective_date: str = None,
        contract_id: str = CONTRACT_ID,
    ) -> Optional[dict]:
        """Look up wage from structured data."""
        self._load_wages(contract_id=contract_id)
        if not self.wages_data:
            return None
        
        # Import here to avoid circular dependency
        from backend.ingest.extract_wages import lookup_wage
        return lookup_wage(self.wages_data, classification, hours_worked, months_employed, effective_date)
    
    def _load_all_chunks_for_contract(self, contract_id: str) -> list:
        """Load and cache chunks scoped to a specific contract."""
        if contract_id in self._all_chunks_by_contract:
            return self._all_chunks_by_contract[contract_id]

        chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
        all_chunks = []
        if chunks_file and chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)

        allow_unscoped = self._allow_legacy_unscoped_chunks()
        required_region = resolve_contract_region_id(contract_id)
        filtered_chunks = []
        for c in all_chunks:
            chunk_contract_id = c.get("contract_id")
            chunk_region = c.get("region_id") or resolve_contract_region_id(str(chunk_contract_id or contract_id))
            if chunk_contract_id == contract_id and str(chunk_region) == str(required_region):
                c_copy = dict(c)
                c_copy["region_id"] = required_region
                filtered_chunks.append(c_copy)
            elif allow_unscoped and chunk_contract_id in (None, ""):
                c_copy = dict(c)
                c_copy["contract_id"] = contract_id
                c_copy["region_id"] = required_region
                filtered_chunks.append(c_copy)
        self._all_chunks_by_contract[contract_id] = filtered_chunks
        return filtered_chunks

    def _expand_with_related_sections(
        self,
        chunks: list,
        contract_id: str = CONTRACT_ID,
        max_total: int = 8,
        preferred_articles: Optional[list[int]] = None,
    ) -> list:
        """
        Expand retrieved chunks with related sections from the same articles.
        
        This enables cross-section synthesis by ensuring that if we retrieve
        Section 49, we also include nearby sections like 44, 45, 46 that may
        contain definitions or related provisions.
        """
        if not chunks:
            return chunks
        
        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        
        # Collect per-article retrieval anchors and cross-referenced sections.
        retrieved_articles = set()
        article_first_rank: dict[int, int] = {}
        retrieved_ids = set()
        retrieved_sections_by_article: dict[int, set[int]] = {}
        referenced_sections_by_article: dict[int, set[int]] = {}
        for idx, chunk in enumerate(chunks):
            article_num = chunk.get('article_num')
            if article_num:
                retrieved_articles.add(article_num)
                try:
                    article_int = int(article_num)
                except (TypeError, ValueError):
                    article_int = None
                if article_int is not None:
                    if article_int not in article_first_rank:
                        article_first_rank[article_int] = idx
                    section_num = chunk.get("section_num")
                    try:
                        sec_int = int(section_num) if section_num is not None else None
                    except (TypeError, ValueError):
                        sec_int = None
                    if sec_int is not None:
                        retrieved_sections_by_article.setdefault(article_int, set()).add(sec_int)

                    content_text = str(
                        chunk.get("content_with_tables")
                        or chunk.get("content")
                        or ""
                    )
                    refs = {
                        int(m.group(1))
                        for m in re.finditer(r'\bSection\s+(\d+)\b', content_text, flags=re.IGNORECASE)
                    }
                    if refs:
                        referenced_sections_by_article.setdefault(article_int, set()).update(refs)
            retrieved_ids.add(chunk.get('chunk_id', chunk.get('citation', '')))
        
        preferred_set = set(_normalize_article_list(preferred_articles or []))

        # Find related sections from same articles (priority order)
        related_chunks = []
        ordered_articles = sorted(
            retrieved_articles,
            key=lambda a: (
                0 if (
                    (lambda ai: ai in preferred_set if ai is not None else False)(
                        (int(a) if str(a).isdigit() else None)
                    )
                ) else 1,
                article_first_rank.get(
                    (int(a) if str(a).isdigit() else -1),
                    9999,
                ),
            ),
        )
        for article_num in ordered_articles:
            try:
                article_int = int(article_num)
            except (TypeError, ValueError):
                article_int = None
            # Get all sections from this article
            article_chunks = []
            for c in contract_chunks:
                c_article = c.get("article_num")
                if article_int is not None:
                    try:
                        c_article_int = int(c_article) if c_article is not None else None
                    except (TypeError, ValueError):
                        c_article_int = None
                    same_article = c_article_int == article_int
                else:
                    same_article = c_article == article_num
                if not same_article:
                    continue
                if c.get('chunk_id', c.get('citation', '')) in retrieved_ids:
                    continue
                article_chunks.append(c)
            
            # Add up to 2 related sections per article, prioritizing:
            # 1) Sections explicitly referenced by retrieved chunks
            # 2) Adjacent sections to retrieved section numbers
            # 3) Lowest section numbers as fallback
            ref_secs = referenced_sections_by_article.get(article_int, set()) if article_int is not None else set()
            base_secs = retrieved_sections_by_article.get(article_int, set()) if article_int is not None else set()

            def _related_priority(c: dict) -> tuple:
                raw = c.get("section_num")
                try:
                    sec = int(raw) if raw is not None else None
                except (TypeError, ValueError):
                    sec = None
                if sec is None:
                    return (3, 9999, 9999)

                if sec in ref_secs:
                    return (0, 0, sec)
                if base_secs:
                    distance = min(abs(sec - b) for b in base_secs)
                    if distance <= 1:
                        return (1, distance, sec)
                    return (2, distance, sec)
                return (2, 999, sec)

            article_chunks.sort(key=_related_priority)
            chosen_sections: set[int] = set()
            selected_for_article = 0
            for c in article_chunks:
                if len(chunks) + len(related_chunks) >= max_total:
                    break
                if selected_for_article >= 2:
                    break

                raw_sec = c.get("section_num")
                try:
                    sec = int(raw_sec) if raw_sec is not None else None
                except (TypeError, ValueError):
                    sec = None
                if sec is not None and sec in chosen_sections:
                    continue

                # Mark as related context
                c_copy = dict(c)
                c_copy['similarity'] = 0.5  # Lower score to indicate it's supplemental
                c_copy['is_related'] = True
                related_chunks.append(c_copy)
                selected_for_article += 1
                if sec is not None:
                    chosen_sections.add(sec)
        
        # Combine: original chunks first, then related
        return chunks + related_chunks

    def _ensure_topic_article_coverage(
        self,
        chunks: list,
        article_numbers: list[int] | None,
        contract_id: str = CONTRACT_ID,
        max_additional: int = 3,
        query_text: Optional[str] = None,
    ) -> list:
        """
        Ensure at least one chunk from topic-relevant articles is present.

        This stabilizes retrieval when lexical matches are noisy (for example,
        "vacation pay" can over-match generic "pay" language).
        """
        if not chunks or not article_numbers or max_additional <= 0:
            return chunks

        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        if not contract_chunks:
            return chunks

        existing_ids = {c.get("chunk_id", c.get("citation", "")) for c in chunks}
        existing_articles = {
            int(c.get("article_num")) for c in chunks if isinstance(c.get("article_num"), int)
        }

        additions: list[dict] = []
        query_tokens = [
            t for t in re.findall(r"[a-z0-9]+", (query_text or "").lower())
            if len(t) >= 3
        ]

        def _candidate_score(c: dict) -> tuple[int, int, str]:
            section_num = c.get("section_num") or 0
            subsection = str(c.get("subsection") or "")
            text = (
                f"{c.get('citation', '')} "
                f"{c.get('article_title', '')} "
                f"{(c.get('content', '') or '')[:600]}"
            ).lower()
            overlap = 0
            if query_tokens:
                overlap = sum(1 for tok in query_tokens if tok in text)
            # Higher overlap first, then lower section number for determinism.
            return (overlap, -int(section_num), subsection)

        for article_num in _normalize_article_list(article_numbers):
            if len(additions) >= max_additional:
                break
            if article_num in existing_articles:
                continue

            candidates = [
                c for c in contract_chunks
                if c.get("article_num") == article_num
                and c.get("chunk_id", c.get("citation", "")) not in existing_ids
            ]
            if not candidates:
                continue

            candidates.sort(key=_candidate_score, reverse=True)
            seed = dict(candidates[0])
            seed["similarity"] = max(0.42, float(seed.get("similarity", 0) or 0))
            seed["is_topic_seed"] = True
            additions.append(seed)
            existing_ids.add(seed.get("chunk_id", seed.get("citation", "")))

        return chunks + additions

    def _prioritize_topic_articles(
        self,
        chunks: list,
        article_numbers: list[int] | None,
        topic: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> list:
        """Reorder chunks so topic-relevant articles appear first."""
        if not chunks or not article_numbers:
            return chunks
        preferred = set(_normalize_article_list(article_numbers))
        if not preferred:
            return chunks
        signals = _TOPIC_LEXICAL_SIGNALS.get(str(topic or "").strip().lower(), ())
        query_lower = (query_text or "").lower()
        # Only apply lexical signal boosts when the query itself indicates the topic.
        apply_signal_boost = bool(signals and any(s in query_lower for s in signals))

        def _sort_key(c: dict) -> tuple:
            in_topic = 0 if c.get("article_num") in preferred else 1
            base_score = float(c.get("similarity", 0) or 0)
            bonus = 0.0
            if apply_signal_boost:
                text = (
                    f"{c.get('citation', '')} "
                    f"{(c.get('content', '') or '')[:800]}"
                ).lower()
                hits = sum(1 for s in signals if s in text)
                bonus = min(0.18, 0.06 * hits)
            eff = base_score + bonus
            return (
                in_topic,
                -eff,
                int(c.get("section_num") or 0),
                str(c.get("subsection") or ""),
            )

        return sorted(chunks, key=_sort_key)

    def _ensure_wage_table_context(
        self,
        chunks: list,
        classification: Optional[str],
        contract_id: str = CONTRACT_ID,
        max_additional: int = 2,
    ) -> list:
        """
        Ensure wage answers include table-backed appendix evidence when available.

        This is deterministic and contract-scoped: it only pulls from the active
        contract chunk set and never uses cross-contract fallback.
        """
        if not chunks or not classification or max_additional <= 0:
            return chunks

        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        if not contract_chunks:
            return chunks

        existing_ids = {c.get("chunk_id", c.get("citation", "")) for c in chunks}
        norm_class = re.sub(r"[^a-z0-9]+", "_", str(classification).lower()).strip("_")
        class_tokens = [t for t in norm_class.split("_") if len(t) > 2]

        scored: list[tuple[float, dict]] = []
        for chunk in contract_chunks:
            chunk_id = chunk.get("chunk_id", chunk.get("citation", ""))
            if chunk_id in existing_ids:
                continue
            if not chunk.get("table_refs"):
                continue

            citation = str(chunk.get("citation") or "")
            doc_type = str(chunk.get("doc_type") or "").lower()
            text = (chunk.get("content_with_tables") or chunk.get("content") or "").lower()

            # Require strong appendix/table signal.
            if doc_type != "appendix" and "appendix" not in citation.lower() and "table" not in citation.lower():
                continue

            token_hits = sum(1 for token in class_tokens if token in text)
            has_wage_grid_signal = "classification" in text and "$" in text
            if token_hits == 0 and not has_wage_grid_signal:
                continue

            score = 0.0
            if token_hits:
                score += 2.0 + min(1.0, token_hits * 0.25)
            if "appendix" in citation.lower() or doc_type == "appendix":
                score += 1.5
            if "$" in text or "effective" in text:
                score += 1.0
            if "classification" in text:
                score += 0.5

            if score <= 0:
                continue
            scored.append((score, chunk))

        if not scored:
            return chunks

        scored.sort(key=lambda x: x[0], reverse=True)
        additions: list[dict] = []
        for _, chunk in scored[:max_additional]:
            c = dict(chunk)
            c["similarity"] = max(float(c.get("similarity", 0) or 0), 0.46)
            c["is_wage_table_context"] = True
            additions.append(c)

        return chunks + additions

    @staticmethod
    def _is_vacation_entitlement_query(query_text: Optional[str]) -> bool:
        """Detect vacation-amount questions that need entitlement schedule coverage."""
        q = _normalize_query_text(query_text or "")
        if not q:
            return False
        return bool(re.search(VACATION_ENTITLEMENT_QUERY_PATTERN, q))

    def _ensure_vacation_entitlement_coverage(
        self,
        chunks: list,
        contract_id: str = CONTRACT_ID,
        preferred_articles: Optional[list[int]] = None,
        topic: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> list:
        """
        Ensure vacation entitlement queries include accrual schedule language.

        Some vacation prompts (for example "How much vacation do I get per year?")
        can over-retrieve scheduling/payment sections while missing entitlement
        schedule text. This adds one deterministic entitlement seed chunk from
        preferred vacation articles when absent.
        """
        if not chunks:
            return chunks
        if str(topic or "").strip().lower() != "vacation":
            return chunks
        if not self._is_vacation_entitlement_query(query_text):
            return chunks

        preferred = _normalize_article_list(preferred_articles or [])
        if not preferred:
            return chunks

        entitlement_markers = (
            "paid vacation after",
            "anniversary year",
            "continuous service",
            "years service",
        )

        def _norm_text(chunk: dict) -> str:
            text = (
                f"{chunk.get('citation', '')} "
                f"{chunk.get('article_title', '')} "
                f"{chunk.get('content_with_tables') or chunk.get('content') or ''}"
            )
            # punctuation-insensitive lexical checks
            return re.sub(r"[^a-z0-9]+", " ", text.lower())

        # If an entitlement-like chunk is already present in preferred articles,
        # keep existing ordering.
        for c in chunks:
            article_num = c.get("article_num")
            if article_num not in preferred:
                continue
            text = _norm_text(c)
            hits = sum(1 for marker in entitlement_markers if marker in text)
            if "paid vacation after" in text or hits >= 2:
                return chunks

        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        if not contract_chunks:
            return chunks

        existing_ids = {c.get("chunk_id", c.get("citation", "")) for c in chunks}
        candidates: list[tuple[float, dict]] = []
        for c in contract_chunks:
            if c.get("article_num") not in preferred:
                continue
            cid = c.get("chunk_id", c.get("citation", ""))
            if cid in existing_ids:
                continue
            text = _norm_text(c)
            hits = sum(1 for marker in entitlement_markers if marker in text)
            if hits == 0:
                continue
            if ("paid vacation after" not in text) and hits < 2:
                continue
            section_num = int(c.get("section_num") or 0)
            score = float(hits) + (0.5 if section_num <= 45 else 0.0)
            candidates.append((score, c))

        if not candidates:
            return chunks

        candidates.sort(key=lambda x: (x[0], -int(x[1].get("section_num") or 0)), reverse=True)
        seed = dict(candidates[0][1])
        seed["similarity"] = max(0.53, float(seed.get("similarity", 0) or 0))
        seed["is_topic_entitlement_seed"] = True
        return chunks + [seed]

    def _expand_to_full_article(
        self,
        chunks: list,
        contract_id: str = CONTRACT_ID,
        n_results: int = 5,
        preferred_articles: Optional[list[int]] = None,
    ) -> list:
        """
        Expand retrieval to include ALL chunks from the "winning" article.

        The "winning" article is defined as the article that appears most
        frequently in the top results. This ensures the LLM gets complete
        context for synthesizing an accurate answer.

        This is the "Breadcrumb Retrieval" technique: follow the trail
        to the full source.

        Args:
            chunks: Initial retrieved chunks
            n_results: Number of top results to analyze for winning article
            preferred_articles: Topic-relevant article numbers to prioritize

        Returns:
            Original chunks + all chunks from winning article (up to max limit)
        """
        if not CAG_ENABLE_FULL_ARTICLE_EXPANSION or not chunks:
            return chunks

        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        if not contract_chunks:
            return chunks

        # Count article occurrences in top-N results
        from collections import Counter
        top_chunks = chunks[:n_results]
        article_counts = Counter()

        for chunk in top_chunks:
            article_num = chunk.get('article_num')
            if article_num:
                article_counts[article_num] += 1

        if not article_counts:
            return chunks

        preferred_set = set(_normalize_article_list(preferred_articles or []))
        winning_article = None
        count = 0
        chosen_by_preference = False

        # Prefer topic-relevant articles that already appear in top chunks.
        if preferred_set:
            for chunk in top_chunks:
                article_num = chunk.get("article_num")
                try:
                    article_int = int(article_num) if article_num is not None else None
                except (TypeError, ValueError):
                    article_int = None
                if article_int in preferred_set:
                    winning_article = article_int
                    count = article_counts.get(article_num, article_counts.get(article_int, 0))
                    chosen_by_preference = True
                    break

        # Fallback to the most frequent article.
        if winning_article is None:
            winning_article, count = article_counts.most_common(1)[0]

        # Only expand if the winning article appears enough times
        if (not chosen_by_preference) and count < FULL_ARTICLE_MIN_TOP_K_MATCH:
            return chunks

        # Get chunk IDs already in results
        existing_ids = {c.get('chunk_id', c.get('citation', '')) for c in chunks}

        # Fetch ALL chunks from the winning article
        article_chunks = [
            c for c in contract_chunks
            if c.get('article_num') == winning_article
            and c.get('chunk_id', c.get('citation', '')) not in existing_ids
        ]

        # Prioritize sections explicitly referenced by retrieved winning-article
        # chunks (e.g., Section 49 referencing Section 48), then near-neighbors.
        winning_top_chunks = []
        for c in top_chunks:
            try:
                c_article = int(c.get("article_num")) if c.get("article_num") is not None else None
            except (TypeError, ValueError):
                c_article = None
            if c_article == winning_article:
                winning_top_chunks.append(c)

        retrieved_secs: set[int] = set()
        referenced_secs: set[int] = set()
        for c in winning_top_chunks:
            raw_sec = c.get("section_num")
            try:
                sec_int = int(raw_sec) if raw_sec is not None else None
            except (TypeError, ValueError):
                sec_int = None
            if sec_int is not None:
                retrieved_secs.add(sec_int)

            content_text = str(
                c.get("content_with_tables")
                or c.get("content")
                or ""
            )
            for m in re.finditer(r'\bSection\s+(\d+)\b', content_text, flags=re.IGNORECASE):
                try:
                    ref_sec = int(m.group(1))
                except (TypeError, ValueError):
                    continue
                referenced_secs.add(ref_sec)

        def _article_chunk_priority(c: dict) -> tuple:
            raw = c.get("section_num")
            try:
                sec = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                sec = None
            if sec is None:
                return (3, 9999, 9999, str(c.get("subsection") or ""))
            if sec in referenced_secs:
                return (0, 0, sec, str(c.get("subsection") or ""))
            if retrieved_secs:
                distance = min(abs(sec - base) for base in retrieved_secs)
                if distance <= 1:
                    return (1, distance, sec, str(c.get("subsection") or ""))
                return (2, distance, sec, str(c.get("subsection") or ""))
            return (2, 999, sec, str(c.get("subsection") or ""))

        article_chunks.sort(key=_article_chunk_priority)

        # Limit total chunks to avoid context overflow
        available_slots = FULL_ARTICLE_MAX_CHUNKS - len(chunks)
        if available_slots <= 0:
            return chunks

        # Mark as full-article context and add
        for chunk in article_chunks[:available_slots]:
            chunk_copy = dict(chunk)
            chunk_copy['similarity'] = 0.4  # Lower score to indicate supplemental
            chunk_copy['is_full_article_context'] = True
            chunk_copy['winning_article'] = winning_article
            chunks.append(chunk_copy)

        return chunks

    def retrieve(
        self,
        query: str,
        intent: QueryIntent = None,
        n_results: int = 5,
        hours_worked: int = 0,
        months_employed: int = 0,
        use_hybrid: bool = True,
        contract_id: str = CONTRACT_ID,
    ) -> dict:
        """
        Retrieve relevant context for a query.

        Uses the "Rosetta Stone" CAG pipeline:
        1. Hypothesis Layer - LLM predicts section titles (bridges vocabulary gap)
        2. Hybrid Search - Vector + BM25 with RRF fusion
        3. Title Boosting - Boost chunks matching hypothesized titles
        4. Full Article Expansion - Fetch complete article for context

        Returns:
            dict with:
            - chunks: List of relevant contract chunks
            - wage_info: Wage lookup result (if applicable)
            - intent: Query intent classification
            - escalation_required: Whether to add escalation language
            - query_expansions: List of slang->contract term expansions applied
            - hypothesis_result: HypothesisResult from pre-retrieval reasoning (NEW)
        """
        # Expand query with contract terminology (static mappings)
        ensure_contract_manifest(contract_id)

        expanded_query, expansions = expand_query(query, contract_id=contract_id)

        # Detect LOU keywords in query
        lou_keywords = ['letter of understanding', 'lou', 'letters of understanding']
        query_lower = query.lower()
        lou_detected = any(kw in query_lower for kw in lou_keywords)

        if intent is None:
            # Use expanded query for intent classification
            intent = classify_intent(expanded_query, contract_id=contract_id)

        # ===== PHASE 2: HYPOTHESIS LAYER (Rosetta Stone Brain) =====
        # Use LLM to predict likely section titles before searching
        hypothesis_result = None
        if CAG_ENABLE_HYPOTHESIS_LAYER:
            hypothesis_generator = get_hypothesis_generator()
            hypothesis_result = hypothesis_generator.generate_sync(query)

            if hypothesis_result.success and hypothesis_result.hypothesized_titles:
                # Append hypothesized titles to the query for better matching
                # e.g., "When do I get a break?" -> "When do I get a break? (Relief Periods Rest Intervals Meal Periods)"
                expanded_query = hypothesis_result.query_expansion
        # ===== END HYPOTHESIS LAYER =====

        result = {
            "chunks": [],
            "wage_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "contract_id": contract_id,
            "query_expansions": expansions,  # Track what slang was expanded
            "hypothesis_result": hypothesis_result  # Track hypothesis for metrics
        }

        # Use hybrid search (vector + BM25 with RRF)
        # Weights configured in config.py (default: equal 1.0/1.0 for balanced fusion)
        doc_type_filter = "lou" if lou_detected else None

        if use_hybrid:
            region_id = resolve_contract_region_id(contract_id)
            self._ensure_hybrid_searcher()
            chunks = self.hybrid_searcher.search_to_chunks(
                query=expanded_query,  # Use expanded query with hypotheses
                n_results=n_results,
                use_expansion=True,
                vector_weight=HYBRID_VECTOR_WEIGHT,
                keyword_weight=HYBRID_KEYWORD_WEIGHT,
                contract_id=contract_id,
                region_id=region_id,
                boost_articles=intent.relevant_articles,
                concept_query=query,  # Use original query for stable concept matching
                doc_type=doc_type_filter,  # Filter by doc_type if LOU detected
            )
        else:
            # Fallback to vector-only search
            if self.vector_store is None:
                self._ensure_hybrid_searcher()
            region_id = resolve_contract_region_id(contract_id)
            chunks = self.vector_store.search(
                query=expanded_query,  # Use expanded query with hypotheses
                n_results=n_results,
                contract_id=contract_id,
                region_id=region_id,
                classification=intent.classification,
                topic=intent.topic,  # Pass detected topic for boosting
                boost_articles=intent.relevant_articles,
                doc_type=doc_type_filter,  # Filter by doc_type if LOU detected
            )

        chunks = self._ensure_topic_article_coverage(
            chunks,
            intent.relevant_articles,
            contract_id=contract_id,
            max_additional=3,
            query_text=query,
        )

        # ===== PHASE 2: TITLE BOOSTING =====
        # Boost chunks whose article_title matches hypothesized titles
        if hypothesis_result and hypothesis_result.success:
            chunks = apply_title_boosting(
                chunks,
                hypothesis_result.hypothesized_titles
            )
        # ===== END TITLE BOOSTING =====

        # ===== PHASE 3: FULL ARTICLE EXPANSION =====
        # Fetch all chunks from the "winning" article for complete context
        chunks = self._expand_to_full_article(
            chunks,
            contract_id=contract_id,
            n_results=n_results,
            preferred_articles=intent.relevant_articles,
        )
        # ===== END FULL ARTICLE EXPANSION =====

        # Expand context: fetch related sections from the same articles
        chunks = self._expand_with_related_sections(
            chunks,
            contract_id=contract_id,
            max_total=max(len(chunks) + 3, n_results + 6),
            preferred_articles=intent.relevant_articles,
        )
        chunks = self._ensure_vacation_entitlement_coverage(
            chunks,
            contract_id=contract_id,
            preferred_articles=intent.relevant_articles,
            topic=intent.topic,
            query_text=query,
        )
        chunks = self._prioritize_topic_articles(
            chunks,
            intent.relevant_articles,
            topic=intent.topic,
            query_text=query,
        )

        result["chunks"] = chunks

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=hours_worked,
                months_employed=months_employed,
                contract_id=contract_id,
            )
            result["wage_info"] = wage_info
            if wage_info:
                result["chunks"] = self._ensure_wage_table_context(
                    result["chunks"],
                    classification=intent.classification,
                    contract_id=contract_id,
                )

        return result

    def multi_angle_retrieve(
        self,
        query: str,
        intent: QueryIntent = None,
        n_results: int = 5,
        hours_worked: int = 0,
        months_employed: int = 0,
        contract_id: str = CONTRACT_ID,
    ) -> dict:
        """
        Retrieve using multiple search angles from query interpretation.

        This is the Phase 4 enhancement that:
        1. Uses LLM to deeply interpret the query
        2. Generates multiple search queries (original, hypothetical answers, alternatives)
        3. Runs retrieval for each angle
        4. Merges and deduplicates results using score fusion
        5. Optionally fetches explicit article references directly

        Args:
            query: User's question
            intent: Pre-classified intent (optional)
            n_results: Number of results to return
            hours_worked: For wage lookups
            months_employed: For wage lookups

        Returns:
            dict with chunks, interpretation, and metadata
        """
        ensure_contract_manifest(contract_id)

        if not CAG_ENABLE_QUERY_INTERPRETER:
            # Fall back to standard retrieval
            return self.retrieve(
                query=query,
                intent=intent,
                n_results=n_results,
                hours_worked=hours_worked,
                months_employed=months_employed,
                contract_id=contract_id,
            )

        # ===== PHASE 4: QUERY INTERPRETATION =====
        interpreter = get_interpreter()
        interpretation = interpreter.interpret(query, contract_id=contract_id)

        # Get all search queries to try
        search_queries = interpreter.get_all_search_queries(interpretation)

        # Limit to configured max
        search_queries = search_queries[:MULTI_QUERY_MAX_SEARCHES]

        # Ensure vector store is initialized for direct HyDE searches
        self._ensure_hybrid_searcher()

        # ===== MULTI-ANGLE RETRIEVAL =====
        all_chunks = []
        chunk_scores = {}  # chunk_id -> best score

        # Add explicit article lookups first (highest priority)
        if interpretation.explicit_articles:
            self._load_all_chunks(contract_id=contract_id)
            for article_num in interpretation.explicit_articles:
                article_chunks = [
                    c for c in self._all_chunks
                    if c.get('article_num') == article_num
                    and c.get("contract_id") == contract_id
                ]
                # Sort by section number
                article_chunks.sort(key=lambda x: (
                    x.get('section_num') or 0,
                    x.get('subsection') or ''
                ))
                # Add with high score
                for chunk in article_chunks[:MULTI_QUERY_RESULTS_PER_SEARCH]:
                    chunk_id = chunk.get('chunk_id', chunk.get('citation', ''))
                    chunk_copy = dict(chunk)
                    chunk_copy['similarity'] = 0.95  # High score for explicit reference
                    chunk_copy['search_angle'] = f"explicit_article_{article_num}"

                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = chunk_copy
                        all_chunks.append(chunk_copy)
                    elif chunk_copy['similarity'] > chunk_scores[chunk_id].get('similarity', 0):
                        # Update with better score
                        idx = next(i for i, c in enumerate(all_chunks)
                                   if c.get('chunk_id', c.get('citation', '')) == chunk_id)
                        all_chunks[idx] = chunk_copy
                        chunk_scores[chunk_id] = chunk_copy

        # Run retrieval for each search angle
        for i, search_query in enumerate(search_queries):
            # For hypothetical answers (HyDE), use direct vector search
            # This avoids score distortion from hybrid fusion
            is_hypothetical = i > 0 and i <= len(interpretation.hypothetical_answers)

            if is_hypothetical and self.vector_store:
                # Direct vector search for hypothetical answers
                region_id = resolve_contract_region_id(contract_id)
                angle_chunks = self.vector_store.search(
                    query=search_query,
                    n_results=MULTI_QUERY_RESULTS_PER_SEARCH,
                    contract_id=contract_id,
                    region_id=region_id,
                )
                angle_result = {"chunks": angle_chunks}
            else:
                # Use standard retrieval for original query and search queries
                angle_result = self.retrieve(
                    query=search_query,
                    intent=intent,
                    n_results=MULTI_QUERY_RESULTS_PER_SEARCH,
                    hours_worked=hours_worked,
                    months_employed=months_employed,
                    use_hybrid=True,
                    contract_id=contract_id,
                )

            # Merge chunks with score tracking
            for chunk in angle_result.get('chunks', []):
                chunk_id = chunk.get('chunk_id', chunk.get('citation', ''))
                chunk_copy = dict(chunk)
                chunk_copy['search_angle'] = f"angle_{i}_{search_query[:30]}"

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = chunk_copy
                    all_chunks.append(chunk_copy)
                elif chunk_copy.get('similarity', 0) > chunk_scores[chunk_id].get('similarity', 0):
                    # Update with better score
                    idx = next((i for i, c in enumerate(all_chunks)
                                if c.get('chunk_id', c.get('citation', '')) == chunk_id), None)
                    if idx is not None:
                        all_chunks[idx] = chunk_copy
                        chunk_scores[chunk_id] = chunk_copy

        # Sort by similarity and limit results
        all_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        final_chunks = all_chunks[:MULTI_QUERY_TOTAL_RESULTS]

        # ===== PHASE 5: LLM RERANKING =====
        # Reorder chunks by semantic relevance before expansion
        reranker_result = None
        if CAG_ENABLE_RERANKER and not self._should_skip_reranker_for_query(query):
            from backend.retrieval.reranker import get_reranker
            reranker_result = get_reranker().rerank(
                query=query,
                chunks=final_chunks,
                interpretation=interpretation
            )
            if reranker_result.success:
                final_chunks = reranker_result.chunks
        # ===== END RERANKING =====

        # Apply full article expansion on merged results
        if intent is None:
            expanded_query = " ".join([query] + interpretation.key_concepts)
            intent = classify_intent(expanded_query, contract_id=contract_id)

        final_chunks = self._ensure_topic_article_coverage(
            final_chunks,
            intent.relevant_articles,
            contract_id=contract_id,
            max_additional=3,
            query_text=query,
        )

        # Apply full article expansion on merged results
        final_chunks = self._expand_to_full_article(
            final_chunks,
            contract_id=contract_id,
            n_results=n_results,
            preferred_articles=intent.relevant_articles,
        )
        final_chunks = self._expand_with_related_sections(
            final_chunks,
            contract_id=contract_id,
            max_total=max(len(final_chunks) + 4, n_results + 4),
            preferred_articles=intent.relevant_articles,
        )
        final_chunks = self._ensure_vacation_entitlement_coverage(
            final_chunks,
            contract_id=contract_id,
            preferred_articles=intent.relevant_articles,
            topic=intent.topic,
            query_text=query,
        )
        final_chunks = self._prioritize_topic_articles(
            final_chunks,
            intent.relevant_articles,
            topic=intent.topic,
            query_text=query,
        )

        # Build result
        result = {
            "chunks": final_chunks,
            "wage_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "contract_id": contract_id,
            "query_expansions": [],
            "interpretation": interpretation,  # Include interpretation for debugging
            "search_angles_used": len(search_queries),
            "explicit_articles_fetched": interpretation.explicit_articles,
            "reranker_result": reranker_result,  # Include reranker metrics
        }

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=hours_worked,
                months_employed=months_employed,
                contract_id=contract_id,
            )
            result["wage_info"] = wage_info
            if wage_info:
                result["chunks"] = self._ensure_wage_table_context(
                    result["chunks"],
                    classification=intent.classification,
                    contract_id=contract_id,
                )

        return result

    def _load_all_chunks(self, contract_id: str = CONTRACT_ID):
        """Load all chunks for direct article lookup."""
        self._all_chunks = self._load_all_chunks_for_contract(contract_id)

    @staticmethod
    def _should_skip_reranker_for_query(query: str) -> bool:
        """
        Skip LLM reranker for ultra-short queries.

        Very short prompts (for example, "float days") are better handled by
        deterministic topic routing and lexical relevance than by stochastic
        semantic reranking.
        """
        token_count = len(re.findall(r"[a-z0-9]+", (query or "").lower()))
        return token_count <= 3


def main():
    """Test the router and retriever."""
    
    # Test query expansion first
    print("=" * 60)
    print("Testing Query Expansion (Slang -> Contract Terms)")
    print("=" * 60)
    
    expansion_tests = [
        "do i get float days?",
        "what about DUG employees?",
        "can i get written up for being late?",
        "what's the policy on tardiness?",
        "how many floaters do i get?",
        "what are my pto options?",
        "when do i get a raise?",
    ]
    
    for query in expansion_tests:
        expanded, expansions = expand_query(query)
        print(f"\nOriginal: {query}")
        if expansions:
            print(f"Expanded: {expanded}")
            print(f"  Mapped: {expansions}")
        else:
            print("  (no expansion needed)")
    
    print("\n" + "=" * 60)
    print("Testing Intent Router")
    print("=" * 60)
    
    test_queries = [
        "What is my pay rate as a courtesy clerk?",
        "How much do I make after 3000 hours?",
        "I'm being called into a meeting with my manager about my performance",
        "How does seniority work for scheduling?",
        "What are my Weingarten rights?",
        "When can I take vacation?",
        "My manager is harassing me",
        "What are the duties of a head clerk?",
    ]
    
    for query in test_queries:
        intent = classify_intent(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {intent.intent_type} (conf: {intent.confidence:.2f})")
        print(f"  Classification: {intent.classification}")
        print(f"  Topic: {intent.topic}")
        print(f"  Escalation: {intent.requires_escalation}")
        if intent.keywords_matched:
            print(f"  Matched: {intent.keywords_matched}")
    
    # Test hybrid retrieval
    print("\n\n--- Testing Hybrid Retrieval ---")
    retriever = HybridRetriever()
    
    test_cases = [
        ("What is the starting pay for a courtesy clerk?", 0, 0),
        ("How much does an all purpose clerk make after 5000 hours?", 5000, 0),
        ("What are my rights during a disciplinary meeting?", 0, 0),
    ]
    
    for query, hours, months in test_cases:
        print(f"\nQuery: {query}")
        result = retriever.retrieve(query, hours_worked=hours, months_employed=months)
        
        if result["wage_info"]:
            wi = result["wage_info"]
            print(f"  Wage: ${wi['rate']:.2f}/hr ({wi['step']}) - {wi['citation']}")
        
        print(f"  Escalation: {result['escalation_required']}")
        print(f"  Top chunks:")
        for i, chunk in enumerate(result["chunks"][:2]):
            print(f"    {i+1}. [{chunk['citation']}] (sim: {chunk['similarity']:.3f})")


if __name__ == "__main__":
    main()
