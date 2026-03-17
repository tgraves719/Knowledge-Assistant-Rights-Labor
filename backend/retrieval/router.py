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
from typing import Optional, Tuple, Any
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
from backend.entitlement_files import resolve_entitlement_file
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
    "stewards conference": "annual union stewards conference union stewards work schedules",
    "rep": "union representative steward",
    "dues": "union dues",
    "union meeting": "union business leave",
    "union meeting night": "regular local union meeting scheduled later than 6:00 p.m.",

    # Contract term phrasing
    "run through": "term of agreement expiration date end date",
    "in force through": "term of agreement expiration date end date",

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
    "premiums": (
        "premium",
        "holiday",
        "holidays",
        "holiday pay",
        "holiday premium",
        "when a holiday is worked",
        "per hour worked",
        "night premium",
        "sunday premium",
    ),
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
    "overtime": ("overtime", "time and one-half", "time and a half", "ot"),
}

_ROLE_COMPARISON_TITLE_HINTS: tuple[str, ...] = (
    "check-off",
    "new employees, transferred employees, promoted or demoted",
    "definitions of classifications",
    "rates of pay",
    "scheduling and assignment of hours",
)


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


@lru_cache(maxsize=16)
def infer_role_comparison_articles(contract_id: str = CONTRACT_ID) -> list[int]:
    """
    Infer stable fallback anchors for role-comparison questions.

    Used when manifests do not define `classification_to_articles`.
    """
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if not manifest_path.exists():
        return []

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return []

    article_titles = manifest.get("article_titles", {}) or {}
    inferred: list[int] = []
    for article_key, raw_title in article_titles.items():
        try:
            article_num = int(article_key)
        except (TypeError, ValueError):
            continue
        title = str(raw_title or "").strip().lower()
        if not title:
            continue
        if any(hint in title for hint in _ROLE_COMPARISON_TITLE_HINTS):
            inferred.append(article_num)
    return _normalize_article_list(inferred)


def _allow_legacy_unscoped_chunks_for_routing() -> bool:
    """Allow unscoped chunk fallback only in single-manifest mode."""
    return len(list(MANIFESTS_DIR.glob("*.json"))) == 1


def _infer_side_letter_doc_type_from_text(text: str) -> Optional[str]:
    lower = str(text or "").lower()
    if not lower:
        return None
    has_loa = "letter of agreement" in lower
    has_lou = "letter of understanding" in lower
    if has_loa and not has_lou:
        return "loa"
    if has_lou and not has_loa:
        return "lou"
    if has_loa:
        return "loa"
    if has_lou:
        return "lou"
    return None


def _resolved_side_letter_doc_type(row: Optional[dict]) -> str:
    chunk = row if isinstance(row, dict) else {}
    raw_doc_type = str(chunk.get("doc_type") or "").strip().lower()
    if raw_doc_type in {"loa", "lou"}:
        return raw_doc_type

    meta_text = "\n".join(
        [
            str(chunk.get("citation") or ""),
            str(chunk.get("article_title") or ""),
            str(chunk.get("parent_context") or ""),
            str(chunk.get("content_with_tables") or ""),
            str(chunk.get("content") or ""),
        ]
    )
    inferred = _infer_side_letter_doc_type_from_text(meta_text)
    return inferred or raw_doc_type


def _normalize_resolved_doc_types(chunks: Optional[list[dict]]) -> list[dict]:
    normalized: list[dict] = []
    for chunk in list(chunks or []):
        if not isinstance(chunk, dict):
            continue
        chunk_copy = dict(chunk)
        resolved_doc_type = _resolved_side_letter_doc_type(chunk_copy)
        if resolved_doc_type in {"loa", "lou"}:
            chunk_copy["doc_type"] = resolved_doc_type
        normalized.append(chunk_copy)
    return normalized


@lru_cache(maxsize=16)
def contract_supports_side_letter_doc_type_filter(contract_id: str = CONTRACT_ID) -> bool:
    """
    Only apply retrieval-time doc_type filters when the pack actually materialized
    LOA/LOU buckets. Older packs still need lexical side-letter retrieval.
    """
    chunks_path = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    if not chunks_path or not chunks_path.exists():
        return False

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False

    if not isinstance(payload, list):
        return False

    target_region = str(resolve_contract_region_id(contract_id))
    allow_unscoped = _allow_legacy_unscoped_chunks_for_routing()
    for row in payload:
        if not isinstance(row, dict):
            continue
        chunk_contract_id = row.get("contract_id")
        if chunk_contract_id != contract_id:
            if not (allow_unscoped and chunk_contract_id in (None, "")):
                continue
        chunk_region = row.get("region_id") or resolve_contract_region_id(str(chunk_contract_id or contract_id))
        if str(chunk_region) != target_region and not (allow_unscoped and chunk_contract_id in (None, "")):
            continue
        raw_doc_type = str(row.get("doc_type") or "").strip().lower()
        if raw_doc_type in {"loa", "lou"}:
            return True
    return False


@lru_cache(maxsize=16)
def infer_side_letter_articles(contract_id: str = CONTRACT_ID) -> list[int]:
    """
    Infer side-letter-heavy article anchors from chunk artifacts.

    Uses `doc_type` first (`loa`/`lou`) with lexical fallback for older packs.
    """
    chunks_path = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    if not chunks_path or not chunks_path.exists():
        return []

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    target_region = str(resolve_contract_region_id(contract_id))
    allow_unscoped = _allow_legacy_unscoped_chunks_for_routing()
    doc_type_counts: dict[int, int] = {}
    lexical_counts: dict[int, int] = {}

    for row in payload:
        if not isinstance(row, dict):
            continue

        chunk_contract_id = row.get("contract_id")
        if chunk_contract_id != contract_id:
            if not (allow_unscoped and chunk_contract_id in (None, "")):
                continue

        chunk_region = row.get("region_id") or resolve_contract_region_id(str(chunk_contract_id or contract_id))
        if str(chunk_region) != target_region and not (allow_unscoped and chunk_contract_id in (None, "")):
            continue

        doc_type = _resolved_side_letter_doc_type(row)

        try:
            article_num = int(row.get("article_num") or 0)
        except (TypeError, ValueError):
            article_num = 0
        if article_num <= 0:
            continue

        if doc_type in {"loa", "lou"}:
            doc_type_counts[article_num] = int(doc_type_counts.get(article_num, 0)) + 1
            continue

        blob = (
            str(row.get("citation") or "")
            + "\n"
            + str(row.get("content_with_tables") or "")
            + "\n"
            + str(row.get("content") or "")
        ).lower()
        if ("letter of agreement" in blob) or ("letter of understanding" in blob):
            lexical_counts[article_num] = int(lexical_counts.get(article_num, 0)) + 1

    counts = doc_type_counts if doc_type_counts else lexical_counts
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    return [int(article_num) for article_num, _ in ranked]


def _is_side_letter_explicit_query(query_text: str) -> bool:
    q = _normalize_query_text(query_text or "")
    if not q:
        return False
    return bool(
        re.search(
            r"\b(letter(?:s)?\s+of\s+(?:agreement|understanding)|side[-\s]?letter|lou|loa)\b",
            q,
        )
    )


def _is_side_letter_followup_query(query_text: str) -> bool:
    q = _normalize_query_text(query_text or "")
    if not q or _is_side_letter_explicit_query(q):
        return False

    if re.search(CONTRACT_TERM_CUE_PATTERN, q):
        return False

    has_reference = bool(
        re.search(r"\b(this|that|the)\s+agreement\b|\b(this|that|the)\s+letter\b|\bagreement\b", q)
    )
    has_followup_cue = bool(
        re.search(
            r"\b(written\s+notice|30\s*days|either\s+party|cancel|discontinue|discontinuing|discontinued|implement\s+this\s+procedure)\b",
            q,
        )
    )
    return has_reference and has_followup_cue

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
            r"|\bterm\s*of\s*(?:agreement|contract)\b|\beffective\s*date\b|\bexpiration\s*date\b"
            r"|\brun\s*through\b|\bin\s*force\s*through\b",
            "term of agreement effective date expiration date start date end date",
            "contract term pattern",
        ),
        (
            r"\bstewards?\b.*\b(conference|meeting)\b|\bunion\s*meeting\b.*\bstewards?\b",
            "annual union stewards conference adjust union stewards work schedules regular local union meeting scheduled later than 6:00 p.m.",
            "steward schedule pattern",
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
    options = get_classification_options(contract_id=contract_id, include_unmapped=True) or []
    alias_to_value: dict[str, str] = {}

    def _register_alias(alias: str, value: str) -> None:
        key = _normalize_query_text(alias)
        if key:
            alias_to_value[key] = value

    def _alias_variants(raw_text: str) -> set[str]:
        text = _normalize_query_text(raw_text)
        if not text:
            return set()
        variants: set[str] = {text}

        # Keep a plain form without parenthetical hints.
        no_paren = _normalize_query_text(re.sub(r"\([^)]*\)", " ", text))
        if no_paren:
            variants.add(no_paren)

        # Split common list separators while preserving full phrase.
        for part in re.split(r"[/,;]", text):
            token = _normalize_query_text(part)
            if len(token) >= 2:
                variants.add(token)

        # Acronym aliases: "courtesy clerk" -> "cc", "drive up and go" -> "dug".
        for candidate in list(variants):
            words = [
                w for w in re.findall(r"[a-z0-9]+", candidate)
                if w not in {"and", "or", "of", "the", "to", "for"}
            ]
            if len(words) >= 2:
                acronym = "".join(w[0] for w in words)
                if 2 <= len(acronym) <= 6:
                    variants.add(acronym)
                    variants.add(f"{acronym}s")

        return {v for v in variants if len(v) >= 2}

    for opt in options:
        value = str(opt.get("value") or "").strip().lower()
        label = str(opt.get("label") or "").strip().lower()
        if not value:
            continue

        value_phrase = value.replace("_", " ")
        for alias in _alias_variants(value):
            _register_alias(alias, value)
        for alias in _alias_variants(value_phrase):
            _register_alias(alias, value)
        for alias in _alias_variants(label):
            _register_alias(alias, value)

        # Normalize common '&' variants for stable matching.
        if " & " in value_phrase:
            _register_alias(value_phrase.replace(" & ", " and "), value)
        if " and " in value_phrase:
            _register_alias(value_phrase.replace(" and ", " & "), value)
        if label:
            if " & " in label:
                _register_alias(label.replace(" & ", " and "), value)
            if " and " in label:
                _register_alias(label.replace(" and ", " & "), value)

        for raw_alias in (opt.get("alias_labels") or []):
            for alias in _alias_variants(str(raw_alias or "")):
                _register_alias(alias, value)

    return alias_to_value


def _normalize_query_text(text: str) -> str:
    """
    Normalize free text for deterministic lexical matching.

    Notes:
    - Keeps alphanumerics only (punctuation-insensitive matching).
    - Collapses possessives/plurals like "cc's" -> "ccs" so acronym aliases
      remain matchable.
    """
    raw = str(text or "").lower()
    raw = raw.replace("’", "'").replace("`", "'")
    raw = re.sub(r"(?<=\w)'(?=\w)", "", raw)
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _classification_alias_candidates(raw: str) -> list[str]:
    """
    Produce deterministic lexical variants for contract alias resolution.

    Keep this conservative; semantic remapping belongs in contract data.
    """
    pending: list[str] = [_normalize_query_text(raw)]
    seen: set[str] = set()
    ordered: list[str] = []

    while pending:
        candidate = _normalize_query_text(pending.pop(0))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

        derived = [
            candidate.replace("nonfood", "non food"),
            candidate.replace("non food", "nonfood"),
            re.sub(r"\bfoods\b", "food", candidate),
            re.sub(r"\bfood\b", "foods", candidate),
        ]
        for value in derived:
            value = _normalize_query_text(value)
            if value and value not in seen:
                pending.append(value)

    return ordered


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
    for candidate in _classification_alias_candidates(raw):
        if candidate in aliases:
            return aliases[candidate]

        snake = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
        if snake in aliases:
            return aliases[snake]

        underscored = candidate.replace(" ", "_")
        if underscored in aliases:
            return aliases[underscored]

    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or raw


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
    mentioned_classifications: list = None
    comparison_mode: bool = False
    required_evidence_slots: list = None
    
    def __post_init__(self):
        if self.relevant_articles is None:
            self.relevant_articles = []
        if self.mentioned_classifications is None:
            self.mentioned_classifications = []
        if self.required_evidence_slots is None:
            self.required_evidence_slots = []


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

# Topics where "rate" language is often legal-calculation text, not wage lookup.
WAGE_SUPPRESS_TOPICS = {
    "overtime",
    "premiums",
    "breaks",
    "grievance",
    "discipline",
    "term",
}

# Classification extraction patterns
CLASSIFICATION_PATTERNS = {
    "courtesy_clerk": r"courtesy\s*clerk|bagger",
    "head_clerk": r"head\s*clerk",
    "all_purpose_clerk": r"all[\s-]*purpose\s*clerk",
    "produce_manager": r"produce\s*(department)?\s*manager",
    "bakery_manager": r"bakery\s*manager",
    "cake_decorator": r"cake\s*decorator",
    "pharmacy_tech": r"pharmacy\s*tech",
    "other_assistant_managers": r"other\s*assistant\s*managers?|assistant\s*managers?",
    "non_foods_clerk": r"non.?food|gm\s*clerk|general\s*merchandise|non.?food.*gm.*floral|floral",
}

CONTRACT_TERM_CUE_PATTERN = (
    r"term\s*of|contract\s*term|agreement\s*term|term\s*of\s*(agreement|contract)|"
    r"expir|effective\s*date|"
    r"run\s*through|in\s*force\s*through|"
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
    "discipline": r"disciplin|warning|write\s*up|written up|tardiness|tardy|\blate\b|attendance",
    "grievance": r"grievance|arbitration|file\s*a\s*complaint",
    "breaks": rf"break|lunch|meal\s*period|relief|rest\s*period|{INTER_SHIFT_REST_CUE_PATTERN}",
    "premiums": (
        r"premium|night\s*pay|sunday\s*pay|sunday\s*premium|holiday\s*premium|"
        r"holiday.*(premium|pay).*worked|worked.*holiday.*(premium|pay)|"
        r"in\s+addition\s+to\s+(their|the)\s+hourly\s+rate|"
        r"12\s*00\s*midnight.*6\s*00\s*a\s*m"
    ),
    "weingarten": r"weingarten|right\s*to\s*representation|union\s*rep",
    "health_benefits": r"health\s*(benefit|insurance|coverage|care)|medical\s*benefit|eligible.*(health|benefit)|benefit.*eligible",
    "promotion": r"promot|advance|move up|basket.*hours|credit.*hours",
    "probation": r"probation|probationary|trial\s*period|new\s*employee.*hours",
    "term": CONTRACT_TERM_CUE_PATTERN,
    "minimum_wage": r"minimum\s*wage|colorado.*wage|\$15",
    "joint_committee": r"joint.*committee|labor.*management\s*committee",
}

# Topics that represent role/entities, not cross-contract semantic topics.
# These should be handled via contract-scoped entity extraction.
_ROLE_LIKE_TOPICS = {"drive_up_go"}

_ROLE_COMPARISON_PATTERN = (
    r"\b(difference|compare|comparison|vs\.?|versus|between)\b"
    r"|what\s+is\s+the\s+difference\s+between"
)

_TOPIC_EVIDENCE_SLOTS: dict[str, list[str]] = {
    "wages": ["rate_schedule", "progression_rules"],
    "vacation": ["vacation_entitlement", "vacation_scheduling_rules"],
    "personal_holiday": ["personal_holiday_eligibility", "personal_holiday_usage_rules"],
    "overtime": ["overtime_thresholds", "overtime_rate_rules"],
    "breaks": ["break_meal_rules", "inter_shift_rest_rules"],
    "discipline": ["discipline_procedure", "representation_rights"],
    "grievance": ["grievance_steps", "timelines"],
    "term": ["term_start_end_dates"],
    "layoff": ["seniority_bumping_rules", "notice_rules"],
    "scheduling": ["scheduling_rules", "minimum_hours_rules"],
    "bereavement": ["bereavement_eligibility", "bereavement_pay_rules"],
    "sick_leave": ["sick_leave_eligibility", "sick_leave_usage_rules"],
    "health_benefits": ["eligibility_rules", "benefit_scope"],
    "premiums": ["premium_conditions", "premium_rate_rules"],
    "promotion": ["progression_credits", "promotion_rate_rules"],
    "probation": ["probation_duration", "probation_rights"],
}

VACATION_ENTITLEMENT_QUERY_PATTERN = (
    r"(how\s+much|how\s+many).*\bvacation\b"
    r"|\bvacation\b.*(per\s+year|entitlement|accrual|do\s+i\s+get|years?\s+of\s+(continuous\s+)?service|tenure|anniversary\s+year)"
    r"|\bweeks?\s+of\s+vacation\b"
    r"|annual\s+paid\s+vacation"
)

HOLIDAY_WORK_PREMIUM_QUERY_PATTERN = (
    r"\bholiday\b.*\b(work|worked|working)\b.*\b(premium|pay|rate)\b"
    r"|\b(work|worked|working)\b.*\bholiday\b.*\b(premium|pay|rate)\b"
)

STEWARD_UNION_MEETING_SCHEDULE_QUERY_PATTERN = (
    r"\bstewards?\b.*\b(union\s*meeting|meeting\s*nights?)\b"
    r"|\b(union\s*meeting|meeting\s*nights?)\b.*\bstewards?\b"
    r"|\bstewards?\b.*\b6\s*00\b"
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
        if not phrase or len(phrase) < 2:
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


def extract_classifications_for_contract(
    query: str,
    contract_id: str = CONTRACT_ID,
    max_matches: int = 3,
) -> list[str]:
    """
    Extract up to max_matches classification mentions in query order.

    Used for comparison prompts such as "difference between X and Y" so retrieval
    can anchor both role-specific article maps deterministically.
    """
    query_lower = _normalize_query_text(query)
    aliases = _contract_classification_aliases(contract_id)
    matches: list[tuple[int, int, str]] = []  # (start, -len(phrase), value)

    for phrase, value in aliases.items():
        if not phrase or len(phrase) < 2:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        for m in re.finditer(pattern, query_lower):
            matches.append((m.start(), -len(phrase), value))

    if not matches:
        legacy = extract_classification(query_lower)
        if legacy:
            normalized = normalize_classification_for_contract(legacy, contract_id=contract_id)
            return [normalized] if normalized else []
        return []

    matches.sort(key=lambda t: (t[0], t[1]))
    ordered_values: list[str] = []
    seen: set[str] = set()
    for _, _, value in matches:
        if value in seen:
            continue
        seen.add(value)
        ordered_values.append(value)
        if len(ordered_values) >= max_matches:
            break
    return ordered_values


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
        "probation",
        "promotion",
        "term",
        "personal_holiday",  # Check before vacation since it's more specific
        "bereavement",
        "layoff",
        "sick_leave",
        "premiums",
        "vacation",
        "overtime",
        "grievance",
        "discipline",
        "seniority",
        "breaks",
        "scheduling",  # Generic - matches "hours" so put last
    ]

    # First pass: check topics in priority order
    for topic in TOPIC_PRIORITY:
        if topic in _ROLE_LIKE_TOPICS:
            continue
        if topic in topic_patterns:
            pattern = topic_patterns[topic]
            if re.search(pattern, query_lower):
                return topic

    # Second pass: check any remaining topics
    for topic, pattern in topic_patterns.items():
        if topic in _ROLE_LIKE_TOPICS:
            continue
        if topic not in TOPIC_PRIORITY:
            if re.search(pattern, query_lower):
                return topic

    return None


@dataclass
class QueryPlan:
    """Deterministic query plan for evidence orchestration."""
    contract_id: str
    topic: Optional[str]
    primary_classification: Optional[str]
    mentioned_classifications: list[str]
    comparison_mode: bool
    article_anchors: list[int]
    required_evidence_slots: list[str]
    explicit_articles: list[int]


@dataclass
class FollowupRoutingPlan:
    """Router-owned follow-up routing plan derived from prior retrieval context."""
    question: str
    routing_query: str
    strategy: str = "global_default"
    followup_context_used: bool = False
    topic: Optional[str] = None
    article_anchors: list[int] = None
    prior_citations: list[str] = None

    def __post_init__(self):
        if self.article_anchors is None:
            self.article_anchors = []
        if self.prior_citations is None:
            self.prior_citations = []

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "routing_query": self.routing_query,
            "strategy": self.strategy,
            "followup_context_used": self.followup_context_used,
            "topic": self.topic,
            "article_anchors": list(self.article_anchors),
            "prior_citations": list(self.prior_citations),
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "FollowupRoutingPlan":
        data = dict(payload or {})
        return cls(
            question=str(data.get("question") or ""),
            routing_query=str(data.get("routing_query") or data.get("question") or ""),
            strategy=str(data.get("strategy") or "global_default"),
            followup_context_used=bool(data.get("followup_context_used")),
            topic=str(data.get("topic") or "").strip() or None,
            article_anchors=_normalize_article_list(list(data.get("article_anchors") or [])),
            prior_citations=[str(v) for v in list(data.get("prior_citations") or [])],
        )


@dataclass
class RetrievalPlanRecord:
    """Typed retrieval plan before execution."""
    planned_strategy: str
    search_mode: str
    use_hybrid: bool
    use_interpreter: bool
    intent_type: str
    topic: Optional[str]
    article_anchors: list[int]
    article_anchor_count: int
    explicit_articles_requested: list[int]
    explicit_article_request_count: int
    query_expansion_count: int
    doc_type_filter: Optional[str]
    side_letter_filter_supported: bool
    side_letter_query_detected: bool
    apply_topic_seed_coverage: bool
    apply_article_prioritization: bool
    apply_side_letter_promotion: bool
    apply_full_article_expansion: bool
    apply_related_section_expansion: bool
    apply_vacation_entitlement_coverage: bool
    apply_holiday_premium_coverage: bool

    def to_dict(self) -> dict:
        return {
            "planned_strategy": self.planned_strategy,
            "search_mode": self.search_mode,
            "use_hybrid": self.use_hybrid,
            "use_interpreter": self.use_interpreter,
            "intent_type": self.intent_type,
            "topic": self.topic,
            "article_anchors": list(self.article_anchors),
            "article_anchor_count": self.article_anchor_count,
            "explicit_articles_requested": list(self.explicit_articles_requested),
            "explicit_article_request_count": self.explicit_article_request_count,
            "query_expansion_count": self.query_expansion_count,
            "doc_type_filter": self.doc_type_filter,
            "side_letter_filter_supported": self.side_letter_filter_supported,
            "side_letter_query_detected": self.side_letter_query_detected,
            "apply_topic_seed_coverage": self.apply_topic_seed_coverage,
            "apply_article_prioritization": self.apply_article_prioritization,
            "apply_side_letter_promotion": self.apply_side_letter_promotion,
            "apply_full_article_expansion": self.apply_full_article_expansion,
            "apply_related_section_expansion": self.apply_related_section_expansion,
            "apply_vacation_entitlement_coverage": self.apply_vacation_entitlement_coverage,
            "apply_holiday_premium_coverage": self.apply_holiday_premium_coverage,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "RetrievalPlanRecord":
        data = dict(payload or {})
        return cls(
            planned_strategy=str(data.get("planned_strategy") or "global_default"),
            search_mode=str(data.get("search_mode") or "single_angle_hybrid"),
            use_hybrid=bool(data.get("use_hybrid")),
            use_interpreter=bool(data.get("use_interpreter")),
            intent_type=str(data.get("intent_type") or ""),
            topic=str(data.get("topic") or "").strip() or None,
            article_anchors=_normalize_article_list(list(data.get("article_anchors") or [])),
            article_anchor_count=max(0, int(data.get("article_anchor_count") or 0)),
            explicit_articles_requested=_normalize_article_list(list(data.get("explicit_articles_requested") or [])),
            explicit_article_request_count=max(0, int(data.get("explicit_article_request_count") or 0)),
            query_expansion_count=max(0, int(data.get("query_expansion_count") or 0)),
            doc_type_filter=str(data.get("doc_type_filter") or "").strip().lower() or None,
            side_letter_filter_supported=bool(data.get("side_letter_filter_supported")),
            side_letter_query_detected=bool(data.get("side_letter_query_detected")),
            apply_topic_seed_coverage=bool(data.get("apply_topic_seed_coverage")),
            apply_article_prioritization=bool(data.get("apply_article_prioritization")),
            apply_side_letter_promotion=bool(data.get("apply_side_letter_promotion")),
            apply_full_article_expansion=bool(data.get("apply_full_article_expansion")),
            apply_related_section_expansion=bool(data.get("apply_related_section_expansion")),
            apply_vacation_entitlement_coverage=bool(data.get("apply_vacation_entitlement_coverage")),
            apply_holiday_premium_coverage=bool(data.get("apply_holiday_premium_coverage")),
        )


@dataclass
class RetrievalPolicyRecord:
    """Typed retrieval policy summary after execution."""
    strategy: str
    search_mode: str
    article_anchors: list[int]
    article_anchor_count: int
    article_anchor_hits: int
    topic_seeded: bool
    topic_seed_count: int
    side_letter_seeded: bool
    side_letter_seed_count: int
    explicit_articles_fetched: list[int]
    doc_type_filter: Optional[str]
    query_expansion_count: int
    interpreter_used: bool
    executed_stages: list[str]
    executed_stage_count: int

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "search_mode": self.search_mode,
            "article_anchors": list(self.article_anchors),
            "article_anchor_count": self.article_anchor_count,
            "article_anchor_hits": self.article_anchor_hits,
            "topic_seeded": self.topic_seeded,
            "topic_seed_count": self.topic_seed_count,
            "side_letter_seeded": self.side_letter_seeded,
            "side_letter_seed_count": self.side_letter_seed_count,
            "explicit_articles_fetched": list(self.explicit_articles_fetched),
            "doc_type_filter": self.doc_type_filter,
            "query_expansion_count": self.query_expansion_count,
            "interpreter_used": self.interpreter_used,
            "executed_stages": list(self.executed_stages),
            "executed_stage_count": self.executed_stage_count,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "RetrievalPolicyRecord":
        data = dict(payload or {})
        return cls(
            strategy=str(data.get("strategy") or "global_default"),
            search_mode=str(data.get("search_mode") or "single_angle_hybrid"),
            article_anchors=_normalize_article_list(list(data.get("article_anchors") or [])),
            article_anchor_count=max(0, int(data.get("article_anchor_count") or 0)),
            article_anchor_hits=max(0, int(data.get("article_anchor_hits") or 0)),
            topic_seeded=bool(data.get("topic_seeded")),
            topic_seed_count=max(0, int(data.get("topic_seed_count") or 0)),
            side_letter_seeded=bool(data.get("side_letter_seeded")),
            side_letter_seed_count=max(0, int(data.get("side_letter_seed_count") or 0)),
            explicit_articles_fetched=_normalize_article_list(list(data.get("explicit_articles_fetched") or [])),
            doc_type_filter=str(data.get("doc_type_filter") or "").strip().lower() or None,
            query_expansion_count=max(0, int(data.get("query_expansion_count") or 0)),
            interpreter_used=bool(data.get("interpreter_used")),
            executed_stages=[str(v).strip() for v in list(data.get("executed_stages") or []) if str(v).strip()],
            executed_stage_count=max(0, int(data.get("executed_stage_count") or 0)),
        )


_FOLLOWUP_REFERENCE_TOKENS = {
    "about",
    "that",
    "those",
    "them",
    "there",
    "it",
    "one",
    "ones",
    "what",
    "how",
    "if",
    "and",
}


def _parse_article_numbers_from_citations(citations: list[str]) -> list[int]:
    articles: list[int] = []
    seen: set[int] = set()
    for citation in citations or []:
        for match in re.finditer(r"\barticle\s+(\d+)\b", str(citation or ""), flags=re.IGNORECASE):
            article = int(match.group(1))
            if article in seen:
                continue
            seen.add(article)
            articles.append(article)
    return articles


def is_followup_query_text(text: str) -> bool:
    normalized = _normalize_query_text(text or "")
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) <= 8 and any(token in _FOLLOWUP_REFERENCE_TOKENS for token in tokens):
        return True
    return bool(
        re.search(
            r"^(what|how)\s+about\b|^and\b|^what\s+if\b|^if\s+i\b|^\d+\s+(year|month|hour)s?\b",
            normalized,
        )
    )


def build_followup_routing_plan(
    question: str,
    prior_topic: Optional[str],
    prior_citations: Optional[list[str]] = None,
    prior_article_anchors: Optional[list[int]] = None,
) -> FollowupRoutingPlan:
    normalized_question = _normalize_query_text(question or "")
    topic = str(prior_topic or "").strip().lower() or None
    article_anchors = _normalize_article_list(
        list(prior_article_anchors or []) + _parse_article_numbers_from_citations(list(prior_citations or []))
    )
    if not topic or not is_followup_query_text(question):
        return FollowupRoutingPlan(
            question=question,
            routing_query=question,
            strategy="global_default",
            followup_context_used=False,
            topic=topic,
            article_anchors=article_anchors,
            prior_citations=list(prior_citations or []),
        )

    pieces: list[str] = []
    if topic and topic not in normalized_question:
        pieces.append(topic)
    if article_anchors and len(normalized_question.split()) <= 10:
        pieces.append(" ".join(f"Article {article}" for article in article_anchors[:2]))
    if normalized_question:
        pieces.append(normalized_question)
    routing_query = " ".join(piece for piece in pieces if piece).strip() or question
    strategy = "followup_anchor_seeded" if article_anchors else "followup_topic_seeded"
    return FollowupRoutingPlan(
        question=question,
        routing_query=routing_query,
        strategy=strategy,
        followup_context_used=True,
        topic=topic,
        article_anchors=article_anchors,
        prior_citations=list(prior_citations or []),
    )


def _is_role_comparison_query(query: str) -> bool:
    return bool(re.search(_ROLE_COMPARISON_PATTERN, _normalize_query_text(query)))


def _is_steward_union_meeting_schedule_query(query_text: str) -> bool:
    """Detect steward scheduling prompts for regular union meeting nights."""
    q = _normalize_query_text(query_text or "")
    if not q:
        return False
    if not re.search(STEWARD_UNION_MEETING_SCHEDULE_QUERY_PATTERN, q):
        return False
    has_schedule_cue = bool(re.search(r"\b(schedule|scheduled|scheduling|later)\b", q))
    has_meeting_cue = bool(re.search(r"\bunion\s*meeting|meeting\s*nights?\b", q))
    has_time_cue = bool(re.search(r"\b6\s*00\b", q))
    return has_meeting_cue and (has_schedule_cue or has_time_cue)


def _required_evidence_slots_for_plan(
    topic: Optional[str],
    comparison_mode: bool,
    mentioned_classifications: list[str],
) -> list[str]:
    slots: list[str] = []
    if comparison_mode and len(mentioned_classifications) >= 2:
        slots.extend(
            [
                "classification_definition",
                "classification_rate_rules",
                "classification_scheduling_rules",
            ]
        )
    topic_key = str(topic or "").strip().lower()
    slots.extend(_TOPIC_EVIDENCE_SLOTS.get(topic_key, []))
    # Stable de-dup preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        if not slot or slot in seen:
            continue
        seen.add(slot)
        deduped.append(slot)
    return deduped


def build_query_plan(
    query: str,
    contract_id: str = CONTRACT_ID,
    user_classification: Optional[str] = None,
) -> QueryPlan:
    """
    Build deterministic query plan for retrieval orchestration.

    Roles/entities are extracted from contract-scoped role catalog and used as
    article anchors; semantic topics remain contract-domain concepts.
    """
    ensure_contract_manifest(contract_id)
    query_text = str(query or "")

    mentioned_classes = extract_classifications_for_contract(query_text, contract_id=contract_id, max_matches=4)
    primary_class = normalize_classification_for_contract(user_classification, contract_id=contract_id)
    if not primary_class:
        primary_class = extract_classification_for_contract(query_text, contract_id=contract_id)
    if not primary_class and mentioned_classes:
        primary_class = mentioned_classes[0]

    topic = extract_topic(query_text, contract_id=contract_id)
    if topic in _ROLE_LIKE_TOPICS:
        topic = None

    topic_article_map = get_topic_article_map(contract_id)
    class_article_map = get_classification_article_map(contract_id)
    side_letter_query = _is_side_letter_explicit_query(query_text) or _is_side_letter_followup_query(query_text)
    if side_letter_query:
        # For explicit/follow-up side-letter questions, prioritize side-letter
        # anchors over generic topic anchors (for example grievance Article 48).
        anchors = list(infer_side_letter_articles(contract_id)[:6])
    else:
        anchors = list(topic_article_map.get(topic, []) if topic else [])
    query_mentioned_classes = list(mentioned_classes)
    if not side_letter_query:
        classes_for_anchors = list(query_mentioned_classes)
        if primary_class and primary_class not in classes_for_anchors:
            classes_for_anchors.insert(0, primary_class)
        for cls in classes_for_anchors:
            anchors.extend(class_article_map.get(cls, []) or [])
    if _is_steward_union_meeting_schedule_query(query_text):
        anchors.append(45)

    explicit_articles: list[int] = []
    for match in re.findall(r"\b(?:article|art\.?)\s*(\d+)\b", query_text.lower()):
        try:
            explicit_articles.append(int(match))
        except ValueError:
            continue
    anchors.extend(explicit_articles)
    anchors = _normalize_article_list(anchors)

    comparison_mode = _is_role_comparison_query(query_text) and len(query_mentioned_classes) >= 2
    if comparison_mode and not anchors:
        anchors = infer_role_comparison_articles(contract_id)
    required_slots = _required_evidence_slots_for_plan(
        topic=topic,
        comparison_mode=comparison_mode,
        mentioned_classifications=query_mentioned_classes,
    )
    return QueryPlan(
        contract_id=contract_id,
        topic=topic,
        primary_classification=primary_class,
        mentioned_classifications=query_mentioned_classes,
        comparison_mode=comparison_mode,
        article_anchors=anchors,
        required_evidence_slots=required_slots,
        explicit_articles=_normalize_article_list(explicit_articles),
    )


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
        r"what (is|are) .+ (pay|wage|rate)",
        r"what (do|does) .+ (make|earn|get paid)",
        r"what .+ (pay|wage|rate) .+ for",
        r"what (is|are|'s) the .+ rate of pay",
        r"what should i (make|be making|earn|be earning)",
        r"\$\d+.*hour",  # Dollar amounts with hour
    ]

    for pattern in wage_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"pattern:{pattern}")

    # Keep this strict: legal prose often contains "rate ... for" but is not
    # asking for a personal wage lookup.
    if re.search(r"\b(pay|wage|hourly)\b", query_lower) and re.search(r"\b(for|as)\b", query_lower):
        matched.append("pattern:role_targeted_pay_for")

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
    plan = build_query_plan(
        query=query,
        contract_id=contract_id,
        user_classification=user_classification,
    )
    classification = plan.primary_classification
    topic = plan.topic
    relevant_articles = list(plan.article_anchors)
    classes_for_routing = list(plan.mentioned_classifications)
    
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
    if not is_wage and classes_for_routing:
        query_norm = _normalize_query_text(query)
        has_wage_token = bool(re.search(r"\b(pay|wage|rate|make|earn|paid|hourly)\b", query_norm))
        has_personal_signal = bool(re.search(r"\b(my|me|i)\b", query_norm))
        has_role_wage_question = bool(
            re.search(r"\b(what|how much)\b.*\b(do|does|is|are)\b.*\b(make|earn|paid|pay|wage|rate)\b", query_norm)
        )
        legal_clause_cue = bool(
            re.search(
                r"\b(shall|section|article|hereof|thereof|for all work performed|in addition to)\b",
                query_norm,
            )
        )
        if has_wage_token and (has_personal_signal or has_role_wage_question) and not legal_clause_cue:
            is_wage = True
            wage_matches = wage_matches + ["contextual_role_targeted_wage"]

    # Suppress wage routing for legal premium/overtime calculations unless the
    # user is explicitly asking about their own compensation.
    if is_wage and topic in WAGE_SUPPRESS_TOPICS:
        q_norm = _normalize_query_text(query)
        has_personal_comp_signal = bool(
            re.search(r"\b(my|me|i)\b", q_norm)
            and re.search(r"\b(pay|wage|salary|hourly|make|earn|rate)\b", q_norm)
        )
        has_explicit_wage_phrase = bool(
            re.search(
                r"\b(what (do|am|should) i (make|earn|be making)|what'?s my pay|my pay|my wage|my salary)\b",
                q_norm,
            )
        )
        if not has_personal_comp_signal and not has_explicit_wage_phrase:
            is_wage = False
            wage_matches = []

    # Global suppression for legal-clause quoting with wage-like words.
    if is_wage:
        q_norm = _normalize_query_text(query)
        has_personal_comp_signal = bool(
            re.search(r"\b(my|me|i)\b", q_norm)
            and re.search(r"\b(pay|wage|salary|hourly|make|earn|rate)\b", q_norm)
        )
        has_role_wage_question = bool(
            re.search(r"\b(what|how much)\b.*\b(do|does|is|are)\b.*\b(make|earn|paid|pay|wage|rate)\b", q_norm)
        )
        legal_clause_cue = bool(
            re.search(
                r"\b(shall|section|article|hereof|thereof|for all work performed|in addition to)\b",
                q_norm,
            )
        )
        if legal_clause_cue and not has_personal_comp_signal and not has_role_wage_question:
            is_wage = False
            wage_matches = []

    if is_wage and classes_for_routing:
        # Explicit role mention in the user query should override profile role
        # for wage targeting.
        classification = classes_for_routing[0]
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
            relevant_articles=relevant_articles,
            mentioned_classifications=classes_for_routing,
            comparison_mode=plan.comparison_mode,
            required_evidence_slots=plan.required_evidence_slots,
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
            relevant_articles=relevant_articles,
            mentioned_classifications=classes_for_routing,
            comparison_mode=plan.comparison_mode,
            required_evidence_slots=plan.required_evidence_slots,
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
        relevant_articles=relevant_articles,
        mentioned_classifications=classes_for_routing,
        comparison_mode=plan.comparison_mode,
        required_evidence_slots=plan.required_evidence_slots,
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
        self.entitlements_data = None
        self._entitlements_by_contract = {}
        self._all_chunks_by_contract = {}
        self._load_wages(CONTRACT_ID)
        self._load_entitlements(CONTRACT_ID)
    
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

    def _load_entitlements(self, contract_id: str = CONTRACT_ID):
        """Load entitlement data for a contract from JSON."""
        if contract_id in self._entitlements_by_contract:
            self.entitlements_data = self._entitlements_by_contract[contract_id]
            return

        entitlement_file = resolve_entitlement_file(contract_id=contract_id, allow_shared_fallback=True)
        if entitlement_file and entitlement_file.exists():
            with open(entitlement_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entitlements_by_contract[contract_id] = data
            self.entitlements_data = data
            return

        self._entitlements_by_contract[contract_id] = None
        self.entitlements_data = None
    
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

    def lookup_vacation_entitlement(
        self,
        months_employed: int = 0,
        hours_worked: int = 0,
        hire_date: Optional[str] = None,
        contract_id: str = CONTRACT_ID,
    ) -> Optional[dict]:
        """Look up deterministic vacation entitlement schedule data."""
        self._load_entitlements(contract_id=contract_id)
        if not self.entitlements_data:
            return None
        from backend.ingest.extract_entitlements import lookup_vacation_entitlement
        result = lookup_vacation_entitlement(
            self.entitlements_data,
            months_employed=months_employed,
            hours_worked=hours_worked,
            hire_date=hire_date,
        )
        if not result:
            return None
        if isinstance(self.entitlements_data, dict):
            result = dict(result)
            result["effective_version_id"] = (
                str(self.entitlements_data.get("effective_version_id") or "").strip() or None
            )
            result["amendments_applied"] = list(self.entitlements_data.get("amendments_applied") or [])
        return result
    
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

    @staticmethod
    def _side_letter_query_targets(query_text: Optional[str], seed_chunks: Optional[list[dict]] = None) -> dict:
        """
        Extract deterministic side-letter targeting hints from the query and
        nearby retrieved chunks (for CBA -> attached LOU cross-reference prompts).
        """
        query_raw = str(query_text or "")
        q_norm = _normalize_query_text(query_raw)

        explicit_type: Optional[str] = None
        explicit_number: Optional[int] = None
        m_lou = re.search(r"\b(?:letter(?:s)?\s+of\s+understanding|lou)\s+(\d{1,3})\b", q_norm)
        m_loa = re.search(r"\b(?:letter(?:s)?\s+of\s+agreement|loa)\s+(\d{1,3})\b", q_norm)
        if m_lou:
            explicit_type = "lou"
            explicit_number = int(m_lou.group(1))
        elif m_loa:
            explicit_type = "loa"
            explicit_number = int(m_loa.group(1))

        article_mentions: set[int] = set()
        for m in re.finditer(r"\barticle\s+(\d+)\b", q_norm):
            try:
                article_mentions.add(int(m.group(1)))
            except (TypeError, ValueError):
                continue

        # Focus terms: coarse lexical hints for picking the right LOU part.
        stop = {
            "what", "whats", "does", "the", "this", "that", "about", "with", "for", "from",
            "into", "your", "have", "tell", "show", "which", "where", "when", "will", "shall",
            "would", "could", "should", "there", "they", "them", "their", "agreement", "letter",
            "understanding", "letters", "part", "pursuant", "section", "article", "requirements",
        }
        query_terms = [
            tok for tok in re.findall(r"[a-z0-9]+", query_raw.lower())
            if len(tok) >= 4 and tok not in stop
        ]
        detail_prompt = bool(
            re.search(r"\b(wear|wearing|shirt|pants|shoes?|color|colours?|grooming|dress\s*code)\b", query_raw, flags=re.IGNORECASE)
        )
        existence_prompt = bool(
            re.search(r"\b(do\s+you\s+have|which\s+letter|what\s+letter|is\s+there)\b", query_raw, flags=re.IGNORECASE)
        )

        title_phrases: set[str] = set()
        query_lower = query_raw.lower()
        if "dress code" in query_lower:
            title_phrases.add("dress requirements")

        for chunk in list(seed_chunks or [])[:10]:
            blob = (
                f"{chunk.get('citation', '')}\n"
                f"{chunk.get('parent_context', '')}\n"
                f"{chunk.get('content_with_tables') or chunk.get('content') or ''}"
            )
            blob_lower = blob.lower()

            # Article cross-ref pattern: Letter of Understanding, "Dress Requirements,"
            for m in re.finditer(
                r'letter\s+of\s+(?:understanding|agreement)\s*,?\s*["“]([^"”]{3,120})["”]',
                blob,
                flags=re.IGNORECASE,
            ):
                phrase = _normalize_query_text(m.group(1))
                if len(phrase) >= 3:
                    title_phrases.add(phrase)

            # Side-letter citation title pattern.
            for m in re.finditer(
                r'letter\s+of\s+(?:understanding|agreement)\s+\d+\s*:\s*([^\n,]{3,140})',
                blob,
                flags=re.IGNORECASE,
            ):
                phrase = _normalize_query_text(m.group(1))
                if len(phrase) >= 3:
                    title_phrases.add(phrase)

            # Extract explicit number/type from retrieved side-letter citations if
            # the query is a generic follow-up and we already have a strong side-letter clue.
            if explicit_number is None and _is_side_letter_followup_query(query_raw):
                m_side = re.search(
                    r"letter of (understanding|agreement)\s+(\d{1,3})",
                    blob_lower,
                    flags=re.IGNORECASE,
                )
                if m_side:
                    explicit_type = "lou" if m_side.group(1).lower() == "understanding" else "loa"
                    try:
                        explicit_number = int(m_side.group(2))
                    except (TypeError, ValueError):
                        explicit_number = None

        return {
            "explicit_type": explicit_type,
            "explicit_number": explicit_number,
            "article_mentions": sorted(article_mentions),
            "title_phrases": sorted(p for p in title_phrases if len(p) >= 3),
            "query_terms": query_terms,
            "explicit_query": _is_side_letter_explicit_query(query_raw),
            "followup_query": _is_side_letter_followup_query(query_raw),
            "detail_prompt": detail_prompt,
            "existence_prompt": existence_prompt,
        }

    @staticmethod
    def _side_letter_chunk_score(
        chunk: dict,
        *,
        targets: dict,
        seed_articles: set[int],
    ) -> tuple[float, float]:
        """
        Return (bonus, lexical_overlap_score) for side-letter-aware ordering.
        """
        citation = str(chunk.get("citation") or "")
        parent = str(chunk.get("parent_context") or "")
        content = str(chunk.get("content_with_tables") or chunk.get("content") or "")
        blob = f"{citation}\n{parent}\n{content}"
        blob_lower = blob.lower()
        doc_type = _resolved_side_letter_doc_type(chunk)
        similarity = float(chunk.get("similarity", 0) or 0.0)

        explicit_type = targets.get("explicit_type")
        explicit_number = targets.get("explicit_number")
        title_phrases = list(targets.get("title_phrases") or [])
        query_terms = list(targets.get("query_terms") or [])
        article_mentions = set(targets.get("article_mentions") or [])
        explicit_query = bool(targets.get("explicit_query"))
        followup_query = bool(targets.get("followup_query"))
        detail_prompt = bool(targets.get("detail_prompt"))
        existence_prompt = bool(targets.get("existence_prompt"))

        bonus = 0.0
        lexical = 0.0
        norm_blob = _normalize_query_text(blob)

        part_num = None
        m_part = re.search(r"\bpart\s+(\d+)\b", citation, flags=re.IGNORECASE)
        if m_part:
            try:
                part_num = int(m_part.group(1))
            except (TypeError, ValueError):
                part_num = None
        is_headerish = bool(
            ("##" in str(content or ""))
            or (len(str(content or "")) <= 260 and any(p in norm_blob for p in title_phrases))
        )

        if doc_type in {"lou", "loa"}:
            bonus += 0.35
            if explicit_query or followup_query:
                bonus += 0.35

        if explicit_type and doc_type == explicit_type:
            bonus += 1.2

        if explicit_number is not None:
            if re.search(
                rf"\b(?:letter of (?:understanding|agreement)|lou|loa)\s+{int(explicit_number)}\b",
                blob_lower,
                flags=re.IGNORECASE,
            ):
                bonus += 4.0
            if re.search(rf"\bitem\s+{int(explicit_number)}\b", blob_lower):
                bonus += 2.0

        title_hits = sum(1 for phrase in title_phrases if phrase and phrase in norm_blob)
        if title_hits:
            if doc_type in {"lou", "loa"}:
                bonus += 2.0 + min(1.5, 0.5 * title_hits)
            else:
                # Preserve CBA cross-reference anchors but below actual side-letter text.
                bonus += 0.8 + min(0.6, 0.2 * title_hits)
        if doc_type in {"lou", "loa"} and title_hits and article_mentions and not detail_prompt:
            if is_headerish:
                bonus += 1.2
            if part_num == 1:
                bonus += 0.8

        if article_mentions:
            if doc_type in {"lou", "loa"} and any(f"article {a}" in blob_lower for a in article_mentions):
                bonus += 0.8
            try:
                article_num = int(chunk.get("article_num") or 0)
            except (TypeError, ValueError):
                article_num = 0
            if article_num and article_num in article_mentions and "letter of understanding" in blob_lower:
                bonus += 0.9

        overlap = sum(1 for tok in query_terms if tok in blob_lower)
        if overlap:
            lexical = 0.08 * float(overlap)
            if doc_type in {"lou", "loa"}:
                lexical += 0.12 * float(overlap)

        # If query is likely asking for side-letter details, prioritize content-rich
        # side-letter chunks over header/placeholder chunks.
        if doc_type in {"lou", "loa"} and overlap >= 2:
            bonus += 0.5
        if doc_type in {"lou", "loa"} and len(content) >= 300:
            bonus += 0.1
        if doc_type in {"lou", "loa"} and detail_prompt:
            if any(tok in blob_lower for tok in ("wear", "shirt", "pants", "shoe", "shoes", "color", "black", "white")):
                bonus += 1.4
            if part_num == 2:
                bonus += 0.9
        if doc_type in {"lou", "loa"} and (existence_prompt or (explicit_query and not detail_prompt)):
            if is_headerish:
                bonus += 1.5
            if part_num == 1:
                bonus += 1.0
            elif part_num and part_num > 3 and title_hits == 0 and overlap == 0:
                bonus -= 0.35

        # Keep existing high-confidence ranking signal in the ordering mix.
        return bonus + (0.15 * similarity), lexical

    def _promote_side_letter_context(
        self,
        chunks: list,
        *,
        contract_id: str = CONTRACT_ID,
        query_text: Optional[str] = None,
        n_results: int = 8,
        max_additional: int = 3,
    ) -> list:
        """
        Reorder/add side-letter chunks for explicit LOU/LOA prompts and CBA->LOU
        cross-reference prompts.

        This is deterministic and contract-scoped. It does not replace normal
        retrieval scoring; it adds a targeted post-pass for side-letter surfacing.
        """
        if not chunks:
            return chunks

        targets = self._side_letter_query_targets(query_text, seed_chunks=chunks)
        explicit_query = bool(targets.get("explicit_query"))
        followup_query = bool(targets.get("followup_query"))
        title_phrases = list(targets.get("title_phrases") or [])
        explicit_number = targets.get("explicit_number")
        explicit_type = targets.get("explicit_type")

        top_seed = list(chunks[: max(8, n_results)])
        has_side_letter_in_seed = any(
            _resolved_side_letter_doc_type(c) in {"lou", "loa"}
            for c in top_seed
        )
        has_side_letter_reference_in_seed = any(
            ("letter of understanding" in f"{c.get('citation', '')} {c.get('content_with_tables') or c.get('content') or ''}".lower())
            or ("letter of agreement" in f"{c.get('citation', '')} {c.get('content_with_tables') or c.get('content') or ''}".lower())
            for c in top_seed
        )

        should_engage = (
            explicit_query
            or followup_query
            or bool(title_phrases)
            or has_side_letter_in_seed
            or has_side_letter_reference_in_seed
        )
        if not should_engage:
            return chunks

        # Track seed article references for cross-ref anchoring (e.g., Article 52 §158).
        seed_articles: set[int] = set()
        for c in top_seed:
            try:
                article_int = int(c.get("article_num") or 0)
            except (TypeError, ValueError):
                article_int = 0
            if article_int > 0:
                seed_articles.add(article_int)

        existing_ids = {c.get("chunk_id", c.get("citation", "")) for c in chunks}
        enriched_chunks: list[dict] = list(chunks)

        # If we have explicit side-letter targeting or a harvested title phrase,
        # add a few matching side-letter chunks from the contract corpus when they
        # are not already present in the retrieved set.
        if explicit_query or followup_query or title_phrases or explicit_number is not None:
            contract_chunks = self._load_all_chunks_for_contract(contract_id)
            additions: list[tuple[float, dict]] = []
            for c in contract_chunks:
                cid = c.get("chunk_id", c.get("citation", ""))
                if cid in existing_ids:
                    continue
                doc_type = _resolved_side_letter_doc_type(c)
                if doc_type not in {"lou", "loa"}:
                    continue

                bonus, lexical = self._side_letter_chunk_score(
                    c,
                    targets=targets,
                    seed_articles=seed_articles,
                )
                blob_lower = (
                    f"{c.get('citation', '')}\n{c.get('parent_context', '')}\n"
                    f"{c.get('content_with_tables') or c.get('content') or ''}"
                ).lower()

                strong_match = False
                if explicit_number is not None and re.search(
                    rf"\b(?:letter of (?:understanding|agreement)|lou|loa)\s+{int(explicit_number)}\b",
                    blob_lower,
                    flags=re.IGNORECASE,
                ):
                    strong_match = True
                if not strong_match and title_phrases:
                    norm_blob = _normalize_query_text(blob_lower)
                    strong_match = any(p in norm_blob for p in title_phrases)
                if not strong_match and lexical >= 0.4:
                    strong_match = True
                if not strong_match:
                    continue

                score = float(bonus + lexical)
                additions.append((score, c))

            if additions:
                additions.sort(
                    key=lambda x: (
                        -float(x[0]),
                        -len(str(x[1].get("content_with_tables") or x[1].get("content") or "")),
                        str(x[1].get("citation") or ""),
                    )
                )
                for _, c in additions[: max(1, int(max_additional))]:
                    c_copy = dict(c)
                    resolved_doc_type = _resolved_side_letter_doc_type(c_copy)
                    if resolved_doc_type in {"lou", "loa"}:
                        c_copy["doc_type"] = resolved_doc_type
                    c_copy["similarity"] = max(float(c_copy.get("similarity", 0) or 0.0), 0.74)
                    c_copy["is_side_letter_seed"] = True
                    enriched_chunks.append(c_copy)
                    existing_ids.add(c_copy.get("chunk_id", c_copy.get("citation", "")))

        scored_rows: list[tuple[float, float, float, int, dict]] = []
        for idx, c in enumerate(enriched_chunks):
            bonus, lexical = self._side_letter_chunk_score(
                c,
                targets=targets,
                seed_articles=seed_articles,
            )
            base_similarity = float(c.get("similarity", 0) or 0.0)
            total = base_similarity + bonus + lexical
            c_copy = dict(c)
            resolved_doc_type = _resolved_side_letter_doc_type(c_copy)
            if resolved_doc_type in {"lou", "loa"}:
                c_copy["doc_type"] = resolved_doc_type
            if (bonus + lexical) >= 1.0 and resolved_doc_type in {"lou", "loa"}:
                c_copy["is_side_letter_seed"] = True
                c_copy["similarity"] = max(base_similarity, min(0.98, total))
            scored_rows.append((total, bonus, lexical, idx, c_copy))

        scored_rows.sort(
            key=lambda row: (
                -float(row[0]),
                -float(row[1]),
                -float(row[2]),
                row[3],  # stable original order tie-breaker
            )
        )
        return [row[4] for row in scored_rows]

    def _build_retrieval_policy(
        self,
        *,
        chunks: list,
        intent: QueryIntent,
        search_mode: str,
        doc_type_filter: Optional[str] = None,
        explicit_articles: Optional[list[int]] = None,
        query_expansions: Optional[list[str]] = None,
        executed_stages: Optional[list[str]] = None,
    ) -> dict:
        """Summarize the deterministic retrieval stages that actually fired."""
        normalized_chunks = list(chunks or [])
        article_anchors = _normalize_article_list(list(getattr(intent, "relevant_articles", []) or []))
        anchor_set = set(article_anchors)
        explicit_article_list = _normalize_article_list(list(explicit_articles or []))
        executed_stage_list = [str(stage).strip() for stage in (executed_stages or []) if str(stage).strip()]
        topic_seed_count = sum(1 for chunk in normalized_chunks if chunk.get("is_topic_seed"))
        side_letter_seed_count = sum(1 for chunk in normalized_chunks if chunk.get("is_side_letter_seed"))
        anchor_hits = {
            int(chunk.get("article_num"))
            for chunk in normalized_chunks
            if isinstance(chunk.get("article_num"), int) and int(chunk.get("article_num")) in anchor_set
        }
        strategy = search_mode
        if explicit_article_list:
            strategy = "explicit_article_fetch"
        elif side_letter_seed_count:
            strategy = "side_letter_promoted"
        elif topic_seed_count:
            strategy = "topic_article_seeded"
        elif doc_type_filter:
            strategy = "doc_type_filtered"
        return RetrievalPolicyRecord(
            strategy=strategy,
            search_mode=search_mode,
            article_anchors=article_anchors,
            article_anchor_count=len(article_anchors),
            article_anchor_hits=len(anchor_hits),
            topic_seeded=topic_seed_count > 0,
            topic_seed_count=topic_seed_count,
            side_letter_seeded=side_letter_seed_count > 0,
            side_letter_seed_count=side_letter_seed_count,
            explicit_articles_fetched=explicit_article_list,
            doc_type_filter=str(doc_type_filter or "").strip().lower() or None,
            query_expansion_count=len(list(query_expansions or [])),
            interpreter_used=search_mode == "multi_angle_interpreted",
            executed_stages=executed_stage_list,
            executed_stage_count=len(executed_stage_list),
        ).to_dict()

    def _build_retrieval_plan(
        self,
        *,
        intent: QueryIntent,
        search_mode: str,
        use_hybrid: bool,
        explicit_articles: Optional[list[int]] = None,
        query_expansions: Optional[list[str]] = None,
        doc_type_filter: Optional[str] = None,
        supports_side_letter_filter: bool = False,
        lou_detected: bool = False,
        loa_detected: bool = False,
        side_letter_detected: bool = False,
    ) -> dict:
        """Describe the deterministic retrieval stages planned before execution."""
        article_anchors = _normalize_article_list(list(getattr(intent, "relevant_articles", []) or []))
        explicit_article_list = _normalize_article_list(list(explicit_articles or []))
        planned_strategy = search_mode
        if explicit_article_list:
            planned_strategy = "explicit_article_fetch"
        elif lou_detected or loa_detected or side_letter_detected:
            planned_strategy = "side_letter_query"
        elif article_anchors:
            planned_strategy = "topic_anchor_guided"
        elif doc_type_filter:
            planned_strategy = "doc_type_filtered"
        return RetrievalPlanRecord(
            planned_strategy=planned_strategy,
            search_mode=search_mode,
            use_hybrid=bool(use_hybrid),
            use_interpreter=search_mode == "multi_angle_interpreted",
            intent_type=str(getattr(intent, "intent_type", "") or ""),
            topic=str(getattr(intent, "topic", "") or "") or None,
            article_anchors=article_anchors,
            article_anchor_count=len(article_anchors),
            explicit_articles_requested=explicit_article_list,
            explicit_article_request_count=len(explicit_article_list),
            query_expansion_count=len(list(query_expansions or [])),
            doc_type_filter=str(doc_type_filter or "").strip().lower() or None,
            side_letter_filter_supported=bool(supports_side_letter_filter),
            side_letter_query_detected=bool(lou_detected or loa_detected or side_letter_detected),
            apply_topic_seed_coverage=bool(article_anchors),
            apply_article_prioritization=bool(article_anchors),
            apply_side_letter_promotion=bool(lou_detected or loa_detected or side_letter_detected or explicit_article_list),
            apply_full_article_expansion=True,
            apply_related_section_expansion=True,
            apply_vacation_entitlement_coverage=str(getattr(intent, "topic", "") or "").strip().lower() == "vacation",
            apply_holiday_premium_coverage=True,
        ).to_dict()

    def _execute_retrieval_plan(
        self,
        *,
        chunks: list,
        plan: Optional[dict],
        contract_id: str = CONTRACT_ID,
        query_text: Optional[str] = None,
        n_results: int = 8,
        related_section_max_total: Optional[int] = None,
    ) -> tuple[list, list[str]]:
        """Execute the deterministic post-retrieval stages described by a plan."""
        working_chunks = list(chunks or [])
        plan_data = dict(plan or {})
        article_anchors = _normalize_article_list(list(plan_data.get("article_anchors") or []))
        topic = str(plan_data.get("topic") or "").strip().lower() or None
        executed_stages: list[str] = []

        if plan_data.get("apply_topic_seed_coverage"):
            working_chunks = self._ensure_topic_article_coverage(
                working_chunks,
                article_anchors,
                contract_id=contract_id,
                max_additional=3,
                query_text=query_text,
            )
            executed_stages.append("topic_seed_coverage")

        if plan_data.get("apply_full_article_expansion"):
            working_chunks = self._expand_to_full_article(
                working_chunks,
                contract_id=contract_id,
                n_results=n_results,
                preferred_articles=article_anchors,
            )
            executed_stages.append("full_article_expansion")

        if plan_data.get("apply_related_section_expansion"):
            max_total = related_section_max_total or max(len(working_chunks) + 3, n_results + 6)
            working_chunks = self._expand_with_related_sections(
                working_chunks,
                contract_id=contract_id,
                max_total=max_total,
                preferred_articles=article_anchors,
            )
            executed_stages.append("related_section_expansion")

        if plan_data.get("apply_vacation_entitlement_coverage"):
            working_chunks = self._ensure_vacation_entitlement_coverage(
                working_chunks,
                contract_id=contract_id,
                preferred_articles=article_anchors,
                topic=topic,
                query_text=query_text,
            )
            executed_stages.append("vacation_entitlement_coverage")

        if plan_data.get("apply_holiday_premium_coverage"):
            working_chunks = self._ensure_holiday_work_premium_coverage(
                working_chunks,
                contract_id=contract_id,
                query_text=query_text,
            )
            executed_stages.append("holiday_premium_coverage")

        if plan_data.get("apply_article_prioritization"):
            working_chunks = self._prioritize_topic_articles(
                working_chunks,
                article_anchors,
                topic=topic,
                query_text=query_text,
            )
            executed_stages.append("article_prioritization")

        if plan_data.get("apply_side_letter_promotion"):
            working_chunks = self._promote_side_letter_context(
                working_chunks,
                contract_id=contract_id,
                query_text=query_text,
                n_results=n_results,
            )
            executed_stages.append("side_letter_promotion")

        return working_chunks, executed_stages

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
        topic_key = str(topic or "").strip().lower()
        signals = _TOPIC_LEXICAL_SIGNALS.get(topic_key, ())
        title_hints = _TOPIC_ARTICLE_TITLE_HINTS.get(topic_key, ())
        query_lower = (query_text or "").lower()
        # Only apply lexical signal boosts when the query itself indicates the topic.
        apply_signal_boost = bool(signals and any(s in query_lower for s in signals))
        is_locator_query = bool(
            re.search(r"\b(where|which)\b", query_lower)
            and re.search(r"\b(defined|definition|rules?)\b", query_lower)
        )

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
            title_text = (
                f"{c.get('article_title', '')} "
                f"{c.get('citation', '')}"
            ).lower()
            title_hits = sum(1 for hint in title_hints if hint in title_text)
            if title_hits:
                bonus += min(0.36, 0.12 * title_hits)
                if is_locator_query:
                    bonus += 0.25
            if c.get("is_holiday_premium_seed"):
                bonus += 0.35
            if c.get("is_topic_seed"):
                bonus += 0.12
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
        wage_info: Optional[dict] = None,
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
        preferred_table_ids: set[str] = set()
        for row in (wage_info or {}).get("table_evidence", []) or []:
            table_id = str((row or {}).get("table_id") or "").strip()
            if table_id:
                preferred_table_ids.add(table_id)

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
            table_refs = {str(tid).strip() for tid in (chunk.get("table_refs") or []) if str(tid).strip()}

            # Require strong appendix/table signal.
            if doc_type != "appendix" and "appendix" not in citation.lower() and "table" not in citation.lower():
                continue

            token_hits = sum(1 for token in class_tokens if token in text)
            has_wage_grid_signal = "classification" in text and "$" in text
            has_preferred_table = bool(preferred_table_ids and table_refs.intersection(preferred_table_ids))
            if token_hits == 0 and not has_wage_grid_signal and not has_preferred_table:
                continue

            score = 0.0
            if has_preferred_table:
                score += 4.0
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

        return additions + chunks

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
            "years of service",
            "weeks of vacation",
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

    @staticmethod
    def _is_holiday_work_premium_query(query_text: Optional[str]) -> bool:
        """Detect holiday-work premium questions that need premium-rule coverage."""
        q = _normalize_query_text(query_text or "")
        if not q:
            return False
        return bool(re.search(HOLIDAY_WORK_PREMIUM_QUERY_PATTERN, q))

    @staticmethod
    def _query_targets_post_2005_holiday_rate(query_text: Optional[str]) -> bool:
        q = _normalize_query_text(query_text or "")
        if not q:
            return False
        return bool(
            re.search(
                r"\b(post|after|on or after)\b.*\b2005\b"
                r"|\bpost\s*2005\b"
                r"|\bon or after march\b",
                q,
            )
        )

    @staticmethod
    def _infer_tenure_from_query(query_text: Optional[str]) -> tuple[int, int]:
        """
        Infer hours/months tenure hints embedded directly in the query.

        This is a deterministic fallback only when explicit caller values are
        missing.
        """
        q = _normalize_query_text(query_text or "")
        if not q:
            return 0, 0

        hours = 0
        months = 0

        hour_match = re.search(r"\b(?:after|at|with)\s+(\d{2,6})\s+hours?\b", q)
        if not hour_match:
            hour_match = re.search(r"\b(\d{2,6})\s+hours?\b", q)
        if hour_match:
            try:
                hours = int(hour_match.group(1))
            except (TypeError, ValueError):
                hours = 0

        month_match = re.search(r"\b(?:after|at|with)\s+(\d{1,3})\s+months?\b", q)
        if not month_match:
            month_match = re.search(r"\b(\d{1,3})\s+months?\b", q)
        if month_match:
            try:
                months = int(month_match.group(1))
            except (TypeError, ValueError):
                months = 0

        return max(0, hours), max(0, months)

    def _ensure_holiday_work_premium_coverage(
        self,
        chunks: list,
        contract_id: str = CONTRACT_ID,
        query_text: Optional[str] = None,
    ) -> list:
        """
        Ensure holiday-work premium queries include premium-rule evidence.

        Holiday prompts can over-retrieve eligibility/scheduling sections
        while missing the premium-pay section. This adds one deterministic
        seed chunk when holiday-work premium markers are absent.
        """
        if not chunks:
            return chunks
        if not self._is_holiday_work_premium_query(query_text):
            return chunks
        query_targets_post_2005 = self._query_targets_post_2005_holiday_rate(query_text)

        premium_markers = (
            "when a holiday is worked",
            "per hour worked",
            "one and one-half",
            "1.00",
            "premium pay for holiday work",
        )
        strong_markers = (
            "per hour worked",
            "one and one-half",
            "premium pay for holiday work",
        )

        def _norm_text(chunk: dict) -> str:
            text = (
                f"{chunk.get('citation', '')} "
                f"{chunk.get('article_title', '')} "
                f"{chunk.get('content_with_tables') or chunk.get('content') or ''}"
            )
            return re.sub(r"[^a-z0-9.]+", " ", text.lower())

        for c in chunks:
            text = _norm_text(c)
            if "holiday" not in text:
                continue
            if "worked" not in text and "work" not in text:
                continue
            marker_hits = sum(1 for marker in premium_markers if marker in text)
            has_strong = any(marker in text for marker in strong_markers)
            if marker_hits >= 2 and has_strong:
                if query_targets_post_2005 and "2005" not in text:
                    continue
                promoted = dict(c)
                promoted["similarity"] = max(0.96, float(promoted.get("similarity", 0) or 0))
                promoted["is_holiday_premium_seed"] = True
                remaining = [row for row in chunks if row is not c]
                return [promoted] + remaining

        contract_chunks = self._load_all_chunks_for_contract(contract_id)
        if not contract_chunks:
            return chunks

        query_tokens = [
            tok for tok in re.findall(r"[a-z0-9]+", (query_text or "").lower())
            if len(tok) >= 3
        ]
        existing_ids = {c.get("chunk_id", c.get("citation", "")) for c in chunks}

        candidates: list[tuple[float, dict]] = []
        for c in contract_chunks:
            cid = c.get("chunk_id", c.get("citation", ""))
            if cid in existing_ids:
                continue
            text = _norm_text(c)
            if "holiday" not in text:
                continue
            if "worked" not in text and "work" not in text:
                continue

            marker_hits = sum(1 for marker in premium_markers if marker in text)
            if marker_hits == 0:
                continue
            if not any(marker in text for marker in strong_markers):
                continue
            if query_targets_post_2005 and "2005" not in text:
                continue
            overlap = sum(1 for tok in query_tokens if tok in text) if query_tokens else 0
            has_legacy_cutoff = bool(re.search(r"on or after march|march\s+\d+,\s*2005", text))
            has_rate_amount = bool("$1.00" in text or "one dollar" in text or "1.00" in text)
            section_num = int(c.get("section_num") or 0)
            score = float(marker_hits) + (0.05 * overlap) + (0.2 if section_num > 0 else 0.0)
            if has_rate_amount:
                score += 1.25
            if query_targets_post_2005 and has_legacy_cutoff:
                score += 2.5
            candidates.append((score, c))

        if not candidates:
            return chunks

        candidates.sort(key=lambda x: (x[0], -int(x[1].get("section_num") or 0)), reverse=True)
        seed = dict(candidates[0][1])
        seed["similarity"] = max(0.96, float(seed.get("similarity", 0) or 0))
        seed["is_holiday_premium_seed"] = True
        return [seed] + chunks

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

        # Detect explicit side-letter keywords in query.
        # Keep this narrow to avoid clashing with generic agreement phrasing.
        lou_keywords = ['letter of understanding', 'letters of understanding', ' lou ']
        loa_keywords = ['letter of agreement', 'letters of agreement']
        side_letter_keywords = ['side letter', 'side-letter', 'sideletter']
        query_lower = query.lower()
        query_padded = f" {query_lower} "
        lou_detected = any(kw in query_padded for kw in lou_keywords)
        loa_detected = any(kw in query_padded for kw in loa_keywords)
        side_letter_detected = any(kw in query_padded for kw in side_letter_keywords)

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
            "entitlement_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "contract_id": contract_id,
            "query_expansions": expansions,  # Track what slang was expanded
            "hypothesis_result": hypothesis_result,  # Track hypothesis for metrics
            "retrieval_plan": None,
            "retrieval_policy": None,
        }

        # Use hybrid search (vector + BM25 with RRF)
        # Weights configured in config.py (default: equal 1.0/1.0 for balanced fusion)
        doc_type_filter = None
        supports_side_letter_filter = contract_supports_side_letter_doc_type_filter(contract_id)
        if supports_side_letter_filter and lou_detected and not loa_detected and not side_letter_detected:
            doc_type_filter = "lou"
        elif supports_side_letter_filter and loa_detected and not lou_detected and not side_letter_detected:
            doc_type_filter = "loa"
        result["retrieval_plan"] = self._build_retrieval_plan(
            intent=intent,
            search_mode="single_angle_hybrid" if use_hybrid else "single_angle_vector",
            use_hybrid=use_hybrid,
            explicit_articles=[],
            query_expansions=expansions,
            doc_type_filter=doc_type_filter,
            supports_side_letter_filter=supports_side_letter_filter,
            lou_detected=lou_detected,
            loa_detected=loa_detected,
            side_letter_detected=side_letter_detected,
        )

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

        # ===== PHASE 2: TITLE BOOSTING =====
        # Boost chunks whose article_title matches hypothesized titles
        if hypothesis_result and hypothesis_result.success:
            chunks = apply_title_boosting(
                chunks,
                hypothesis_result.hypothesized_titles
            )
        # ===== END TITLE BOOSTING =====

        retrieval_plan = dict(result.get("retrieval_plan") or {})
        chunks, executed_stages = self._execute_retrieval_plan(
            chunks=chunks,
            plan=retrieval_plan,
            contract_id=contract_id,
            query_text=query,
            n_results=n_results,
            related_section_max_total=max(len(chunks) + 3, n_results + 6),
        )

        chunks = _normalize_resolved_doc_types(chunks)
        result["chunks"] = chunks
        result["retrieval_policy"] = self._build_retrieval_policy(
            chunks=chunks,
            intent=intent,
            search_mode="single_angle_hybrid" if use_hybrid else "single_angle_vector",
            doc_type_filter=doc_type_filter,
            explicit_articles=[],
            query_expansions=expansions,
            executed_stages=executed_stages,
        )
        inferred_hours, inferred_months = self._infer_tenure_from_query(query)
        resolved_hours = int(hours_worked or 0) if int(hours_worked or 0) > 0 else inferred_hours
        resolved_months = int(months_employed or 0) if int(months_employed or 0) > 0 else inferred_months

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=resolved_hours,
                months_employed=resolved_months,
                contract_id=contract_id,
            )
            result["wage_info"] = wage_info
            if wage_info:
                result["chunks"] = self._ensure_wage_table_context(
                    result["chunks"],
                    classification=intent.classification,
                    wage_info=wage_info,
                    contract_id=contract_id,
                )

        if (
            str(intent.topic or "").strip().lower() == "vacation"
            and self._is_vacation_entitlement_query(query)
        ):
            result["entitlement_info"] = self.lookup_vacation_entitlement(
                months_employed=resolved_months,
                hours_worked=resolved_hours,
                hire_date=None,
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

        retrieval_plan = self._build_retrieval_plan(
            intent=intent,
            search_mode="multi_angle_interpreted",
            use_hybrid=True,
            explicit_articles=interpretation.explicit_articles,
            query_expansions=search_queries,
        )
        final_chunks, executed_stages = self._execute_retrieval_plan(
            chunks=final_chunks,
            plan=retrieval_plan,
            contract_id=contract_id,
            query_text=query,
            n_results=n_results,
            related_section_max_total=max(len(final_chunks) + 4, n_results + 4),
        )

        # Build result
        final_chunks = _normalize_resolved_doc_types(final_chunks)
        result = {
            "chunks": final_chunks,
            "wage_info": None,
            "entitlement_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "contract_id": contract_id,
            "query_expansions": [],
            "interpretation": interpretation,  # Include interpretation for debugging
            "search_angles_used": len(search_queries),
            "explicit_articles_fetched": interpretation.explicit_articles,
            "reranker_result": reranker_result,  # Include reranker metrics
            "retrieval_plan": retrieval_plan,
            "retrieval_policy": self._build_retrieval_policy(
                chunks=final_chunks,
                intent=intent,
                search_mode="multi_angle_interpreted",
                explicit_articles=interpretation.explicit_articles,
                query_expansions=search_queries,
                executed_stages=executed_stages,
            ),
        }
        inferred_hours, inferred_months = self._infer_tenure_from_query(query)
        resolved_hours = int(hours_worked or 0) if int(hours_worked or 0) > 0 else inferred_hours
        resolved_months = int(months_employed or 0) if int(months_employed or 0) > 0 else inferred_months

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=resolved_hours,
                months_employed=resolved_months,
                contract_id=contract_id,
            )
            result["wage_info"] = wage_info
            if wage_info:
                result["chunks"] = self._ensure_wage_table_context(
                    result["chunks"],
                    classification=intent.classification,
                    wage_info=wage_info,
                    contract_id=contract_id,
                )

        if (
            str(intent.topic or "").strip().lower() == "vacation"
            and self._is_vacation_entitlement_query(query)
        ):
            result["entitlement_info"] = self.lookup_vacation_entitlement(
                months_employed=resolved_months,
                hours_worked=resolved_hours,
                hire_date=None,
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
