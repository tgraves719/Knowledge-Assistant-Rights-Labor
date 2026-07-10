"""
Deterministic language lexicon + concept enrichment for ingestion.

Design:
- Offline probabilistic sources can augment artifacts later.
- Runtime remains deterministic by reading frozen artifacts only.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_STOPWORDS = {
    "the", "and", "for", "from", "with", "without", "into", "onto", "this", "that",
    "shall", "will", "must", "may", "all", "any", "each", "per", "are", "is",
    "article", "section", "part", "agreement", "collective", "bargaining",
    "employee", "employees", "employer", "union", "local",
}

_NON_GEO_TOKENS = {
    "local", "ufcw", "safeway", "kingsoopers", "kroger", "clerks", "clerk", "meat",
    "department", "departments", "contract", "agreement", "retail", "store",
    "stores", "unit", "units",
}

# Canonical concept -> deterministic alias set.
_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "overtime": ("overtime", "ot", "extra hours", "time and a half"),
    "vacation": ("vacation", "vacation time", "vacation pay", "time off"),
    "personal_holiday": ("personal holiday", "personal holidays", "float day", "float days", "floater"),
    "sick_leave": ("sick leave", "sick pay", "sick day", "call in sick"),
    "breaks": ("break", "breaks", "rest break", "relief period", "lunch break", "meal period"),
    "wages": ("wage", "wages", "pay", "pay rate", "hourly rate", "rate of pay"),
    "seniority": ("seniority", "years of service", "service time"),
    "grievance": ("grievance", "dispute", "arbitration", "appeal"),
    "discipline": ("discipline", "warning", "write up", "written up", "discharge", "termination", "fired"),
    "layoff": ("layoff", "laid off", "bumping", "displacement"),
    "health_benefits": ("health benefits", "health and welfare", "insurance", "medical"),
    "pension": ("pension", "retirement"),
    "safety": ("safety", "unsafe", "injury", "accident"),
    "scheduling": ("schedule", "scheduling", "shift", "hours", "posting"),
}

_ARTICLE_TITLE_ALIAS_HINTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("term of agreement",),
        (
            "contract term",
            "start and end",
            "start date",
            "end date",
            "expiration date",
            "effective date",
            "when does this contract end",
        ),
    ),
    (
        ("sunday premium",),
        (
            "sunday pay",
            "sundays",
            "working sundays",
            "extra pay sunday",
            "sunday differential",
        ),
    ),
    (
        ("night premium", "night premiums"),
        (
            "night premium",
            "night shift pay",
            "night differential",
            "graveyard shift",
            "graveyard pay",
            "extra money at night",
        ),
    ),
    (
        ("store closing",),
        (
            "store closes",
            "store is shutting down",
            "store shutting down",
            "severance",
            "severance pay",
        ),
    ),
    (
        ("bereavement leave",),
        (
            "funeral leave",
            "funeral",
            "death in family",
            "died",
            "days off for funeral",
        ),
    ),
    (
        ("lunch breaks", "relief periods"),
        (
            "break between shifts",
            "hours between shifts",
            "close and open",
            "clopen",
            "turnaround shift",
        ),
    ),
]


def _norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: str, fallback: str = "unknown") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return text or fallback


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(value or "").lower())).strip()


def _iter_tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(t) >= 3]


def infer_region_id(contract_id: str, manifest: Optional[dict] = None) -> str:
    """Deterministically derive region_id when not explicitly present in manifest."""
    manifest = manifest or {}
    explicit = _norm_space(manifest.get("region_id"))
    if explicit:
        return _slug(explicit)

    # Prefer explicit location field if available.
    location = _norm_space(manifest.get("location"))
    if location:
        return _slug(location)

    # Fallback to contract id geo-like token.
    tokens = _iter_tokens(contract_id)
    geo_token = None
    for tok in tokens:
        if re.fullmatch(r"local\d+", tok):
            continue
        if tok in _NON_GEO_TOKENS:
            continue
        if re.fullmatch(r"\d{4}", tok):
            continue
        geo_token = tok
        break

    if geo_token:
        return f"region-{geo_token}"
    return f"region-{_slug(contract_id)}"


def ensure_manifest_region_id(manifest: dict, contract_id: str) -> dict:
    out = dict(manifest or {})
    out["region_id"] = infer_region_id(contract_id=contract_id, manifest=out)
    return out


def _extract_title_terms(article_title: str) -> list[str]:
    title = _normalize_phrase(article_title)
    if not title:
        return []
    tokens = [t for t in title.split() if t not in _STOPWORDS and len(t) >= 3]
    terms: list[str] = []
    # Avoid single-token aliases from article titles; they are often too broad
    # (e.g., "store", "pay") and create noisy query rewrites.
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram not in terms:
            terms.append(bigram)
    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        if trigram not in terms:
            terms.append(trigram)
    return terms[:10]


def _derive_article_title_aliases(article_title: str) -> list[str]:
    title = _normalize_phrase(article_title)
    if not title:
        return []
    aliases = {title}
    aliases.update(_extract_title_terms(title))
    for triggers, extra_aliases in _ARTICLE_TITLE_ALIAS_HINTS:
        if any(t in title for t in triggers):
            aliases.update(_normalize_phrase(a) for a in extra_aliases)
    return [a for a in sorted(aliases) if a]


def _concept_hits(text: str) -> list[str]:
    lowered = _normalize_phrase(text)
    tokens = set(lowered.split())
    hits: list[str] = []
    for concept, aliases in _CANONICAL_ALIASES.items():
        matched = False
        for alias in aliases:
            a = _normalize_phrase(alias)
            if not a:
                continue
            if " " in a:
                if f" {a} " in f" {lowered} ":
                    matched = True
                    break
            else:
                if a in tokens:
                    matched = True
                    break
        if matched:
            hits.append(concept)
    return hits


def _canonical_questions_for(concept: str) -> list[str]:
    templates = {
        "overtime": ["Do I get overtime for extra hours?"],
        "vacation": ["How much vacation time do I get?"],
        "personal_holiday": ["How many float days do I get?"],
        "sick_leave": ["Can I use sick time for this absence?"],
        "breaks": ["When do I get a break?"],
        "wages": ["How much should I be making per hour?"],
        "seniority": ["How does seniority affect my rights?"],
        "grievance": ["How do I file a grievance?"],
        "discipline": ["Can I be written up or fired for this?"],
        "layoff": ["What are my rights if I get laid off?"],
        "health_benefits": ["When do my health benefits start?"],
        "pension": ["How does this affect my pension?"],
        "safety": ["What can I do if work is unsafe?"],
        "scheduling": ["Can they change my schedule like this?"],
    }
    return templates.get(concept, [])


def _sanitize_list(values, max_items: int = 12) -> list[str]:
    seen = set()
    out: list[str] = []
    for raw in values or []:
        v = _normalize_phrase(raw)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= max_items:
            break
    return out


def derive_chunk_language_features(
    chunk: dict,
    manifest: Optional[dict] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (alternative_names, worker_questions, concept_hits)."""
    manifest = manifest or {}
    article_title = str(chunk.get("article_title") or "")
    content = str(chunk.get("content_with_tables") or chunk.get("content") or "")
    citation = str(chunk.get("citation") or "")
    summary = str(chunk.get("summary") or "")
    text = f"{article_title}\n{citation}\n{summary}\n{content[:2400]}"

    base_alts = []
    base_alts.extend(chunk.get("alternative_names") or [])
    base_alts.extend(_extract_title_terms(article_title))

    hits = _concept_hits(text)
    for concept in hits:
        base_alts.extend(_CANONICAL_ALIASES.get(concept, ()))

    # classification names from manifest that appear in chunk text
    lowered = _normalize_phrase(text)
    for cls in manifest.get("classifications", []) or []:
        cls_norm = _normalize_phrase(cls)
        if cls_norm and cls_norm in lowered:
            base_alts.append(cls_norm)

    alternative_names = _sanitize_list(base_alts, max_items=14)

    base_questions = []
    base_questions.extend(chunk.get("worker_questions") or [])
    for concept in hits:
        base_questions.extend(_canonical_questions_for(concept))

    if not base_questions:
        topic = alternative_names[0] if alternative_names else "this section"
        if citation:
            base_questions.append(f"What does {citation.lower()} say about {topic}?")
        else:
            base_questions.append(f"What does the contract say about {topic}?")

    worker_questions = _sanitize_list(base_questions, max_items=5)
    return alternative_names, worker_questions, hits


def apply_deterministic_language_enrichment(
    chunks: list[dict],
    contract_id: str,
    manifest: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    """
    Backfill deterministic language metadata on chunks.

    Ensures `region_id`, `alternative_names`, and `worker_questions` exist.
    """
    manifest = ensure_manifest_region_id(manifest or {}, contract_id=contract_id)
    region_id = manifest["region_id"]

    enriched: list[dict] = []
    stats = {
        "chunk_count": len(chunks or []),
        "region_id": region_id,
        "chunks_with_alternative_names": 0,
        "chunks_with_worker_questions": 0,
        "chunks_with_concept_hits": 0,
    }

    for chunk in chunks or []:
        c = dict(chunk)
        c["contract_id"] = str(c.get("contract_id") or contract_id)
        c["region_id"] = str(c.get("region_id") or region_id)

        alt_names, questions, hits = derive_chunk_language_features(c, manifest=manifest)
        c["alternative_names"] = alt_names
        c["worker_questions"] = questions
        if hits and not c.get("topics"):
            c["topics"] = sorted(hits)[:4]

        if alt_names:
            stats["chunks_with_alternative_names"] += 1
        if questions:
            stats["chunks_with_worker_questions"] += 1
        if hits:
            stats["chunks_with_concept_hits"] += 1
        enriched.append(c)

    return enriched, stats


def build_language_lexicon(
    chunks: list[dict],
    contract_id: str,
    manifest: Optional[dict] = None,
) -> dict:
    """Build frozen alias graph artifact for deterministic runtime lookups."""
    manifest = ensure_manifest_region_id(manifest or {}, contract_id=contract_id)
    region_id = manifest["region_id"]
    article_aliases: dict[int, set[str]] = defaultdict(set)
    canonical_to_aliases: dict[str, set[str]] = defaultdict(set)

    routing = (manifest.get("query_routing") or {}).get("slang_to_contract", {}) or {}
    for alias, canonical in routing.items():
        a = _normalize_phrase(alias)
        c = _normalize_phrase(str(canonical).replace("_", " "))
        if a and c:
            canonical_to_aliases[c].add(a)
            canonical_to_aliases[c].add(c)

    article_titles = (manifest.get("article_titles") or {}) if isinstance(manifest, dict) else {}
    for _, raw_title in sorted(article_titles.items(), key=lambda kv: str(kv[0])):
        title_norm = _normalize_phrase(raw_title)
        if not title_norm:
            continue
        for alias in _derive_article_title_aliases(title_norm):
            canonical_to_aliases[title_norm].add(alias)

    for chunk in chunks or []:
        article_num = chunk.get("article_num")
        try:
            article_num_int = int(article_num) if article_num is not None else None
        except (TypeError, ValueError):
            article_num_int = None

        alt_names = _sanitize_list(chunk.get("alternative_names") or [], max_items=20)
        concepts = _concept_hits(
            f"{chunk.get('article_title', '')}\n{chunk.get('content_with_tables') or chunk.get('content') or ''}"
        )
        for concept in concepts:
            for alias in _CANONICAL_ALIASES.get(concept, ()):
                canonical_to_aliases[concept].add(_normalize_phrase(alias))
            canonical_to_aliases[concept].add(_normalize_phrase(concept.replace("_", " ")))

        for alias in alt_names:
            for concept, aliases in _CANONICAL_ALIASES.items():
                if alias in aliases or alias == concept:
                    canonical_to_aliases[concept].add(alias)
            if article_num_int is not None:
                article_aliases[article_num_int].add(alias)

    entries = []
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in sorted(canonical_to_aliases.items()):
        alias_list = sorted(a for a in aliases if a)
        if not alias_list:
            continue
        entries.append({"canonical": canonical, "aliases": alias_list})
        canonical_phrase = _normalize_phrase(str(canonical).replace("_", " "))
        for alias in alias_list:
            alias_to_canonical[alias] = canonical_phrase

    return {
        "schema_version": "language_lexicon_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract_id,
        "region_id": region_id,
        "union_local_id": _norm_space(manifest.get("union_local")),
        "employer": _norm_space(manifest.get("employer")),
        "entries": entries,
        "alias_to_canonical": alias_to_canonical,
        "article_aliases": {
            str(k): sorted(v) for k, v in sorted(article_aliases.items())
        },
    }


def save_language_lexicon(path: Path, lexicon: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2, ensure_ascii=False)
    return path
