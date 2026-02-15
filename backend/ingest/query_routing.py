"""
Deterministic manifest query-routing synthesis.

Builds contract-scoped routing metadata from ingestion-owned artifacts:
- manifest article titles/classifications
- concept index concept/question maps
- frozen language lexicon alias graph
- classification ontology decisions (optional)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from backend.ingest.extract_wages import normalize_classification_name


_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "wages": ("wage", "wages", "rate of pay", "pay rate", "hourly rate", "appendix a"),
    "overtime": ("overtime", "ot", "time and a half", "unscheduled overtime"),
    "scheduling": ("schedule", "scheduling", "shift", "posting", "hours"),
    "seniority": ("seniority", "years of service", "service time"),
    "layoff": ("layoff", "laid off", "bumping", "displacement", "reduction of hours"),
    "personal_holiday": ("personal holiday", "personal holidays", "float day", "float days", "floater"),
    "vacation": ("vacation", "anniversary year", "paid vacation", "vacation roster"),
    "bereavement": ("bereavement", "bereavement leave", "funeral", "funeral leave", "death in family"),
    "sick_leave": ("sick leave", "sick pay", "sick day", "call in sick"),
    "discipline": ("discipline", "write up", "written up", "discharge", "termination", "fired"),
    "grievance": ("grievance", "arbitration", "dispute procedure", "appeal"),
    "breaks": ("lunch break", "relief period", "meal period", "rest period", "between shifts"),
    "premiums": ("premium", "sunday premium", "night premium", "holiday pay"),
    "weingarten": ("weingarten", "right to representation", "union representative"),
    "health_benefits": ("health benefits", "health and welfare", "insurance", "medical"),
    "promotion": ("promotion", "promoted", "demoted", "transferred"),
    "drive_up_go": ("drive up and go", "dug shopper", "clicklist", "personal shopper"),
    "probation": ("probation", "probationary", "trial period", "new employee"),
    "term": ("term of agreement", "contract term", "effective date", "expiration date", "start and end"),
}

_ROUTING_STOPWORDS = {
    "the", "and", "for", "to", "of", "in", "on", "at", "by", "from", "with", "without",
    "article", "section", "part", "agreement", "contract", "union", "local", "employee",
}
_GENERIC_SINGLE_TOKEN_PATTERN_TERMS = {
    "pay", "paid", "days", "day", "hours", "hour", "time", "work", "worker", "workers",
    "store", "manager", "employee", "employees", "rate", "rates",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(value or "").lower())).strip()


def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(value)) if len(t) >= 3}


def _to_int_articles(values) -> list[int]:
    seen = set()
    out: list[int] = []
    for v in values or []:
        try:
            a = int(v)
        except (TypeError, ValueError):
            continue
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    out.sort()
    return out


def _compact_phrase_list(values, max_items: int = 24) -> list[str]:
    seen = set()
    out: list[str] = []
    for raw in values or []:
        phrase = _norm(raw)
        if not phrase:
            continue
        if phrase in seen:
            continue
        if len(phrase) < 3:
            continue
        toks = re.findall(r"[a-z0-9]+", phrase)
        if len(toks) == 1:
            tok = toks[0]
            if tok in _GENERIC_SINGLE_TOKEN_PATTERN_TERMS:
                continue
            if len(tok) < 6:
                continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= max_items:
            break
    return out


def _phrase_to_regex(phrase: str) -> str:
    toks = re.findall(r"[a-z0-9]+", _norm(phrase))
    if not toks:
        return ""
    if len(toks) == 1:
        return rf"\b{re.escape(toks[0])}\b"
    joined = r"\s*".join(re.escape(t) for t in toks)
    return rf"\b{joined}\b"


def _matches_hint(text: str, hint: str) -> bool:
    t = _norm(text)
    h = _norm(hint)
    if not t or not h:
        return False
    if h in t:
        return True
    h_tokens = _tokens(h)
    t_tokens = _tokens(t)
    if not h_tokens or not t_tokens:
        return False
    return len(h_tokens & t_tokens) >= min(2, len(h_tokens))


def _build_index_map(raw_map, valid_articles: set[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if not isinstance(raw_map, dict):
        return out
    for key, values in raw_map.items():
        norm_key = _norm(str(key or ""))
        if not norm_key:
            continue
        articles = [a for a in _to_int_articles(values) if a in valid_articles]
        if articles:
            out[norm_key] = articles
    return out


def _score_topic_articles(
    hints: tuple[str, ...],
    article_titles: dict[int, str],
    concept_to_articles: dict[str, list[int]],
    question_to_articles: dict[str, list[int]],
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    title_matches: set[int] = set()

    for article_num, title in article_titles.items():
        title_norm = _norm(title)
        for hint in hints:
            if _matches_hint(title_norm, hint):
                scores[article_num] += 4.0
                title_matches.add(article_num)

    for concept, articles in concept_to_articles.items():
        for hint in hints:
            if _matches_hint(concept, hint):
                for a in articles:
                    scores[a] += 2.0

    for question, articles in question_to_articles.items():
        for hint in hints:
            if _matches_hint(question, hint):
                for a in articles:
                    scores[a] += 1.25

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    picked: list[int] = []
    if title_matches:
        for article_num, score in ranked:
            if score < 2.0:
                continue
            if article_num in title_matches:
                picked.append(article_num)
            if len(picked) >= 4:
                break
    for article_num, score in ranked:
        if len(picked) >= 4:
            break
        if score < 2.0 or article_num in picked:
            continue
        picked.append(article_num)
    if not picked and ranked:
        picked = [ranked[0][0]]
    return picked


def _topic_alias_pool(
    topic: str,
    hints: tuple[str, ...],
    selected_articles: list[int],
    article_titles: dict[int, str],
    lexicon_alias_to_canonical: dict[str, str],
) -> list[str]:
    aliases: list[str] = []
    aliases.extend(hints)
    for article_num in selected_articles:
        aliases.append(article_titles.get(article_num, ""))

    for alias, canonical in sorted(lexicon_alias_to_canonical.items()):
        canonical_norm = _norm(canonical)
        alias_norm = _norm(alias)
        if not alias_norm or not canonical_norm:
            continue
        if any(_matches_hint(canonical_norm, hint) for hint in hints):
            aliases.append(alias_norm)
        elif _matches_hint(canonical_norm, topic.replace("_", " ")):
            aliases.append(alias_norm)

    return _compact_phrase_list(aliases, max_items=22)


def _build_topic_patterns(
    topic_to_articles: dict[str, list[int]],
    article_titles: dict[int, str],
    lexicon_alias_to_canonical: dict[str, str],
) -> dict[str, str]:
    patterns: dict[str, str] = {}
    for topic, articles in sorted(topic_to_articles.items()):
        hints = _TOPIC_HINTS.get(topic, ())
        if not hints:
            continue
        pool = _topic_alias_pool(
            topic=topic,
            hints=hints,
            selected_articles=articles,
            article_titles=article_titles,
            lexicon_alias_to_canonical=lexicon_alias_to_canonical,
        )
        fragments = []
        for phrase in pool:
            frag = _phrase_to_regex(phrase)
            if frag:
                fragments.append(frag)
        fragments = list(dict.fromkeys(fragments))
        if len(fragments) < 2:
            continue
        patterns[topic] = "|".join(fragments[:18])
    return patterns


def _build_slang_map(lexicon_alias_to_canonical: dict[str, str], max_entries: int = 320) -> dict[str, str]:
    scored: list[tuple[float, str, str]] = []
    for alias, canonical in sorted((lexicon_alias_to_canonical or {}).items()):
        a = _norm(alias)
        c = _norm(canonical)
        if not a or not c or a == c:
            continue
        if a in _ROUTING_STOPWORDS:
            continue
        if len(a) < 3:
            continue
        a_tokens = [t for t in re.findall(r"[a-z0-9]+", a) if t]
        if len(a_tokens) == 1 and len(a_tokens[0]) < 5:
            continue
        score = 0.0
        if len(a_tokens) > 1:
            score += 2.0
        if len(c.split()) > 1:
            score += 1.0
        if len(a) >= 8:
            score += 0.5
        scored.append((score, a, c))

    scored.sort(key=lambda row: (-row[0], row[1]))
    out: dict[str, str] = {}
    for _, alias, canonical in scored[:max_entries]:
        out[alias] = canonical
    return out


def _build_classification_article_map(
    manifest: dict,
    concept_to_articles: dict[str, list[int]],
    topic_to_articles: dict[str, list[int]],
    classification_ontology: Optional[dict],
    valid_articles: set[int],
) -> dict[str, list[int]]:
    classes = list(manifest.get("classifications") or [])
    if not classes and isinstance(classification_ontology, dict):
        classes = [c.get("normalized") for c in (classification_ontology.get("manifest_classifications") or []) if c.get("normalized")]

    title_map = {int(k): str(v or "") for k, v in (manifest.get("article_titles") or {}).items() if str(k).isdigit()}
    wage_articles = {
        a for a, title in title_map.items()
        if any(_matches_hint(title, hint) for hint in _TOPIC_HINTS["wages"])
    }
    wage_articles.update(_to_int_articles(topic_to_articles.get("wages")))
    wage_articles = {a for a in wage_articles if a in valid_articles}

    out: dict[str, list[int]] = {}
    for raw in classes:
        key = normalize_classification_name(str(raw or ""))
        if not key:
            continue
        aliases = {
            _norm(str(raw or "")),
            _norm(key.replace("_", " ")),
        }
        scores: dict[int, float] = defaultdict(float)

        for concept, articles in concept_to_articles.items():
            for alias in aliases:
                if not alias:
                    continue
                if _matches_hint(concept, alias):
                    for article_num in articles:
                        if article_num in valid_articles:
                            scores[article_num] += 1.5

        for article_num in wage_articles:
            scores[article_num] += 2.0

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        picked = [a for a, score in ranked if score >= 2.0][:4]
        if not picked and wage_articles:
            picked = sorted(wage_articles)[:2]
        if picked:
            out[key] = picked

    return out


def synthesize_query_routing(
    manifest: dict,
    concept_index: Optional[dict] = None,
    language_lexicon: Optional[dict] = None,
    classification_ontology: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    Build deterministic query-routing payload for manifest.

    Returns:
        (routing_payload, stats)
    """
    manifest = manifest or {}
    article_titles_raw = manifest.get("article_titles") or {}
    article_titles: dict[int, str] = {}
    for raw_key, raw_title in article_titles_raw.items():
        try:
            article_num = int(raw_key)
        except (TypeError, ValueError):
            continue
        article_titles[article_num] = str(raw_title or "")
    valid_articles = set(article_titles.keys())

    concept_payload = concept_index or {}
    concept_to_articles = _build_index_map(
        (concept_payload or {}).get("concept_to_articles", {}),
        valid_articles=valid_articles,
    )
    question_to_articles = _build_index_map(
        (concept_payload or {}).get("question_to_articles", {}),
        valid_articles=valid_articles,
    )

    topic_to_articles: dict[str, list[int]] = {}
    for topic, hints in _TOPIC_HINTS.items():
        picked = _score_topic_articles(
            hints=hints,
            article_titles=article_titles,
            concept_to_articles=concept_to_articles,
            question_to_articles=question_to_articles,
        )
        if picked:
            topic_to_articles[topic] = picked

    alias_to_canonical = {}
    if isinstance(language_lexicon, dict):
        alias_to_canonical = language_lexicon.get("alias_to_canonical") or {}
        if not isinstance(alias_to_canonical, dict):
            alias_to_canonical = {}

    slang_to_contract = _build_slang_map(alias_to_canonical)
    topic_patterns = _build_topic_patterns(
        topic_to_articles=topic_to_articles,
        article_titles=article_titles,
        lexicon_alias_to_canonical=alias_to_canonical,
    )
    classification_to_articles = _build_classification_article_map(
        manifest=manifest,
        concept_to_articles=concept_to_articles,
        topic_to_articles=topic_to_articles,
        classification_ontology=classification_ontology,
        valid_articles=valid_articles,
    )

    routing = {
        "slang_to_contract": slang_to_contract,
        "topic_to_articles": topic_to_articles,
        "topic_patterns": topic_patterns,
        "classification_to_articles": classification_to_articles,
    }
    stats = {
        "topic_entries": len(topic_to_articles),
        "topic_pattern_entries": len(topic_patterns),
        "slang_entries": len(slang_to_contract),
        "classification_entries": len(classification_to_articles),
        "article_count": len(valid_articles),
    }
    return routing, stats


def merge_query_routing(generated: dict, existing: Optional[dict] = None) -> dict:
    """
    Merge generated routing with existing routing.

    Existing values are treated as overrides for string maps.
    Article-list maps are merged (union) to preserve manual curation.
    """
    generated = generated or {}
    existing = existing or {}

    def _merge_article_map(g_map, e_map):
        out: dict[str, list[int]] = {}
        keys = set((g_map or {}).keys()) | set((e_map or {}).keys())
        for key in sorted(keys):
            g_vals = _to_int_articles((g_map or {}).get(key, []))
            e_vals = _to_int_articles((e_map or {}).get(key, []))
            merged: list[int] = []
            seen = set()
            # Keep existing-manual ordering first, then generated fill.
            for value in e_vals + g_vals:
                if value in seen:
                    continue
                seen.add(value)
                merged.append(value)
            if merged:
                out[key] = merged
        return out

    return {
        "slang_to_contract": {
            **(generated.get("slang_to_contract") or {}),
            **(existing.get("slang_to_contract") or {}),
        },
        "topic_to_articles": _merge_article_map(
            generated.get("topic_to_articles") or {},
            existing.get("topic_to_articles") or {},
        ),
        "topic_patterns": {
            **(generated.get("topic_patterns") or {}),
            **(existing.get("topic_patterns") or {}),
        },
        "classification_to_articles": _merge_article_map(
            generated.get("classification_to_articles") or {},
            existing.get("classification_to_articles") or {},
        ),
    }
