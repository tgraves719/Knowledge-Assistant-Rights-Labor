"""
Deterministic classification ontology builder.

Maps manifest-facing classification labels to wage-table keys and emits
an auditable artifact for contract-pack ingestion quality gates.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from backend.ingest.extract_wages import normalize_classification_name


ONTOLOGY_SCHEMA_VERSION = "classification_ontology_v1"
MANUAL_OVERRIDE_SCHEMA_VERSION = "classification_manual_overrides_v1"

_STOP_TOKENS = {
    "department", "deparment", "unit", "employee", "employees", "and", "of",
    "the", "position", "positions", "job", "jobs", "worker", "workers",
}

_TOKEN_CANON = {
    "technician": "tech",
    "clerks": "clerk",
    "cutters": "cutter",
    "managers": "manager",
    "assist": "asst",
    "assistant": "asst",
    "asst": "asst",
    "non": "nonfood",
    "food": "nonfood",
}

_EXPLICIT_ALIASES: dict[str, tuple[str, ...]] = {
    # Keep explicit aliases strictly for lexical variants, not semantic
    # cross-class remapping. Contract-specific semantics belong in per-pack
    # manual_classification_overrides.json artifacts.
    "pharmacy_technician": ("pharmacy_tech",),
    "pharmacy_tech": ("pharmacy_technician",),
    "meat_cutter": ("meat_cutters", "meatcutter"),
    "meatcutter": ("meat_cutters", "meat_cutter"),
}


def _tokenize(value: str) -> set[str]:
    key = normalize_classification_name(value)
    if not key:
        return set()
    raw_tokens = [t for t in key.split("_") if t]
    tokens: set[str] = set()
    for token in raw_tokens:
        if token in _STOP_TOKENS:
            continue
        canon = _TOKEN_CANON.get(token, token)
        if canon and canon not in _STOP_TOKENS:
            tokens.add(canon)
    return tokens


def _score_candidate(source_key: str, source_tokens: set[str], target_key: str, target_tokens: set[str]) -> float:
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = len(source_tokens & target_tokens)
    if overlap == 0:
        return 0.0

    recall = overlap / max(len(source_tokens), 1)
    precision = overlap / max(len(target_tokens), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    seq = SequenceMatcher(None, source_key, target_key).ratio()
    prefix = 1.0 if (source_key.startswith(target_key) or target_key.startswith(source_key)) else 0.0

    score = 0.55 * f1 + 0.30 * seq + 0.15 * prefix

    source_is_manager = "manager" in source_tokens
    target_is_manager = "manager" in target_tokens
    source_is_clerk = "clerk" in source_tokens
    target_is_clerk = "clerk" in target_tokens
    source_is_cutter = "cutter" in source_tokens
    target_is_cutter = "cutter" in target_tokens

    if source_is_manager != target_is_manager:
        score -= 0.18
    if source_is_clerk and target_is_manager:
        score -= 0.12
    if source_is_cutter != target_is_cutter:
        score -= 0.10

    return max(0.0, min(1.0, round(score, 6)))


def _resolve_mapping(
    source_key: str,
    wage_keys: list[str],
    wage_tokens: dict[str, set[str]],
) -> tuple[str | None, str, float, list[dict[str, Any]]]:
    if source_key in wage_tokens:
        return source_key, "exact", 1.0, []

    for alias in _EXPLICIT_ALIASES.get(source_key, ()):
        if alias in wage_tokens:
            return alias, "explicit_alias", 0.99, [{"wage_key": alias, "score": 0.99}]

    source_tokens = _tokenize(source_key)
    scored: list[tuple[str, float]] = []
    for target_key in wage_keys:
        score = _score_candidate(source_key, source_tokens, target_key, wage_tokens[target_key])
        if score > 0:
            scored.append((target_key, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    candidates = [{"wage_key": k, "score": s} for k, s in scored[:5]]
    if not scored:
        return None, "unresolved", 0.0, candidates

    top_key, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = top_score - second_score

    if top_score < 0.55:
        return None, "unresolved_low_score", top_score, candidates
    if margin < 0.08 and top_score < 0.80:
        return None, "unresolved_ambiguous", top_score, candidates

    return top_key, "token_similarity", top_score, candidates


def build_classification_ontology(
    contract_id: str,
    manifest_classifications: list[str],
    wages_data: dict,
    manual_alias_overrides: dict[str, str] | None = None,
) -> dict:
    classes = (wages_data or {}).get("classifications", {}) or {}
    wage_keys = sorted(str(k) for k in classes.keys())
    wage_tokens = {k: _tokenize(k) | _tokenize(classes.get(k, {}).get("name", "")) for k in wage_keys}
    manual_alias_overrides = {
        normalize_classification_name(k): normalize_classification_name(v)
        for k, v in (manual_alias_overrides or {}).items()
        if str(k).strip() and str(v).strip()
    }

    source_labels_by_key: dict[str, list[str]] = {}
    for raw in manifest_classifications or []:
        label = str(raw or "").strip()
        if not label:
            continue
        key = normalize_classification_name(label)
        if not key:
            continue
        source_labels_by_key.setdefault(key, [])
        if label not in source_labels_by_key[key]:
            source_labels_by_key[key].append(label)

    decisions: list[dict] = []
    alias_map: dict[str, str] = {}
    unresolved: list[str] = []
    manual_override_hits = 0

    for source_key in sorted(source_labels_by_key.keys()):
        override_target = manual_alias_overrides.get(source_key)
        if override_target and override_target in wage_tokens:
            mapped_key, method, score, candidates = (
                override_target,
                "manual_override",
                1.0,
                [{"wage_key": override_target, "score": 1.0}],
            )
            manual_override_hits += 1
        else:
            mapped_key, method, score, candidates = _resolve_mapping(source_key, wage_keys, wage_tokens)
        decision = {
            "source_key": source_key,
            "source_labels": source_labels_by_key[source_key],
            "mapped_wage_key": mapped_key,
            "mapping_method": method,
            "score": round(float(score), 4),
            "candidate_scores": candidates,
        }
        decisions.append(decision)
        if mapped_key:
            alias_map[source_key] = mapped_key
        else:
            unresolved.append(source_key)

    # Identity aliases for direct wage keys.
    for wage_key in wage_keys:
        alias_map.setdefault(wage_key, wage_key)
        wage_name = (classes.get(wage_key, {}) or {}).get("name")
        if wage_name:
            normalized_name = normalize_classification_name(str(wage_name))
            if normalized_name:
                alias_map.setdefault(normalized_name, wage_key)

    # Preserve manual overrides as aliases even when source is outside manifest list.
    for source_key, target_key in manual_alias_overrides.items():
        if target_key in wage_tokens:
            alias_map[source_key] = target_key

    total = len(source_labels_by_key)
    resolved = total - len(unresolved)
    coverage = (resolved / total) if total else 1.0

    return {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract_id,
        "manifest_classifications": [
            {"normalized": key, "labels": labels}
            for key, labels in sorted(source_labels_by_key.items())
        ],
        "wage_classifications": [
            {
                "wage_key": wage_key,
                "name": (classes.get(wage_key, {}) or {}).get("name", wage_key),
            }
            for wage_key in wage_keys
        ],
        "decisions": decisions,
        "alias_to_wage_key": alias_map,
        "summary": {
            "total_manifest_classes": total,
            "resolved_manifest_classes": resolved,
            "unresolved_manifest_classes": len(unresolved),
            "coverage": round(coverage, 4),
            "unresolved_manifest_keys": unresolved,
            "manual_override_count": manual_override_hits,
        },
    }


def apply_ontology_aliases(wages_data: dict, ontology: dict) -> dict:
    out = dict(wages_data or {})
    aliases = dict(out.get("classification_aliases") or {})
    for source_key, target_key in (ontology.get("alias_to_wage_key") or {}).items():
        aliases[str(source_key)] = str(target_key)
    out["classification_aliases"] = aliases
    out["classification_ontology_summary"] = dict(ontology.get("summary") or {})
    return out


def save_classification_ontology(path: Path, ontology: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)


def load_manual_classification_overrides(
    path: Path,
    contract_id: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Load and normalize manual classification alias overrides.

    Accepts either:
    - schema envelope with `classification_alias_overrides`
    - plain key->value mapping for compatibility
    """
    if not path.exists():
        return {}, []

    warnings: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return {}, [f"manual_override_json_load_failed: {exc}"]

    mapping = {}
    if isinstance(raw, dict) and "classification_alias_overrides" in raw:
        if contract_id and raw.get("contract_id") and str(raw.get("contract_id")) != str(contract_id):
            warnings.append(
                f"manual_override_contract_id_mismatch: expected={contract_id} actual={raw.get('contract_id')}"
            )
        raw_map = raw.get("classification_alias_overrides") or {}
        if not isinstance(raw_map, dict):
            return {}, warnings + ["manual_override_alias_map_invalid_type"]
        mapping = raw_map
    elif isinstance(raw, dict):
        mapping = raw
    else:
        return {}, warnings + ["manual_override_root_invalid_type"]

    normalized: dict[str, str] = {}
    for src, dst in mapping.items():
        src_key = normalize_classification_name(str(src or ""))
        dst_key = normalize_classification_name(str(dst or ""))
        if not src_key or not dst_key:
            warnings.append(f"manual_override_invalid_entry: {src}->{dst}")
            continue
        normalized[src_key] = dst_key
    return normalized, warnings


def write_manual_override_template(path: Path, contract_id: str) -> None:
    if path.exists():
        return
    payload = {
        "schema_version": MANUAL_OVERRIDE_SCHEMA_VERSION,
        "contract_id": contract_id,
        "classification_alias_overrides": {},
        "notes": "Add source_key -> wage_key overrides after reviewing ingestion_review_queue.json",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
