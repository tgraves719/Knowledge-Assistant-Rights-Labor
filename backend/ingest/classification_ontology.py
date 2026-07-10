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


ONTOLOGY_SCHEMA_VERSION = "classification_ontology_v2"
MANUAL_OVERRIDE_SCHEMA_VERSION = "classification_manual_overrides_v2"
_REVIEW_OVERRIDE_ACTIONS = {"out_of_scope", "needs_clarification"}

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


def _normalize_review_override_entry(
    source_key: str,
    raw_value: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if isinstance(raw_value, str):
        action = normalize_classification_name(raw_value)
        payload: dict[str, Any] = {"action": action}
    elif isinstance(raw_value, dict):
        action = normalize_classification_name(str(raw_value.get("action") or ""))
        payload = {"action": action}
        candidate_keys = raw_value.get("clarification_wage_keys") or []
        if candidate_keys and not isinstance(candidate_keys, list):
            warnings.append(f"manual_review_override_invalid_candidates_type: {source_key}")
            candidate_keys = []
        normalized_candidates = []
        for candidate in candidate_keys:
            candidate_key = normalize_classification_name(str(candidate or ""))
            if candidate_key:
                normalized_candidates.append(candidate_key)
        if normalized_candidates:
            payload["clarification_wage_keys"] = normalized_candidates
        comment = str(raw_value.get("comment") or "").strip()
        if comment:
            payload["comment"] = comment
    else:
        return None, [f"manual_review_override_invalid_entry_type: {source_key}"]

    action = str(payload.get("action") or "")
    if action not in _REVIEW_OVERRIDE_ACTIONS:
        return None, [f"manual_review_override_invalid_action: {source_key}->{action or 'missing'}"]
    return payload, warnings


def build_classification_ontology(
    contract_id: str,
    manifest_classifications: list[str],
    wages_data: dict,
    manual_alias_overrides: dict[str, str] | None = None,
    manual_review_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict:
    classes = (wages_data or {}).get("classifications", {}) or {}
    wage_keys = sorted(str(k) for k in classes.keys())
    wage_tokens = {k: _tokenize(k) | _tokenize(classes.get(k, {}).get("name", "")) for k in wage_keys}
    manual_alias_overrides = {
        normalize_classification_name(k): normalize_classification_name(v)
        for k, v in (manual_alias_overrides or {}).items()
        if str(k).strip() and str(v).strip()
    }
    manual_review_overrides = {
        normalize_classification_name(k): dict(v)
        for k, v in (manual_review_overrides or {}).items()
        if str(k).strip() and isinstance(v, dict)
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
    out_of_scope: list[str] = []
    clarification_required: list[str] = []
    manual_override_hits = 0
    manual_review_hits = 0

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
            review_state = "resolved"
            clarification_wage_keys: list[str] = []
            review_comment = ""
        elif manual_review_overrides.get(source_key, {}).get("action") == "out_of_scope":
            mapped_key, method, score, candidates = (
                None,
                "reviewed_out_of_scope",
                1.0,
                [],
            )
            manual_review_hits += 1
            review_state = "out_of_scope"
            clarification_wage_keys = []
            review_comment = str(manual_review_overrides.get(source_key, {}).get("comment") or "").strip()
        elif manual_review_overrides.get(source_key, {}).get("action") == "needs_clarification":
            _, _, inferred_score, inferred_candidates = _resolve_mapping(source_key, wage_keys, wage_tokens)
            configured_candidates = [
                candidate_key
                for candidate_key in (manual_review_overrides.get(source_key, {}).get("clarification_wage_keys") or [])
                if candidate_key in wage_tokens
            ]
            clarification_wage_keys = configured_candidates or [
                str(row.get("wage_key") or "")
                for row in inferred_candidates
                if str(row.get("wage_key") or "") in wage_tokens
            ]
            candidates = [
                {
                    "wage_key": candidate_key,
                    "score": next(
                        (
                            float(row.get("score") or 0.0)
                            for row in inferred_candidates
                            if str(row.get("wage_key") or "") == candidate_key
                        ),
                        1.0 if configured_candidates else 0.0,
                    ),
                }
                for candidate_key in clarification_wage_keys[:5]
            ]
            mapped_key, method, score = None, "reviewed_needs_clarification", inferred_score
            manual_review_hits += 1
            review_state = "needs_clarification"
            review_comment = str(manual_review_overrides.get(source_key, {}).get("comment") or "").strip()
        else:
            mapped_key, method, score, candidates = _resolve_mapping(source_key, wage_keys, wage_tokens)
            review_state = "resolved" if mapped_key else "unresolved"
            clarification_wage_keys = []
            review_comment = ""
        decision = {
            "source_key": source_key,
            "source_labels": source_labels_by_key[source_key],
            "mapped_wage_key": mapped_key,
            "mapping_method": method,
            "score": round(float(score), 4),
            "candidate_scores": candidates,
            "review_state": review_state,
        }
        if clarification_wage_keys:
            decision["clarification_wage_keys"] = clarification_wage_keys[:10]
        if review_comment:
            decision["review_comment"] = review_comment
        decisions.append(decision)
        if mapped_key:
            alias_map[source_key] = mapped_key
        elif review_state == "out_of_scope":
            out_of_scope.append(source_key)
        elif review_state == "needs_clarification":
            clarification_required.append(source_key)
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
    actionable_total = max(total - len(out_of_scope), 0)
    resolved = total - len(unresolved) - len(out_of_scope) - len(clarification_required)
    covered = resolved + len(clarification_required)
    coverage = (covered / actionable_total) if actionable_total else 1.0

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
            "clarification_manifest_classes": len(clarification_required),
            "out_of_scope_manifest_classes": len(out_of_scope),
            "unresolved_manifest_classes": len(unresolved),
            "actionable_manifest_classes": actionable_total,
            "covered_manifest_classes": covered,
            "coverage": round(coverage, 4),
            "unresolved_manifest_keys": unresolved,
            "clarification_manifest_keys": clarification_required,
            "out_of_scope_manifest_keys": out_of_scope,
            "manual_override_count": manual_override_hits,
            "manual_review_override_count": manual_review_hits,
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


def load_manual_classification_review_overrides(
    path: Path,
    contract_id: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    """
    Load and normalize manual classification alias overrides.

    Accepts either:
    - schema envelope with `classification_alias_overrides`
    - plain key->value mapping for compatibility
    """
    if not path.exists():
        return {}, {}, []

    warnings: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return {}, {}, [f"manual_override_json_load_failed: {exc}"]

    mapping = {}
    review_mapping = {}
    if isinstance(raw, dict) and "classification_alias_overrides" in raw:
        if contract_id and raw.get("contract_id") and str(raw.get("contract_id")) != str(contract_id):
            warnings.append(
                f"manual_override_contract_id_mismatch: expected={contract_id} actual={raw.get('contract_id')}"
            )
        raw_map = raw.get("classification_alias_overrides") or {}
        if not isinstance(raw_map, dict):
            return {}, {}, warnings + ["manual_override_alias_map_invalid_type"]
        mapping = raw_map
        raw_review_map = raw.get("classification_review_overrides") or {}
        if raw_review_map and not isinstance(raw_review_map, dict):
            return {}, {}, warnings + ["manual_review_override_map_invalid_type"]
        review_mapping = raw_review_map
    elif isinstance(raw, dict):
        mapping = raw
    else:
        return {}, {}, warnings + ["manual_override_root_invalid_type"]

    normalized: dict[str, str] = {}
    for src, dst in mapping.items():
        src_key = normalize_classification_name(str(src or ""))
        dst_key = normalize_classification_name(str(dst or ""))
        if not src_key or not dst_key:
            warnings.append(f"manual_override_invalid_entry: {src}->{dst}")
            continue
        normalized[src_key] = dst_key
    normalized_review_overrides: dict[str, dict[str, Any]] = {}
    for src, raw_value in (review_mapping or {}).items():
        src_key = normalize_classification_name(str(src or ""))
        if not src_key:
            warnings.append(f"manual_review_override_invalid_entry: {src}")
            continue
        normalized_value, value_warnings = _normalize_review_override_entry(src_key, raw_value)
        warnings.extend(value_warnings)
        if normalized_value is not None:
            normalized_review_overrides[src_key] = normalized_value
    return normalized, normalized_review_overrides, warnings


def load_manual_classification_overrides(
    path: Path,
    contract_id: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    aliases, _, warnings = load_manual_classification_review_overrides(path, contract_id=contract_id)
    return aliases, warnings


def write_manual_override_template(path: Path, contract_id: str) -> None:
    if path.exists():
        return
    payload = {
        "schema_version": MANUAL_OVERRIDE_SCHEMA_VERSION,
        "contract_id": contract_id,
        "classification_alias_overrides": {},
        "classification_review_overrides": {},
        "notes": (
            "Add source_key -> wage_key overrides or reviewed out_of_scope/"
            "needs_clarification states after reviewing ingestion_review_queue.json"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
