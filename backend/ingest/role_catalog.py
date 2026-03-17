"""
Deterministic contract-scoped role catalog builder.

Produces a role artifact that separates "role referenced in contract metadata"
from "role has a contract wage-table row", so onboarding/runtime can avoid
surfacing non-wage roles as wage-selectable defaults.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.ingest.extract_wages import normalize_classification_name


ROLE_CATALOG_SCHEMA_VERSION = "role_catalog_v2"
_AMBIGUOUS_MANAGEMENT_VALUES = {
    "assistant_manager",
    "manager_trainee",
    "other_assistant_managers",
}


def _prettify_role_label(value: str) -> str:
    cleaned = re.sub(r"[_\s]+", " ", str(value or "").strip()).strip()
    if not cleaned:
        return "Unknown Role"
    words = []
    for token in cleaned.split(" "):
        upper = token.upper()
        if upper in {"HR", "GM", "DUG", "PTO", "ASST"}:
            words.append(upper)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(label or "").strip()).strip(" .")
    if not cleaned:
        return ""
    letters = [ch for ch in cleaned if ch.isalpha()]
    mostly_upper = bool(letters) and (
        sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.6
    )
    if mostly_upper:
        cleaned = cleaned.title()
        for acronym in ("Hr", "Gm", "Dug", "Pto", "Ufcw", "Asst"):
            cleaned = re.sub(rf"\b{acronym}\b", acronym.upper(), cleaned)
    return cleaned


def _manifest_label_map(manifest_classifications: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in manifest_classifications or []:
        label = str(raw or "").strip()
        if not label:
            continue
        key = normalize_classification_name(label)
        if not key:
            continue
        out.setdefault(key, _normalize_label(label) or _prettify_role_label(key))
    return out


def _index_decisions(ontology: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (ontology or {}).get("decisions", []) or []:
        if not isinstance(row, dict):
            continue
        source = normalize_classification_name(str(row.get("source_key") or ""))
        if source:
            out[source] = row
    return out


def _sorted_unique_nonempty(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _requires_role_clarification(role: dict) -> bool:
    value = normalize_classification_name(str(role.get("value") or ""))
    label = str(role.get("label") or "").strip().lower()
    wage_key = normalize_classification_name(str(role.get("wage_key") or ""))
    review_state = str(role.get("review_state") or "").strip().lower()
    blob = " ".join(part for part in [value, label, wage_key] if part)
    if review_state == "needs_clarification":
        return True
    if value in _AMBIGUOUS_MANAGEMENT_VALUES:
        return True
    if "assistant manager" in blob or "manager trainee" in blob:
        return True
    return False


def _collapse_wage_equivalent_onboarding_roles(role_rows: list[dict]) -> list[dict]:
    """
    Ensure one onboarding-default role per wage_key.

    Keep wage-equivalent aliases in the catalog for deterministic matching, but
    mark only one as onboarding default to prevent duplicate role choices.
    """
    by_wage_key: dict[str, list[dict]] = {}
    for role in role_rows:
        if not bool(role.get("wage_available")):
            continue
        wage_key = normalize_classification_name(str(role.get("wage_key") or ""))
        if not wage_key:
            continue
        role["wage_key"] = wage_key
        by_wage_key.setdefault(wage_key, []).append(role)

    for wage_key, group in by_wage_key.items():
        if len(group) <= 1:
            continue

        # Prefer canonical wage-key role for deterministic wage lookup identity.
        ranked = sorted(
            group,
            key=lambda r: (
                0 if str(r.get("value") or "") == wage_key else 1,
                0 if str(r.get("source") or "") == "wage_table" else 1,
                0 if bool(r.get("manifest_present")) else 1,
                0 if str(r.get("mapping_method") or "") == "manual_override" else 1,
                len(str(r.get("label") or "")),
                str(r.get("value") or ""),
            ),
        )
        primary = ranked[0]
        primary["onboarding_default"] = True

        alias_values: list[str] = []
        alias_labels: list[str] = []
        for alias_role in ranked[1:]:
            alias_role["onboarding_default"] = False
            alias_role["alias_of_wage_key"] = wage_key
            alias_values.append(str(alias_role.get("value") or ""))
            alias_labels.append(str(alias_role.get("label") or ""))

        alias_values = _sorted_unique_nonempty(alias_values)
        alias_labels = _sorted_unique_nonempty(
            [label for label in alias_labels if label != str(primary.get("label") or "")]
        )
        if alias_values:
            primary["alias_values"] = alias_values
        if alias_labels:
            primary["alias_labels"] = alias_labels

    return role_rows


def build_role_catalog(
    contract_id: str,
    manifest: dict,
    wages_data: dict,
    classification_ontology: dict,
) -> dict:
    manifest_labels = _manifest_label_map((manifest or {}).get("classifications") or [])
    decisions_by_source = _index_decisions(classification_ontology or {})
    wage_classes = (wages_data or {}).get("classifications", {}) or {}

    roles: dict[str, dict[str, Any]] = {}

    def _upsert(role: dict) -> None:
        key = normalize_classification_name(str(role.get("value") or ""))
        if not key:
            return
        role = dict(role)
        role["value"] = key
        role.setdefault("label", _prettify_role_label(key))
        role.setdefault("wage_available", False)
        role.setdefault("wage_key", None)
        role.setdefault("source", "manifest")
        role.setdefault("mapping_method", "unknown")
        role.setdefault("review_state", "unresolved")
        role.setdefault("manifest_present", False)
        role.setdefault("onboarding_default", bool(role.get("wage_available")))

        existing = roles.get(key)
        if existing is None:
            roles[key] = role
            return

        merged = dict(existing)
        if not merged.get("label") and role.get("label"):
            merged["label"] = role["label"]
        if role.get("manifest_present"):
            merged["manifest_present"] = True
        if role.get("wage_available"):
            merged["wage_available"] = True
            merged["onboarding_default"] = True
        if not merged.get("wage_key") and role.get("wage_key"):
            merged["wage_key"] = role.get("wage_key")
        # Prefer explicit provenance/method if previous value was generic.
        if merged.get("source") in {"manifest", "unknown"} and role.get("source"):
            merged["source"] = role.get("source")
        if merged.get("mapping_method") in {"unknown", "manifest_only"} and role.get("mapping_method"):
            merged["mapping_method"] = role.get("mapping_method")
        if merged.get("review_state") in {"unknown", "unresolved"} and role.get("review_state"):
            merged["review_state"] = role.get("review_state")
        if not merged.get("clarification_wage_keys") and role.get("clarification_wage_keys"):
            merged["clarification_wage_keys"] = list(role.get("clarification_wage_keys") or [])
        roles[key] = merged

    for wage_key, cls in sorted(wage_classes.items()):
        norm_key = normalize_classification_name(str(cls.get("normalized_name") or wage_key or ""))
        if not norm_key:
            continue
        label = (
            _normalize_label(str(cls.get("name") or ""))
            or manifest_labels.get(norm_key)
            or _prettify_role_label(norm_key)
        )
        _upsert(
            {
                "value": norm_key,
                "label": label,
                "wage_available": True,
                "wage_key": norm_key,
                "source": "wage_table",
                "mapping_method": "exact_wage_key",
                "review_state": "resolved",
                "manifest_present": norm_key in manifest_labels,
                "onboarding_default": True,
            }
        )

    for source_key, label in sorted(manifest_labels.items()):
        decision = decisions_by_source.get(source_key, {})
        mapped_key = normalize_classification_name(str(decision.get("mapped_wage_key") or ""))
        wage_available = bool(mapped_key and mapped_key in wage_classes)
        review_state = str(decision.get("review_state") or ("resolved" if wage_available else "unresolved"))
        clarification_wage_keys = [
            normalize_classification_name(str(value or ""))
            for value in (decision.get("clarification_wage_keys") or [])
            if normalize_classification_name(str(value or ""))
        ]
        _upsert(
            {
                "value": source_key,
                "label": label,
                "wage_available": wage_available,
                "wage_key": mapped_key if wage_available else None,
                "source": (
                    "ontology_resolved"
                    if wage_available
                    else "ontology_reviewed_out_of_scope"
                    if review_state == "out_of_scope"
                    else "ontology_reviewed_clarification"
                    if review_state == "needs_clarification"
                    else "ontology_unresolved"
                ),
                "mapping_method": str(decision.get("mapping_method") or "manifest_only"),
                "review_state": review_state,
                "clarification_wage_keys": clarification_wage_keys,
                "manifest_present": True,
                "onboarding_default": wage_available,
            }
        )

    role_list = _collapse_wage_equivalent_onboarding_roles(list(roles.values()))
    for role in role_list:
        if _requires_role_clarification(role):
            role["onboarding_default"] = False

    role_list = sorted(
        role_list,
        key=lambda r: (
            0 if r.get("onboarding_default") else 1,
            0 if r.get("wage_available") else 1,
            str(r.get("label") or r.get("value") or ""),
        ),
    )

    unresolved_manifest = sorted(
        r["value"]
        for r in role_list
        if (
            r.get("manifest_present")
            and not r.get("wage_available")
            and str(r.get("review_state") or "unresolved") not in {"out_of_scope", "needs_clarification"}
        )
    )
    clarification_manifest = sorted(
        r["value"]
        for r in role_list
        if r.get("manifest_present") and str(r.get("review_state") or "") == "needs_clarification"
    )
    out_of_scope_manifest = sorted(
        r["value"]
        for r in role_list
        if r.get("manifest_present") and str(r.get("review_state") or "") == "out_of_scope"
    )
    actionable_manifest_roles = max(
        sum(1 for r in role_list if r.get("manifest_present")) - len(out_of_scope_manifest),
        0,
    )

    return {
        "schema_version": ROLE_CATALOG_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract_id,
        "roles": role_list,
        "summary": {
            "total_roles": len(role_list),
            "onboarding_default_roles": sum(1 for r in role_list if r.get("onboarding_default")),
            "wage_available_roles": sum(1 for r in role_list if r.get("wage_available")),
            "manifest_roles": sum(1 for r in role_list if r.get("manifest_present")),
            "actionable_manifest_roles": actionable_manifest_roles,
            "clarification_manifest_roles": clarification_manifest,
            "out_of_scope_manifest_roles": out_of_scope_manifest,
            "unresolved_manifest_roles": unresolved_manifest,
        },
    }


def save_role_catalog(path: Path, catalog: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
