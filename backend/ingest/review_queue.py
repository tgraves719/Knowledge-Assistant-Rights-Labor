"""
Ingestion review queue builder.

Aggregates unresolved/ambiguous extraction and ontology mapping signals into a
single contract-pack artifact for human adjudication.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_QUEUE_SCHEMA_VERSION = "ingestion_review_queue_v1"


def _make_item(
    issue_id: str,
    category: str,
    severity: str,
    summary: str,
    evidence: dict[str, Any],
    recommended_action: str,
) -> dict:
    return {
        "issue_id": issue_id,
        "category": category,
        "severity": severity,
        "summary": summary,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def build_ingestion_review_queue(
    contract_id: str,
    ontology: dict | None,
    wages_data: dict | None,
    manual_override_warnings: list[str] | None = None,
) -> dict:
    items: list[dict] = []
    manual_override_warnings = manual_override_warnings or []

    if manual_override_warnings:
        items.append(
            _make_item(
                issue_id="manual_overrides_warnings",
                category="manual_overrides",
                severity="medium",
                summary="Manual override file contains validation warnings.",
                evidence={"warnings": manual_override_warnings[:100]},
                recommended_action="Fix manual override file schema/keys before next onboarding run.",
            )
        )

    ontology_summary = (ontology or {}).get("summary", {}) or {}
    decisions = (ontology or {}).get("decisions", []) or []
    unresolved_decisions = [
        d for d in decisions
        if (
            isinstance(d, dict)
            and not d.get("mapped_wage_key")
            and str(d.get("review_state") or "unresolved") == "unresolved"
        )
    ]
    if unresolved_decisions:
        items.append(
            _make_item(
                issue_id="ontology_unresolved_manifest_classes",
                category="classification_ontology",
                severity="high",
                summary="Some manifest classifications do not map to wage keys.",
                evidence={
                    "unresolved_count": len(unresolved_decisions),
                    "unresolved_source_keys": [
                        d.get("source_key") for d in unresolved_decisions[:100]
                    ],
                },
                recommended_action=(
                    "Review candidate_scores and add manual_classification_overrides "
                    "or reviewed classification_review_overrides entries where appropriate."
                ),
            )
        )

    low_confidence_mappings = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        mapped = d.get("mapped_wage_key")
        if not mapped:
            continue
        method = str(d.get("mapping_method") or "")
        score = float(d.get("score") or 0.0)
        if method == "token_similarity" and score < 0.75:
            low_confidence_mappings.append(
                {
                    "source_key": d.get("source_key"),
                    "mapped_wage_key": mapped,
                    "score": score,
                    "candidate_scores": (d.get("candidate_scores") or [])[:5],
                }
            )
    if low_confidence_mappings:
        items.append(
            _make_item(
                issue_id="ontology_low_confidence_mappings",
                category="classification_ontology",
                severity="medium",
                summary="Some ontology mappings were accepted with low similarity confidence.",
                evidence={
                    "count": len(low_confidence_mappings),
                    "mappings": low_confidence_mappings[:100],
                },
                recommended_action="Review low-confidence mappings; add manual overrides or leave unresolved explicitly.",
            )
        )

    extraction_meta = (wages_data or {}).get("extraction_metadata", {}) or {}
    unresolved_rows = extraction_meta.get("unresolved_rows", []) or []
    if unresolved_rows:
        items.append(
            _make_item(
                issue_id="wage_extraction_unresolved_rows",
                category="wage_extraction",
                severity="medium",
                summary="Deterministic row classifier produced unresolved wage-like rows.",
                evidence={"count": len(unresolved_rows), "rows": unresolved_rows[:100]},
                recommended_action="Inspect source tables and add parser rule or manual override where needed.",
            )
        )

    canonical_conflicts = extraction_meta.get("canonical_conflicts", []) or []
    if canonical_conflicts:
        items.append(
            _make_item(
                issue_id="canonical_wage_conflicts",
                category="wage_canonical_rows",
                severity="high",
                summary="Canonical wage rows encountered conflicting rates for same key.",
                evidence={"count": len(canonical_conflicts), "conflicts": canonical_conflicts[:100]},
                recommended_action="Resolve conflicting source rows and confirm correct rate selection.",
            )
        )

    canonical_ambiguities = extraction_meta.get("canonical_ambiguities", []) or []
    if canonical_ambiguities:
        items.append(
            _make_item(
                issue_id="canonical_wage_ambiguities",
                category="wage_canonical_rows",
                severity="medium",
                summary="Canonical wage rows had ambiguity tie cases.",
                evidence={"count": len(canonical_ambiguities), "ambiguities": canonical_ambiguities[:100]},
                recommended_action="Review ambiguity tie cases; add deterministic rule or manual correction.",
            )
        )

    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for item in items:
        sev = str(item.get("severity") or "low")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    unresolved_manifest = int(ontology_summary.get("unresolved_manifest_classes", 0) or 0)

    return {
        "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract_id,
        "summary": {
            "total_items": len(items),
            "severity_counts": severity_counts,
            "unresolved_manifest_classes": unresolved_manifest,
            "canonical_conflicts": len(canonical_conflicts),
            "canonical_ambiguities": len(canonical_ambiguities),
            "unresolved_wage_rows": len(unresolved_rows),
        },
        "items": items,
    }


def save_ingestion_review_queue(path: Path, queue: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)

