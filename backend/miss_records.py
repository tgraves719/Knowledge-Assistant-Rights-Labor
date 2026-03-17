"""Structured real-user miss records for local error-correction workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


SCHEMA_VERSION = "real_miss_record_v3"

MISS_TAXONOMY_TYPES = {
    "onboarding_taxonomy_defect",
    "deterministic_answer_binding_defect",
    "trigger_intent_defect",
    "retrieval_followup_defect",
    "artifact_scope_defect",
    "genuine_corpus_gap",
    "extraction_indexing_gap",
}

ROOT_CAUSE_TYPES = {
    "ingestion_defect",
    "retrieval_defect",
    "deterministic_logic_defect",
    "generation_binding_defect",
    "genuine_corpus_gap",
    "unknown",
}

EXPORT_MODES = {
    "private_local",
    "structured_export_opt_in",
}

CLASSIFICATION_REVIEW_STATES = {
    "resolved",
    "needs_clarification",
    "out_of_scope",
    "unresolved",
}

REGRESSION_READINESS = {
    "triage_only",
    "ready_for_regression",
    "regression_added",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any, *, max_length: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > max_length:
        return text[: max_length - 3].rstrip() + "..."
    return text


def _clean_optional_text(value: Any, *, max_length: int) -> Optional[str]:
    text = _clean_text(value, max_length=max_length)
    return text or None


def _clean_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _clean_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_string_list(values: Any, *, max_items: int, max_item_length: int) -> list[str]:
    items: list[str] = []
    for value in list(values or [])[:max_items]:
        text = _clean_text(value, max_length=max_item_length)
        if text:
            items.append(text)
    return items


def _clean_chunk_list(values: Any, *, max_items: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in list(values or [])[:max_items]:
        if not isinstance(value, dict):
            continue
        row = {
            "citation": _clean_optional_text(value.get("citation"), max_length=200),
            "doc_type": _clean_optional_text(value.get("doc_type"), max_length=32),
            "score": None,
            "chunk_id": _clean_optional_text(value.get("chunk_id"), max_length=120),
        }
        score = value.get("score")
        try:
            if score is not None:
                row["score"] = round(float(score), 6)
        except (TypeError, ValueError):
            row["score"] = None
        if any(v is not None for v in row.values()):
            rows.append(row)
    return rows


def normalize_miss_record(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Miss record payload must be a JSON object.")

    taxonomy_type = str(payload.get("taxonomy_type") or "").strip().lower()
    if taxonomy_type not in MISS_TAXONOMY_TYPES:
        raise ValueError(
            "taxonomy_type must be one of: "
            + ", ".join(sorted(MISS_TAXONOMY_TYPES))
        )

    root_cause = str(payload.get("root_cause_type") or "unknown").strip().lower()
    if root_cause not in ROOT_CAUSE_TYPES:
        raise ValueError(
            "root_cause_type must be one of: "
            + ", ".join(sorted(ROOT_CAUSE_TYPES))
        )

    export_mode = str(payload.get("export_mode") or "private_local").strip().lower()
    if export_mode not in EXPORT_MODES:
        raise ValueError(
            "export_mode must be one of: "
            + ", ".join(sorted(EXPORT_MODES))
        )

    regression_status = str(payload.get("regression_status") or "triage_only").strip().lower()
    if regression_status not in REGRESSION_READINESS:
        raise ValueError(
            "regression_status must be one of: "
            + ", ".join(sorted(REGRESSION_READINESS))
        )

    contract_id = _clean_text(payload.get("contract_id"), max_length=120)
    question = _clean_text(payload.get("question"), max_length=2000)
    operator_label = _clean_text(payload.get("operator_label"), max_length=80)
    miss_summary = _clean_text(payload.get("miss_summary"), max_length=2000)

    if not contract_id:
        raise ValueError("contract_id is required.")
    if not question:
        raise ValueError("question is required.")
    if not operator_label:
        raise ValueError("operator_label is required.")
    if not miss_summary:
        raise ValueError("miss_summary is required.")

    record = {
        "schema_version": SCHEMA_VERSION,
        "miss_id": _clean_text(payload.get("miss_id") or f"miss_{uuid4().hex[:12]}", max_length=40),
        "recorded_at_utc": _clean_text(payload.get("recorded_at_utc") or _utc_now(), max_length=64),
        "export_mode": export_mode,
        "regression_status": regression_status,
        "taxonomy_type": taxonomy_type,
        "root_cause_type": root_cause,
        "operator_label": operator_label,
        "miss_summary": miss_summary,
        "contract_id": contract_id,
        "union_local_id": _clean_optional_text(payload.get("union_local_id"), max_length=80),
        "contract_version": _clean_optional_text(payload.get("contract_version"), max_length=80),
        "session_id": _clean_optional_text(payload.get("session_id"), max_length=120),
        "question": question,
        "user_classification": _clean_optional_text(payload.get("user_classification"), max_length=120),
        "employment_type": _clean_optional_text(payload.get("employment_type"), max_length=40),
        "hire_date": _clean_optional_text(payload.get("hire_date"), max_length=32),
        "months_employed": _clean_int(payload.get("months_employed")),
        "estimated_hours": _clean_int(payload.get("estimated_hours")),
        "intent_type": _clean_optional_text(payload.get("intent_type"), max_length=40),
        "topic": _clean_optional_text(payload.get("topic"), max_length=80),
        "retrieval_strategy": _clean_optional_text(payload.get("retrieval_strategy"), max_length=80),
        "followup_context_used": _clean_bool(payload.get("followup_context_used")),
        "retrieval_anchor_count": _clean_int(payload.get("retrieval_anchor_count")),
        "high_stakes_topic": _clean_optional_text(payload.get("high_stakes_topic"), max_length=80),
        "active_urgent_context": _clean_bool(payload.get("active_urgent_context")),
        "search_angles_used": _clean_int(payload.get("search_angles_used")),
        "classification_review_state": _clean_optional_text(payload.get("classification_review_state"), max_length=40),
        "clarification_wage_keys": _clean_string_list(
            payload.get("clarification_wage_keys"),
            max_items=10,
            max_item_length=120,
        ),
        "top_retrieved_chunks": _clean_chunk_list(payload.get("top_retrieved_chunks")),
        "wage_info": payload.get("wage_info") if isinstance(payload.get("wage_info"), dict) else None,
        "entitlement_info": payload.get("entitlement_info") if isinstance(payload.get("entitlement_info"), dict) else None,
        "effective_version_id": _clean_optional_text(payload.get("effective_version_id"), max_length=120),
        "amendments_applied": _clean_string_list(payload.get("amendments_applied"), max_items=20, max_item_length=120),
        "final_answer": _clean_optional_text(payload.get("final_answer"), max_length=3000),
        "final_citations": _clean_string_list(payload.get("final_citations"), max_items=20, max_item_length=200),
        "deterministic_fallback_ran": _clean_bool(payload.get("deterministic_fallback_ran")),
        "retrieval_retry_ran": _clean_bool(payload.get("retrieval_retry_ran")),
        "notes": _clean_optional_text(payload.get("notes"), max_length=3000),
        "regression_case_id": _clean_optional_text(payload.get("regression_case_id"), max_length=120),
    }
    review_state = str(record.get("classification_review_state") or "").strip().lower()
    if review_state and review_state not in CLASSIFICATION_REVIEW_STATES:
        raise ValueError(
            "classification_review_state must be one of: "
            + ", ".join(sorted(CLASSIFICATION_REVIEW_STATES))
        )
    if review_state:
        record["classification_review_state"] = review_state
    return record


def build_regression_stub(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_miss_record(record)
    return {
        "miss_id": normalized["miss_id"],
        "contract_id": normalized["contract_id"],
        "taxonomy_type": normalized["taxonomy_type"],
        "root_cause_type": normalized["root_cause_type"],
        "operator_label": normalized["operator_label"],
        "question": normalized["question"],
        "expected_behavior": normalized["miss_summary"],
        "final_citations": list(normalized.get("final_citations") or []),
        "classification_review_state": normalized.get("classification_review_state"),
        "clarification_wage_keys": list(normalized.get("clarification_wage_keys") or []),
        "retrieval_strategy": normalized.get("retrieval_strategy"),
        "regression_case_id": normalized.get("regression_case_id"),
    }


def load_miss_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with open(target, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return normalize_miss_record(payload)


def write_miss_record(record: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_miss_record(record)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return target
