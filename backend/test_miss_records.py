"""Deterministic checks for structured miss-record normalization."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.miss_records import (
    SCHEMA_VERSION,
    build_regression_stub,
    normalize_miss_record,
    write_miss_record,
)


def _sample_payload() -> dict:
    return {
        "taxonomy_type": "deterministic_answer_binding_defect",
        "root_cause_type": "generation_binding_defect",
        "contract_id": "local7_safeway_pueblo_clerks_2022",
        "union_local_id": "local7",
        "question": "what is my pay right now?",
        "operator_label": "courtesy_clerk_wage_consistency_19_months",
        "miss_summary": "Narrative wage answer must match deterministic wage metadata and estimate card.",
        "session_id": "sess_123",
        "regression_status": "regression_added",
        "regression_case_id": "wage_answer_binding",
        "intent_type": "wage",
        "topic": "wages",
        "retrieval_strategy": "followup_anchor_seeded",
        "followup_context_used": True,
        "retrieval_anchor_count": 2,
        "months_employed": 19,
        "estimated_hours": 2800,
        "classification_review_state": "needs_clarification",
        "clarification_wage_keys": ["courtesy_clerk", "all_purpose_clerk"],
        "search_angles_used": 3,
        "top_retrieved_chunks": [
            {
                "citation": "Appendix A",
                "doc_type": "appendix",
                "score": 0.9821,
                "chunk_id": "chunk_1",
                "extra": "ignored",
            }
        ],
        "wage_info": {"rate": 17.5, "step": "1 year 6 months"},
        "final_answer": "You make $17.00/hr.",
        "final_citations": ["Appendix A"],
        "deterministic_fallback_ran": True,
        "retrieval_retry_ran": False,
        "notes": "Reviewed by steward; user consented to structured export only.",
    }


def _test_normalize_miss_record_applies_schema_defaults() -> None:
    record = normalize_miss_record(_sample_payload())
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["taxonomy_type"] == "deterministic_answer_binding_defect"
    assert record["root_cause_type"] == "generation_binding_defect"
    assert record["export_mode"] == "private_local"
    assert record["regression_status"] == "regression_added"
    assert record["top_retrieved_chunks"][0]["citation"] == "Appendix A"
    assert "extra" not in record["top_retrieved_chunks"][0]
    assert record["months_employed"] == 19
    assert record["classification_review_state"] == "needs_clarification"
    assert record["clarification_wage_keys"] == ["courtesy_clerk", "all_purpose_clerk"]
    assert record["retrieval_strategy"] == "followup_anchor_seeded"
    assert record["regression_case_id"] == "wage_answer_binding"


def _test_build_regression_stub_is_minimal_and_stable() -> None:
    stub = build_regression_stub(_sample_payload())
    assert stub["contract_id"] == "local7_safeway_pueblo_clerks_2022"
    assert stub["operator_label"] == "courtesy_clerk_wage_consistency_19_months"
    assert stub["taxonomy_type"] == "deterministic_answer_binding_defect"
    assert stub["expected_behavior"].startswith("Narrative wage answer")
    assert stub["classification_review_state"] == "needs_clarification"
    assert stub["clarification_wage_keys"] == ["courtesy_clerk", "all_purpose_clerk"]
    assert stub["regression_case_id"] == "wage_answer_binding"


def _test_normalize_miss_record_rejects_invalid_review_state() -> None:
    payload = _sample_payload()
    payload["classification_review_state"] = "mystery_state"
    try:
        normalize_miss_record(payload)
    except ValueError as exc:
        assert "classification_review_state must be one of" in str(exc)
        return
    raise AssertionError("Expected invalid classification_review_state to be rejected.")


def _test_write_miss_record_round_trips_normalized_json() -> None:
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="miss_records_", dir=str(root)))
    try:
        out_path = write_miss_record(_sample_payload(), tmp_dir / "miss_record.json")
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["contract_id"] == "local7_safeway_pueblo_clerks_2022"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _test_normalize_miss_record_requires_core_fields() -> None:
    payload = _sample_payload()
    payload["question"] = ""
    try:
        normalize_miss_record(payload)
    except ValueError as exc:
        assert "question is required" in str(exc)
        return
    raise AssertionError("Expected normalize_miss_record to reject empty question.")


def main() -> None:
    _test_normalize_miss_record_applies_schema_defaults()
    _test_build_regression_stub_is_minimal_and_stable()
    _test_write_miss_record_round_trips_normalized_json()
    _test_normalize_miss_record_requires_core_fields()
    _test_normalize_miss_record_rejects_invalid_review_state()
    print("[OK] miss-record checks passed")


if __name__ == "__main__":
    main()
