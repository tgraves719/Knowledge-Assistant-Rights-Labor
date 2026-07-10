"""Regression coverage for reviewed classification override states."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from uuid import uuid4

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.classification_ontology import (
    build_classification_ontology,
    load_manual_classification_review_overrides,
)
from backend.ingest.role_catalog import build_role_catalog


def _workspace_tempdir(prefix: str) -> Path:
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _test_reviewed_out_of_scope_and_clarification_states_drive_coverage() -> None:
    wages_data = {
        "classifications": {
            "meat_cutters": {"name": "MEAT CUTTERS", "normalized_name": "meat_cutters"},
            "meat_head_clerk_asst": {"name": "MEAT HEAD CLERK/ASST", "normalized_name": "meat_head_clerk_asst"},
            "deli_head_clerk_asst": {"name": "DELI HEAD CLERK/ASST", "normalized_name": "deli_head_clerk_asst"},
        }
    }
    ontology = build_classification_ontology(
        contract_id="review_override_contract",
        manifest_classifications=["Cake Decorator", "Head Clerk", "Meat Cutter"],
        wages_data=wages_data,
        manual_review_overrides={
            "cake_decorator": {"action": "out_of_scope"},
            "head_clerk": {
                "action": "needs_clarification",
                "clarification_wage_keys": ["meat_head_clerk_asst", "deli_head_clerk_asst"],
            },
        },
    )
    summary = ontology.get("summary") or {}
    assert summary.get("coverage") == 1.0
    assert summary.get("actionable_manifest_classes") == 2
    assert summary.get("covered_manifest_classes") == 2
    assert summary.get("out_of_scope_manifest_classes") == 1
    assert summary.get("clarification_manifest_classes") == 1
    assert summary.get("unresolved_manifest_classes") == 0
    assert summary.get("out_of_scope_manifest_keys") == ["cake_decorator"]
    assert summary.get("clarification_manifest_keys") == ["head_clerk"]

    decisions = {
        str(row.get("source_key") or ""): row
        for row in (ontology.get("decisions") or [])
        if isinstance(row, dict)
    }
    assert decisions["cake_decorator"]["review_state"] == "out_of_scope"
    assert decisions["head_clerk"]["review_state"] == "needs_clarification"
    assert decisions["head_clerk"]["clarification_wage_keys"] == [
        "meat_head_clerk_asst",
        "deli_head_clerk_asst",
    ]

    role_catalog = build_role_catalog(
        contract_id="review_override_contract",
        manifest={"classifications": ["Cake Decorator", "Head Clerk", "Meat Cutter"]},
        wages_data=wages_data,
        classification_ontology=ontology,
    )
    role_summary = role_catalog.get("summary") or {}
    assert "head_clerk" in set(role_summary.get("clarification_manifest_roles") or [])
    assert "cake_decorator" in set(role_summary.get("out_of_scope_manifest_roles") or [])
    assert not role_summary.get("unresolved_manifest_roles")


def _test_manual_review_override_loader_parses_v2_schema() -> None:
    tmp_dir = _workspace_tempdir("review_override_loader_")
    try:
        path = tmp_dir / "manual_classification_overrides.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "classification_manual_overrides_v2",
                    "contract_id": "demo_contract",
                    "classification_alias_overrides": {"Meat Cutter": "meat_cutters"},
                    "classification_review_overrides": {
                        "Cake Decorator": {"action": "out_of_scope"},
                        "Head Clerk": {
                            "action": "needs_clarification",
                            "clarification_wage_keys": ["meat_head_clerk_asst", "deli_head_clerk_asst"],
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        aliases, reviews, warnings = load_manual_classification_review_overrides(
            path,
            contract_id="demo_contract",
        )
        assert not warnings
        assert aliases == {"meat_cutter": "meat_cutters"}
        assert reviews["cake_decorator"]["action"] == "out_of_scope"
        assert reviews["head_clerk"]["action"] == "needs_clarification"
        assert reviews["head_clerk"]["clarification_wage_keys"] == [
            "meat_head_clerk_asst",
            "deli_head_clerk_asst",
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    _test_reviewed_out_of_scope_and_clarification_states_drive_coverage()
    _test_manual_review_override_loader_parses_v2_schema()
    print("[OK] classification review override checks passed")


if __name__ == "__main__":
    main()
