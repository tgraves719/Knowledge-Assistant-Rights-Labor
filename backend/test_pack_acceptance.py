"""Deterministic checks for refreshed contract pack acceptance capability gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.pack_acceptance import _effective_moa_provenance_page_metrics, evaluate_contract_pack


CONTRACTS_ROOT = Path("data/contracts")


def _check_map(scorecard: dict) -> dict[str, dict]:
    return {
        str(row.get("id") or ""): row
        for row in (scorecard.get("checks") or [])
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _test_refreshed_older_pack_now_materializes_side_letter_doc_types() -> None:
    scorecard = evaluate_contract_pack(
        CONTRACTS_ROOT / "local7_kingsoopers_loveland_meat_2019",
        write_scorecard=False,
    )
    checks = _check_map(scorecard)
    row = checks["side_letter_doc_type_materialization"]
    assert row["severity"] == "required"
    assert row["status"] == "pass"
    assert bool((scorecard.get("summary") or {}).get("pass")) is True


def _test_effective_packs_now_include_entitlement_index_inputs() -> None:
    for contract_id in (
        "local7_safeway_pueblo_clerks_2022",
        "local7_safeway_pueblo_meat_2022",
    ):
        scorecard = evaluate_contract_pack(
            CONTRACTS_ROOT / contract_id,
            write_scorecard=False,
        )
        checks = _check_map(scorecard)
        exists_row = checks["effective_entitlement_index_input_exists"]
        non_empty_row = checks["effective_entitlement_index_input_non_empty"]
        assert exists_row["status"] == "pass"
        assert non_empty_row["status"] == "pass"


def _test_refreshed_local_packs_have_no_required_failures() -> None:
    expected_advisories = {
        "local7_kingsoopers_loveland_meat_2019": set(),
        "local7_safeway_pueblo_clerks_2022": set(),
        "local7_safeway_pueblo_meat_2022": set(),
    }
    for contract_id, advisory_ids in expected_advisories.items():
        scorecard = evaluate_contract_pack(
            CONTRACTS_ROOT / contract_id,
            write_scorecard=False,
        )
        summary = scorecard.get("summary") or {}
        assert summary.get("required_failed") == []
        assert set(summary.get("advisory_failed") or []) == advisory_ids
        assert bool(summary.get("pass")) is True


def _test_pack_capabilities_surface_green_required_state() -> None:
    scorecard = evaluate_contract_pack(
        CONTRACTS_ROOT / "local7_safeway_pueblo_clerks_2022",
        write_scorecard=False,
    )
    checks = _check_map(scorecard)
    capabilities = scorecard.get("capabilities") or {}
    assert capabilities.get("side_letter_doc_type_materialization") == "pass"
    assert capabilities.get("effective_snapshot_present") is True
    assert capabilities.get("effective_entitlement_index_input") == "pass"
    assert capabilities.get("moa_wage_schedule_metadata_integrity") == "pass"
    assert capabilities.get("moa_wage_schedule_sync_registration") == "pass"
    assert capabilities.get("effective_moa_provenance_page_integrity") == "pass"
    assert checks["effective_moa_provenance_page_integrity"]["severity"] == "required"
    assert capabilities.get("miss_records_present") is True
    assert capabilities.get("miss_record_count") == 2
    assert capabilities.get("miss_record_backlog_linkage") == "pass"
    assert capabilities.get("miss_record_signal_alignment") == "pass"


def _test_pack_without_local_miss_records_stays_green_on_miss_linkage_checks() -> None:
    scorecard = evaluate_contract_pack(
        CONTRACTS_ROOT / "local7_safeway_pueblo_meat_2022",
        write_scorecard=False,
    )
    checks = _check_map(scorecard)
    capabilities = scorecard.get("capabilities") or {}
    assert checks["miss_record_backlog_linkage"]["status"] == "pass"
    assert checks["miss_record_signal_alignment"]["status"] == "pass"
    assert checks["moa_wage_schedule_metadata_integrity"]["status"] == "pass"
    assert checks["moa_wage_schedule_sync_registration"]["status"] == "pass"
    assert checks["effective_moa_provenance_page_integrity"]["status"] == "pass"
    assert checks["effective_moa_provenance_page_integrity"]["severity"] == "required"
    assert capabilities.get("miss_records_present") is False
    assert capabilities.get("miss_record_count") == 0


def _test_effective_moa_provenance_metrics_detect_missing_pages() -> None:
    with tempfile.TemporaryDirectory(prefix="karl_pack_acceptance_") as tmpdir:
        package_dir = Path(tmpdir) / "demo_contract"
        _write_json(
            package_dir / "effective" / "latest.json",
            {
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "a" * 64,
            },
        )
        _write_json(
            package_dir / "effective" / "effective_2025_07_05" / "effective_contract.json",
            {
                "contract_id": "demo_contract",
                "sections": [
                    {
                        "citation": "Article 15, Section 34",
                        "provenance": [
                            {
                                "source_type": "base",
                                "pdf": "Base-CBA.pdf",
                                "pdf_page": 55,
                            },
                            {
                                "source_type": "moa",
                                "pdf": "Signed-MOA.pdf",
                                "pdf_page": None,
                                "source_doc_id": "demo_moa_2025",
                            },
                        ],
                    }
                ],
                "tables": {
                    "appendix_a_wage_rows": {
                        "table_id": "appendix_a_wage_rows",
                        "rows": [
                            {
                                "row_key": "cc_start",
                                "provenance": [
                                    {
                                        "source_type": "moa",
                                        "pdf": "Signed-MOA.pdf",
                                        "pdf_page": 7,
                                        "source_doc_id": "demo_moa_2025",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        )
        metrics = _effective_moa_provenance_page_metrics(package_dir)
        assert metrics["effective_version_id"] == "effective_2025_07_05"
        assert metrics["moa_ref_count"] == 2
        assert metrics["missing_page_ref_count"] == 1
        assert metrics["sections_missing_page"] == ["Article 15, Section 34"]
        assert metrics["tables_missing_page"] == []
        assert metrics["source_doc_ids"] == ["demo_moa_2025"]


def main() -> None:
    _test_refreshed_older_pack_now_materializes_side_letter_doc_types()
    _test_effective_packs_now_include_entitlement_index_inputs()
    _test_refreshed_local_packs_have_no_required_failures()
    _test_pack_capabilities_surface_green_required_state()
    _test_pack_without_local_miss_records_stays_green_on_miss_linkage_checks()
    _test_effective_moa_provenance_metrics_detect_missing_pages()
    print("[OK] pack acceptance capability checks passed")


if __name__ == "__main__":
    main()
