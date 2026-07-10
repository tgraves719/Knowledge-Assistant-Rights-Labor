"""
Deterministic canonical wage-row lookup checks.

Ensures runtime wage lookup resolves through canonical row artifacts (when
present) and includes structured table evidence metadata.
"""

from pathlib import Path
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.extract_wages import lookup_wage
from backend.wage_files import resolve_wage_file


def _load_wages(contract_id: str) -> dict:
    path = Path(f"data/wages/wage_tables_{contract_id}.json")
    assert path.exists(), f"Missing wage artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _test_canonical_lookup_includes_table_evidence() -> None:
    wages = _load_wages("local7_safeway_pueblo_clerks_2022")
    result = lookup_wage(
        wages_data=wages,
        classification="dug_shopper",
        hours_worked=0,
        months_employed=0,
    )
    assert result is not None, "Expected wage lookup result for dug_shopper"
    assert result.get("source_method") == "canonical_rows", (
        "Expected canonical rows to be primary wage lookup source."
    )
    evidence = result.get("table_evidence") or []
    assert evidence, "Expected table evidence rows in canonical lookup."
    assert any(str((row or {}).get("table_id") or "").strip() for row in evidence), (
        "Expected at least one table-evidence row with table_id."
    )


def _test_fallback_without_canonical_rows() -> None:
    wages = _load_wages("local7_safeway_pueblo_clerks_2022")
    fallback_wages = copy.deepcopy(wages)
    fallback_wages["canonical_wage_rows"] = []
    result = lookup_wage(
        wages_data=fallback_wages,
        classification="dug_shopper",
        hours_worked=0,
        months_employed=0,
    )
    assert result is not None, "Expected fallback wage lookup to still resolve."
    assert result.get("source_method") == "classification_steps", (
        "Expected deterministic step-based fallback when canonical rows are absent."
    )
    assert result.get("table_evidence") == [], "Fallback lookup should not include table evidence rows."


def _test_contract_scoped_effective_wage_lookup_prefers_latest_snapshot() -> None:
    path = resolve_wage_file(
        contract_id="local7_safeway_pueblo_clerks_2022",
        allow_shared_fallback=False,
    )
    assert path is not None and path.exists(), "Expected contract-scoped wage artifact to resolve."
    normalized = str(path).replace("\\", "/")
    assert "effective_local7_safeway_moa_2025_07_05" in normalized, (
        "Expected latest effective wage snapshot to be preferred for contract-scoped lookup."
    )
    with open(path, "r", encoding="utf-8") as f:
        wages = json.load(f)

    result = lookup_wage(
        wages_data=wages,
        classification="courtesy_clerk",
        hours_worked=0,
        months_employed=1,
    )
    assert result is not None, "Expected contract-scoped effective wage lookup result."
    assert result.get("effective_date") == "2025-07-05", "Expected latest effective date from resolved artifact."
    assert float(result.get("rate") or 0.0) == 17.25, "Expected Courtesy Clerk start rate to resolve from FSAR, not pre-ratification Current."
    assert str(result.get("selected_schedule_label") or "") == "FSAR", (
        "Expected effective wage lookup to preserve the selected MOA schedule label."
    )
    source_schedule = dict(result.get("source_rate_schedule") or {})
    assert float(source_schedule.get("Current") or 0.0) == 17.0, "Expected source schedule to preserve the pre-ratification Current column."
    assert float(source_schedule.get("FSAR") or 0.0) == 17.25, "Expected source schedule to preserve the ratified FSAR column."
    evidence = result.get("table_evidence") or []
    assert evidence, "Expected table evidence rows for effective wage lookup."
    assert any(str((row or {}).get("table_id") or "").strip() == "tbl_art58_3" for row in evidence), (
        "Expected Courtesy Clerk evidence to retain Appendix table id."
    )
    assert any(str((row or {}).get("source_type") or "").strip().lower() == "moa" for row in evidence), (
        "Expected effective wage evidence to expose MOA provenance."
    )
    assert any(
        str((row or {}).get("source_doc_id") or "").strip() == "albertsons_safeway_moa_2025_07_05"
        for row in evidence
    ), "Expected effective wage evidence to carry MOA source_doc_id."


def main() -> None:
    _test_canonical_lookup_includes_table_evidence()
    _test_fallback_without_canonical_rows()
    _test_contract_scoped_effective_wage_lookup_prefers_latest_snapshot()
    print("[OK] Canonical wage lookup checks passed")


if __name__ == "__main__":
    main()
