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


def main() -> None:
    _test_canonical_lookup_includes_table_evidence()
    _test_fallback_without_canonical_rows()
    print("[OK] Canonical wage lookup checks passed")


if __name__ == "__main__":
    main()
