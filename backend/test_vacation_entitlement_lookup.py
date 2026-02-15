"""Deterministic tests for vacation entitlement lookup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.retrieval.router import HybridRetriever


def _assert_weeks(
    retriever: HybridRetriever,
    contract_id: str,
    hire_date: str,
    months_employed: int,
    expected_weeks: int,
) -> None:
    result = retriever.lookup_vacation_entitlement(
        contract_id=contract_id,
        hire_date=hire_date,
        months_employed=months_employed,
        hours_worked=1800,
    )
    assert result is not None, f"Expected entitlement result for {contract_id}"
    assert result.get("source_method") == "vacation_entitlement_tiers"
    assert result.get("selected_schedule"), f"Expected selected schedule for {contract_id}"
    assert int(result.get("estimated_weeks_per_year") or -1) == expected_weeks
    assert "Article" in str(result.get("citation") or "")


def test_vacation_entitlement_exact_resolution() -> None:
    retriever = HybridRetriever(vector_store=None)

    _assert_weeks(
        retriever=retriever,
        contract_id="local7_safeway_pueblo_clerks_2022",
        hire_date="2004-01-15",
        months_employed=24,
        expected_weeks=2,
    )
    _assert_weeks(
        retriever=retriever,
        contract_id="local7_safeway_pueblo_clerks_2022",
        hire_date="2006-01-15",
        months_employed=24,
        expected_weeks=1,
    )
    _assert_weeks(
        retriever=retriever,
        contract_id="local7_kingsoopers_loveland_meat_2019",
        hire_date="2005-03-10",
        months_employed=36,
        expected_weeks=2,
    )


def test_vacation_entitlement_requires_hire_date_for_multi_schedule_contract() -> None:
    retriever = HybridRetriever(vector_store=None)
    result = retriever.lookup_vacation_entitlement(
        contract_id="local7_safeway_pueblo_meat_2022",
        hire_date=None,
        months_employed=36,
        hours_worked=1800,
    )
    assert result is not None
    assert result.get("selected_schedule") is None, "Should not guess schedule without hire_date"
    assert len(result.get("schedules_considered") or []) >= 2


def main() -> None:
    test_vacation_entitlement_exact_resolution()
    test_vacation_entitlement_requires_hire_date_for_multi_schedule_contract()
    print("[OK] Vacation entitlement lookup tests passed")


if __name__ == "__main__":
    main()
