"""
Validate multi-contract unanswerable dataset structure and coverage floors.

Release intent:
- prevent silent benchmark drift
- enforce schema and distribution minimums
- ensure active contract coverage
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.contracts import list_contract_catalog


DEFAULT_SCHEMA_VERSION = "unanswerable_multi_contract_test_v1"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _active_contract_ids() -> list[str]:
    return sorted(
        str(c.get("contract_id"))
        for c in list_contract_catalog()
        if c.get("contract_id")
    )


def run(
    dataset_path: Path,
    required_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_scenario_types: int,
) -> tuple[bool, list[str], dict]:
    issues: list[str] = []
    data = _load_json(dataset_path)
    test_cases = list(data.get("test_cases") or [])

    schema_version = str(data.get("schema_version") or "")
    if schema_version != required_schema_version:
        issues.append(
            f"schema_version mismatch: expected '{required_schema_version}', got '{schema_version or '<missing>'}'"
        )

    if len(test_cases) < min_total_cases:
        issues.append(f"total test cases too low: {len(test_cases)} < {min_total_cases}")

    by_contract = Counter()
    by_scenario = Counter()
    seen_ids: set[str] = set()

    active_contracts = set(_active_contract_ids())
    if not active_contracts:
        issues.append("no active contracts found in runtime catalog")

    for idx, case in enumerate(test_cases):
        case_id = str(case.get("id") or "").strip()
        contract_id = str(case.get("contract_id") or "").strip()
        question = str(case.get("question") or "").strip()
        scenario = str(case.get("scenario") or "").strip().lower()
        expected_behavior = str(case.get("expected_behavior") or "").strip().lower()
        expected_escalation = case.get("expected_escalation")

        if not case_id:
            issues.append(f"case[{idx}] missing id")
        elif case_id in seen_ids:
            issues.append(f"duplicate case id: {case_id}")
        else:
            seen_ids.add(case_id)

        if not contract_id:
            issues.append(f"{case_id or f'case[{idx}]'} missing contract_id")
        elif active_contracts and contract_id not in active_contracts:
            issues.append(f"{case_id or f'case[{idx}]'} contract_id not active: {contract_id}")

        if not question:
            issues.append(f"{case_id or f'case[{idx}]'} missing question")

        if expected_behavior != "uncertain":
            issues.append(f"{case_id or f'case[{idx}]'} expected_behavior must be 'uncertain'")

        if expected_escalation is not None and not isinstance(expected_escalation, bool):
            issues.append(f"{case_id or f'case[{idx}]'} expected_escalation must be boolean when provided")

        if not scenario:
            issues.append(f"{case_id or f'case[{idx}]'} missing scenario")
        else:
            by_scenario[scenario] += 1

        by_contract[contract_id] += 1

    if len(by_scenario) < min_scenario_types:
        issues.append(
            f"scenario diversity too low: {len(by_scenario)} < {min_scenario_types}"
        )

    for contract_id in sorted(active_contracts):
        total = int(by_contract[contract_id])
        if total < min_cases_per_contract:
            issues.append(
                f"{contract_id} total cases too low: {total} < {min_cases_per_contract}"
            )

    summary = {
        "schema_version": schema_version,
        "total_cases": len(test_cases),
        "scenario_counts": dict(by_scenario),
        "by_contract_total": dict(by_contract),
        "active_contract_ids": sorted(active_contracts),
    }
    return (len(issues) == 0), issues, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate unanswerable dataset structure/coverage.")
    parser.add_argument(
        "--dataset",
        default=str(DATA_DIR / "test_set" / "unanswerable_multi_contract_test.json"),
    )
    parser.add_argument("--required-schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--min-total-cases", type=int, default=12)
    parser.add_argument("--min-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-scenario-types", type=int, default=3)
    args = parser.parse_args()

    ok, issues, summary = run(
        dataset_path=Path(args.dataset),
        required_schema_version=args.required_schema_version,
        min_total_cases=args.min_total_cases,
        min_cases_per_contract=args.min_cases_per_contract,
        min_scenario_types=args.min_scenario_types,
    )

    print("=" * 72)
    print("KARL Unanswerable Dataset Validation")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Schema version: {summary.get('schema_version')}")
    print(f"Total cases: {summary.get('total_cases')}")
    print(f"Scenario counts: {summary.get('scenario_counts')}")
    print(f"By contract total: {summary.get('by_contract_total')}")

    if not ok:
        print("\nValidation status: BLOCKED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("\nValidation status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
