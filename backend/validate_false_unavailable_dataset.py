"""
Validate false-unavailable dataset structure and coverage floors.

Release intent:
- prevent silent benchmark drift
- enforce schema and distribution minimums
- ensure active contract coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.contracts import list_contract_catalog


DEFAULT_SCHEMA_VERSION = "false_unavailable_test_v1"


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
    min_recover_cases: int,
    min_uncertain_cases: int,
    min_cases_per_contract: int,
    min_recover_per_contract: int,
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

    by_expectation = Counter()
    by_contract_total = Counter()
    by_contract_recover = Counter()
    seen_ids = set()

    active_contracts = set(_active_contract_ids())
    if not active_contracts:
        issues.append("no active contracts found in runtime catalog")

    for idx, case in enumerate(test_cases):
        case_id = str(case.get("id") or "").strip()
        expectation = str(case.get("expectation") or "").strip().lower()
        contract_id = str(case.get("contract_id") or "").strip()
        question = str(case.get("question") or "").strip()
        expected_citations = list(case.get("expected_citations") or [])

        if not case_id:
            issues.append(f"case[{idx}] missing id")
        elif case_id in seen_ids:
            issues.append(f"duplicate case id: {case_id}")
        else:
            seen_ids.add(case_id)

        if expectation not in {"recover", "uncertain"}:
            issues.append(f"{case_id or f'case[{idx}]'} invalid expectation '{expectation}'")
            continue

        if not contract_id:
            issues.append(f"{case_id or f'case[{idx}]'} missing contract_id")
        elif active_contracts and contract_id not in active_contracts:
            issues.append(f"{case_id or f'case[{idx}]'} contract_id not active: {contract_id}")

        if not question:
            issues.append(f"{case_id or f'case[{idx}]'} missing question")

        if expectation == "recover" and not expected_citations:
            issues.append(f"{case_id or f'case[{idx}]'} recover case missing expected_citations")
        if expectation == "uncertain" and expected_citations:
            issues.append(f"{case_id or f'case[{idx}]'} uncertain case should not have expected_citations")

        by_expectation[expectation] += 1
        by_contract_total[contract_id] += 1
        if expectation == "recover":
            by_contract_recover[contract_id] += 1

    if by_expectation["recover"] < min_recover_cases:
        issues.append(
            f"recover case count too low: {by_expectation['recover']} < {min_recover_cases}"
        )
    if by_expectation["uncertain"] < min_uncertain_cases:
        issues.append(
            f"uncertain case count too low: {by_expectation['uncertain']} < {min_uncertain_cases}"
        )

    for contract_id in sorted(active_contracts):
        total = int(by_contract_total[contract_id])
        recover = int(by_contract_recover[contract_id])
        if total < min_cases_per_contract:
            issues.append(
                f"{contract_id} total cases too low: {total} < {min_cases_per_contract}"
            )
        if recover < min_recover_per_contract:
            issues.append(
                f"{contract_id} recover cases too low: {recover} < {min_recover_per_contract}"
            )

    summary = {
        "schema_version": schema_version,
        "total_cases": len(test_cases),
        "expectation_counts": dict(by_expectation),
        "by_contract_total": dict(by_contract_total),
        "by_contract_recover": dict(by_contract_recover),
        "active_contract_ids": sorted(active_contracts),
    }
    return (len(issues) == 0), issues, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate false-unavailable dataset structure/coverage.")
    parser.add_argument(
        "--dataset",
        default=str(DATA_DIR / "test_set" / "false_unavailable_test.json"),
    )
    parser.add_argument("--required-schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--min-total-cases", type=int, default=12)
    parser.add_argument("--min-recover-cases", type=int, default=9)
    parser.add_argument("--min-uncertain-cases", type=int, default=3)
    parser.add_argument("--min-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-recover-per-contract", type=int, default=2)
    args = parser.parse_args()

    ok, issues, summary = run(
        dataset_path=Path(args.dataset),
        required_schema_version=args.required_schema_version,
        min_total_cases=args.min_total_cases,
        min_recover_cases=args.min_recover_cases,
        min_uncertain_cases=args.min_uncertain_cases,
        min_cases_per_contract=args.min_cases_per_contract,
        min_recover_per_contract=args.min_recover_per_contract,
    )

    print("=" * 72)
    print("KARL False-Unavailable Dataset Validation")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Schema version: {summary.get('schema_version')}")
    print(f"Total cases: {summary.get('total_cases')}")
    print(f"Expectation counts: {summary.get('expectation_counts')}")
    print(f"By contract total: {summary.get('by_contract_total')}")
    print(f"By contract recover: {summary.get('by_contract_recover')}")

    if not ok:
        print("\nValidation status: BLOCKED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("\nValidation status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
