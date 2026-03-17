"""
Canonical role-catalog integrity evaluator.

Validates that contract-scoped role options remain deterministic and safe:
- default onboarding roles are wage-resolvable
- unresolved manifest roles do not leak into default options
- onboarding-default role aliases collapse to one default per wage_key
- targeted cross-contract role expectations hold (dataset driven)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.contracts import get_contract_catalog_entry
from backend.ingest.extract_wages import normalize_classification_name
from backend.role_catalog_files import resolve_role_catalog_file
from backend.user.profile import get_classification_options
from backend.wage_files import resolve_wage_file


def _canonical_contract_id(contract_id: str) -> str:
    entry = get_contract_catalog_entry(contract_id)
    if entry and entry.get("contract_id"):
        return str(entry["contract_id"])
    return str(contract_id)


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _option_map(options: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for opt in options or []:
        value = str((opt or {}).get("value") or "").strip().lower()
        if value:
            out[value] = opt
    return out


def _contract_wage_keys(contract_id: str) -> set[str]:
    wage_file = resolve_wage_file(contract_id=contract_id, allow_shared_fallback=False)
    if not wage_file or not wage_file.exists():
        return set()
    try:
        wages = _load_json(wage_file)
    except Exception:
        return set()

    out: set[str] = set()
    classes = (wages or {}).get("classifications") or {}
    for key, cls in classes.items():
        norm = normalize_classification_name(str((cls or {}).get("normalized_name") or key or ""))
        if norm:
            out.add(norm)
    return out


def _evaluate_contract_invariants(contract_id: str) -> list[dict]:
    rows: list[dict] = []
    default_opts = get_classification_options(contract_id=contract_id)
    all_opts = get_classification_options(contract_id=contract_id, include_unmapped=True)
    default_values = {str(opt.get("value") or "").strip().lower() for opt in default_opts}
    wage_keys = _contract_wage_keys(contract_id)

    role_catalog_path = resolve_role_catalog_file(
        contract_id=contract_id,
        allow_shared_fallback=False,
    )
    role_catalog = {}
    role_catalog_ok = False
    if role_catalog_path and role_catalog_path.exists():
        try:
            role_catalog = _load_json(role_catalog_path)
            role_catalog_ok = True
        except Exception:
            role_catalog_ok = False

    roles = role_catalog.get("roles") if isinstance(role_catalog, dict) else []
    schema_valid = (
        role_catalog_ok
        and role_catalog.get("schema_version") in {"role_catalog_v1", "role_catalog_v2"}
        and role_catalog.get("contract_id") == contract_id
        and isinstance(roles, list)
        and len(roles) > 0
    )
    rows.append(
        {
            "kind": "contract_invariant",
            "check_id": "role_catalog_schema_valid",
            "contract_id": contract_id,
            "pass": bool(schema_valid),
            "details": {
                "role_catalog_path": str(role_catalog_path) if role_catalog_path else None,
                "schema_version": role_catalog.get("schema_version") if isinstance(role_catalog, dict) else None,
                "catalog_contract_id": role_catalog.get("contract_id") if isinstance(role_catalog, dict) else None,
                "role_count": len(roles) if isinstance(roles, list) else 0,
            },
        }
    )

    default_unmapped = []
    missing_wage_keys = []
    for opt in default_opts:
        value = str(opt.get("value") or "").strip().lower()
        wage_available = bool(opt.get("wage_available"))
        wage_key = normalize_classification_name(str(opt.get("wage_key") or value))
        if not wage_available:
            default_unmapped.append(value)
            continue
        if wage_key and wage_key not in wage_keys:
            missing_wage_keys.append({"value": value, "wage_key": wage_key})

    default_wage_ready = bool(default_opts) and not default_unmapped and not missing_wage_keys
    rows.append(
        {
            "kind": "contract_invariant",
            "check_id": "default_options_wage_ready",
            "contract_id": contract_id,
            "pass": bool(default_wage_ready),
            "details": {
                "default_option_count": len(default_opts),
                "default_unmapped_values": sorted(default_unmapped),
                "missing_wage_keys": missing_wage_keys[:20],
            },
        }
    )

    unresolved_manifest = {
        str(opt.get("value") or "").strip().lower()
        for opt in all_opts
        if bool(opt.get("manifest_present")) and not bool(opt.get("wage_available"))
    }
    unresolved_not_default = unresolved_manifest.isdisjoint(default_values)
    rows.append(
        {
            "kind": "contract_invariant",
            "check_id": "unresolved_manifest_not_default",
            "contract_id": contract_id,
            "pass": bool(unresolved_not_default),
            "details": {
                "unresolved_manifest_values": sorted(unresolved_manifest),
                "default_values": sorted(default_values),
                "overlap": sorted(unresolved_manifest & default_values),
            },
        }
    )

    duplicate_default_wage_keys = []
    if isinstance(roles, list):
        counts: dict[str, int] = {}
        for role in roles:
            if not isinstance(role, dict) or not bool(role.get("onboarding_default")):
                continue
            wage_key = normalize_classification_name(str(role.get("wage_key") or ""))
            if not wage_key:
                continue
            counts[wage_key] = counts.get(wage_key, 0) + 1
        duplicate_default_wage_keys = sorted(k for k, c in counts.items() if c > 1)

    default_wage_key_unique = schema_valid and len(duplicate_default_wage_keys) == 0
    rows.append(
        {
            "kind": "contract_invariant",
            "check_id": "default_wage_key_unique",
            "contract_id": contract_id,
            "pass": bool(default_wage_key_unique),
            "details": {
                "duplicate_default_wage_keys": duplicate_default_wage_keys,
            },
        }
    )

    return rows


def _evaluate_dataset_cases(test_cases: list[dict]) -> list[dict]:
    rows: list[dict] = []
    options_cache: dict[str, dict[str, dict[str, dict]]] = {}

    def _contract_options(contract_id: str) -> tuple[dict[str, dict], dict[str, dict]]:
        if contract_id in options_cache:
            cached = options_cache[contract_id]
            return cached["default"], cached["all"]
        default_map = _option_map(get_classification_options(contract_id=contract_id))
        all_map = _option_map(get_classification_options(contract_id=contract_id, include_unmapped=True))
        options_cache[contract_id] = {"default": default_map, "all": all_map}
        return default_map, all_map

    for case in test_cases:
        case_id = str(case.get("id") or "")
        contract_id = _canonical_contract_id(str(case.get("contract_id") or ""))
        role_value = normalize_classification_name(str(case.get("role_value") or ""))

        default_map, all_map = _contract_options(contract_id)
        in_default = role_value in default_map
        in_all = role_value in all_map
        wage_available = None
        if in_default:
            wage_available = bool((default_map.get(role_value) or {}).get("wage_available"))
        elif in_all:
            wage_available = bool((all_map.get(role_value) or {}).get("wage_available"))

        expected_default = _bool_or_none(case.get("expect_default_present"))
        expected_all = _bool_or_none(case.get("expect_all_present"))
        expected_wage = _bool_or_none(case.get("expect_wage_available"))

        checks: list[bool] = []
        if expected_default is not None:
            checks.append(in_default == expected_default)
        if expected_all is not None:
            checks.append(in_all == expected_all)
        if expected_wage is not None:
            checks.append(wage_available == expected_wage)
        passed = all(checks) if checks else False

        rows.append(
            {
                "kind": "dataset_case",
                "check_id": case_id or "dataset_case",
                "contract_id": contract_id,
                "role_value": role_value,
                "pass": bool(passed),
                "details": {
                    "in_default": in_default,
                    "in_all": in_all,
                    "wage_available": wage_available,
                    "expected_default_present": expected_default,
                    "expected_all_present": expected_all,
                    "expected_wage_available": expected_wage,
                },
            }
        )

    return rows


def run(test_file: Optional[Path] = None) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "role_catalog_integrity_test.json"

    payload = _load_json(test_file)
    test_cases = list(payload.get("test_cases") or [])

    contract_ids = sorted(
        {
            _canonical_contract_id(str(case.get("contract_id") or ""))
            for case in test_cases
            if str(case.get("contract_id") or "").strip()
        }
    )
    rows = []
    rows.extend(_evaluate_dataset_cases(test_cases))
    for contract_id in contract_ids:
        rows.extend(_evaluate_contract_invariants(contract_id))

    by_contract = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in rows:
        cid = str(row.get("contract_id") or "")
        if not cid:
            continue
        by_contract[cid]["total"] += 1
        if row.get("pass"):
            by_contract[cid]["passed"] += 1

    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass"))

    dataset_rows = [r for r in rows if r.get("kind") == "dataset_case"]
    default_ready_rows = [r for r in rows if r.get("check_id") == "default_options_wage_ready"]
    unresolved_rows = [r for r in rows if r.get("check_id") == "unresolved_manifest_not_default"]
    unique_rows = [r for r in rows if r.get("check_id") == "default_wage_key_unique"]

    def _rate(subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return round(sum(1 for r in subset if r.get("pass")) / len(subset), 4)

    by_contract_summary = {}
    for contract_id, stats in sorted(by_contract.items()):
        t = int(stats["total"])
        p = int(stats["passed"])
        by_contract_summary[contract_id] = {
            "passed": p,
            "total": t,
            "pass_rate": round((p / t) if t else 0.0, 4),
        }

    return {
        "schema_version": "role_catalog_integrity_eval_v1",
        "dataset_schema_version": str(payload.get("schema_version") or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_file": str(test_file),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
            "dataset_case_total": len(dataset_rows),
            "dataset_case_pass_rate": _rate(dataset_rows),
            "default_wage_ready_rate": _rate(default_ready_rows),
            "unresolved_not_default_rate": _rate(unresolved_rows),
            "default_wage_key_unique_rate": _rate(unique_rows),
        },
        "by_contract": by_contract_summary,
        "results": rows,
    }


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "role_catalog_integrity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate role-catalog integrity slice.")
    parser.add_argument("--input", type=str, default=None, help="Path to role-catalog-integrity benchmark JSON")
    args = parser.parse_args()

    test_file = Path(args.input) if args.input else None
    report = run(test_file=test_file)
    out_path = _write_report(report)

    overall = report["overall"]
    print("=" * 72)
    print("KARL Role-Catalog Integrity Evaluation")
    print("=" * 72)
    print(f"Overall: {overall['passed']}/{overall['total']} ({overall['pass_rate']:.1%})")
    print(f"Dataset-case pass rate: {overall['dataset_case_pass_rate']:.1%}")
    print(f"Default wage-ready rate: {overall['default_wage_ready_rate']:.1%}")
    print(f"Unresolved-not-default rate: {overall['unresolved_not_default_rate']:.1%}")
    print(f"Default wage-key-unique rate: {overall['default_wage_key_unique_rate']:.1%}")
    for cid, stats in report["by_contract"].items():
        print(f"- {cid}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
