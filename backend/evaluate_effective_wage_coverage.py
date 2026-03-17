"""Validate MOA-effective wage chronology in latest effective snapshots.

Purpose:
- Catch cases where wage table patches were applied but the effective wage artifact
  did not materialize a row at the patch effective date (e.g., historical row was
  overwritten instead of superseded).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.effective_contracts import resolve_latest_effective_version_id


WAGE_TABLE_ID = "appendix_a_wage_rows"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _discover_contract_ids(explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return sorted({str(v).strip() for v in explicit if str(v).strip()})
    out: list[str] = []
    contracts_root = DATA_DIR / "contracts"
    if not contracts_root.exists():
        return []
    for contract_dir in sorted(contracts_root.iterdir()):
        if not contract_dir.is_dir():
            continue
        if (contract_dir / "effective" / "latest.json").exists():
            out.append(contract_dir.name)
    return out


def run(contract_ids: Optional[list[str]] = None) -> dict:
    results: list[dict] = []

    for contract_id in _discover_contract_ids(contract_ids):
        effective_version_id = resolve_latest_effective_version_id(contract_id)
        version_dir = DATA_DIR / "contracts" / contract_id / "effective" / str(effective_version_id or "")
        build_log = _load_json(version_dir / "build_log.json") if effective_version_id else {}
        patch_chain = _load_json(version_dir / "patch_chain.json") if effective_version_id else {}
        wages = _load_json(version_dir / "index_inputs" / f"wage_tables_{contract_id}.json") if effective_version_id else {}

        missing_files: list[str] = []
        if not effective_version_id:
            missing_files.append("latest effective version")
        else:
            for rel in [
                "build_log.json",
                "patch_chain.json",
                f"index_inputs/wage_tables_{contract_id}.json",
            ]:
                if not (version_dir / rel).exists():
                    missing_files.append(str(version_dir / rel))

        patch_dates_by_id: dict[str, str] = {}
        for row in (patch_chain.get("patches") or []):
            if not isinstance(row, dict):
                continue
            patch_id = str(row.get("patch_id") or "").strip()
            eff_date = str(row.get("effective_date") or "").strip()
            if patch_id and eff_date:
                patch_dates_by_id[patch_id] = eff_date

        wage_rows = wages.get("canonical_wage_rows") or []
        if not isinstance(wage_rows, list):
            wage_rows = []

        rows_by_patch_and_date: set[tuple[str, str]] = set()
        all_wage_dates: set[str] = set()
        for row in wage_rows:
            if not isinstance(row, dict):
                continue
            eff_date = str(row.get("effective_date") or "").strip()
            if eff_date:
                all_wage_dates.add(eff_date)
            for patch_id in (row.get("amendments_applied") or []):
                patch_id_norm = str(patch_id or "").strip()
                if patch_id_norm and eff_date:
                    rows_by_patch_and_date.add((patch_id_norm, eff_date))

        wage_patch_ops: list[dict] = []
        for op in (build_log.get("operations") or []):
            if not isinstance(op, dict):
                continue
            target = op.get("target") if isinstance(op.get("target"), dict) else {}
            if str(op.get("op") or "").strip() != "replace_table_row":
                continue
            if str(target.get("table_id") or "").strip() != WAGE_TABLE_ID:
                continue
            if not bool(op.get("applied")):
                continue
            op_id = str(op.get("op_id") or "").strip()
            patch_id = op_id.split("#", 1)[0] if "#" in op_id else op_id
            patch_effective_date = patch_dates_by_id.get(patch_id)
            wage_patch_ops.append(
                {
                    "op_id": op_id,
                    "patch_id": patch_id or None,
                    "patch_effective_date": patch_effective_date,
                    "target_row_key": str(target.get("row_key") or "").strip() or None,
                }
            )

        missing_patch_dates: list[dict] = []
        for op in wage_patch_ops:
            patch_id = str(op.get("patch_id") or "").strip()
            patch_effective_date = str(op.get("patch_effective_date") or "").strip()
            if not patch_id or not patch_effective_date:
                missing_patch_dates.append({**op, "reason": "missing_patch_effective_date"})
                continue
            if (patch_id, patch_effective_date) not in rows_by_patch_and_date:
                missing_patch_dates.append({**op, "reason": "no_row_materialized_at_patch_effective_date"})

        pass_row = (not missing_files) and (len(missing_patch_dates) == 0)
        results.append(
            {
                "contract_id": contract_id,
                "effective_version_id": effective_version_id,
                "missing_files": missing_files,
                "wage_effective_dates": sorted(all_wage_dates),
                "wage_patch_op_count": len(wage_patch_ops),
                "wage_patch_ops": wage_patch_ops,
                "missing_patch_date_materializations": missing_patch_dates,
                "pass": pass_row,
            }
        )

    passed = sum(1 for row in results if row.get("pass"))
    total = len(results)
    return {
        "schema_version": "effective_wage_snapshot_coverage_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
        },
        "results": results,
    }


def _write_report(report: dict, out_path: Optional[Path] = None) -> Path:
    target = out_path or (DATA_DIR / "test_set" / "effective_wage_snapshot_coverage_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate effective wage snapshot chronology for MOA table-row patches.")
    parser.add_argument("--contract-id", action="append", default=None, help="Contract ID to validate (repeatable)")
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    report = run(contract_ids=args.contract_id)
    out_path = _write_report(report, out_path=Path(args.output) if args.output else None)
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL Effective Wage Snapshot Coverage")
    print("=" * 72)
    print(
        f"Pass: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    print(f"Results: {out_path}")
    for row in report.get("results") or []:
        print(
            f"- {row.get('contract_id')}: pass={row.get('pass')} "
            f"wage_patch_ops={row.get('wage_patch_op_count')} "
            f"missing_materializations={len(row.get('missing_patch_date_materializations') or [])}"
        )
    return 0 if int(overall.get("passed", 0)) == int(overall.get("total", 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
