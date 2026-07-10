"""Seed MOA patch with Appendix A wage-row roll-forward ops at patch effective date.

Use case:
- An MOA restates/extends current Appendix A wage schedules, but the approved patch only
  includes a small number of manual wage row ops.
- We want KARL to treat all Appendix A rows as current-effective at the MOA effective date
  (so runtime lookups stop citing old base dates) without hand-authoring hundreds of row ops.

This utility appends `replace_table_row` ops for the latest existing wage rows (typically the
base contract's most recent effective date) so the materializer supersedes them into the MOA
effective date.

Notes:
- It uses existing row values (current-rate roll-forward). It does not infer future schedule
  columns (e.g., +52 / +104 week rates) from the MOA text.
- It is idempotent with respect to row keys already targeted by the patch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


WAGE_TABLE_ID = "appendix_a_wage_rows"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _iso_dates(rows: list[dict[str, Any]]) -> list[str]:
    out = sorted(
        {
            str(r.get("effective_date") or "").strip()
            for r in rows
            if isinstance(r, dict) and str(r.get("effective_date") or "").strip()
        }
    )
    return [d for d in out if len(d) == 10 and d[4] == "-" and d[7] == "-"]


def _infer_default_wage_artifact(contract_id: str) -> Path:
    return Path("data") / "wages" / f"wage_tables_{contract_id}.json"


def _existing_targeted_row_keys(patch_payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for op in patch_payload.get("operations") or []:
        if not isinstance(op, dict):
            continue
        if str(op.get("op") or "").strip() != "replace_table_row":
            continue
        target = op.get("target") if isinstance(op.get("target"), dict) else {}
        if str(target.get("table_id") or "").strip() != WAGE_TABLE_ID:
            continue
        row_key = str(target.get("row_key") or "").strip()
        if row_key:
            out.add(row_key)
    return out


def _seed_ops(
    *,
    patch_payload: dict[str, Any],
    wages_payload: dict[str, Any],
    source_effective_date: str | None,
    review_status: str,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patch_effective_date = str(patch_payload.get("effective_date") or "").strip()
    if not patch_effective_date:
        raise ValueError("Patch missing effective_date")
    source_pdf = str(patch_payload.get("source_pdf") or "").strip() or None
    source_doc_id = str(patch_payload.get("source_doc_id") or "").strip() or None

    rows = wages_payload.get("canonical_wage_rows") or []
    if not isinstance(rows, list):
        raise ValueError("wage artifact canonical_wage_rows must be a list")

    dates = _iso_dates(rows)
    if not dates:
        raise ValueError("No effective dates found in canonical_wage_rows")

    if source_effective_date:
        seed_date = source_effective_date
    else:
        prior = [d for d in dates if d < patch_effective_date]
        seed_date = prior[-1] if prior else dates[-1]

    targeted = _existing_targeted_row_keys(patch_payload)
    new_ops: list[dict[str, Any]] = []
    skipped_already_targeted = 0
    skipped_missing_hash = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("effective_date") or "").strip() != seed_date:
            continue
        row_key = str(row.get("row_key") or "").strip()
        if not row_key:
            continue
        if row_key in targeted:
            skipped_already_targeted += 1
            continue
        row_hash = str(row.get("row_hash") or "").strip().lower()
        if not row_hash:
            skipped_missing_hash += 1
            continue
        rate = row.get("rate")
        if rate is None:
            continue
        source_refs = []
        ref = {}
        if source_pdf:
            ref["pdf"] = source_pdf
        if source_doc_id:
            ref["source_doc_id"] = source_doc_id
        ref["source_type"] = "moa"
        source_refs.append(ref)
        new_ops.append(
            {
                "op": "replace_table_row",
                "target": {
                    "table_id": WAGE_TABLE_ID,
                    "row_key": row_key,
                },
                "expected_prev_hash": row_hash,
                # Same rate value intentionally: materializer will supersede row to patch effective date.
                "new_row": {
                    "rate": float(rate),
                },
                "source_refs": source_refs,
                "confidence": float(confidence),
                "review_status": review_status,
            }
        )
        selected_schedule_label = str(row.get("selected_schedule_label") or "").strip()
        if selected_schedule_label:
            new_ops[-1]["new_row"]["selected_schedule_label"] = selected_schedule_label
        if isinstance(row.get("source_rate_schedule"), dict):
            new_ops[-1]["new_row"]["source_rate_schedule"] = dict(row.get("source_rate_schedule") or {})
        targeted.add(row_key)

    summary = {
        "seed_source_effective_date": seed_date,
        "patch_effective_date": patch_effective_date,
        "generated_ops": len(new_ops),
        "skipped_already_targeted": skipped_already_targeted,
        "skipped_missing_hash": skipped_missing_hash,
    }
    return new_ops, summary


def run(
    *,
    patch_file: Path,
    wage_file: Path | None = None,
    source_effective_date: str | None = None,
    review_status: str = "approved",
    confidence: float = 0.7,
    in_place: bool = False,
    output: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    patch_payload = _load_json(patch_file)
    contract_id = str(patch_payload.get("contract_id") or "").strip()
    if not contract_id:
        raise ValueError("Patch file missing contract_id")

    wage_path = wage_file or _infer_default_wage_artifact(contract_id)
    if not wage_path.exists():
        raise FileNotFoundError(f"Wage artifact not found: {wage_path}")
    wages_payload = _load_json(wage_path)

    new_ops, summary = _seed_ops(
        patch_payload=patch_payload,
        wages_payload=wages_payload,
        source_effective_date=source_effective_date,
        review_status=review_status,
        confidence=confidence,
    )

    patch_payload.setdefault("operations", [])
    if not isinstance(patch_payload["operations"], list):
        raise ValueError("patch operations must be a list")
    patch_payload["operations"].extend(new_ops)

    target_path: Path
    if in_place:
        target_path = patch_file
    else:
        target_path = output or patch_file.with_name(f"{patch_file.stem}.seeded{patch_file.suffix}")

    _write_json(target_path, patch_payload)
    return target_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed MOA patch with Appendix A wage roll-forward table-row ops.")
    parser.add_argument("--patch-file", required=True, help="Path to MOA patch JSON")
    parser.add_argument("--wage-file", default=None, help="Optional wage artifact JSON (defaults to data/wages/wage_tables_<contract>.json)")
    parser.add_argument("--source-effective-date", default=None, help="Explicit source effective date to clone from (default: latest < patch effective date)")
    parser.add_argument("--review-status", default="approved", choices=["approved", "pending"])
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--in-place", action="store_true", help="Overwrite patch file")
    parser.add_argument("--output", default=None, help="Output file path (ignored with --in-place)")
    args = parser.parse_args()

    out_path, summary = run(
        patch_file=Path(args.patch_file),
        wage_file=Path(args.wage_file) if args.wage_file else None,
        source_effective_date=args.source_effective_date,
        review_status=args.review_status,
        confidence=float(args.confidence),
        in_place=bool(args.in_place),
        output=Path(args.output) if args.output else None,
    )
    print("=" * 72)
    print("KARL MOA Wage Roll-Forward Seeder")
    print("=" * 72)
    print(f"Output: {out_path}")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
