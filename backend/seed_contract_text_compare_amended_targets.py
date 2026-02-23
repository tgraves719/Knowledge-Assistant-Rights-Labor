"""Seed/expand amended-section compare regression targets from approved MOA patches.

Deterministically samples approved `replace_section` ops from ingested contracts and writes
`contract_text_compare_amended_targets_test.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.evaluate_contract_text_compare_amended import (
    DEFAULT_INPUT,
    discover_approved_replace_section_inventory,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return path


def _case_id(contract_id: str, patch_id: str, article_num: int, section_num: int) -> str:
    return f"{contract_id}_{patch_id}_a{article_num}_s{section_num}".replace("-", "_")


def _case_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("contract_id") or "").strip(),
        int(row.get("article_num") or 0),
        int(row.get("section_num") or 0),
    )


def _deterministic_selection(
    inventory_ops: list[dict[str, Any]],
    *,
    per_contract: int,
    include_contracts: set[str],
) -> list[dict[str, Any]]:
    by_contract: dict[str, list[dict[str, Any]]] = {}
    for row in inventory_ops:
        cid = str(row.get("contract_id") or "").strip()
        if not cid:
            continue
        if include_contracts and cid not in include_contracts:
            continue
        by_contract.setdefault(cid, []).append(row)

    selected: list[dict[str, Any]] = []
    for cid in sorted(by_contract):
        rows = sorted(
            by_contract[cid],
            key=lambda r: (
                int(r.get("article_num") or 0),
                int(r.get("section_num") or 0),
                str(r.get("patch_id") or ""),
                int(r.get("op_index") or 0),
            ),
        )
        selected.extend(rows[: max(1, int(per_contract))])
    return selected


def _build_case(row: dict[str, Any]) -> dict[str, Any]:
    contract_id = str(row.get("contract_id") or "").strip()
    patch_id = str(row.get("patch_id") or "").strip()
    article_num = int(row.get("article_num") or 0)
    section_num = int(row.get("section_num") or 0)
    return {
        "id": _case_id(contract_id, patch_id, article_num, section_num),
        "description": (
            f"Auto-seeded amended section target from approved patch {patch_id}: "
            f"Article {article_num}, Section {section_num} should differ between base and effective."
        ),
        "contract_id": contract_id,
        "article_num": article_num,
        "section_num": section_num,
        "expect_difference": True,
    }


def run(
    *,
    output_path: Path = DEFAULT_INPUT,
    per_contract: int = 2,
    merge_existing: bool = True,
    prune_unselected: bool = False,
    include_contract_ids: list[str] | None = None,
) -> dict[str, Any]:
    inventory = discover_approved_replace_section_inventory(data_dir=DATA_DIR)
    inventory_ops = list(inventory.get("operations") or [])
    include_set = {c.strip() for c in (include_contract_ids or []) if str(c).strip()}
    selected_ops = _deterministic_selection(inventory_ops, per_contract=per_contract, include_contracts=include_set)

    existing_payload: dict[str, Any] = {}
    existing_cases: list[dict[str, Any]] = []
    if merge_existing and output_path.exists():
        try:
            existing_payload = _load_json(output_path)
            existing_cases = [row for row in list(existing_payload.get("test_cases") or []) if isinstance(row, dict)]
        except Exception:
            existing_payload = {}
            existing_cases = []

    existing_by_key = {_case_key(row): dict(row) for row in existing_cases if _case_key(row)[0]}
    selected_keys = set()
    generated_cases: list[dict[str, Any]] = []
    reused_count = 0
    added_count = 0

    for row in selected_ops:
        key = (str(row.get("contract_id") or "").strip(), int(row.get("article_num") or 0), int(row.get("section_num") or 0))
        selected_keys.add(key)
        if key in existing_by_key:
            generated_cases.append(existing_by_key[key])
            reused_count += 1
        else:
            generated_cases.append(_build_case(row))
            added_count += 1

    preserved_count = 0
    if merge_existing and not prune_unselected:
        for row in existing_cases:
            key = _case_key(row)
            if key in selected_keys:
                continue
            generated_cases.append(dict(row))
            preserved_count += 1

    generated_cases.sort(key=lambda r: (str(r.get("contract_id") or ""), int(r.get("article_num") or 0), int(r.get("section_num") or 0), str(r.get("id") or "")))

    payload = {
        "schema_version": "contract_text_compare_amended_targets_test_v1",
        "description": "Known amended section targets that should differ between immutable base chunks and effective snapshot chunks.",
        "seed_metadata": {
            "mode": "merge" if merge_existing else "replace",
            "prune_unselected": bool(prune_unselected),
            "per_contract": int(per_contract),
            "include_contract_ids": sorted(include_set),
            "inventory_summary": inventory.get("summary") or {},
            "selected_targets": len(selected_ops),
            "reused_cases": reused_count,
            "added_cases": added_count,
            "preserved_unselected_existing_cases": preserved_count,
        },
        "test_cases": generated_cases,
    }
    _write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed amended compare targets from approved replace_section MOA ops.")
    parser.add_argument("--output", default=str(DEFAULT_INPUT))
    parser.add_argument("--per-contract", type=int, default=2)
    parser.add_argument("--replace", action="store_true", help="Replace dataset instead of merging existing cases.")
    parser.add_argument("--prune-unselected", action="store_true", help="When merging, drop unselected existing cases.")
    parser.add_argument("--contract-id", action="append", default=[], help="Optional contract filter (repeatable).")
    args = parser.parse_args()

    payload = run(
        output_path=Path(args.output),
        per_contract=max(1, int(args.per_contract)),
        merge_existing=not bool(args.replace),
        prune_unselected=bool(args.prune_unselected),
        include_contract_ids=list(args.contract_id or []),
    )
    seed_meta = payload.get("seed_metadata") or {}
    print("=" * 72)
    print("KARL Amended Compare Target Seeder")
    print("=" * 72)
    print(f"Inventory approved replace_section ops: {(seed_meta.get('inventory_summary') or {}).get('approved_replace_section_ops', 0)}")
    print(f"Selected targets: {seed_meta.get('selected_targets', 0)}")
    print(f"Cases total: {len(list(payload.get('test_cases') or []))}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
