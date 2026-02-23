"""CLI for deterministic effective-contract materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.ingest.moa_schema import load_patch_artifact
from backend.ingest.materializer import (
    ContractMaterializer,
    PatchRebaseFailure,
    WAGE_TABLE_ID,
    discover_patch_files,
    ensure_base_snapshot,
    load_base_contract_state,
    materialize_contract,
    rebase_patch_file,
)


def _inspect_hashes(contract_id: str, section_anchor: str | None, row_key: str | None, table_id: str) -> int:
    base_paths = ensure_base_snapshot(contract_id)
    base_state = load_base_contract_state(contract_id=contract_id, base_paths=base_paths)
    out = {}

    if section_anchor:
        match = next(
            (s for s in (base_state.get("sections") or []) if str(s.get("anchor_id") or "") == section_anchor),
            None,
        )
        if match is None:
            print(json.dumps({"error": f"section anchor '{section_anchor}' not found"}, indent=2))
            return 1
        out["section"] = {
            "anchor_id": section_anchor,
            "expected_prev_hash": ContractMaterializer.hash_text(match.get("content_markdown", "")),
            "citation": match.get("citation"),
        }

    if row_key:
        table = (base_state.get("tables") or {}).get(table_id) or {}
        row = next(
            (r for r in (table.get("rows") or []) if str(r.get("row_key") or "") == row_key),
            None,
        )
        if row is None:
            print(json.dumps({"error": f"row_key '{row_key}' not found in table '{table_id}'"}, indent=2))
            return 1
        out["table_row"] = {
            "table_id": table_id,
            "row_key": row_key,
            "expected_prev_hash": ContractMaterializer.hash_row(row.get("columns") or {}),
            "columns": row.get("columns") or {},
        }

    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize effective contract snapshots from MOA patches")
    parser.add_argument("--contract-id", required=True, help="Contract ID")
    parser.add_argument("--effective-version-id", default=None, help="Effective version ID output folder")
    parser.add_argument(
        "--patch-id",
        action="append",
        default=None,
        help="Patch ID to include (repeatable). Defaults to all patch files in amendments/",
    )
    parser.add_argument(
        "--no-update-latest",
        action="store_true",
        help="Do not update effective/latest.json pointer",
    )
    parser.add_argument("--hash-section-anchor", default=None, help="Print expected_prev_hash for section anchor_id and exit")
    parser.add_argument("--hash-row-key", default=None, help="Print expected_prev_hash for table row_key and exit")
    parser.add_argument("--table-id", default=WAGE_TABLE_ID, help="Table ID for --hash-row-key")
    parser.add_argument("--rebase-patch-id", default=None, help="Patch ID to rebase expected_prev_hash values against prior patches")
    parser.add_argument("--rebase-output", default=None, help="Output file path for rebased patch JSON")
    parser.add_argument("--rebase-in-place", action="store_true", help="Overwrite source patch JSON with rebased payload")
    args = parser.parse_args()

    if args.hash_section_anchor or args.hash_row_key:
        return _inspect_hashes(
            contract_id=args.contract_id,
            section_anchor=args.hash_section_anchor,
            row_key=args.hash_row_key,
            table_id=args.table_id,
        )

    if args.rebase_patch_id:
        if args.rebase_in_place and args.rebase_output:
            print(json.dumps({"error": "Use either --rebase-in-place or --rebase-output, not both"}, indent=2))
            return 1

        all_paths = discover_patch_files(contract_id=args.contract_id, patch_ids=None)
        if not all_paths:
            print(
                json.dumps(
                    {
                        "error": f"No patch files found for contract_id='{args.contract_id}'",
                        "amendments_dir": str(Path("data/contracts") / args.contract_id / "amendments"),
                    },
                    indent=2,
                )
            )
            return 1

        loaded = []
        for path in all_paths:
            artifact = load_patch_artifact(path)
            loaded.append((artifact, path))
        loaded.sort(key=lambda row: (str(row[0].effective_date), str(row[0].patch_id)))

        target_idx = None
        for idx, (artifact, _path) in enumerate(loaded):
            if artifact.patch_id == args.rebase_patch_id:
                target_idx = idx
                break
        if target_idx is None:
            print(json.dumps({"error": f"Patch '{args.rebase_patch_id}' not found for contract '{args.contract_id}'"}, indent=2))
            return 1

        target_artifact, target_path = loaded[target_idx]
        prior_paths = [path for _artifact, path in loaded[:target_idx]]

        try:
            result = rebase_patch_file(
                contract_id=args.contract_id,
                patch_path=target_path,
                prior_patch_paths=prior_paths,
            )
        except PatchRebaseFailure as exc:
            print(json.dumps(exc.report, indent=2, ensure_ascii=False, sort_keys=True))
            return 1
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

        rebased_payload = result.pop("rebased_patch_payload")
        if args.rebase_in_place:
            out_path = target_path
        elif args.rebase_output:
            out_path = Path(args.rebase_output)
        else:
            out_path = target_path.with_name(f"{target_artifact.patch_id}.rebased.json")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rebased_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

        result["output_path"] = str(out_path)
        result["changed_operations"] = sum(1 for row in result.get("changes", []) if row.get("changed"))
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    patch_paths = discover_patch_files(contract_id=args.contract_id, patch_ids=args.patch_id)
    if not patch_paths:
        print(
            json.dumps(
                {
                    "error": f"No patch files found for contract_id='{args.contract_id}'",
                    "amendments_dir": str(Path("data/contracts") / args.contract_id / "amendments"),
                },
                indent=2,
            )
        )
        return 1

    effective_version_id = args.effective_version_id
    if not effective_version_id:
        # Deterministic default from final patch id.
        effective_version_id = f"effective_{Path(patch_paths[-1]).stem}"

    result = materialize_contract(
        contract_id=args.contract_id,
        effective_version_id=effective_version_id,
        patch_paths=patch_paths,
        write_latest_pointer=not args.no_update_latest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
