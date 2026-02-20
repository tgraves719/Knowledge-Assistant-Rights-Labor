"""Build table navigation indices for one or all contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import DATA_DIR, ONTOLOGIES_DIR
from backend.contracts import list_contract_catalog
from backend.table_nav_index import build_table_nav_index, save_table_nav_index


def _discover_contract_ids(contract_id: str | None = None) -> list[str]:
    if contract_id:
        return [contract_id]
    ids: list[str] = []
    for row in list_contract_catalog():
        cid = str(row.get("contract_id") or "").strip()
        if cid:
            ids.append(cid)
    return sorted(set(ids))


def build_indices(contract_id: str | None = None) -> int:
    contract_ids = _discover_contract_ids(contract_id=contract_id)
    if not contract_ids:
        print("[FAIL] No contracts discovered for table-nav build.")
        return 1

    for cid in contract_ids:
        index = build_table_nav_index(contract_id=cid)
        shared_path = ONTOLOGIES_DIR / f"table_nav_index_{cid}.json"
        package_path = DATA_DIR / "contracts" / cid / "ontology" / "table_nav_index.json"
        save_table_nav_index(shared_path, index)
        save_table_nav_index(package_path, index)
        stats = index.get("stats") or {}
        print(
            f"[OK] {cid}: tables={stats.get('tables_total', 0)} "
            f"mapped={stats.get('tables_with_page', 0)}"
        )
        print(f"     shared : {shared_path}")
        print(f"     package: {package_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build table navigation index artifacts.")
    parser.add_argument("--contract-id", type=str, default=None, help="Contract ID to build")
    args = parser.parse_args()
    return build_indices(contract_id=args.contract_id)


if __name__ == "__main__":
    raise SystemExit(main())
