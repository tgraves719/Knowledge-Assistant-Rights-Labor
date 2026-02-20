"""Build contract outline artifacts from current ingestion outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.contract_outline import (
    build_contract_outline,
    package_contract_outline_path,
    save_contract_outline,
    shared_contract_outline_path,
)
from backend.contracts import list_contract_catalog
from backend.config import MANIFESTS_DIR


def _discover_contract_ids(contract_id: str | None = None) -> list[str]:
    if contract_id:
        return [contract_id]

    ids: list[str] = []
    for row in list_contract_catalog():
        cid = str(row.get("contract_id") or "").strip()
        if cid:
            ids.append(cid)
    if ids:
        return sorted(set(ids))
    return sorted(p.stem for p in MANIFESTS_DIR.glob("*.json"))


def build_outlines(contract_id: str | None = None) -> int:
    contract_ids = _discover_contract_ids(contract_id=contract_id)
    if not contract_ids:
        print("[FAIL] No contracts discovered for outline build.")
        return 1

    for cid in contract_ids:
        outline = build_contract_outline(contract_id=cid)
        package_path = package_contract_outline_path(cid)
        shared_path = shared_contract_outline_path(cid)
        save_contract_outline(package_path, outline)
        save_contract_outline(shared_path, outline)
        stats = outline.get("stats") or {}
        print(
            f"[OK] {cid}: articles={stats.get('article_count', 0)} "
            f"sections={stats.get('section_count', 0)}"
        )
        print(f"     package: {package_path}")
        print(f"     shared : {shared_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contract outline artifacts.")
    parser.add_argument("--contract-id", type=str, default=None, help="Contract ID to build")
    args = parser.parse_args()
    return build_outlines(contract_id=args.contract_id)


if __name__ == "__main__":
    raise SystemExit(main())
