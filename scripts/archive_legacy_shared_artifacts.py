"""
Archive legacy shared runtime artifacts in multi-contract mode.

Why:
- Shared files like data/chunks/contract_chunks_enriched.json were useful for
  single-contract mode, but become ambiguous once multiple contract manifests
  exist.
- Contract-scoped artifacts remain canonical:
  - contract_chunks_*_<contract_id>.json
  - concept_index_<contract_id>.json

Usage:
  python scripts/archive_legacy_shared_artifacts.py          # dry run
  python scripts/archive_legacy_shared_artifacts.py --apply  # move files
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR, MANIFESTS_DIR


LEGACY_SHARED_FILES = [
    DATA_DIR / "chunks" / "contract_chunks.json",
    DATA_DIR / "chunks" / "contract_chunks_smart.json",
    DATA_DIR / "chunks" / "contract_chunks_enriched.json",
    DATA_DIR / "chunks" / "concept_index.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive legacy shared artifacts in multi-contract mode.")
    parser.add_argument("--apply", action="store_true", help="Move artifacts into archive folder.")
    args = parser.parse_args()

    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    if manifest_count <= 1:
        print(f"Single-contract mode detected (manifest_count={manifest_count}); no archive needed.")
        return 0

    existing = [p for p in LEGACY_SHARED_FILES if p.exists()]
    if not existing:
        print("No legacy shared artifacts found.")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = DATA_DIR / "legacy" / "shared_artifacts" / ts
    plan = []
    for src in existing:
        rel = src.relative_to(DATA_DIR)
        dst = archive_root / rel
        plan.append({"source": str(src), "dest": str(dst)})

    print("Legacy shared artifacts detected:")
    for row in plan:
        print(f"- {row['source']} -> {row['dest']}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to archive.")
        return 0

    archive_root.mkdir(parents=True, exist_ok=True)
    for row in plan:
        src = Path(row["source"])
        dst = Path(row["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    report = {
        "schema_version": "legacy_shared_archive_v1",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_count": manifest_count,
        "moved": plan,
    }
    report_path = archive_root / "archive_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nArchived {len(plan)} files to {archive_root}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

