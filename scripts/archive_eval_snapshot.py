"""
Archive evaluation artifacts into a timestamped, git-friendly snapshot folder.

Usage:
  python scripts/archive_eval_snapshot.py --label v0_9_step1
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SET_DIR = PROJECT_ROOT / "data" / "test_set"
HISTORY_DIR = TEST_SET_DIR / "history"


DEFAULT_FILES = [
    "evaluation_results.json",
    "comprehensive_results.json",
    "escalation_precision_results.json",
    "eval_run_metadata_latest.json",
]


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description="Archive KARL evaluation artifacts.")
    parser.add_argument("--label", default="manual", help="Short label for this snapshot (e.g., v0_9_step1).")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_name = f"{ts}_{args.label}"
    snapshot_dir = HISTORY_DIR / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for name in DEFAULT_FILES:
        src = TEST_SET_DIR / name
        dst = snapshot_dir / name
        if _safe_copy(src, dst):
            copied.append(name)
        else:
            missing.append(name)

    manifest = {
        "snapshot": snapshot_name,
        "created_utc": ts,
        "label": args.label,
        "copied_files": copied,
        "missing_files": missing,
        "source_dir": str(TEST_SET_DIR.relative_to(PROJECT_ROOT)),
    }

    with open(snapshot_dir / "snapshot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Keep an index for easy changelog/history browsing.
    index_path = HISTORY_DIR / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"snapshots": []}

    index["snapshots"].append(
        {
            "snapshot": snapshot_name,
            "created_utc": ts,
            "label": args.label,
            "copied_files": copied,
        }
    )

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("=" * 72)
    print("KARL Evaluation Snapshot Archive")
    print("=" * 72)
    print(f"Snapshot: {snapshot_dir}")
    print(f"Copied: {copied}")
    if missing:
        print(f"Missing: {missing}")
    print("Done.")


if __name__ == "__main__":
    main()
