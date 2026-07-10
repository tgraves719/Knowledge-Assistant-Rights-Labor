from __future__ import annotations

import argparse
import json

from backend.ingest.moa_wage_schedule_configs import PUEBLO_CLERKS_2025_07_05
from backend.ingest.moa_wage_schedule_sync import update_patch_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Pueblo Clerks Appendix A wage row patch ops from MOA output.json wage schedule tables."
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute coverage and operation counts without writing patch file.")
    args = parser.parse_args()

    summary = update_patch_file(PUEBLO_CLERKS_2025_07_05, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
