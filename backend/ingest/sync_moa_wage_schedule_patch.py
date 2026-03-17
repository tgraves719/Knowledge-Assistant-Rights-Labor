from __future__ import annotations

import argparse
import json

from backend.ingest.moa_wage_schedule_configs import CONFIGS, get_config
from backend.ingest.moa_wage_schedule_sync import update_patch_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync contract Appendix A wage row patch ops from MOA wage schedule tables.")
    parser.add_argument("--config-id", required=True, choices=sorted(CONFIGS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Compute coverage and operation counts without writing patch file.")
    args = parser.parse_args()

    summary = update_patch_file(get_config(args.config_id), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
