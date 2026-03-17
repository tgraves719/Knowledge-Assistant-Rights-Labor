"""Create a structured local miss record from a reviewed JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.miss_records import build_regression_stub, normalize_miss_record, write_miss_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and write a structured KARL miss record.")
    parser.add_argument("--input", required=True, help="Path to a reviewed miss payload JSON file")
    parser.add_argument("--output", required=True, help="Output path for the normalized miss record")
    parser.add_argument(
        "--emit-regression-stub",
        default=None,
        help="Optional output path for a small regression-stub JSON artifact",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    normalized = normalize_miss_record(payload)
    out_path = write_miss_record(normalized, args.output)

    print(f"[OK] Wrote normalized miss record: {out_path}")

    if args.emit_regression_stub:
        stub_path = Path(args.emit_regression_stub)
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub = build_regression_stub(normalized)
        with open(stub_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(stub, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        print(f"[OK] Wrote regression stub: {stub_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
