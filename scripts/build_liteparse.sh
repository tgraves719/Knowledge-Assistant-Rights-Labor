#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITEPARSE_DIR="$ROOT_DIR/vendor/liteparse"

if [[ ! -d "$LITEPARSE_DIR" ]]; then
  echo "LiteParse vendor directory not found: $LITEPARSE_DIR" >&2
  exit 1
fi

cd "$LITEPARSE_DIR"
npm ci
npm run build

echo "LiteParse built at $LITEPARSE_DIR/dist/src/index.js"
