"""CLI for registering shared source documents (MOA/CBA/LOU/etc)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.source_docs import register_source_doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a shared source document under data/source_docs/")
    parser.add_argument("--source-doc-id", required=True, help="Stable source document id (e.g. albertsons_safeway_moa_2025_07_05)")
    parser.add_argument("--doc-type", default="moa", help="Document type folder (moa, cba, lou, side_letter, etc)")
    parser.add_argument("--title", default=None, help="Optional title")
    parser.add_argument("--document-date", default=None, help="Document date (YYYY-MM-DD)")
    parser.add_argument("--ratified-date", default=None, help="Ratified date (YYYY-MM-DD)")
    parser.add_argument("--pdf-path", default=None, help="Path to source PDF")
    parser.add_argument("--json-path", default=None, help="Path to extracted JSON")
    parser.add_argument("--md-path", default=None, help="Path to extracted markdown")
    parser.add_argument("--party", action="append", default=None, help="Party name (repeatable)")
    parser.add_argument("--contract-id", action="append", default=None, help="Impacted contract id (repeatable)")
    parser.add_argument(
        "--applies-to-contract-id",
        action="append",
        default=None,
        help="Canonical applicability contract id (repeatable). If omitted, contract_ids are also used.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing source doc folder")
    args = parser.parse_args()

    result = register_source_doc(
        source_doc_id=args.source_doc_id,
        doc_type=args.doc_type,
        title=args.title,
        document_date=args.document_date,
        ratified_date=args.ratified_date,
        source_pdf_path=Path(args.pdf_path) if args.pdf_path else None,
        extracted_json_path=Path(args.json_path) if args.json_path else None,
        extracted_md_path=Path(args.md_path) if args.md_path else None,
        parties=args.party,
        contract_ids=args.contract_id,
        applies_to_contract_ids=args.applies_to_contract_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
