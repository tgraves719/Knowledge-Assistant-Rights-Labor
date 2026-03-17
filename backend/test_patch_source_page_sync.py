"""Checks for syncing missing patch source pages from regenerated drafts."""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.sync_patch_source_pages import sync_patch_source_pages  # noqa: E402


def _test_sync_patch_source_pages_copies_missing_pdf_page() -> None:
    approved = {
        "operations": [
            {
                "op": "replace_section",
                "target": {"anchor_id": "a15_s34", "article_num": 15, "section_num": 34},
                "source_refs": [
                    {
                        "source_type": "moa",
                        "source_doc_id": "demo_moa",
                        "pdf": "demo.pdf",
                    }
                ],
            }
        ]
    }
    draft = {
        "operations": [
            {
                "op": "replace_section",
                "target": {"anchor_id": "a15_s34", "article_num": 15, "section_num": 34},
                "source_refs": [
                    {
                        "source_type": "moa",
                        "source_doc_id": "demo_moa",
                        "pdf": "demo.pdf",
                        "pdf_page": 12,
                    }
                ],
            }
        ]
    }
    synced, changes = sync_patch_source_pages(approved, draft)
    assert synced["operations"][0]["source_refs"][0]["pdf_page"] == 12
    assert len(changes) == 1


def _test_sync_patch_source_pages_leaves_existing_pdf_page_alone() -> None:
    approved = {
        "operations": [
            {
                "op": "replace_section",
                "target": {"anchor_id": "a15_s34"},
                "source_refs": [
                    {
                        "source_type": "moa",
                        "source_doc_id": "demo_moa",
                        "pdf_page": 9,
                    }
                ],
            }
        ]
    }
    draft = {
        "operations": [
            {
                "op": "replace_section",
                "target": {"anchor_id": "a15_s34"},
                "source_refs": [
                    {
                        "source_type": "moa",
                        "source_doc_id": "demo_moa",
                        "pdf_page": 12,
                    }
                ],
            }
        ]
    }
    synced, changes = sync_patch_source_pages(approved, draft)
    assert synced["operations"][0]["source_refs"][0]["pdf_page"] == 9
    assert changes == []


def main() -> None:
    _test_sync_patch_source_pages_copies_missing_pdf_page()
    _test_sync_patch_source_pages_leaves_existing_pdf_page_alone()
    print("[OK] patch source page sync checks passed")


if __name__ == "__main__":
    main()
