"""Contract-history endpoint checks for Contract tab MOA/base/effective source modes."""

from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.api as api
import backend.effective_contracts as effective_contracts
import backend.pdf_nav_index as pdf_nav_index
import backend.source_docs as source_docs


@contextmanager
def _workspace_tempdir(prefix: str):
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _touch_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%EOF\n")


def _write_source_doc(
    *,
    data_root: Path,
    source_doc_id: str,
    doc_type: str,
    pdf_name: str,
    applies_to_contract_ids: list[str] | None = None,
) -> None:
    doc_dir = data_root / "source_docs" / doc_type / source_doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    _touch_pdf(doc_dir / "original.pdf")
    _write_json(
        doc_dir / "metadata.json",
        {
            "schema_version": "source_doc_v0_9_0",
            "source_doc_id": source_doc_id,
            "doc_type": doc_type,
            "source_pdf_filename": pdf_name,
            "applies_to_contract_ids": list(applies_to_contract_ids or []),
        },
    )


def _test_contract_history_payload_and_endpoint() -> None:
    contract_id = "history_contract"
    with _workspace_tempdir("contract_history_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        source_dir = contract_root / "source"
        effective_dir = contract_root / "effective" / "effective_2025_07_05"
        source_doc_id = "albertsons_safeway_moa_2025_07_05"

        base_pdf = "Base-CBA.pdf"
        moa_pdf = "Signed+MOA+-+July+5,+2025+(Safeway).pdf"
        _touch_pdf(source_dir / base_pdf)
        _write_source_doc(
            data_root=data_root,
            source_doc_id=source_doc_id,
            doc_type="moa",
            pdf_name=moa_pdf,
        )

        _write_json(
            contract_root / "effective" / "latest.json",
            {
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "a" * 64,
            },
        )
        _write_json(
            effective_dir / "effective_contract.json",
            {
                "contract_id": contract_id,
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "a" * 64,
                "amendments_applied": ["patch_2025_07_05"],
                "source_documents": {
                    "base_pdf": base_pdf,
                    "amendment_pdfs": [],
                    "amendment_source_doc_ids": [source_doc_id],
                },
            },
        )
        _write_json(
            effective_dir / "patch_chain.json",
            {
                "contract_id": contract_id,
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "a" * 64,
                "applied_patch_ids": ["patch_2025_07_05"],
                "patches": [
                    {
                        "patch_id": "patch_2025_07_05",
                        "effective_date": "2025-07-05",
                        "ratified_date": "2025-07-05",
                        "source_pdf": None,
                        "source_doc_id": source_doc_id,
                        "operation_count": 2,
                        "approved_operation_count": 2,
                        "patch_file_sha256": "b" * 64,
                        "patch_payload_sha256": "c" * 64,
                    }
                ],
            },
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(api, "DATA_DIR", data_root))
            stack.enter_context(patch.object(effective_contracts, "DATA_DIR", data_root))
            stack.enter_context(patch.object(source_docs, "SOURCE_DOCS_DIR", data_root / "source_docs"))
            stack.enter_context(patch.object(api, "get_contract_catalog_entry", lambda _cid: {"contract_id": contract_id}))

            payload = api._build_contract_history_payload(contract_id)
            assert payload["contract_id"] == contract_id
            assert payload["effective_version_id"] == "effective_2025_07_05"
            assert payload["effective_content_hash"] == "a" * 64
            assert payload["base_pdf"] == base_pdf
            assert payload["amendment_pdfs"] == [moa_pdf]
            assert payload["amendment_source_doc_ids"] == [source_doc_id]
            assert payload["applied_patch_ids"] == ["patch_2025_07_05"]
            assert payload["patch_count"] == 1
            assert payload["patches"][0]["patch_id"] == "patch_2025_07_05"
            assert payload["patches"][0]["source_doc_id"] == source_doc_id
            assert "moa" in payload["source_modes"]

            response = asyncio.run(api.get_contract_history(contract_id=contract_id))
            assert response.contract_id == contract_id
            assert response.patch_count == 1
            assert response.patches[0].source_pdf is None
            assert response.patches[0].source_doc_id == source_doc_id


def _test_pdf_location_moa_without_page_avoids_base_fallback() -> None:
    contract_id = "history_contract_pdf_nav"
    with _workspace_tempdir("contract_history_pdf_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        source_dir = contract_root / "source"
        effective_dir = contract_root / "effective" / "effective_2025_07_05"
        source_doc_id = "albertsons_safeway_moa_2025_07_05"

        base_pdf = "Base-CBA.pdf"
        moa_pdf = "Signed+MOA+-+July+5,+2025+(Safeway).pdf"
        _touch_pdf(source_dir / base_pdf)
        _write_source_doc(
            data_root=data_root,
            source_doc_id=source_doc_id,
            doc_type="moa",
            pdf_name=moa_pdf,
        )

        _write_json(
            contract_root / "effective" / "latest.json",
            {
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "d" * 64,
            },
        )
        _write_json(
            effective_dir / "effective_contract.json",
            {
                "contract_id": contract_id,
                "effective_version_id": "effective_2025_07_05",
                "sections": [
                    {
                        "article_num": 15,
                        "section_num": 34,
                        "anchor_id": "a15_s34",
                        "provenance": [
                            {
                                "source_type": "moa",
                                "pdf": moa_pdf,
                                "pdf_page": None,
                                "source_doc_id": source_doc_id,
                            }
                        ],
                    }
                ],
            },
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(api, "DATA_DIR", data_root))
            stack.enter_context(patch.object(effective_contracts, "DATA_DIR", data_root))
            stack.enter_context(patch.object(source_docs, "SOURCE_DOCS_DIR", data_root / "source_docs"))
            stack.enter_context(patch.object(api, "get_contract_catalog_entry", lambda _cid: {"contract_id": contract_id}))

            response = asyncio.run(
                api.get_pdf_location(
                    contract_id=contract_id,
                    article_num=15,
                    section_num=34,
                    source_type="moa",
                    source_doc_id=source_doc_id,
                )
            )
            assert response.contract_id == contract_id
            assert response.page_number is None
            assert response.matched_by == "provenance_missing"
            assert response.pdf_url and "source_doc_id=" in response.pdf_url

            pdf_response = asyncio.run(
                api.get_contract_pdf(
                    contract_id=contract_id,
                    source_doc_id=source_doc_id,
                    source_type="moa",
                )
            )
            assert str(getattr(pdf_response, "path", "")).endswith("original.pdf")


def _test_foreign_source_doc_is_blocked_for_contract_pdf_and_history() -> None:
    contract_id = "history_contract_scope_guard"
    foreign_contract_id = "different_contract"
    with _workspace_tempdir("contract_history_scope_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        effective_dir = contract_root / "effective" / "effective_2025_07_05"
        source_doc_id = "shared_moa_2025_07_05"
        moa_pdf = "Shared+MOA.pdf"

        _write_source_doc(
            data_root=data_root,
            source_doc_id=source_doc_id,
            doc_type="moa",
            pdf_name=moa_pdf,
            applies_to_contract_ids=[foreign_contract_id],
        )
        _write_json(
            contract_root / "effective" / "latest.json",
            {
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "e" * 64,
            },
        )
        _write_json(
            effective_dir / "effective_contract.json",
            {
                "contract_id": contract_id,
                "effective_version_id": "effective_2025_07_05",
                "effective_content_hash": "e" * 64,
                "source_documents": {
                    "amendment_source_doc_ids": [source_doc_id],
                },
                "amendments_applied": ["patch_2025_07_05"],
            },
        )
        _write_json(
            effective_dir / "patch_chain.json",
            {
                "contract_id": contract_id,
                "effective_version_id": "effective_2025_07_05",
                "applied_patch_ids": ["patch_2025_07_05"],
                "patches": [
                    {
                        "patch_id": "patch_2025_07_05",
                        "effective_date": "2025-07-05",
                        "source_doc_id": source_doc_id,
                        "operation_count": 1,
                        "approved_operation_count": 1,
                    }
                ],
            },
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(api, "DATA_DIR", data_root))
            stack.enter_context(patch.object(effective_contracts, "DATA_DIR", data_root))
            stack.enter_context(patch.object(source_docs, "SOURCE_DOCS_DIR", data_root / "source_docs"))
            stack.enter_context(patch.object(api, "get_contract_catalog_entry", lambda _cid: {"contract_id": contract_id}))

            payload = api._build_contract_history_payload(contract_id)
            assert payload["amendment_source_doc_ids"] == []
            assert payload["patch_count"] == 0 or all(
                (row.get("source_doc_id") or "") != source_doc_id for row in (payload.get("patches") or [])
            )

            try:
                asyncio.run(
                    api.get_contract_pdf(
                        contract_id=contract_id,
                        source_doc_id=source_doc_id,
                        source_type="moa",
                    )
                )
            except api.HTTPException as e:
                assert e.status_code == 404
            else:
                raise AssertionError("Expected HTTPException 404 for inapplicable foreign source_doc_id")


def _test_contract_browse_groups_and_item_endpoint() -> None:
    contract_id = "browse_contract"
    with _workspace_tempdir("contract_browse_") as tmp:
        data_root = tmp / "data"
        manifests_dir = tmp / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        source_dir = data_root / "contracts" / contract_id / "source"
        chunks_path = data_root / "contracts" / contract_id / "chunks" / "contract_chunks_enriched.json"
        _touch_pdf(source_dir / "Base-CBA.pdf")
        _write_json(
            source_dir / f"{contract_id}.json",
            {
                "pages": [
                    {
                        "page_number": 1,
                        "items": [{"value": "Table of Contents"}],
                    },
                    {
                        "page_number": 2,
                        "items": [{"value": "Appendix A Wage Table - Basket Hours"}],
                    },
                    {
                        "page_number": 3,
                        "items": [{"value": "8. Dress Requirements letter pursuant to Article 52 of this Agreement."}],
                    },
                ]
            },
        )

        _write_json(
            manifests_dir / f"{contract_id}.json",
            {
                "contract_id": contract_id,
                "article_titles": {"1": "Recognition"},
                "total_articles": 1,
            },
        )
        _write_json(
            chunks_path,
            [
                {
                    "contract_id": contract_id,
                    "doc_type": "cba",
                    "article_num": 1,
                    "section_num": 1,
                    "citation": "Article 1, Section 1",
                    "content": "Recognition clause.",
                    "article_title": "Recognition",
                },
                {
                    "contract_id": contract_id,
                    "doc_type": "lou",
                    "chunk_id": "lou_8_part1",
                    "citation": "Letter of Understanding 8: Dress Requirements, Part 1",
                    "content": "Employees must comply with dress requirements.",
                },
                {
                    "contract_id": contract_id,
                    "doc_type": "lou",
                    "chunk_id": "lou_8_part2",
                    "citation": "Letter of Understanding 8: Dress Requirements, Part 2",
                    "content": "Black shoes are required.",
                },
                {
                    "contract_id": contract_id,
                    "doc_type": "appendix",
                    "chunk_id": "appendix_a_chunk",
                    "citation": "Appendix A Wage Table - Basket Hours",
                    "content": "Appendix wage table content.",
                },
            ],
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(api, "DATA_DIR", data_root))
            stack.enter_context(patch.object(api, "MANIFESTS_DIR", manifests_dir))
            stack.enter_context(patch.object(api, "get_contract_catalog_entry", lambda _cid: {"contract_id": contract_id}))
            stack.enter_context(patch.object(api, "resolve_chunk_file", lambda *args, **kwargs: chunks_path))
            stack.enter_context(patch.object(effective_contracts, "DATA_DIR", data_root))
            stack.enter_context(patch.object(pdf_nav_index, "DATA_DIR", data_root))

            browse = asyncio.run(api.get_contract_browse(contract_id=contract_id))
            group_keys = [g.key for g in browse.groups]
            assert "articles" in group_keys
            assert "side_letters" in group_keys
            assert "appendices" in group_keys

            side_letter_group = next(g for g in browse.groups if g.key == "side_letters")
            lou_keys = [item.key for item in side_letter_group.items]
            assert "lou:8" in lou_keys

            item = asyncio.run(api.get_contract_browse_item(contract_id=contract_id, kind="lou", key="lou:8"))
            assert item.kind == "lou"
            assert item.key == "lou:8"
            assert len(item.sections) == 2
            assert "Dress Requirements" in item.sections[0].citation

            loc = asyncio.run(
                api.get_pdf_location(contract_id=contract_id, browse_kind="lou", browse_key="lou:8")
            )
            assert loc.contract_id == contract_id
            assert loc.page_number == 3
            assert loc.matched_by == "browse_item"


def _test_article_and_browse_item_support_source_view_base() -> None:
    contract_id = "text_source_view_contract"
    with _workspace_tempdir("contract_text_view_") as tmp:
        effective_chunks = tmp / "effective_chunks.json"
        base_chunks = tmp / "base_chunks.json"

        _write_json(
            effective_chunks,
            [
                {
                    "contract_id": contract_id,
                    "doc_type": "cba",
                    "article_num": 15,
                    "section_num": 34,
                    "citation": "Article 15, Section 34",
                    "content": "Courtesy Clerks shall receive fifty cents ($0.50) per hour.",
                    "article_title": "Premiums",
                },
                {
                    "contract_id": contract_id,
                    "doc_type": "lou",
                    "chunk_id": "lou_8_part1",
                    "citation": "Letter of Understanding 8: Dress Requirements, Part 1",
                    "content": "Effective text version of dress requirements.",
                },
            ],
        )
        _write_json(
            base_chunks,
            [
                {
                    "contract_id": contract_id,
                    "doc_type": "cba",
                    "article_num": 15,
                    "section_num": 34,
                    "citation": "Article 15, Section 34",
                    "content": "Courtesy Clerks shall receive twenty-five cents ($0.25) per hour.",
                    "article_title": "Premiums",
                },
                {
                    "contract_id": contract_id,
                    "doc_type": "lou",
                    "chunk_id": "lou_8_part1",
                    "citation": "Letter of Understanding 8: Dress Requirements, Part 1",
                    "content": "Base text version of dress requirements.",
                },
            ],
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(api, "get_contract_catalog_entry", lambda _cid: {"contract_id": contract_id}))
            stack.enter_context(patch.object(api, "resolve_chunk_file", lambda *args, **kwargs: effective_chunks))
            stack.enter_context(patch.object(api, "_resolve_base_chunk_file", lambda _cid: base_chunks))

            eff_article = asyncio.run(api.get_article(article_num=15, contract_id=contract_id, source_view="effective"))
            base_article = asyncio.run(api.get_article(article_num=15, contract_id=contract_id, source_view="base"))
            assert "0.50" in eff_article.sections[0].content
            assert "0.25" in base_article.sections[0].content

            eff_lou = asyncio.run(api.get_contract_browse_item(kind="lou", key="lou:8", contract_id=contract_id, source_view="effective"))
            base_lou = asyncio.run(api.get_contract_browse_item(kind="lou", key="lou:8", contract_id=contract_id, source_view="base"))
            assert "Effective text version" in eff_lou.sections[0].content
            assert "Base text version" in base_lou.sections[0].content


def main() -> None:
    _test_contract_history_payload_and_endpoint()
    _test_pdf_location_moa_without_page_avoids_base_fallback()
    _test_foreign_source_doc_is_blocked_for_contract_pdf_and_history()
    _test_contract_browse_groups_and_item_endpoint()
    _test_article_and_browse_item_support_source_view_base()
    print("[OK] contract history API checks passed")


if __name__ == "__main__":
    main()
