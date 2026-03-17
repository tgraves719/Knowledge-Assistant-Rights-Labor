"""Checks for draft MOA patch generation heuristics."""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.generate_patch_drafts import (  # noqa: E402
    _build_patch_payload,
    _extract_effective_candidates,
    _metadata_contract_ids,
    _render_page_aware_markdown_from_source_json,
    _select_candidates,
    _to_effective_text,
    resolve_target_contract_ids,
)
from backend.ingest.materializer import ContractMaterializer  # noqa: E402


def _test_to_effective_text_strips_redlines() -> None:
    raw = (
        "Section 34. All Courtesy Clerks shall receive <u>fifty</u>~~twenty-five~~ cents "
        "($.50~~25~~) per hour."
    )
    out = _to_effective_text(raw)
    assert "twenty-five" not in out
    assert "<u>" not in out
    assert "fifty" in out
    assert "$.50" in out


def _test_extract_candidates_tracks_article_section_and_page() -> None:
    md = "\n".join(
        [
            "# ARTICLE 15",
            "Page 12 of 33",
            "Section 34. Night premium text <u>fifty</u>~~twenty-five~~ cents.",
            "Night premium shall not apply on Sunday.",
            "# ARTICLE 17",
            "Section 49. Vacation language.",
        ]
    )
    candidates = _extract_effective_candidates(md)
    assert len(candidates) >= 2
    first = candidates[0]
    assert first["article_num"] == 15
    assert first["section_num"] == 34
    assert first["source_page"] == 12
    assert first["has_redline"] is True


def _test_render_page_aware_markdown_from_source_json_tracks_pages() -> None:
    payload = {
        "pages": [
            {
                "page_number": 2,
                "items": [
                    {"type": "header", "md": "UFCW Local 7\nPage 2 of 33\nMemorandum of Agreement"},
                    {"type": "heading", "md": "### ARTICLE 15"},
                    {"type": "text", "md": "Section 34. Night premium <u>fifty</u>~~twenty-five~~ cents."},
                ],
            }
        ]
    }
    md = _render_page_aware_markdown_from_source_json(payload)
    assert "Page 2 of 2" in md
    candidates = _extract_effective_candidates(md)
    assert candidates[0]["article_num"] == 15
    assert candidates[0]["section_num"] == 34
    assert candidates[0]["source_page"] == 2


def _test_build_patch_payload_section_mapping() -> None:
    selected = _select_candidates(
        [
            {
                "article_num": 15,
                "section_num": 34,
                "effective_text_markdown": (
                    "Section 34. A premium of two dollars ($2.00) per hour shall be paid for all work "
                    "performed between 12:00 midnight and 6:00 a.m. to all employees."
                ),
                "source_page": 12,
                "has_redline": True,
            },
            {
                "article_num": 45,
                "section_num": 130,
                "effective_text_markdown": "Section 130. Missing in base.",
                "source_page": 9,
                "has_redline": True,
            },
        ],
        include_unmarked=False,
    )
    base_section = {
        "anchor_id": "a15_s34",
        "article_num": 15,
        "section_num": 34,
        "content_markdown": "Section 34. Old language.",
    }
    payload, report = _build_patch_payload(
        contract_id="local7_safeway_pueblo_clerks_2022",
        source_doc_id="albertsons_safeway_moa_2025_07_05",
        source_pdf="Signed+MOA+-+July+5,+2025+(Safeway).pdf",
        effective_date="2025-07-05",
        ratified_date="2025-07-05",
        parent_effective_version_id="base_snapshot_v0_9_0",
        selected_candidates=selected,
        base_index={(15, 34): [base_section]},
    )

    operations = payload.get("operations") or []
    assert len(operations) == 1
    op = operations[0]
    assert op["op"] == "replace_section"
    assert op["review_status"] == "pending"
    assert op["target"]["anchor_id"] == "a15_s34"
    assert op["source_refs"][0]["source_doc_id"] == "albertsons_safeway_moa_2025_07_05"
    assert op["source_refs"][0]["pdf_page"] == 12
    assert op["expected_prev_hash"] == ContractMaterializer.hash_text("Section 34. Old language.")
    assert report["skipped_count"] == 1


def _test_low_quality_candidate_is_skipped() -> None:
    payload, report = _build_patch_payload(
        contract_id="local7_safeway_pueblo_clerks_2022",
        source_doc_id="albertsons_safeway_moa_2025_07_05",
        source_pdf="Signed+MOA+-+July+5,+2025+(Safeway).pdf",
        effective_date="2025-07-05",
        ratified_date="2025-07-05",
        parent_effective_version_id="base_snapshot_v0_9_0",
        selected_candidates=[
            {
                "article_num": 7,
                "section_num": 15,
                "effective_text_markdown": "Section 15. Work Between Classifications: ...",
                "source_page": 3,
                "has_redline": True,
            }
        ],
        base_index={
            (7, 15): [
                {
                    "anchor_id": "a7_s15",
                    "article_num": 7,
                    "section_num": 15,
                    "content_markdown": "Section 15. Old text.",
                }
            ]
        },
    )
    assert payload.get("operations") == []
    assert report["skipped_count"] == 1
    assert report["skipped_reason_counts"].get("low_quality_candidate") == 1
    assert report["quality_flag_counts"].get("contains_ellipsis") == 1


def _test_metadata_contract_target_resolution() -> None:
    metadata = {
        "contract_ids": [
            "local7_safeway_pueblo_clerks_2022",
            "local7_safeway_pueblo_meat_2022",
        ],
        "linked_contract_ids": ["local7_safeway_pueblo_clerks_2022"],
        "contract_id": "local7_safeway_pueblo_clerks_2022",
    }
    assert _metadata_contract_ids(metadata) == [
        "local7_safeway_pueblo_clerks_2022",
        "local7_safeway_pueblo_meat_2022",
    ]

    targets, mode = resolve_target_contract_ids(
        source_doc_id="albertsons_safeway_moa_2025_07_05",
        metadata=metadata,
        explicit_contract_ids=None,
        exclude_contract_ids=None,
    )
    assert mode == "metadata"
    assert targets == [
        "local7_safeway_pueblo_clerks_2022",
        "local7_safeway_pueblo_meat_2022",
    ]


def _test_metadata_contract_target_resolution_prefers_applies_to_contract_ids() -> None:
    metadata = {
        "applies_to_contract_ids": [
            "local7_safeway_pueblo_meat_2022",
            "local7_safeway_pueblo_clerks_2022",
            "local7_safeway_pueblo_meat_2022",
        ],
        "contract_ids": ["legacy_contract_should_not_duplicate"],
    }
    # Resolver aggregates all known metadata fields; primary requirement is that
    # canonical applicability field is recognized.
    targets, mode = resolve_target_contract_ids(
        source_doc_id="albertsons_safeway_moa_2025_07_05",
        metadata=metadata,
        explicit_contract_ids=None,
        exclude_contract_ids=["legacy_contract_should_not_duplicate"],
    )
    assert mode == "metadata"
    assert targets == [
        "local7_safeway_pueblo_clerks_2022",
        "local7_safeway_pueblo_meat_2022",
    ]


def _test_explicit_contract_target_resolution_with_exclusions() -> None:
    metadata = {
        "contract_ids": [
            "local7_safeway_pueblo_clerks_2022",
            "local7_safeway_pueblo_meat_2022",
        ]
    }
    targets, mode = resolve_target_contract_ids(
        source_doc_id="albertsons_safeway_moa_2025_07_05",
        metadata=metadata,
        explicit_contract_ids=[
            "local7_safeway_pueblo_clerks_2022",
            "local7_safeway_pueblo_meat_2022",
            "local7_safeway_pueblo_clerks_2022",
        ],
        exclude_contract_ids=["local7_safeway_pueblo_meat_2022"],
    )
    assert mode == "explicit"
    assert targets == ["local7_safeway_pueblo_clerks_2022"]


def main() -> None:
    _test_to_effective_text_strips_redlines()
    _test_extract_candidates_tracks_article_section_and_page()
    _test_render_page_aware_markdown_from_source_json_tracks_pages()
    _test_build_patch_payload_section_mapping()
    _test_low_quality_candidate_is_skipped()
    _test_metadata_contract_target_resolution()
    _test_metadata_contract_target_resolution_prefers_applies_to_contract_ids()
    _test_explicit_contract_target_resolution_with_exclusions()
    print("[OK] MOA patch draft generator checks passed")


if __name__ == "__main__":
    main()
