"""
Deterministic MOA materializer and effective-routing checks.

Covers:
1) replace_section correctness + provenance
2) replace_table_row correctness + provenance
3) expected_prev_hash collision failure
4) full materialization determinism (bytes + hashes)
5) retrieval path resolution prefers effective snapshot inputs
"""

from __future__ import annotations

import copy
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.chunk_files as chunk_files
import backend.effective_contracts as effective_contracts
import backend.pdf_nav_files as pdf_nav_files
import backend.pdf_nav_index as pdf_nav_index
import backend.source_docs as source_docs
import backend.table_nav_files as table_nav_files
import backend.table_nav_index as table_nav_index
import backend.wage_files as wage_files
from backend.ingest.materializer import ContractMaterializer, MaterializationFailure
import backend.ingest.materializer as materializer_module
from backend.ingest.moa_schema import PatchArtifact


def _base_contract_fixture() -> dict:
    return {
        "contract_id": "unit_test_contract",
        "base_version_id": "base_2024_01_01",
        "sections": [
            {
                "anchor_id": "a1_s1",
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content_markdown": "Old base section text.",
                "provenance": [
                    {"source_type": "base", "pdf": "Base.pdf", "pdf_page": 2},
                ],
                "amendments": [],
            }
        ],
        "tables": {
            materializer_module.WAGE_TABLE_ID: {
                "table_id": materializer_module.WAGE_TABLE_ID,
                "rows": [
                    {
                        "row_key": "courtesy_clerk|hours:0|2024-01-21",
                        "columns": {
                            "classification_key": "courtesy_clerk",
                            "classification_name": "COURTESY CLERK",
                            "step_name": "Start",
                            "step_type": "hours",
                            "threshold_value": 0,
                            "effective_date": "2024-01-21",
                            "rate": 17.0,
                            "row_type": "step_row",
                            "source_method": "table_registry",
                            "confidence": 0.9,
                            "source_reference": {"table_id": "tbl_art1_1", "row_index": 0},
                        },
                        "provenance": [
                            {
                                "source_type": "base",
                                "pdf": "Base.pdf",
                                "pdf_page": 10,
                                "table_id": "tbl_art1_1",
                                "row_index": 0,
                            }
                        ],
                        "amendments": [],
                    }
                ],
            }
        },
        "amendments_applied": [],
        "source_documents": {
            "base_pdf": "Base.pdf",
            "amendment_pdfs": [],
            "amendment_source_doc_ids": [],
        },
    }


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


def _patch_fixture(
    *,
    expected_section_hash: str,
    expected_row_hash: str,
    new_text: str = "Updated section text from MOA.",
    new_rate: float = 17.5,
) -> PatchArtifact:
    return PatchArtifact.model_validate(
        {
            "schema_version": "moa_patch_v0_9_0",
            "patch_id": "patch_2025_07_05",
            "contract_id": "unit_test_contract",
            "source_pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
            "effective_date": "2025-07-05",
            "ratified_date": "2025-07-05",
            "parent_effective_version_id": "base_2024_01_01",
            "operations": [
                {
                    "op": "replace_section",
                    "target": {
                        "anchor_id": "a1_s1",
                        "article_num": 1,
                        "section_num": 1,
                    },
                    "expected_prev_hash": expected_section_hash,
                    "new_text_markdown": new_text,
                    "source_refs": [
                        {
                            "source_type": "moa",
                            "pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                            "pdf_page": 7,
                        }
                    ],
                    "confidence": 0.95,
                    "review_status": "approved",
                },
                {
                    "op": "replace_table_row",
                    "target": {
                        "table_id": materializer_module.WAGE_TABLE_ID,
                        "row_key": "courtesy_clerk|hours:0|2024-01-21",
                    },
                    "expected_prev_hash": expected_row_hash,
                    "new_row": {"rate": new_rate},
                    "source_refs": [
                        {
                            "source_type": "moa",
                            "pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                            "pdf_page": 12,
                        }
                    ],
                    "confidence": 0.89,
                    "review_status": "approved",
                },
            ],
        }
    )


def _patch_list_for_base(base_contract: dict) -> list[PatchArtifact]:
    materializer = ContractMaterializer()
    section = base_contract["sections"][0]
    row = base_contract["tables"][materializer_module.WAGE_TABLE_ID]["rows"][0]
    patch = _patch_fixture(
        expected_section_hash=materializer.hash_text(section["content_markdown"]),
        expected_row_hash=materializer.hash_row(row["columns"]),
    )
    return [patch]


@contextmanager
def _patched_data_root(data_root: Path):
    with ExitStack() as stack:
        stack.enter_context(patch.object(materializer_module, "DATA_DIR", data_root))
        stack.enter_context(patch.object(effective_contracts, "DATA_DIR", data_root))
        stack.enter_context(patch.object(source_docs, "SOURCE_DOCS_DIR", data_root / "source_docs"))
        stack.enter_context(patch.object(pdf_nav_index, "DATA_DIR", data_root))
        stack.enter_context(patch.object(table_nav_index, "DATA_DIR", data_root))
        stack.enter_context(patch.object(pdf_nav_files, "ONTOLOGIES_DIR", data_root / "ontologies"))
        stack.enter_context(patch.object(pdf_nav_files, "MANIFESTS_DIR", data_root / "manifests"))
        stack.enter_context(patch.object(table_nav_files, "ONTOLOGIES_DIR", data_root / "ontologies"))
        stack.enter_context(patch.object(table_nav_files, "MANIFESTS_DIR", data_root / "manifests"))
        stack.enter_context(patch.object(chunk_files, "CHUNKS_DIR", data_root / "chunks"))
        stack.enter_context(patch.object(chunk_files, "MANIFESTS_DIR", data_root / "manifests"))
        stack.enter_context(patch.object(wage_files, "WAGES_DIR", data_root / "wages"))
        stack.enter_context(patch.object(wage_files, "MANIFESTS_DIR", data_root / "manifests"))
        yield


def _test_replace_section_correctness() -> None:
    base = _base_contract_fixture()
    materializer = ContractMaterializer()
    patches = _patch_list_for_base(base)
    effective, _log = materializer.apply_patch_list(base, patches)

    section = effective["sections"][0]
    assert section["content_markdown"] == "Updated section text from MOA."
    assert "patch_2025_07_05" in section["amendments"]
    assert any(
        (ref.get("source_type") == "moa" and ref.get("pdf_page") == 7)
        for ref in section.get("provenance", [])
    ), "Section provenance must include MOA source ref"


def _test_replace_table_row_correctness() -> None:
    base = _base_contract_fixture()
    materializer = ContractMaterializer()
    patches = _patch_list_for_base(base)
    effective, _log = materializer.apply_patch_list(base, patches)

    rows = effective["tables"][materializer_module.WAGE_TABLE_ID]["rows"]
    assert len(rows) == 2, "MOA wage patch should preserve historical row and add superseding row"

    by_key = {str(row.get("row_key")): row for row in rows}
    base_key = "courtesy_clerk|hours:0|2024-01-21"
    superseded_key = "courtesy_clerk|hours:0|2025-07-05"
    assert base_key in by_key
    assert superseded_key in by_key

    base_row = by_key[base_key]
    new_row = by_key[superseded_key]
    assert float(base_row["columns"]["rate"]) == 17.0
    assert "patch_2025_07_05" not in (base_row.get("amendments") or [])
    assert float(new_row["columns"]["rate"]) == 17.5
    assert str(new_row["columns"]["effective_date"]) == "2025-07-05"
    assert "patch_2025_07_05" in (new_row.get("amendments") or [])
    assert any(
        (ref.get("source_type") == "moa" and ref.get("pdf_page") == 12)
        for ref in new_row.get("provenance", [])
    ), "Superseded wage row provenance must include MOA source ref"


def _test_collision_detection_expected_hash_mismatch() -> None:
    base = _base_contract_fixture()
    patches = _patch_list_for_base(base)
    bad_patch = copy.deepcopy(patches[0])
    bad_patch.operations[0].expected_prev_hash = "0" * 64

    materializer = ContractMaterializer()
    try:
        materializer.apply_patch_list(base, [bad_patch])
        raise AssertionError("Expected MaterializationFailure due to expected_prev_hash mismatch")
    except MaterializationFailure as exc:
        report = exc.report
        assert report.get("status") == "failed"
        assert any(err.get("code") == "operation_failed" for err in report.get("errors", []))
        first_op = (report.get("operations") or [{}])[0]
        assert "expected_prev_hash_mismatch" in (first_op.get("errors") or [])
        diagnostics = first_op.get("diagnostics") or {}
        assert diagnostics.get("expected_prev_hash") == "0" * 64
        assert diagnostics.get("actual_current_hash")
        assert isinstance(diagnostics.get("incoming_vs_current_diff"), list)


def _test_collision_diagnostics_include_last_touch() -> None:
    base = _base_contract_fixture()
    materializer = ContractMaterializer()
    section = base["sections"][0]
    row = base["tables"][materializer_module.WAGE_TABLE_ID]["rows"][0]
    section_hash = materializer.hash_text(section["content_markdown"])
    row_hash = materializer.hash_row(row["columns"])

    first_patch = _patch_fixture(
        expected_section_hash=section_hash,
        expected_row_hash=row_hash,
        new_text="Patch one section update.",
        new_rate=18.0,
    )
    second_patch = PatchArtifact.model_validate(
        {
            "schema_version": "moa_patch_v0_9_0",
            "patch_id": "patch_2026_01_01",
            "contract_id": "unit_test_contract",
            "source_pdf": "MOA-2026.pdf",
            "effective_date": "2026-01-01",
            "ratified_date": "2026-01-01",
            "parent_effective_version_id": "base_2024_01_01",
            "operations": [
                {
                    "op": "replace_section",
                    "target": {"anchor_id": "a1_s1", "article_num": 1, "section_num": 1},
                    "expected_prev_hash": section_hash,  # stale expected hash on purpose
                    "new_text_markdown": "Patch two section update.",
                    "source_refs": [{"source_type": "moa", "pdf": "MOA-2026.pdf", "pdf_page": 2}],
                    "review_status": "approved",
                }
            ],
        }
    )

    try:
        materializer.apply_patch_list(base, [first_patch, second_patch])
        raise AssertionError("Expected MaterializationFailure for stale hash in second patch")
    except MaterializationFailure as exc:
        report = exc.report
        failing = next(
            op for op in (report.get("operations") or [])
            if op.get("op_id") == "patch_2026_01_01#1"
        )
        diagnostics = failing.get("diagnostics") or {}
        last_touch = diagnostics.get("last_touch") or {}
        assert last_touch.get("patch_id") == "patch_2025_07_05"
        assert last_touch.get("op_id") == "patch_2025_07_05#1"
        assert diagnostics.get("current_excerpt")
        assert diagnostics.get("incoming_excerpt")
        assert isinstance(diagnostics.get("incoming_vs_current_diff"), list)
        assert diagnostics.get("incoming_vs_current_diff"), "Expected non-empty mismatch diff for collision diagnostics"


def _test_source_doc_id_resolution_and_tracking() -> None:
    base = _base_contract_fixture()
    materializer = ContractMaterializer()
    section_hash = materializer.hash_text(base["sections"][0]["content_markdown"])
    row_hash = materializer.hash_row(base["tables"][materializer_module.WAGE_TABLE_ID]["rows"][0]["columns"])

    with _workspace_tempdir("moa_source_doc_") as tmp:
        source_doc_root = tmp / "source_docs"
        doc_id = "shared_moa_2025_07_05"
        doc_dir = source_doc_root / "moa" / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        with open(doc_dir / "original.pdf", "wb") as f:
            f.write(b"%PDF-1.4\n%EOF\n")
        with open(doc_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "source_doc_v0_9_0",
                    "source_doc_id": doc_id,
                    "doc_type": "moa",
                    "source_pdf_filename": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                },
                f,
                indent=2,
            )

        patch_obj = PatchArtifact.model_validate(
            {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "patch_shared_doc",
                "contract_id": "unit_test_contract",
                "source_doc_id": doc_id,
                "effective_date": "2025-07-05",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": "a1_s1", "article_num": 1, "section_num": 1},
                        "expected_prev_hash": section_hash,
                        "new_text_markdown": "Updated from shared MOA doc.",
                        "source_refs": [{"source_type": "moa", "source_doc_id": doc_id, "pdf_page": 7}],
                        "review_status": "approved",
                    },
                    {
                        "op": "replace_table_row",
                        "target": {
                            "table_id": materializer_module.WAGE_TABLE_ID,
                            "row_key": "courtesy_clerk|hours:0|2024-01-21",
                        },
                        "expected_prev_hash": row_hash,
                        "new_row": {"rate": 17.75},
                        "source_refs": [{"source_type": "moa", "source_doc_id": doc_id, "pdf_page": 12}],
                        "review_status": "approved",
                    },
                ],
            }
        )

        with patch.object(source_docs, "SOURCE_DOCS_DIR", source_doc_root):
            effective, _log = materializer.apply_patch_list(base, [patch_obj])

    section = effective["sections"][0]
    assert any(ref.get("source_doc_id") == doc_id for ref in (section.get("provenance") or []))
    assert any(ref.get("pdf") == "Signed+MOA+-+July+5,+2025+(Safeway).pdf" for ref in (section.get("provenance") or []))
    assert effective["source_documents"]["amendment_source_doc_ids"] == [doc_id]


def _test_missing_source_refs_fails_build() -> None:
    base = _base_contract_fixture()
    materializer = ContractMaterializer()
    bad_patch = PatchArtifact.model_validate(
        {
            "schema_version": "moa_patch_v0_9_0",
            "patch_id": "patch_missing_sources",
            "contract_id": "unit_test_contract",
            "source_doc_id": "missing_source_doc",
            "effective_date": "2025-08-01",
            "operations": [
                {
                    "op": "replace_section",
                    "target": {"anchor_id": "a1_s1", "article_num": 1, "section_num": 1},
                    "expected_prev_hash": materializer.hash_text(base["sections"][0]["content_markdown"]),
                    "new_text_markdown": "Should fail source resolution.",
                    "source_refs": [],
                    "review_status": "approved",
                }
            ],
        }
    )

    with _workspace_tempdir("moa_missing_sources_") as tmp:
        missing_source_doc_root = tmp / "no_source_docs"
        with patch.object(source_docs, "SOURCE_DOCS_DIR", missing_source_doc_root):
            try:
                materializer.apply_patch_list(base, [bad_patch])
                raise AssertionError("Expected MaterializationFailure for missing source refs")
            except MaterializationFailure as exc:
                report = exc.report
                assert report.get("status") == "failed"
                first_op = (report.get("operations") or [{}])[0]
                assert "missing_source_refs" in (first_op.get("errors") or [])
                diagnostics = first_op.get("diagnostics") or {}
                assert diagnostics.get("patch_source_doc_id") == "missing_source_doc"


def _test_materialization_determinism_bytes_and_hashes() -> None:
    contract_id = "determinism_contract"
    with _workspace_tempdir("moa_det_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (contract_root / "amendments").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks = [
            {
                "contract_id": contract_id,
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content": "Base text for determinism test.",
                "content_with_tables": "Base text for determinism test.",
                "chunk_id": "chunk_1",
            }
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": ["2024-01-21"],
            "classifications": {
                "courtesy_clerk": {
                    "name": "COURTESY CLERK",
                    "normalized_name": "courtesy_clerk",
                    "steps": [
                        {
                            "step_name": "Start",
                            "hours_required": 0,
                            "months_required": 0,
                            "rates": {"2024-01-21": 17.0},
                        }
                    ],
                }
            },
            "canonical_wage_rows": [
                {
                    "classification_key": "courtesy_clerk",
                    "classification_name": "COURTESY CLERK",
                    "step_name": "Start",
                    "step_type": "hours",
                    "threshold_value": 0,
                    "effective_date": "2024-01-21",
                    "rate": 17.0,
                    "row_type": "step_row",
                    "source_method": "table_registry",
                    "confidence": 0.9,
                    "source_reference": {"table_id": "tbl_art1_1", "row_index": 0},
                }
            ],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)
            section = base_state["sections"][0]
            row = base_state["tables"][materializer_module.WAGE_TABLE_ID]["rows"][0]

            patch_payload = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "determinism_patch_1",
                "contract_id": contract_id,
                "source_pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                "effective_date": "2025-07-05",
                "ratified_date": "2025-07-05",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": section["anchor_id"], "article_num": 1, "section_num": 1},
                        "expected_prev_hash": ContractMaterializer.hash_text(section["content_markdown"]),
                        "new_text_markdown": "Updated deterministic text.",
                        "source_refs": [
                            {
                                "source_type": "moa",
                                "pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                                "pdf_page": 7,
                            }
                        ],
                        "review_status": "approved",
                    },
                    {
                        "op": "replace_table_row",
                        "target": {
                            "table_id": materializer_module.WAGE_TABLE_ID,
                            "row_key": row["row_key"],
                        },
                        "expected_prev_hash": ContractMaterializer.hash_row(row["columns"]),
                        "new_row": {"rate": 17.5},
                        "source_refs": [
                            {
                                "source_type": "moa",
                                "pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
                                "pdf_page": 12,
                            }
                        ],
                        "review_status": "approved",
                    },
                ],
            }
            patch_path = contract_root / "amendments" / "determinism_patch_1.json"
            with open(patch_path, "w", encoding="utf-8") as f:
                json.dump(patch_payload, f, indent=2)

            result1 = materializer_module.materialize_contract(
                contract_id=contract_id,
                effective_version_id="effective_test",
                patch_paths=[patch_path],
                write_latest_pointer=True,
            )
            contract_bytes_1 = Path(result1["effective_contract_path"]).read_bytes()
            markdown_bytes_1 = Path(result1["effective_markdown_path"]).read_bytes()
            log_bytes_1 = Path(result1["build_log_path"]).read_bytes()
            patch_chain_bytes_1 = Path(result1["patch_chain_path"]).read_bytes()
            index_chunks_bytes_1 = Path(result1["index_chunks_path"]).read_bytes()
            index_wages_bytes_1 = Path(result1["index_wages_path"]).read_bytes()

            result2 = materializer_module.materialize_contract(
                contract_id=contract_id,
                effective_version_id="effective_test",
                patch_paths=[patch_path],
                write_latest_pointer=True,
            )
            contract_bytes_2 = Path(result2["effective_contract_path"]).read_bytes()
            markdown_bytes_2 = Path(result2["effective_markdown_path"]).read_bytes()
            log_bytes_2 = Path(result2["build_log_path"]).read_bytes()
            patch_chain_bytes_2 = Path(result2["patch_chain_path"]).read_bytes()
            index_chunks_bytes_2 = Path(result2["index_chunks_path"]).read_bytes()
            index_wages_bytes_2 = Path(result2["index_wages_path"]).read_bytes()

            assert contract_bytes_1 == contract_bytes_2
            assert markdown_bytes_1 == markdown_bytes_2
            assert log_bytes_1 == log_bytes_2
            assert patch_chain_bytes_1 == patch_chain_bytes_2
            assert index_chunks_bytes_1 == index_chunks_bytes_2
            assert index_wages_bytes_1 == index_wages_bytes_2
            assert result1["artifact_hashes"] == result2["artifact_hashes"]
            assert result1["effective_content_hash"] == result2["effective_content_hash"]
            assert len(str(result1["effective_content_hash"])) == 64
            assert result1["artifact_hashes"].get("patch_chain_sha256")

            latest_pointer = json.loads(
                (data_root / "contracts" / contract_id / "effective" / "latest.json").read_text(encoding="utf-8")
            )
            assert latest_pointer.get("effective_content_hash") == result1["effective_content_hash"]

            patch_chain = json.loads(Path(result1["patch_chain_path"]).read_text(encoding="utf-8"))
            assert patch_chain.get("effective_content_hash") == result1["effective_content_hash"]
            assert patch_chain.get("applied_patch_ids") == ["determinism_patch_1"]
            assert patch_chain.get("patch_count") == 1

            effective_wages = json.loads(Path(result1["index_wages_path"]).read_text(encoding="utf-8"))
            assert effective_wages.get("effective_version_id") == "effective_test"
            assert "determinism_patch_1" in (effective_wages.get("amendments_applied") or [])
            assert "2025-07-05" in (effective_wages.get("effective_dates") or [])
            rows_by_date = {
                str(row.get("effective_date")): row
                for row in (effective_wages.get("canonical_wage_rows") or [])
                if str(row.get("classification_key") or "") == "courtesy_clerk"
            }
            assert float(rows_by_date["2024-01-21"]["rate"]) == 17.0
            assert float(rows_by_date["2025-07-05"]["rate"]) == 17.5
            assert rows_by_date["2024-01-21"].get("amendments_applied") in ([], None)
            assert rows_by_date["2025-07-05"].get("amendments_applied") == ["determinism_patch_1"]


def _test_retrieval_prefers_effective_snapshot_inputs() -> None:
    contract_id = "router_contract"
    with _workspace_tempdir("moa_router_") as tmp:
        data_root = tmp / "data"
        effective_index_dir = data_root / "contracts" / contract_id / "effective" / "effective_v1" / "index_inputs"
        shared_chunks_dir = data_root / "chunks"
        shared_wages_dir = data_root / "wages"
        manifests_dir = data_root / "manifests"

        effective_index_dir.mkdir(parents=True, exist_ok=True)
        shared_chunks_dir.mkdir(parents=True, exist_ok=True)
        shared_wages_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        with open(data_root / "contracts" / contract_id / "effective" / "latest.json", "w", encoding="utf-8") as f:
            json.dump({"effective_version_id": "effective_v1"}, f)
        with open(manifests_dir / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        effective_chunk_path = effective_index_dir / f"contract_chunks_enriched_{contract_id}.json"
        shared_chunk_path = shared_chunks_dir / f"contract_chunks_enriched_{contract_id}.json"
        with open(effective_chunk_path, "w", encoding="utf-8") as f:
            json.dump([{"source": "effective"}], f)
        with open(shared_chunk_path, "w", encoding="utf-8") as f:
            json.dump([{"source": "base"}], f)

        effective_wage_path = effective_index_dir / f"wage_tables_{contract_id}.json"
        shared_wage_path = shared_wages_dir / f"wage_tables_{contract_id}.json"
        with open(effective_wage_path, "w", encoding="utf-8") as f:
            json.dump({"source": "effective"}, f)
        with open(shared_wage_path, "w", encoding="utf-8") as f:
            json.dump({"source": "base"}, f)

        with _patched_data_root(data_root):
            resolved_chunks = chunk_files.resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
            resolved_wages = wage_files.resolve_wage_file(contract_id=contract_id, allow_shared_fallback=True)

            assert resolved_chunks == effective_chunk_path
            assert resolved_wages == effective_wage_path


def _test_ensure_base_snapshot_refreshes_when_package_artifacts_change() -> None:
    contract_id = "base_refresh_contract"
    with _workspace_tempdir("moa_base_refresh_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks_path = contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json"
        source_wages_path = contract_root / "wages" / f"wage_tables_{contract_id}.json"
        with open(source_chunks_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "chunk_id": "c1",
                        "contract_id": contract_id,
                        "region_id": "region-test",
                        "doc_type": "cba",
                        "article_num": 1,
                        "article_title": "General",
                        "section_num": 1,
                        "citation": "Article 1, Section 1",
                        "content": "Base text",
                        "content_with_tables": "Base text",
                    }
                ],
                f,
                indent=2,
            )
        with open(source_wages_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "contract_id": contract_id,
                    "effective_dates": ["2024-01-21"],
                    "classifications": {},
                    "canonical_wage_rows": [],
                },
                f,
                indent=2,
            )

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            original_base_chunks = json.loads(base_paths["chunks"].read_text(encoding="utf-8"))
            assert str(original_base_chunks[0].get("doc_type") or "") == "cba"

            with open(source_chunks_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "chunk_id": "c1",
                            "contract_id": contract_id,
                            "region_id": "region-test",
                            "doc_type": "lou",
                            "article_num": 1,
                            "article_title": "General",
                            "section_num": 1,
                            "citation": "Letter of Understanding",
                            "content": "Updated LOU text",
                            "content_with_tables": "Updated LOU text",
                        }
                    ],
                    f,
                    indent=2,
                )

            refreshed_paths = materializer_module.ensure_base_snapshot(contract_id)
            refreshed_base_chunks = json.loads(refreshed_paths["chunks"].read_text(encoding="utf-8"))
            assert str(refreshed_base_chunks[0].get("doc_type") or "") == "lou"


def _test_rebase_patch_file_updates_stale_expected_hash() -> None:
    contract_id = "rebase_contract"
    with _workspace_tempdir("moa_rebase_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (contract_root / "amendments").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks = [
            {
                "contract_id": contract_id,
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content": "Base text for rebase test.",
                "content_with_tables": "Base text for rebase test.",
                "chunk_id": "chunk_1",
            }
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": ["2024-01-21"],
            "classifications": {},
            "canonical_wage_rows": [],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)
            section = base_state["sections"][0]
            base_hash = ContractMaterializer.hash_text(section["content_markdown"])

            patch_a = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "patch_a",
                "contract_id": contract_id,
                "source_pdf": "MOA-A.pdf",
                "effective_date": "2025-01-01",
                "ratified_date": "2025-01-01",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": section["anchor_id"], "article_num": 1, "section_num": 1},
                        "expected_prev_hash": base_hash,
                        "new_text_markdown": "Patch A text",
                        "source_refs": [{"source_type": "moa", "pdf": "MOA-A.pdf", "pdf_page": 2}],
                        "review_status": "approved",
                    }
                ],
            }
            patch_a_path = contract_root / "amendments" / "patch_a.json"
            with open(patch_a_path, "w", encoding="utf-8") as f:
                json.dump(patch_a, f, indent=2)

            patch_b = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "patch_b",
                "contract_id": contract_id,
                "source_pdf": "MOA-B.pdf",
                "effective_date": "2025-02-01",
                "ratified_date": "2025-02-01",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": section["anchor_id"], "article_num": 1, "section_num": 1},
                        "expected_prev_hash": base_hash,  # stale against patch_a-applied state
                        "new_text_markdown": "Patch B text",
                        "source_refs": [{"source_type": "moa", "pdf": "MOA-B.pdf", "pdf_page": 3}],
                        "review_status": "approved",
                    }
                ],
            }
            patch_b_path = contract_root / "amendments" / "patch_b.json"
            with open(patch_b_path, "w", encoding="utf-8") as f:
                json.dump(patch_b, f, indent=2)

            rebase_result = materializer_module.rebase_patch_file(
                contract_id=contract_id,
                patch_path=patch_b_path,
                prior_patch_paths=[patch_a_path],
            )
            rebased_payload = rebase_result["rebased_patch_payload"]
            change = rebase_result["changes"][0]
            assert change["changed"] is True
            assert change["old_expected_prev_hash"] == base_hash
            assert change["last_touch"]["patch_id"] == "patch_a"
            assert rebased_payload["operations"][0]["expected_prev_hash"] != base_hash

            materializer = ContractMaterializer()
            patch_a_model = PatchArtifact.model_validate(patch_a)
            patch_b_rebased_model = PatchArtifact.model_validate(rebased_payload)
            effective, _log = materializer.apply_patch_list(base_state, [patch_a_model, patch_b_rebased_model])
            assert effective["sections"][0]["content_markdown"] == "Patch B text"


def _test_patch_chain_manifest_orders_patches_deterministically() -> None:
    contract_id = "patch_chain_contract"
    with _workspace_tempdir("moa_chain_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (contract_root / "amendments").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks = [
            {
                "contract_id": contract_id,
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content": "Base text for chain test.",
                "content_with_tables": "Base text for chain test.",
                "chunk_id": "chunk_1",
            }
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": ["2024-01-21"],
            "classifications": {},
            "canonical_wage_rows": [],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)
            section = base_state["sections"][0]
            base_hash = ContractMaterializer.hash_text(section["content_markdown"])
            patch_a_text = "Patch A chain text."
            patch_b_text = "Patch B chain text."
            patch_a_hash = ContractMaterializer.hash_text(patch_a_text)

            patch_a = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "patch_a",
                "contract_id": contract_id,
                "source_pdf": "MOA-A.pdf",
                "effective_date": "2025-01-01",
                "ratified_date": "2025-01-01",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": section["anchor_id"], "article_num": 1, "section_num": 1},
                        "expected_prev_hash": base_hash,
                        "new_text_markdown": patch_a_text,
                        "source_refs": [{"source_type": "moa", "pdf": "MOA-A.pdf", "pdf_page": 2}],
                        "review_status": "approved",
                    }
                ],
            }
            patch_b = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "patch_b",
                "contract_id": contract_id,
                "source_pdf": "MOA-B.pdf",
                "effective_date": "2025-02-01",
                "ratified_date": "2025-02-01",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": section["anchor_id"], "article_num": 1, "section_num": 1},
                        "expected_prev_hash": patch_a_hash,
                        "new_text_markdown": patch_b_text,
                        "source_refs": [{"source_type": "moa", "pdf": "MOA-B.pdf", "pdf_page": 3}],
                        "review_status": "approved",
                    }
                ],
            }
            patch_a_path = contract_root / "amendments" / "patch_a.json"
            patch_b_path = contract_root / "amendments" / "patch_b.json"
            with open(patch_a_path, "w", encoding="utf-8") as f:
                json.dump(patch_a, f, indent=2)
            with open(patch_b_path, "w", encoding="utf-8") as f:
                json.dump(patch_b, f, indent=2)

            result = materializer_module.materialize_contract(
                contract_id=contract_id,
                effective_version_id="effective_chain",
                patch_paths=[patch_b_path, patch_a_path],  # intentionally unsorted input
                write_latest_pointer=True,
            )
            patch_chain = json.loads(Path(result["patch_chain_path"]).read_text(encoding="utf-8"))
            assert patch_chain.get("applied_patch_ids") == ["patch_a", "patch_b"]
            assert [row.get("patch_id") for row in (patch_chain.get("patches") or [])] == ["patch_a", "patch_b"]
            assert patch_chain.get("patch_count") == 2


def _test_load_base_state_keeps_lou_chunks() -> None:
    contract_id = "lou_base_state_contract"
    with _workspace_tempdir("moa_lou_state_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks = [
            {
                "contract_id": contract_id,
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content": "Base article text.",
                "content_with_tables": "Base article text.",
                "chunk_id": "chunk_1",
                "doc_type": "cba",
            },
            {
                "contract_id": contract_id,
                "article_num": None,
                "section_num": None,
                "subsection": None,
                "citation": "Letter of Understanding 1: Test Item, Part 1",
                "article_title": "Letter of Understanding 1",
                "content": "LOU item part one.",
                "content_with_tables": "LOU item part one.",
                "chunk_id": "lou_1_part1",
                "doc_type": "lou",
                "parent_context": "Letters of Understanding > Item 1: Test Item",
            },
            {
                "contract_id": contract_id,
                "article_num": None,
                "section_num": None,
                "subsection": None,
                "citation": "Letter of Understanding 1: Test Item, Part 2",
                "article_title": "Letter of Understanding 1",
                "content": "LOU item part two.",
                "content_with_tables": "LOU item part two.",
                "chunk_id": "lou_1_part2",
                "doc_type": "lou",
                "parent_context": "Letters of Understanding > Item 1: Test Item",
            },
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": [],
            "classifications": {},
            "canonical_wage_rows": [],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)

            sections = list(base_state.get("sections") or [])
            assert len(sections) == 3
            lou_sections = [row for row in sections if str(row.get("doc_type") or "").lower() == "lou"]
            assert len(lou_sections) == 2
            anchor_ids = [str(row.get("anchor_id") or "") for row in sections]
            assert len(anchor_ids) == len(set(anchor_ids))
            assert all(anchor.startswith("lou_") for anchor in [row.get("anchor_id") for row in lou_sections])


def _test_effective_materialization_preserves_lou_chunks_in_index() -> None:
    contract_id = "lou_effective_contract"
    with _workspace_tempdir("moa_lou_effective_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (contract_root / "amendments").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump({"contract_id": contract_id}, f)

        source_chunks = [
            {
                "contract_id": contract_id,
                "article_num": 1,
                "section_num": 1,
                "subsection": None,
                "citation": "Article 1, Section 1",
                "article_title": "Term",
                "content": "Base article text.",
                "content_with_tables": "Base article text.",
                "chunk_id": "chunk_1",
                "doc_type": "cba",
            },
            {
                "contract_id": contract_id,
                "article_num": None,
                "section_num": None,
                "subsection": None,
                "citation": "Letter of Understanding 4: Test LOU",
                "article_title": "Letter of Understanding 4",
                "content": "LOU rule text.",
                "content_with_tables": "LOU rule text.",
                "chunk_id": "lou_4_part1",
                "doc_type": "lou",
                "parent_context": "Letters of Understanding > Item 4: Test LOU",
            },
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": [],
            "classifications": {},
            "canonical_wage_rows": [],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)
            article_section = next(
                row for row in (base_state.get("sections") or [])
                if str(row.get("citation") or "").strip() == "Article 1, Section 1"
            )
            patch_payload = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "lou_keep_patch",
                "contract_id": contract_id,
                "source_pdf": "LOU-KEEP-MOA.pdf",
                "effective_date": "2025-08-01",
                "ratified_date": "2025-08-01",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {
                            "anchor_id": article_section["anchor_id"],
                            "article_num": 1,
                            "section_num": 1,
                        },
                        "expected_prev_hash": ContractMaterializer.hash_text(article_section["content_markdown"]),
                        "new_text_markdown": "Updated article text.",
                        "source_refs": [
                            {
                                "source_type": "moa",
                                "pdf": "LOU-KEEP-MOA.pdf",
                                "pdf_page": 2,
                            }
                        ],
                        "review_status": "approved",
                    }
                ],
            }
            patch_path = contract_root / "amendments" / "lou_keep_patch.json"
            with open(patch_path, "w", encoding="utf-8") as f:
                json.dump(patch_payload, f, indent=2)

            result = materializer_module.materialize_contract(
                contract_id=contract_id,
                effective_version_id="effective_lou_test",
                patch_paths=[patch_path],
                write_latest_pointer=True,
            )
            effective_chunks = json.loads(Path(result["index_chunks_path"]).read_text(encoding="utf-8"))
            doc_counts = {}
            for row in effective_chunks:
                key = str(row.get("doc_type") or "").strip().lower() or "unknown"
                doc_counts[key] = int(doc_counts.get(key, 0)) + 1
            assert doc_counts.get("cba") == 1
            assert doc_counts.get("lou") == 1


def _test_effective_materialization_writes_entitlement_index_input() -> None:
    contract_id = "effective_entitlement_contract"
    with _workspace_tempdir("moa_entitlement_effective_") as tmp:
        data_root = tmp / "data"
        contract_root = data_root / "contracts" / contract_id
        (contract_root / "chunks").mkdir(parents=True, exist_ok=True)
        (contract_root / "wages").mkdir(parents=True, exist_ok=True)
        (contract_root / "amendments").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests").mkdir(parents=True, exist_ok=True)
        (data_root / "ontologies").mkdir(parents=True, exist_ok=True)
        (data_root / "chunks").mkdir(parents=True, exist_ok=True)
        (data_root / "wages").mkdir(parents=True, exist_ok=True)

        with open(data_root / "manifests" / f"{contract_id}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "contract_id": contract_id,
                    "region_id": "region-test",
                    "article_titles": {"17": "Vacations"},
                },
                f,
                indent=2,
            )

        source_chunks = [
            {
                "chunk_id": "vac_1",
                "contract_id": contract_id,
                "region_id": "region-test",
                "doc_type": "cba",
                "article_num": 17,
                "article_title": "Vacations",
                "section_num": 1,
                "subsection": None,
                "citation": "Article 17, Section 1",
                "parent_context": "Vacation eligibility",
                "content": (
                    "All regular full-time employees shall receive one (1) week's paid vacation "
                    "after one (1) year of continuous service and two (2) weeks' paid vacation "
                    "after two (2) years of continuous service."
                ),
                "content_with_tables": (
                    "All regular full-time employees shall receive one (1) week's paid vacation "
                    "after one (1) year of continuous service and two (2) weeks' paid vacation "
                    "after two (2) years of continuous service."
                ),
            }
        ]
        with open(
            contract_root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_chunks, f, indent=2)

        source_wages = {
            "contract_id": contract_id,
            "effective_dates": ["2024-01-21"],
            "classifications": {},
            "canonical_wage_rows": [],
        }
        with open(
            contract_root / "wages" / f"wage_tables_{contract_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(source_wages, f, indent=2)

        with _patched_data_root(data_root):
            base_paths = materializer_module.ensure_base_snapshot(contract_id)
            base_state = materializer_module.load_base_contract_state(contract_id, base_paths)
            section = base_state["sections"][0]
            patch_payload = {
                "schema_version": "moa_patch_v0_9_0",
                "patch_id": "ent_patch",
                "contract_id": contract_id,
                "source_pdf": "ENT-MOA.pdf",
                "effective_date": "2025-07-05",
                "ratified_date": "2025-07-05",
                "parent_effective_version_id": "base_snapshot_v0_9_0",
                "operations": [
                    {
                        "op": "replace_section",
                        "target": {"anchor_id": str(section.get("anchor_id") or "")},
                        "expected_prev_hash": ContractMaterializer.hash_text(section["content_markdown"]),
                        "new_text_markdown": section["content_markdown"] + "\nVacation language remains in force.",
                        "source_refs": [
                            {
                                "source_type": "moa",
                                "pdf": "ENT-MOA.pdf",
                                "pdf_page": 4,
                            }
                        ],
                        "review_status": "approved",
                    }
                ],
            }
            patch_path = contract_root / "amendments" / "ent_patch.json"
            with open(patch_path, "w", encoding="utf-8") as f:
                json.dump(patch_payload, f, indent=2)

            result = materializer_module.materialize_contract(
                contract_id=contract_id,
                effective_version_id="effective_ent_test",
                patch_paths=[patch_path],
                write_latest_pointer=True,
            )
            effective_entitlements = json.loads(
                Path(result["index_entitlements_path"]).read_text(encoding="utf-8")
            )
            assert effective_entitlements.get("contract_id") == contract_id
            assert len(effective_entitlements.get("vacation_entitlements") or []) >= 1


def main() -> None:
    _test_replace_section_correctness()
    _test_replace_table_row_correctness()
    _test_collision_detection_expected_hash_mismatch()
    _test_collision_diagnostics_include_last_touch()
    _test_source_doc_id_resolution_and_tracking()
    _test_missing_source_refs_fails_build()
    _test_materialization_determinism_bytes_and_hashes()
    _test_retrieval_prefers_effective_snapshot_inputs()
    _test_ensure_base_snapshot_refreshes_when_package_artifacts_change()
    _test_rebase_patch_file_updates_stale_expected_hash()
    _test_patch_chain_manifest_orders_patches_deterministically()
    _test_load_base_state_keeps_lou_chunks()
    _test_effective_materialization_preserves_lou_chunks_in_index()
    _test_effective_materialization_writes_entitlement_index_input()
    print("[OK] MOA materializer and effective-routing checks passed")


if __name__ == "__main__":
    main()
