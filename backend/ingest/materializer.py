"""Deterministic MOA materialization compiler (v0.9.0)."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.config import DATA_DIR
from backend.effective_contracts import write_latest_effective_pointer
from backend.ingest.extract_entitlements import extract_entitlements
from backend.ingest.moa_schema import (
    APPROVED_REVIEW_STATUS,
    PatchArtifact,
    ReplaceSectionOperation,
    ReplaceTableRowOperation,
    load_patch_artifact,
    load_patch_artifacts,
)
from backend.pdf_nav_files import resolve_pdf_nav_index_file
from backend.pdf_nav_index import build_pdf_nav_index, load_pdf_nav_index, to_runtime_navigation_maps
from backend.source_docs import resolve_source_doc_pdf_name
from backend.table_nav_files import resolve_table_nav_index_file
from backend.table_nav_index import build_table_nav_index, load_table_nav_index, to_runtime_table_navigation_maps


EFFECTIVE_CONTRACT_SCHEMA_VERSION = "effective_contract_v0_9_0"
BUILD_LOG_SCHEMA_VERSION = "materializer_build_log_v0_9_0"
MATERIALIZER_VERSION = "0.9.0"
WAGE_TABLE_ID = "appendix_a_wage_rows"
PATCH_CHAIN_SCHEMA_VERSION = "patch_chain_manifest_v0_9_0"


@dataclass
class MaterializationFailure(Exception):
    report: dict

    def __str__(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, sort_keys=True)


@dataclass
class PatchRebaseFailure(Exception):
    report: dict

    def __str__(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, sort_keys=True)


class ContractMaterializer:
    """Pure patch application compiler with deterministic canonicalization."""

    @staticmethod
    def canonical_text(text: Any) -> str:
        value = str(text or "")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t]+", " ", value)
        value = "\n".join(line.rstrip() for line in value.split("\n"))
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    @staticmethod
    def canonical_row(row: dict[str, Any]) -> str:
        def normalize(v: Any) -> Any:
            if isinstance(v, dict):
                return {str(k): normalize(v[k]) for k in sorted(v)}
            if isinstance(v, list):
                return [normalize(item) for item in v]
            if isinstance(v, str):
                return ContractMaterializer.canonical_text(v)
            return v

        normalized = normalize(row or {})
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def hash_text(cls, text: Any) -> str:
        return cls.sha256_hex(cls.canonical_text(text))

    @classmethod
    def hash_row(cls, row: dict[str, Any]) -> str:
        return cls.sha256_hex(cls.canonical_row(row))

    def apply_patch_list(
        self,
        base_contract: dict[str, Any],
        patches: list[PatchArtifact],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = copy.deepcopy(base_contract)
        contract_id = str(state.get("contract_id") or "")
        build_entries: list[dict] = []
        errors: list[dict] = []
        applied_patch_ids: list[str] = []
        applied_by_target: dict[str, dict[str, Optional[str]]] = {}

        for patch in sorted(patches, key=lambda p: (str(p.effective_date), str(p.patch_id))):
            if str(patch.contract_id) != contract_id:
                errors.append(
                    {
                        "code": "contract_id_mismatch",
                        "patch_id": patch.patch_id,
                        "message": f"Patch contract_id '{patch.contract_id}' does not match '{contract_id}'",
                    }
                )
                continue
            patch_source_pdf = _resolve_patch_source_pdf_name(patch)

            patch_entries: list[dict] = []
            had_approved = False

            for idx, operation in enumerate(patch.operations, start=1):
                op_id = f"{patch.patch_id}#{idx}"
                op_sources = _normalized_source_refs(
                    source_refs=operation.source_refs,
                    default_pdf=patch_source_pdf,
                    default_source_doc_id=patch.source_doc_id,
                )
                entry = {
                    "op_id": op_id,
                    "op": operation.op,
                    "target": operation.target.model_dump(),
                    "before_hash": None,
                    "after_hash": None,
                    "applied": False,
                    "sources": op_sources,
                    "errors": [],
                    "diagnostics": {},
                }

                if str(operation.review_status).lower() != APPROVED_REVIEW_STATUS:
                    entry["errors"].append("review_status_not_approved")
                    patch_entries.append(entry)
                    continue

                had_approved = True
                if not op_sources:
                    entry["errors"].append("missing_source_refs")
                    entry["diagnostics"] = {
                        "patch_source_pdf": str(patch.source_pdf or "") or None,
                        "patch_source_doc_id": str(patch.source_doc_id or "") or None,
                        "message": "No resolvable source refs (requires pdf and/or resolvable source_doc_id).",
                    }
                    errors.append(
                        {
                            "code": "operation_failed",
                            "patch_id": patch.patch_id,
                            "op_id": op_id,
                            "target": entry["target"],
                            "errors": list(entry["errors"]),
                            "diagnostics": dict(entry.get("diagnostics") or {}),
                        }
                    )
                    patch_entries.append(entry)
                    continue

                if isinstance(operation, ReplaceSectionOperation):
                    self._apply_replace_section(
                        state=state,
                        patch=patch,
                        operation=operation,
                        entry=entry,
                        applied_by_target=applied_by_target,
                    )
                elif isinstance(operation, ReplaceTableRowOperation):
                    self._apply_replace_table_row(
                        state=state,
                        patch=patch,
                        operation=operation,
                        entry=entry,
                        applied_by_target=applied_by_target,
                    )
                else:
                    entry["errors"].append("unsupported_operation")

                if entry["errors"]:
                    errors.append(
                        {
                            "code": "operation_failed",
                            "patch_id": patch.patch_id,
                            "op_id": op_id,
                            "target": entry["target"],
                            "errors": list(entry["errors"]),
                            "diagnostics": dict(entry.get("diagnostics") or {}),
                        }
                    )
                patch_entries.append(entry)

            build_entries.extend(patch_entries)

            if had_approved and not any(e for e in patch_entries if e["errors"]):
                applied_patch_ids.append(patch.patch_id)

            if any(e for e in patch_entries if e["errors"] and "review_status_not_approved" not in e["errors"]):
                # Apply-or-fail invariant: stop as soon as a clean application fails.
                break

        build_log = {
            "schema_version": BUILD_LOG_SCHEMA_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "contract_id": contract_id,
            "parent_effective_version_id": str(base_contract.get("base_version_id") or ""),
            "operations": build_entries,
            "status": "failed" if errors else "success",
            "errors": errors,
        }

        if errors:
            raise MaterializationFailure(report=build_log)

        state["amendments_applied"] = applied_patch_ids
        state["parent_effective_version_id"] = str(base_contract.get("base_version_id") or "")
        state["source_documents"] = _merge_source_documents_with_patches(
            state.get("source_documents") or {},
            patches=patches,
            applied_patch_ids=applied_patch_ids,
        )
        return state, build_log

    def _apply_replace_section(
        self,
        state: dict[str, Any],
        patch: PatchArtifact,
        operation: ReplaceSectionOperation,
        entry: dict[str, Any],
        applied_by_target: dict[str, dict[str, Optional[str]]],
    ) -> None:
        section, idx = _resolve_section_target(
            sections=state.get("sections") or [],
            target=operation.target.model_dump(),
        )
        if section is None or idx is None:
            entry["errors"].append("section_target_not_found")
            return

        before_hash = self.hash_text(section.get("content_markdown", ""))
        entry["before_hash"] = before_hash
        if before_hash != operation.expected_prev_hash:
            canonical_current = self.canonical_text(section.get("content_markdown", ""))
            canonical_incoming = self.canonical_text(operation.new_text_markdown)
            target_key = _section_target_key(section)
            last_touch = _resolve_last_touch(
                applied_by_target=applied_by_target,
                target_key=target_key,
                existing_amendments=section.get("amendments") or [],
            )
            entry["diagnostics"] = {
                "expected_prev_hash": operation.expected_prev_hash,
                "actual_current_hash": before_hash,
                "target_key": target_key,
                "last_touch": last_touch,
                "current_excerpt": _excerpt(canonical_current),
                "incoming_excerpt": _excerpt(canonical_incoming),
                "incoming_vs_current_diff": _short_unified_diff(
                    old_text=canonical_current,
                    new_text=canonical_incoming,
                ),
            }
            entry["errors"].append("expected_prev_hash_mismatch")
            return

        updated = dict(section)
        updated["content_markdown"] = str(operation.new_text_markdown)
        updated["span_id"] = self.sha256_hex(
            "|".join(
                [
                    self.canonical_text(updated["content_markdown"]),
                    str(updated.get("article_title") or ""),
                    str(updated.get("citation") or ""),
                ]
            )
        )
        updated["provenance"] = _merge_provenance(
            existing=updated.get("provenance") or [],
            incoming=entry.get("sources") or [],
        )
        amendments = list(updated.get("amendments") or [])
        if patch.patch_id not in amendments:
            amendments.append(patch.patch_id)
        updated["amendments"] = amendments

        state["sections"][idx] = updated
        entry["after_hash"] = self.hash_text(updated.get("content_markdown", ""))
        entry["applied"] = True
        applied_by_target[_section_target_key(updated)] = {
            "patch_id": patch.patch_id,
            "op_id": str(entry.get("op_id") or ""),
        }

    def _apply_replace_table_row(
        self,
        state: dict[str, Any],
        patch: PatchArtifact,
        operation: ReplaceTableRowOperation,
        entry: dict[str, Any],
        applied_by_target: dict[str, dict[str, Optional[str]]],
    ) -> None:
        table_id = str(operation.target.table_id or "").strip()
        row_key = str(operation.target.row_key or "").strip()
        table = (state.get("tables") or {}).get(table_id)
        if not isinstance(table, dict):
            entry["errors"].append("table_target_not_found")
            return

        rows = table.get("rows") or []
        row_idx = None
        for i, row in enumerate(rows):
            if str(row.get("row_key") or "") == row_key:
                row_idx = i
                break
        if row_idx is None:
            entry["errors"].append("row_key_not_found")
            return

        row = dict(rows[row_idx])
        columns = dict(row.get("columns") or {})
        before_hash = self.hash_row(columns)
        entry["before_hash"] = before_hash
        if before_hash != operation.expected_prev_hash:
            canonical_current = self.canonical_row(columns)
            canonical_incoming = self.canonical_row(dict(operation.new_row or {}))
            target_key = _table_row_target_key(table_id=table_id, row_key=row_key)
            last_touch = _resolve_last_touch(
                applied_by_target=applied_by_target,
                target_key=target_key,
                existing_amendments=row.get("amendments") or [],
            )
            entry["diagnostics"] = {
                "expected_prev_hash": operation.expected_prev_hash,
                "actual_current_hash": before_hash,
                "target_key": target_key,
                "last_touch": last_touch,
                "current_excerpt": _excerpt(canonical_current),
                "incoming_excerpt": _excerpt(canonical_incoming),
                "incoming_vs_current_diff": _short_unified_diff(
                    old_text=canonical_current,
                    new_text=canonical_incoming,
                ),
            }
            entry["errors"].append("expected_prev_hash_mismatch")
            return

        # Wage rows often need MOA-effective supersession (new effective date) rather than
        # overwriting the historical row in-place. Preserve chronology by cloning when the
        # patch date is newer than the target row date and no explicit historical backfill date
        # was provided in the patch payload.
        supersede_mode = False
        patch_effective_date = str(patch.effective_date or "").strip()
        current_effective_date = str(columns.get("effective_date") or "").strip()
        explicit_new_effective_date = str((operation.new_row or {}).get("effective_date") or "").strip()
        if (
            table_id == WAGE_TABLE_ID
            and _is_iso_date(patch_effective_date)
            and _is_iso_date(current_effective_date)
            and patch_effective_date > current_effective_date
            and not explicit_new_effective_date
        ):
            supersede_mode = True

        merged_columns = dict(columns)
        for k, v in (operation.new_row or {}).items():
            merged_columns[str(k)] = v
        if supersede_mode:
            merged_columns["effective_date"] = patch_effective_date
            new_row_key = _wage_row_key(merged_columns)
            if any(
                idx != row_idx and str(existing.get("row_key") or "") == new_row_key
                for idx, existing in enumerate(rows)
            ):
                entry["diagnostics"] = {
                    **dict(entry.get("diagnostics") or {}),
                    "target_key": _table_row_target_key(table_id=table_id, row_key=row_key),
                    "computed_new_row_key": new_row_key,
                    "patch_effective_date": patch_effective_date,
                    "current_effective_date": current_effective_date,
                }
                entry["errors"].append("supersede_row_key_conflict")
                return

            cloned_row = copy.deepcopy(row)
            cloned_row["row_key"] = new_row_key
            cloned_row["columns"] = merged_columns
            cloned_row["provenance"] = _merge_provenance(
                existing=cloned_row.get("provenance") or [],
                incoming=entry.get("sources") or [],
            )
            amendments = list(cloned_row.get("amendments") or [])
            if patch.patch_id not in amendments:
                amendments.append(patch.patch_id)
            cloned_row["amendments"] = amendments
            rows.append(cloned_row)
            entry["diagnostics"] = {
                **dict(entry.get("diagnostics") or {}),
                "supersede_mode": True,
                "historical_row_key": row_key,
                "new_row_key": new_row_key,
                "patch_effective_date": patch_effective_date,
            }
            entry["after_hash"] = self.hash_row(merged_columns)
            entry["applied"] = True
            applied_by_target[_table_row_target_key(table_id=table_id, row_key=new_row_key)] = {
                "patch_id": patch.patch_id,
                "op_id": str(entry.get("op_id") or ""),
            }
        else:
            row["columns"] = merged_columns
            if table_id == WAGE_TABLE_ID and str(merged_columns.get("effective_date") or "").strip() != row_key.rsplit("|", 1)[-1]:
                row["row_key"] = _wage_row_key(merged_columns)
            row["provenance"] = _merge_provenance(
                existing=row.get("provenance") or [],
                incoming=entry.get("sources") or [],
            )
            amendments = list(row.get("amendments") or [])
            if patch.patch_id not in amendments:
                amendments.append(patch.patch_id)
            row["amendments"] = amendments

            rows[row_idx] = row
            entry["after_hash"] = self.hash_row(merged_columns)
            entry["applied"] = True
            applied_by_target[_table_row_target_key(table_id=table_id, row_key=str(row.get("row_key") or row_key))] = {
                "patch_id": patch.patch_id,
                "op_id": str(entry.get("op_id") or ""),
            }

        table["rows"] = rows
        state["tables"][table_id] = table


def materialize_contract(
    contract_id: str,
    effective_version_id: str,
    patch_paths: list[Path],
    write_latest_pointer: bool = True,
) -> dict[str, Any]:
    base_paths = ensure_base_snapshot(contract_id)
    base_state = load_base_contract_state(contract_id=contract_id, base_paths=base_paths)
    with open(base_paths["meta"], "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    patches = load_patch_artifacts(patch_paths)
    patch_file_info = _build_patch_file_info(patch_paths)
    materializer = ContractMaterializer()
    effective_state, build_log = materializer.apply_patch_list(base_state, patches)

    # Normalize deterministic ordering before serialization.
    effective_state["effective_version_id"] = effective_version_id
    effective_state["schema_version"] = EFFECTIVE_CONTRACT_SCHEMA_VERSION
    effective_state["sections"] = sorted(
        effective_state.get("sections") or [],
        key=_section_sort_key,
    )
    table = (effective_state.get("tables") or {}).get(WAGE_TABLE_ID) or {}
    table["rows"] = sorted(
        table.get("rows") or [],
        key=lambda r: str(r.get("row_key") or ""),
    )
    effective_state.setdefault("tables", {})[WAGE_TABLE_ID] = table
    effective_state["source_documents"] = _normalize_source_documents(effective_state.get("source_documents") or {})

    wages_effective = _build_effective_wages_data(
        base_wages=effective_state.get("base_wages") or {},
        rows=(effective_state["tables"][WAGE_TABLE_ID].get("rows") or []),
        effective_version_id=effective_version_id,
        amendments_applied=list(effective_state.get("amendments_applied") or []),
    )
    chunks_effective = _build_effective_chunks(
        sections=effective_state.get("sections") or [],
        contract_id=contract_id,
        effective_version_id=effective_version_id,
        amendments_applied=list(effective_state.get("amendments_applied") or []),
    )
    manifest = _load_contract_manifest(contract_id)
    entitlements_effective = extract_entitlements(
        chunks=chunks_effective,
        contract_id=contract_id,
        manifest=manifest,
    )
    markdown_effective = _build_effective_markdown(
        sections=effective_state.get("sections") or [],
        contract_id=contract_id,
        effective_version_id=effective_version_id,
    )

    # Remove transient base payload from persisted effective contract.
    effective_contract_out = dict(effective_state)
    effective_contract_out.pop("base_wages", None)

    version_dir = DATA_DIR / "contracts" / contract_id / "effective" / effective_version_id
    index_dir = version_dir / "index_inputs"
    version_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    effective_contract_path = version_dir / "effective_contract.json"
    effective_markdown_path = version_dir / "effective_markdown.md"
    build_log_path = version_dir / "build_log.json"
    patch_chain_path = version_dir / "patch_chain.json"
    chunks_path = index_dir / f"contract_chunks_enriched_{contract_id}.json"
    wages_path = index_dir / f"wage_tables_{contract_id}.json"
    entitlements_path = index_dir / f"entitlement_tables_{contract_id}.json"

    chunks_bytes = _dump_json_bytes(chunks_effective)
    wages_bytes = _dump_json_bytes(wages_effective)
    entitlements_bytes = _dump_json_bytes(entitlements_effective)
    markdown_bytes = markdown_effective.encode("utf-8")
    contract_bytes_for_hash = _dump_json_bytes(effective_contract_out)
    effective_content_hash = _compute_effective_content_hash(
        contract_bytes=contract_bytes_for_hash,
        markdown_bytes=markdown_bytes,
        chunks_bytes=chunks_bytes,
        wages_bytes=wages_bytes,
        entitlements_bytes=entitlements_bytes,
    )
    effective_contract_out["effective_content_hash"] = effective_content_hash
    patch_chain_out = _build_patch_chain_manifest(
        contract_id=contract_id,
        effective_version_id=effective_version_id,
        effective_content_hash=effective_content_hash,
        base_meta=base_meta,
        patches=patches,
        patch_file_info=patch_file_info,
        applied_patch_ids=list(effective_state.get("amendments_applied") or []),
    )
    patch_chain_bytes = _dump_json_bytes(patch_chain_out)
    effective_contract_bytes = _dump_json_bytes(effective_contract_out)
    build_log_out = dict(build_log)
    build_log_out["effective_version_id"] = effective_version_id
    build_log_out["effective_content_hash"] = effective_content_hash
    build_log_out["amendments_applied"] = list(effective_state.get("amendments_applied") or [])
    build_log_out["artifact_hashes"] = {
        "effective_contract_sha256": _sha256_bytes(effective_contract_bytes),
        "effective_markdown_sha256": _sha256_bytes(markdown_bytes),
        "index_chunks_sha256": _sha256_bytes(chunks_bytes),
        "index_wages_sha256": _sha256_bytes(wages_bytes),
        "index_entitlements_sha256": _sha256_bytes(entitlements_bytes),
        "patch_chain_sha256": _sha256_bytes(patch_chain_bytes),
    }
    build_log_bytes = _dump_json_bytes(build_log_out)

    with open(effective_contract_path, "wb") as f:
        f.write(effective_contract_bytes)
    with open(effective_markdown_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown_effective)
    with open(build_log_path, "wb") as f:
        f.write(build_log_bytes)
    with open(patch_chain_path, "wb") as f:
        f.write(patch_chain_bytes)
    with open(chunks_path, "wb") as f:
        f.write(chunks_bytes)
    with open(wages_path, "wb") as f:
        f.write(wages_bytes)
    with open(entitlements_path, "wb") as f:
        f.write(entitlements_bytes)

    if write_latest_pointer:
        write_latest_effective_pointer(
            contract_id=contract_id,
            effective_version_id=effective_version_id,
            effective_content_hash=effective_content_hash,
        )

    return {
        "contract_id": contract_id,
        "effective_version_id": effective_version_id,
        "effective_content_hash": effective_content_hash,
        "effective_contract_path": str(effective_contract_path),
        "effective_markdown_path": str(effective_markdown_path),
        "build_log_path": str(build_log_path),
        "patch_chain_path": str(patch_chain_path),
        "index_chunks_path": str(chunks_path),
        "index_wages_path": str(wages_path),
        "index_entitlements_path": str(entitlements_path),
        "amendments_applied": list(effective_state.get("amendments_applied") or []),
        "artifact_hashes": build_log_out["artifact_hashes"],
    }


def ensure_base_snapshot(contract_id: str) -> dict[str, Path]:
    contract_root = DATA_DIR / "contracts" / contract_id
    base_dir = contract_root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    base_chunks = base_dir / "contract_chunks_enriched.json"
    base_wages = base_dir / "wage_tables.json"
    base_meta = base_dir / "base_metadata.json"

    source_chunks = _discover_contract_chunk_source(contract_id)
    if source_chunks is None:
        raise FileNotFoundError(f"Unable to discover source chunks for contract_id={contract_id}")
    source_wages = _discover_contract_wage_source(contract_id)
    if source_wages is None:
        raise FileNotFoundError(f"Unable to discover source wages for contract_id={contract_id}")

    _sync_base_snapshot_artifact(source_path=source_chunks, target_path=base_chunks)
    _sync_base_snapshot_artifact(source_path=source_wages, target_path=base_wages)

    meta = {
        "contract_id": contract_id,
        "base_version_id": "base_snapshot_v0_9_0",
        "base_chunks_sha256": _sha256_bytes(base_chunks.read_bytes()),
        "base_wages_sha256": _sha256_bytes(base_wages.read_bytes()),
    }
    with open(base_meta, "wb") as f:
        f.write(_dump_json_bytes(meta))

    return {
        "base_dir": base_dir,
        "chunks": base_chunks,
        "wages": base_wages,
        "meta": base_meta,
    }


def load_base_contract_state(contract_id: str, base_paths: dict[str, Path]) -> dict[str, Any]:
    with open(base_paths["chunks"], "r", encoding="utf-8") as f:
        base_chunks = json.load(f)
    with open(base_paths["wages"], "r", encoding="utf-8") as f:
        base_wages = json.load(f)
    with open(base_paths["meta"], "r", encoding="utf-8") as f:
        base_meta = json.load(f)

    pdf_nav = _load_runtime_pdf_nav(contract_id)
    table_nav = _load_runtime_table_nav(contract_id)
    base_pdf = _resolve_base_pdf_filename(contract_id)

    sections: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for chunk in base_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_contract_id = str(chunk.get("contract_id") or "")
        if chunk_contract_id and chunk_contract_id != contract_id:
            continue
        content_markdown = str(chunk.get("content_with_tables") or chunk.get("content") or "")
        if not content_markdown.strip():
            continue
        base_anchor = _base_anchor_id(chunk)
        grouped.setdefault(base_anchor, []).append(dict(chunk))

    for base_anchor, group in grouped.items():
        ordered = sorted(group, key=lambda c: str(c.get("chunk_id") or c.get("citation") or ""))
        for idx, chunk in enumerate(ordered, start=1):
            anchor_id = _anchor_id_with_part(base_anchor, chunk, idx if len(ordered) > 1 else None)
            article_num = _to_int(chunk.get("article_num"))
            section_num = _to_int(chunk.get("section_num"))
            citation = str(chunk.get("citation") or "").strip()
            content_markdown = str(chunk.get("content_with_tables") or chunk.get("content") or "")
            section_key = (
                f"{article_num}:{section_num}"
                if article_num is not None and section_num is not None
                else ""
            )
            page = None
            if section_key:
                page = (pdf_nav.get("section_pages") or {}).get(section_key)
            if page is None and article_num is not None:
                page = (pdf_nav.get("article_pages") or {}).get(article_num)
            section = {
                "anchor_id": anchor_id,
                "article_num": article_num,
                "section_num": section_num,
                "subsection": chunk.get("subsection"),
                "citation": citation,
                "article_title": chunk.get("article_title"),
                "chunk_id": chunk.get("chunk_id"),
                "parent_context": chunk.get("parent_context"),
                "doc_type": chunk.get("doc_type", "cba"),
                "content_markdown": content_markdown,
                "raw_chunk": dict(chunk),
                "provenance": [
                    {
                        "source_type": "base",
                        "pdf": base_pdf,
                        "pdf_page": page,
                    }
                ],
                "amendments": [],
            }
            section["span_id"] = ContractMaterializer.sha256_hex(
                "|".join(
                    [
                        ContractMaterializer.canonical_text(section["content_markdown"]),
                        str(section.get("article_title") or ""),
                        str(section.get("citation") or ""),
                    ]
                )
            )
            sections.append(section)

    rows = []
    for canonical in (base_wages.get("canonical_wage_rows") or []):
        if not isinstance(canonical, dict):
            continue
        columns = {
            "classification_key": canonical.get("classification_key"),
            "classification_name": canonical.get("classification_name"),
            "step_name": canonical.get("step_name"),
            "step_type": canonical.get("step_type"),
            "threshold_value": canonical.get("threshold_value"),
            "effective_date": canonical.get("effective_date"),
            "rate": canonical.get("rate"),
            "row_type": canonical.get("row_type"),
            "source_method": canonical.get("source_method"),
            "confidence": canonical.get("confidence"),
            "source_reference": canonical.get("source_reference") if isinstance(canonical.get("source_reference"), dict) else {},
        }
        if canonical.get("selected_schedule_label") is not None:
            columns["selected_schedule_label"] = canonical.get("selected_schedule_label")
        if isinstance(canonical.get("source_rate_schedule"), dict):
            columns["source_rate_schedule"] = copy.deepcopy(canonical.get("source_rate_schedule") or {})
        source_ref = columns.get("source_reference") or {}
        table_id = str(source_ref.get("table_id") or "").strip()
        page = (table_nav.get("table_pages") or {}).get(table_id) if table_id else None
        rows.append(
            {
                "row_key": _wage_row_key(columns),
                "columns": columns,
                "provenance": [
                    {
                        "source_type": "base",
                        "pdf": base_pdf,
                        "pdf_page": page,
                        "table_id": table_id or None,
                        "row_index": _to_int(source_ref.get("row_index")),
                    }
                ],
                "amendments": [],
            }
        )

    return {
        "contract_id": contract_id,
        "base_version_id": str(base_meta.get("base_version_id") or "base_snapshot_v0_9_0"),
        "sections": sorted(sections, key=_section_sort_key),
        "tables": {
            WAGE_TABLE_ID: {
                "table_id": WAGE_TABLE_ID,
                "columns": [
                    "classification_key",
                    "classification_name",
                    "step_name",
                    "step_type",
                    "threshold_value",
                    "effective_date",
                    "rate",
                    "row_type",
                    "source_method",
                "confidence",
                "source_reference",
                "selected_schedule_label",
                "source_rate_schedule",
                ],
                "rows": sorted(rows, key=lambda r: str(r.get("row_key") or "")),
            }
        },
        "base_wages": base_wages,
        "amendments_applied": [],
        "source_documents": {
            "base_pdf": base_pdf,
            "amendment_pdfs": [],
            "amendment_source_doc_ids": [],
        },
    }


def discover_patch_files(contract_id: str, patch_ids: Optional[list[str]] = None) -> list[Path]:
    amend_dir = DATA_DIR / "contracts" / contract_id / "amendments"
    if not amend_dir.exists():
        return []
    if patch_ids:
        out: list[Path] = []
        for patch_id in patch_ids:
            path = amend_dir / f"{patch_id}.json"
            if path.exists():
                out.append(path)
        return sorted(out)
    return sorted(amend_dir.glob("*.json"))


def rebase_patch_file(
    contract_id: str,
    patch_path: Path,
    prior_patch_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    base_paths = ensure_base_snapshot(contract_id)
    base_state = load_base_contract_state(contract_id=contract_id, base_paths=base_paths)
    materializer = ContractMaterializer()

    prior_artifacts = load_patch_artifacts(prior_patch_paths or [])
    rebased_state = base_state
    if prior_artifacts:
        rebased_state, _ = materializer.apply_patch_list(base_state, prior_artifacts)

    with open(patch_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise PatchRebaseFailure(
            report={
                "status": "failed",
                "code": "invalid_patch_payload",
                "message": f"Patch payload must be a JSON object: {patch_path}",
            }
        )

    rebased_payload = copy.deepcopy(payload)
    artifact = PatchArtifact.model_validate(payload)
    operations_payload = rebased_payload.get("operations")
    if not isinstance(operations_payload, list):
        operations_payload = []
        rebased_payload["operations"] = operations_payload

    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, operation in enumerate(artifact.operations, start=1):
        op_id = f"{artifact.patch_id}#{idx}"
        op_payload = operations_payload[idx - 1] if idx - 1 < len(operations_payload) else None
        if not isinstance(op_payload, dict):
            errors.append(
                {
                    "code": "operation_payload_missing",
                    "op_id": op_id,
                    "message": "Unable to map validated operation back to raw payload",
                }
            )
            continue

        if isinstance(operation, ReplaceSectionOperation):
            section, _ = _resolve_section_target(
                sections=rebased_state.get("sections") or [],
                target=operation.target.model_dump(),
            )
            if section is None:
                errors.append(
                    {
                        "code": "section_target_not_found",
                        "op_id": op_id,
                        "target": operation.target.model_dump(),
                    }
                )
                continue
            new_hash = materializer.hash_text(section.get("content_markdown", ""))
            last_touch = _resolve_last_touch(
                applied_by_target={},
                target_key=_section_target_key(section),
                existing_amendments=section.get("amendments") or [],
            )
            target_key = _section_target_key(section)
        elif isinstance(operation, ReplaceTableRowOperation):
            table_id = str(operation.target.table_id or "").strip()
            row_key = str(operation.target.row_key or "").strip()
            table = (rebased_state.get("tables") or {}).get(table_id)
            if not isinstance(table, dict):
                errors.append(
                    {
                        "code": "table_target_not_found",
                        "op_id": op_id,
                        "target": operation.target.model_dump(),
                    }
                )
                continue
            row = next(
                (r for r in (table.get("rows") or []) if str(r.get("row_key") or "") == row_key),
                None,
            )
            if not isinstance(row, dict):
                errors.append(
                    {
                        "code": "row_key_not_found",
                        "op_id": op_id,
                        "target": operation.target.model_dump(),
                    }
                )
                continue
            new_hash = materializer.hash_row(dict(row.get("columns") or {}))
            last_touch = _resolve_last_touch(
                applied_by_target={},
                target_key=_table_row_target_key(table_id=table_id, row_key=row_key),
                existing_amendments=row.get("amendments") or [],
            )
            target_key = _table_row_target_key(table_id=table_id, row_key=row_key)
        else:
            errors.append(
                {
                    "code": "unsupported_operation",
                    "op_id": op_id,
                    "target": operation.target.model_dump(),
                }
            )
            continue

        old_hash = str(op_payload.get("expected_prev_hash") or "").strip().lower()
        op_payload["expected_prev_hash"] = new_hash
        changes.append(
            {
                "op_id": op_id,
                "op": operation.op,
                "target": operation.target.model_dump(),
                "target_key": target_key,
                "old_expected_prev_hash": old_hash or None,
                "new_expected_prev_hash": new_hash,
                "changed": old_hash != new_hash,
                "last_touch": last_touch,
            }
        )

    if errors:
        raise PatchRebaseFailure(
            report={
                "status": "failed",
                "code": "patch_rebase_failed",
                "contract_id": contract_id,
                "patch_path": str(patch_path),
                "errors": errors,
                "changes": changes,
            }
        )

    prior_patch_ids = [p.patch_id for p in prior_artifacts]
    return {
        "status": "success",
        "contract_id": contract_id,
        "patch_id": str(artifact.patch_id),
        "patch_path": str(patch_path),
        "prior_patch_ids": prior_patch_ids,
        "changes": changes,
        "rebased_patch_payload": rebased_payload,
    }


def _build_patch_file_info(patch_paths: list[Path]) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for patch_path in sorted(patch_paths, key=lambda p: str(p)):
        artifact = load_patch_artifact(patch_path)
        patch_id = str(artifact.patch_id)
        if patch_id in info:
            raise ValueError(f"Duplicate patch_id '{patch_id}' in patch_paths")
        raw_bytes = patch_path.read_bytes()
        canonical_payload = artifact.model_dump(mode="json")
        canonical_bytes = _dump_json_bytes(canonical_payload)
        info[patch_id] = {
            "patch_file_name": patch_path.name,
            "patch_file_sha256": _sha256_bytes(raw_bytes),
            "patch_payload_sha256": _sha256_bytes(canonical_bytes),
        }
    return info


def _build_patch_chain_manifest(
    *,
    contract_id: str,
    effective_version_id: str,
    effective_content_hash: str,
    base_meta: dict[str, Any],
    patches: list[PatchArtifact],
    patch_file_info: dict[str, dict[str, Any]],
    applied_patch_ids: list[str],
) -> dict[str, Any]:
    applied_set = set(str(patch_id or "").strip() for patch_id in (applied_patch_ids or []))
    ordered_applied = [str(patch_id or "").strip() for patch_id in (applied_patch_ids or []) if str(patch_id or "").strip()]

    entries: list[dict[str, Any]] = []
    for patch in sorted(patches, key=lambda p: (str(p.effective_date), str(p.patch_id))):
        patch_id = str(patch.patch_id or "").strip()
        if not patch_id or patch_id not in applied_set:
            continue
        patch_info = patch_file_info.get(patch_id) or {}
        entries.append(
            {
                "patch_id": patch_id,
                "effective_date": str(patch.effective_date or ""),
                "ratified_date": str(patch.ratified_date or "") or None,
                "parent_effective_version_id": str(patch.parent_effective_version_id or "") or None,
                "source_pdf": _resolve_patch_source_pdf_name(patch),
                "source_doc_id": str(patch.source_doc_id or "") or None,
                "operation_count": len(patch.operations or []),
                "approved_operation_count": sum(
                    1 for op in (patch.operations or [])
                    if str(getattr(op, "review_status", "")).strip().lower() == APPROVED_REVIEW_STATUS
                ),
                "patch_file_name": patch_info.get("patch_file_name"),
                "patch_file_sha256": patch_info.get("patch_file_sha256"),
                "patch_payload_sha256": patch_info.get("patch_payload_sha256"),
            }
        )

    return {
        "schema_version": PATCH_CHAIN_SCHEMA_VERSION,
        "contract_id": contract_id,
        "effective_version_id": effective_version_id,
        "effective_content_hash": effective_content_hash,
        "base_snapshot": {
            "base_version_id": str(base_meta.get("base_version_id") or ""),
            "base_chunks_sha256": str(base_meta.get("base_chunks_sha256") or ""),
            "base_wages_sha256": str(base_meta.get("base_wages_sha256") or ""),
        },
        "applied_patch_ids": ordered_applied,
        "patches": entries,
        "patch_count": len(entries),
    }


def _discover_contract_chunk_source(contract_id: str) -> Optional[Path]:
    package_dir = DATA_DIR / "contracts" / contract_id / "chunks"
    candidates = [
        package_dir / f"contract_chunks_enriched_{contract_id}.json",
        package_dir / "contract_chunks_enriched.json",
        package_dir / f"contract_chunks_smart_{contract_id}.json",
        package_dir / "contract_chunks_smart.json",
        package_dir / f"contract_chunks_{contract_id}.json",
        package_dir / "contract_chunks.json",
        DATA_DIR / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _sync_base_snapshot_artifact(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        try:
            if _sha256_bytes(source_path.read_bytes()) == _sha256_bytes(target_path.read_bytes()):
                return
        except Exception:
            pass
    shutil.copyfile(source_path, target_path)


def _discover_contract_wage_source(contract_id: str) -> Optional[Path]:
    package_dir = DATA_DIR / "contracts" / contract_id / "wages"
    candidates = [
        package_dir / f"wage_tables_{contract_id}.json",
        package_dir / "wage_tables.json",
        DATA_DIR / "wages" / f"wage_tables_{contract_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_section_target(sections: list[dict], target: dict) -> tuple[Optional[dict], Optional[int]]:
    anchor_id = str(target.get("anchor_id") or "").strip()
    if anchor_id:
        for idx, section in enumerate(sections):
            if str(section.get("anchor_id") or "") == anchor_id:
                return section, idx

    article_num = _to_int(target.get("article_num"))
    section_num = _to_int(target.get("section_num"))
    subsection = str(target.get("subsection") or "").strip().lower()
    if article_num is None or section_num is None:
        return None, None

    matches: list[tuple[int, dict]] = []
    for idx, section in enumerate(sections):
        if _to_int(section.get("article_num")) != article_num:
            continue
        if _to_int(section.get("section_num")) != section_num:
            continue
        raw_sub = str(section.get("subsection") or "").strip().lower()
        if subsection and raw_sub != subsection:
            continue
        matches.append((idx, section))

    if len(matches) == 1:
        idx, section = matches[0]
        return section, idx
    return None, None


def _section_target_key(section: dict[str, Any]) -> str:
    anchor = str(section.get("anchor_id") or "").strip()
    if anchor:
        return f"section:{anchor}"
    article_num = _to_int(section.get("article_num"))
    section_num = _to_int(section.get("section_num"))
    subsection = str(section.get("subsection") or "").strip().lower()
    return f"section:{article_num}:{section_num}:{subsection}"


def _table_row_target_key(table_id: str, row_key: str) -> str:
    return f"table_row:{str(table_id or '').strip()}:{str(row_key or '').strip()}"


def _resolve_last_touch(
    applied_by_target: dict[str, dict[str, Optional[str]]],
    target_key: str,
    existing_amendments: list[Any],
) -> Optional[dict]:
    touched = applied_by_target.get(target_key)
    if touched:
        return {
            "patch_id": str(touched.get("patch_id") or "").strip() or None,
            "op_id": str(touched.get("op_id") or "").strip() or None,
        }
    if existing_amendments:
        last_patch = str(existing_amendments[-1] or "").strip()
        if last_patch:
            return {"patch_id": last_patch, "op_id": None}
    return None


def _resolve_patch_source_pdf_name(patch: PatchArtifact) -> Optional[str]:
    source_pdf = str(patch.source_pdf or "").strip()
    if source_pdf:
        return source_pdf
    source_doc_id = str(patch.source_doc_id or "").strip()
    if not source_doc_id:
        return None
    return resolve_source_doc_pdf_name(source_doc_id)


def _normalized_source_refs(
    source_refs: list,
    default_pdf: Optional[str],
    default_source_doc_id: Optional[str] = None,
) -> list[dict]:
    out: list[dict] = []
    for ref in source_refs or []:
        if hasattr(ref, "model_dump"):
            row = ref.model_dump()
        elif isinstance(ref, dict):
            row = dict(ref)
        else:
            continue
        source_doc_id = str(row.get("source_doc_id") or default_source_doc_id or "").strip() or None
        pdf_name = str(row.get("pdf") or "").strip()
        if not pdf_name and source_doc_id:
            pdf_name = str(resolve_source_doc_pdf_name(source_doc_id) or "").strip()
        if not pdf_name and default_pdf:
            pdf_name = str(default_pdf).strip()
        if not pdf_name:
            continue
        source_type = str(row.get("source_type") or "").strip().lower()
        if not source_type:
            source_type = "moa" if "moa" in pdf_name.lower() else "base"
        out.append(
            {
                "source_type": source_type,
                "pdf": pdf_name,
                "pdf_page": _to_int(row.get("pdf_page")),
                "source_doc_id": source_doc_id,
            }
        )
    if not out and default_pdf:
        source_doc_id = str(default_source_doc_id or "").strip() or None
        out.append(
            {
                "source_type": "moa" if "moa" in str(default_pdf).lower() else "base",
                "pdf": str(default_pdf),
                "pdf_page": None,
                "source_doc_id": source_doc_id,
            }
        )
    return out


def _merge_provenance(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen = set()
    for collection in (existing or [], incoming or []):
        for item in collection:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("source_type") or ""),
                str(item.get("pdf") or ""),
                _to_int(item.get("pdf_page")),
                str(item.get("source_doc_id") or ""),
                str(item.get("table_id") or ""),
                _to_int(item.get("row_index")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "source_type": key[0],
                    "pdf": key[1],
                    "pdf_page": key[2],
                    "source_doc_id": key[3] or None,
                    "table_id": key[4] or None,
                    "row_index": key[5],
                }
            )
    return merged


def _merge_source_documents_with_patches(source_documents: dict, patches: list[PatchArtifact], applied_patch_ids: list[str]) -> dict:
    base_pdf = str(source_documents.get("base_pdf") or "base_contract.pdf")
    applied = set(applied_patch_ids)
    amendment_pdfs = set(source_documents.get("amendment_pdfs") or [])
    amendment_source_doc_ids = set(source_documents.get("amendment_source_doc_ids") or [])
    for patch in patches:
        if patch.patch_id not in applied:
            continue
        source_doc_id = str(patch.source_doc_id or "").strip()
        source_pdf = str(_resolve_patch_source_pdf_name(patch) or "").strip()
        if source_pdf:
            amendment_pdfs.add(source_pdf)
        if source_doc_id:
            amendment_source_doc_ids.add(source_doc_id)
    return {
        "base_pdf": base_pdf,
        "amendment_pdfs": sorted(amendment_pdfs),
        "amendment_source_doc_ids": sorted(amendment_source_doc_ids),
    }


def _base_anchor_id(chunk: dict) -> str:
    article_num = _to_int(chunk.get("article_num"))
    section_num = _to_int(chunk.get("section_num"))
    if article_num is not None and section_num is not None:
        subsection = str(chunk.get("subsection") or "").strip().lower()
        base = f"a{article_num}_s{section_num}"
        if subsection:
            subsection = re.sub(r"[^a-z0-9]+", "_", subsection).strip("_")
            if subsection:
                base = f"{base}_sub_{subsection}"
        return base

    doc_type = _slug_token(chunk.get("doc_type"), default="section")
    chunk_token = _slug_token(chunk.get("chunk_id"))
    if chunk_token:
        # Group segmented chunks by logical parent (e.g., lou_1_part1, lou_1_part2 -> lou_1).
        base_chunk = re.sub(r"(?:_part|part|_p)\d+$", "", chunk_token).strip("_")
        if base_chunk:
            if base_chunk.startswith(f"{doc_type}_"):
                return base_chunk
            return f"{doc_type}_{base_chunk}"

    citation_raw = str(chunk.get("citation") or "").strip().lower()
    if citation_raw:
        citation_raw = re.sub(r",?\s*part\s+\d+\b", "", citation_raw, flags=re.IGNORECASE)
    citation_token = _slug_token(citation_raw)
    if citation_token:
        return f"{doc_type}_{citation_token[:72]}"

    parent_token = _slug_token(chunk.get("parent_context"))
    if parent_token:
        return f"{doc_type}_{parent_token[:72]}"

    digest_seed = "|".join(
        [
            str(chunk.get("chunk_id") or ""),
            str(chunk.get("citation") or ""),
            str(chunk.get("parent_context") or ""),
            ContractMaterializer.canonical_text(chunk.get("content_with_tables") or chunk.get("content") or "")[:240],
        ]
    )
    digest = ContractMaterializer.sha256_hex(digest_seed)[:12]
    return f"{doc_type}_{digest}"


def _anchor_id_with_part(base_anchor: str, chunk: dict, part_index: Optional[int]) -> str:
    if part_index is None:
        return base_anchor
    citation = str(chunk.get("citation") or "")
    m = re.search(r"Part\s+(\d+)", citation, re.IGNORECASE)
    if m:
        return f"{base_anchor}_p{int(m.group(1))}"
    chunk_id = str(chunk.get("chunk_id") or "")
    m = re.search(r"seg[_-]?0*([0-9]+)", chunk_id, re.IGNORECASE)
    if m:
        return f"{base_anchor}_p{int(m.group(1))}"
    return f"{base_anchor}_p{part_index}"


def _wage_row_key(columns: dict[str, Any]) -> str:
    class_key = str(columns.get("classification_key") or "").strip().lower()
    step_type = str(columns.get("step_type") or "").strip().lower()
    threshold = columns.get("threshold_value")
    threshold_token = "na" if threshold in (None, "") else str(int(threshold))
    effective_date = str(columns.get("effective_date") or "").strip()
    return f"{class_key}|{step_type}:{threshold_token}|{effective_date}"


def _build_effective_wages_data(
    base_wages: dict[str, Any],
    rows: list[dict[str, Any]],
    effective_version_id: str,
    amendments_applied: list[str],
) -> dict[str, Any]:
    wages = copy.deepcopy(base_wages)
    row_map: dict[tuple[str, str, str, Optional[int], str], dict] = {}
    for row in rows:
        columns = dict(row.get("columns") or {})
        class_key = str(columns.get("classification_key") or "").strip()
        step_name = str(columns.get("step_name") or "").strip()
        step_type = str(columns.get("step_type") or "").strip()
        threshold = _to_int(columns.get("threshold_value"))
        effective_date = str(columns.get("effective_date") or "").strip()
        if not class_key or not effective_date:
            continue
        canonical = {
            "schema_version": "wage_canonical_rows_v1",
            "contract_id": wages.get("contract_id"),
            "classification_key": class_key,
            "classification_name": columns.get("classification_name"),
            "step_name": step_name,
            "step_type": step_type,
            "threshold_value": threshold,
            "effective_date": effective_date,
            "rate": float(columns.get("rate")),
            "source_method": columns.get("source_method"),
            "row_type": columns.get("row_type"),
            "confidence": columns.get("confidence"),
            "source_reference": columns.get("source_reference") if isinstance(columns.get("source_reference"), dict) else {},
            "row_key": str(row.get("row_key") or _wage_row_key(columns)),
            "provenance": row.get("provenance") or [],
            "effective_version_id": effective_version_id,
            "amendments_applied": list(row.get("amendments") or []),
        }
        if columns.get("selected_schedule_label") is not None:
            canonical["selected_schedule_label"] = columns.get("selected_schedule_label")
        if isinstance(columns.get("source_rate_schedule"), dict):
            canonical["source_rate_schedule"] = copy.deepcopy(columns.get("source_rate_schedule") or {})
        key = (class_key, step_name, step_type, threshold, effective_date)
        row_map[key] = canonical

    canonical_rows = [
        row_map[key]
        for key in sorted(
            row_map,
            key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else int(k[3]), k[4]),
        )
    ]
    wages["canonical_wage_rows"] = canonical_rows

    date_set = {str(r.get("effective_date") or "") for r in canonical_rows if str(r.get("effective_date") or "")}
    if date_set:
        wages["effective_dates"] = sorted(date_set)

    classes = wages.get("classifications") or {}
    for class_key, class_data in classes.items():
        for step in class_data.get("steps") or []:
            step_name = str(step.get("step_name") or "").strip()
            if step.get("hours_required") is not None:
                step_type = "hours"
                threshold = _to_int(step.get("hours_required"))
            elif step.get("months_required") is not None:
                step_type = "months"
                threshold = _to_int(step.get("months_required"))
            else:
                step_type = "fixed"
                threshold = None

            rates = dict(step.get("rates") or {})
            for key, canonical in row_map.items():
                k_class, k_step, k_type, k_threshold, k_date = key
                if k_class != class_key:
                    continue
                if k_step != step_name:
                    continue
                if k_type != step_type:
                    continue
                if k_threshold != threshold:
                    continue
                rates[k_date] = canonical.get("rate")
            step["rates"] = {k: rates[k] for k in sorted(rates)}

    wages["effective_version_id"] = effective_version_id
    wages["amendments_applied"] = list(amendments_applied)
    return wages


def _build_effective_chunks(
    sections: list[dict[str, Any]],
    contract_id: str,
    effective_version_id: str,
    amendments_applied: list[str],
) -> list[dict]:
    out: list[dict] = []
    for section in sorted(sections, key=_section_sort_key):
        raw = dict(section.get("raw_chunk") or {})
        raw["contract_id"] = contract_id
        raw["content"] = str(section.get("content_markdown") or "")
        raw["content_with_tables"] = str(section.get("content_markdown") or "")
        raw["anchor_id"] = str(section.get("anchor_id") or "")
        raw["span_id"] = str(section.get("span_id") or "")
        raw["provenance"] = section.get("provenance") or []
        raw["effective_version_id"] = effective_version_id
        raw["amendments_applied"] = list(amendments_applied)
        source_types = {str(r.get("source_type") or "").lower() for r in (raw.get("provenance") or [])}
        raw["source_type"] = "moa" if "moa" in source_types else "base"
        out.append(raw)
    return out


def _build_effective_markdown(sections: list[dict], contract_id: str, effective_version_id: str) -> str:
    lines: list[str] = []
    for section in sorted(sections, key=_section_sort_key):
        sources = []
        for ref in section.get("provenance") or []:
            if not isinstance(ref, dict):
                continue
            source_type = str(ref.get("source_type") or "base")
            pdf_name = str(ref.get("pdf") or "")
            page = _to_int(ref.get("pdf_page"))
            source_doc_id = str(ref.get("source_doc_id") or "").strip()
            token = f"{source_type}:{pdf_name}"
            if page is not None:
                token = f"{token}#p{page}"
            if source_doc_id:
                token = f"{token}@{source_doc_id}"
            sources.append(token)
        source_text = ";".join(sources)
        lines.append(
            "PROV("
            f"contract_id={contract_id}, "
            f"anchor_id={section.get('anchor_id')}, "
            f"effective_version_id={effective_version_id}, "
            f"sources=[{source_text}]"
            ")"
        )
        lines.append("")
        lines.append(str(section.get("content_markdown") or ""))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _load_runtime_pdf_nav(contract_id: str) -> dict:
    nav_path = resolve_pdf_nav_index_file(contract_id=contract_id, allow_shared_fallback=True)
    if nav_path and nav_path.exists():
        loaded = load_pdf_nav_index(nav_path)
        return to_runtime_navigation_maps(loaded)
    generated = build_pdf_nav_index(contract_id=contract_id)
    return to_runtime_navigation_maps(generated)


def _load_runtime_table_nav(contract_id: str) -> dict:
    nav_path = resolve_table_nav_index_file(contract_id=contract_id, allow_shared_fallback=True)
    if nav_path and nav_path.exists():
        loaded = load_table_nav_index(nav_path)
        return to_runtime_table_navigation_maps(loaded)
    generated = build_table_nav_index(contract_id=contract_id)
    return to_runtime_table_navigation_maps(generated)


def _resolve_base_pdf_filename(contract_id: str) -> str:
    source_dir = DATA_DIR / "contracts" / contract_id / "source"
    if not source_dir.exists():
        return "base_contract.pdf"
    candidates = sorted(source_dir.glob("*.pdf"))
    if not candidates:
        return "base_contract.pdf"
    non_moa = [p for p in candidates if "moa" not in p.name.lower()]
    if non_moa:
        return non_moa[0].name
    return candidates[0].name


def _normalize_source_documents(documents: dict) -> dict:
    base_pdf = str(documents.get("base_pdf") or "").strip() or "base_contract.pdf"
    amendment_pdfs = []
    for row in documents.get("amendment_pdfs") or []:
        value = str(row or "").strip()
        if value and value not in amendment_pdfs:
            amendment_pdfs.append(value)
    amendment_source_doc_ids = []
    for row in documents.get("amendment_source_doc_ids") or []:
        value = str(row or "").strip()
        if value and value not in amendment_source_doc_ids:
            amendment_source_doc_ids.append(value)
    return {
        "base_pdf": base_pdf,
        "amendment_pdfs": sorted(amendment_pdfs),
        "amendment_source_doc_ids": sorted(amendment_source_doc_ids),
    }


def _section_sort_key(section: dict) -> tuple:
    doc_type = str(section.get("doc_type") or "").strip().lower()
    doc_rank = {
        "cba": 0,
        "appendix": 1,
        "lou": 2,
    }.get(doc_type, 3)
    article_num = _to_int(section.get("article_num"))
    section_num = _to_int(section.get("section_num"))
    return (
        doc_rank,
        article_num if article_num is not None else 10_000_000,
        section_num if section_num is not None else 10_000_000,
        str(section.get("subsection") or ""),
        str(section.get("anchor_id") or ""),
        str(section.get("citation") or ""),
        str(section.get("chunk_id") or ""),
    )


def _slug_token(value: Any, default: str = "") -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if token:
        return token
    return str(default or "").strip().lower()


def _to_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed


def _is_iso_date(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))


def _dump_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _excerpt(value: str, max_chars: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _short_unified_diff(old_text: str, new_text: str, max_lines: int = 16) -> list[str]:
    old_lines = str(old_text or "").splitlines()
    new_lines = str(new_text or "").splitlines()
    lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="current",
            tofile="incoming",
            lineterm="",
            n=1,
        )
    )
    return lines[:max_lines]


def _compute_effective_content_hash(
    *,
    contract_bytes: bytes,
    markdown_bytes: bytes,
    chunks_bytes: bytes,
    wages_bytes: bytes,
    entitlements_bytes: bytes,
) -> str:
    digest = hashlib.sha256()
    parts = [
        ("effective_contract", contract_bytes),
        ("effective_markdown", markdown_bytes),
        ("index_chunks", chunks_bytes),
        ("index_wages", wages_bytes),
        ("index_entitlements", entitlements_bytes),
    ]
    for name, payload in parts:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_contract_manifest(contract_id: str) -> Optional[dict[str, Any]]:
    manifest_path = DATA_DIR / "manifests" / f"{contract_id}.json"
    if not manifest_path.exists():
        manifest_path = DATA_DIR / "contracts" / contract_id / "manifests" / f"{contract_id}.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
