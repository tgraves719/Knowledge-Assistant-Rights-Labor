"""MOA patch schema v0.9.0."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


PATCH_SCHEMA_VERSION = "moa_patch_v0_9_0"
APPROVED_REVIEW_STATUS = "approved"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceRef(BaseModel):
    pdf: Optional[str] = None
    pdf_page: Optional[int] = None
    source_type: Optional[str] = None
    source_doc_id: Optional[str] = None

    @model_validator(mode="after")
    def _normalize(self):
        self.pdf = str(self.pdf or "").strip() or None
        self.source_type = str(self.source_type or "").strip().lower() or None
        self.source_doc_id = str(self.source_doc_id or "").strip() or None
        return self


class SectionTarget(BaseModel):
    anchor_id: str
    article_num: Optional[int] = None
    section_num: Optional[int] = None
    subsection: Optional[str] = None


class TableRowTarget(BaseModel):
    table_id: str
    row_key: str


class ReplaceSectionOperation(BaseModel):
    op: Literal["replace_section"]
    target: SectionTarget
    expected_prev_hash: str
    new_text_markdown: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: Optional[float] = None
    review_status: str = "pending"

    @field_validator("expected_prev_hash")
    @classmethod
    def _validate_expected_hash(cls, value: str) -> str:
        v = str(value or "").strip().lower()
        if not _SHA256_RE.fullmatch(v):
            raise ValueError("expected_prev_hash must be a lowercase sha256 hex string")
        return v

    @field_validator("review_status")
    @classmethod
    def _normalize_review_status(cls, value: str) -> str:
        return str(value or "").strip().lower() or "pending"


class ReplaceTableRowOperation(BaseModel):
    op: Literal["replace_table_row"]
    target: TableRowTarget
    expected_prev_hash: str
    new_row: dict[str, Any]
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: Optional[float] = None
    review_status: str = "pending"

    @field_validator("expected_prev_hash")
    @classmethod
    def _validate_expected_hash(cls, value: str) -> str:
        v = str(value or "").strip().lower()
        if not _SHA256_RE.fullmatch(v):
            raise ValueError("expected_prev_hash must be a lowercase sha256 hex string")
        return v

    @field_validator("review_status")
    @classmethod
    def _normalize_review_status(cls, value: str) -> str:
        return str(value or "").strip().lower() or "pending"


PatchOperation = Annotated[
    Union[ReplaceSectionOperation, ReplaceTableRowOperation],
    Field(discriminator="op"),
]


class PatchArtifact(BaseModel):
    patch_id: str
    contract_id: str
    source_pdf: Optional[str] = None
    source_doc_id: Optional[str] = None
    effective_date: str
    ratified_date: Optional[str] = None
    parent_effective_version_id: Optional[str] = None
    schema_version: str = PATCH_SCHEMA_VERSION
    operations: list[PatchOperation] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        v = str(value or "").strip() or PATCH_SCHEMA_VERSION
        if v != PATCH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported patch schema_version '{v}'")
        return v

    @model_validator(mode="after")
    def _validate_source_identity(self):
        source_pdf = str(self.source_pdf or "").strip()
        source_doc_id = str(self.source_doc_id or "").strip()
        if not source_pdf and not source_doc_id:
            raise ValueError("Patch artifact requires at least one of source_pdf or source_doc_id")
        self.source_pdf = source_pdf or None
        self.source_doc_id = source_doc_id or None
        return self


def load_patch_artifact(path: Path) -> PatchArtifact:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return PatchArtifact.model_validate(payload)


def load_patch_artifacts(paths: list[Path]) -> list[PatchArtifact]:
    artifacts: list[PatchArtifact] = []
    for path in paths:
        artifacts.append(load_patch_artifact(path))
    artifacts.sort(key=lambda p: (str(p.effective_date), str(p.patch_id)))
    return artifacts


def validate_patch_file(path: Path) -> tuple[bool, Optional[str]]:
    try:
        load_patch_artifact(path)
    except (ValidationError, json.JSONDecodeError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, None
