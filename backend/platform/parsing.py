"""Document parser abstractions for tenant-scoped ingestion."""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ParsedBlock:
    text: str
    kind: str = "text"
    bbox: BoundingBox | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    screenshot_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    parser_name: str
    content_type: str
    text: str
    pages: list[ParsedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def to_dict(self) -> dict:
        return {
            "parser_name": self.parser_name,
            "content_type": self.content_type,
            "text": self.text,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text,
                    "screenshot_path": page.screenshot_path,
                    "metadata": dict(page.metadata),
                    "blocks": [
                        {
                            "text": block.text,
                            "kind": block.kind,
                            "bbox": None
                            if block.bbox is None
                            else {
                                "x1": block.bbox.x1,
                                "y1": block.bbox.y1,
                                "x2": block.bbox.x2,
                                "y2": block.bbox.y2,
                            },
                            "metadata": dict(block.metadata),
                        }
                        for block in page.blocks
                    ],
                }
                for page in self.pages
            ],
        }


class ParserUnavailableError(RuntimeError):
    pass


class DocumentParser(Protocol):
    name: str

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        ...

    def parse_bytes(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        ...

    def parse_file(
        self,
        file_path: Path,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        ...


class PlainTextDocumentParser:
    name = "plain_text"

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        normalized_type = str(content_type or "").strip().lower()
        normalized_name = str(filename or "").strip().lower()
        return normalized_type.startswith("text/") or normalized_name.endswith((".txt", ".md", ".markdown"))

    def parse_bytes(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        text = self._decode(payload)
        return ParsedDocument(
            parser_name=self.name,
            content_type=str(content_type or "text/plain"),
            text=text,
            pages=[
                ParsedPage(
                    page_number=1,
                    text=text,
                    blocks=[ParsedBlock(text=text, kind="text")],
                )
            ],
            metadata={"filename": filename or None},
        )

    def parse_file(
        self,
        file_path: Path,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        return self.parse_bytes(
            file_path.read_bytes(),
            content_type=content_type,
            filename=filename or file_path.name,
        )

    @staticmethod
    def _decode(payload: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="ignore")


class ContractPackDocumentParser:
    """Parses a materialized effective-contract JSON export.

    Registered ahead of the plain-text parser so a contract pack keeps its
    article/section hierarchy instead of being flattened into prose. The
    document text is the concatenated section bodies, so retrieval, safety
    scanning and quality checks all behave exactly as they do for markdown.
    """

    name = "contract_pack"

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        normalized_type = str(content_type or "").strip().lower()
        normalized_name = str(filename or "").strip().lower()
        return normalized_type in {"application/json", "text/json"} or normalized_name.endswith(".json")

    def parse_bytes(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        import json

        from backend.platform.document_structure import _looks_like_contract_pack

        raw = payload.decode("utf-8", errors="ignore")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParserUnavailableError(f"Contract pack JSON could not be parsed: {exc}") from exc

        if not _looks_like_contract_pack(document):
            raise ParserUnavailableError(
                "JSON upload is not a contract pack export (expected contract_id and sections[])."
            )

        blocks: list[ParsedBlock] = []
        parts: list[str] = []
        for section in document.get("sections") or []:
            if not isinstance(section, dict):
                continue
            body = str(section.get("content_markdown") or section.get("raw_chunk") or "").strip()
            if not body:
                continue
            parts.append(body)
            blocks.append(ParsedBlock(text=body, kind="text"))

        text = "\n\n".join(parts)
        return ParsedDocument(
            parser_name=self.name,
            content_type="application/json",
            text=text,
            pages=[ParsedPage(page_number=1, text=text, blocks=blocks)],
            metadata={
                "filename": filename or None,
                "contract_pack": document,
                "contract_id": str(document.get("contract_id") or "").strip() or None,
                "effective_version_id": str(document.get("effective_version_id") or "").strip() or None,
            },
        )


    def parse_file(
        self,
        file_path: Path,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        return self.parse_bytes(
            file_path.read_bytes(),
            content_type=content_type,
            filename=filename,
        )


class LiteParseDocumentParser:
    """CLI-backed adapter boundary for LiteParse."""

    name = "liteparse"

    def __init__(self, executable: str | None, *, ocr_enabled: bool = False):
        self.executable = str(executable or "").strip()
        self.ocr_enabled = bool(ocr_enabled)

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        if not self.executable:
            return False
        normalized_name = str(filename or "").strip().lower()
        normalized_type = str(content_type or "").strip().lower()
        supported_exts = (".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        return normalized_type in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "image/png",
            "image/jpeg",
            "image/tiff",
        } or normalized_name.endswith(supported_exts)

    def parse_bytes(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        suffix = Path(filename or "document").suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        try:
            return self.parse_file(tmp_path, content_type=content_type, filename=filename or tmp_path.name)
        finally:
            tmp_path.unlink(missing_ok=True)

    def parse_file(
        self,
        file_path: Path,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        if not self.executable:
            raise ParserUnavailableError("LiteParse executable is not configured.")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            output_path = Path(out.name)
        try:
            result = subprocess.run(
                [
                    *shlex.split(self.executable),
                    "parse",
                    str(file_path),
                    "--format",
                    "json",
                    "-o",
                    str(output_path),
                    *(["--no-ocr"] if not self.ocr_enabled else []),
                    "--quiet",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or result.stdout.strip() or "unknown parser failure"
                raise ParserUnavailableError(f"LiteParse failed: {stderr}")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive until real install
                raise ParserUnavailableError(f"LiteParse output could not be decoded: {exc}") from exc
            return self._normalize_payload(payload, content_type=content_type, filename=filename)
        finally:
            output_path.unlink(missing_ok=True)

    def _normalize_payload(
        self,
        payload: dict,
        *,
        content_type: str | None,
        filename: str | None,
    ) -> ParsedDocument:
        pages_payload = payload.get("pages") or []
        pages: list[ParsedPage] = []
        full_text_parts: list[str] = []
        for idx, page in enumerate(pages_payload, start=1):
            page_text = str(page.get("text") or "")
            if page_text:
                full_text_parts.append(page_text)
            blocks: list[ParsedBlock] = []
            for item in page.get("textItems") or []:
                bbox = None
                if all(key in item for key in ("x", "y", "width", "height")):
                    x = float(item["x"])
                    y = float(item["y"])
                    width = float(item["width"])
                    height = float(item["height"])
                    bbox = BoundingBox(
                        x1=x,
                        y1=y,
                        x2=x + width,
                        y2=y + height,
                    )
                blocks.append(
                    ParsedBlock(
                        text=str(item.get("text") or ""),
                        kind="text_item",
                        bbox=bbox,
                        metadata={
                            "font_name": item.get("fontName"),
                            "font_size": item.get("fontSize"),
                        },
                    )
                )
            for bbox_data in page.get("boundingBoxes") or []:
                if not all(key in bbox_data for key in ("x1", "y1", "x2", "y2")):
                    continue
                blocks.append(
                    ParsedBlock(
                        text="",
                        kind="bounding_box",
                        bbox=BoundingBox(
                            x1=float(bbox_data["x1"]),
                            y1=float(bbox_data["y1"]),
                            x2=float(bbox_data["x2"]),
                            y2=float(bbox_data["y2"]),
                        ),
                    )
                )
            pages.append(
                ParsedPage(
                    page_number=int(page.get("page") or page.get("pageNum") or idx),
                    text=page_text,
                    blocks=[block for block in blocks if block.text],
                    screenshot_path=page.get("screenshot_path"),
                    metadata={
                        "width": page.get("width"),
                        "height": page.get("height"),
                        "bounding_box_count": len(page.get("boundingBoxes") or []),
                    },
                )
            )
        text = "\n\n".join(full_text_parts)
        return ParsedDocument(
            parser_name=self.name,
            content_type=str(content_type or payload.get("content_type") or ""),
            text=text,
            pages=pages,
            warnings=[str(item) for item in payload.get("warnings") or []],
            metadata={
                "filename": filename or None,
                "raw_keys": sorted(payload.keys()),
                "page_count": len(pages_payload),
            },
        )


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]):
        self.parsers = parsers

    def resolve(self, *, content_type: str | None, filename: str | None) -> DocumentParser | None:
        for parser in self.parsers:
            if parser.can_parse(content_type=content_type, filename=filename):
                return parser
        return None
