"""Tenant-scoped document ingestion and retrieval helpers."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.platform.embeddings import DeterministicTextEmbedder, TextEmbedder
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus
from backend.platform.text_normalization import (
    extract_provenance,
    provenance_metadata,
    summarize_section_label,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str | None
    chunk_index: int
    content: str
    similarity: float
    metadata: dict


class TenantRetrievalService:
    def __init__(
        self,
        *,
        chunk_size: int = 800,
        overlap: int = 120,
        # Must match the pgvector column width in models.ChunkEmbedding.
        embedding_dimensions: int = 768,
        embedder: TextEmbedder | None = None,
    ):
        self.chunk_size = max(200, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size // 2))
        self.embedder = embedder or DeterministicTextEmbedder(dimensions=embedding_dimensions)

    def can_inline_ingest(self, *, content_type: str | None, filename: str | None) -> bool:
        normalized_type = str(content_type or "").strip().lower()
        normalized_name = str(filename or "").strip().lower()
        return normalized_type.startswith("text/") or normalized_name.endswith((".txt", ".md", ".markdown"))

    def decode_text(self, payload: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="ignore")

    def split_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\r\n?", "\n", str(text or "")).strip()
        if not normalized:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
                continue
            candidate = f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            chunks.append(current)
            if self.overlap > 0 and len(current) > self.overlap:
                current = f"{current[-self.overlap:]}\n\n{paragraph}"
            else:
                current = paragraph

        if current:
            chunks.append(current)

        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
                continue
            start = 0
            step = self.chunk_size - self.overlap if self.overlap < self.chunk_size else self.chunk_size
            while start < len(chunk):
                piece = chunk[start : start + self.chunk_size].strip()
                if piece:
                    final_chunks.append(piece)
                start += max(1, step)
        return final_chunks

    def _chunk_page(self, page_text: str) -> list[str]:
        return self.split_text(page_text)

    # Words that appear in nearly every chunk of a contract. Left in the term
    # list they add a flat lexical bonus to everything, which drowns the
    # embedding signal (whose useful spread between right and wrong passages
    # is only ~0.05-0.15).
    _STOPWORDS = frozenset(
        """the and for are but not with that this from have has had was were
        will shall may can could would should when what where which while whom
        whose been being than then them they their there here also any all
        each other into onto upon about after before during under over between
        per you your our its his her him she who how why does did doing get
        got""".split()
    )

    def _search_terms(self, query: str) -> list[str]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
            if len(token) >= 3 and token not in self._STOPWORDS
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered[:8]

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        tokens = [token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) >= 3]
        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    @staticmethod
    def _char_trigrams(value: str) -> set[str]:
        normalized = f"  {str(value or '').strip().lower()}  "
        if len(normalized) < 3:
            return {normalized} if normalized.strip() else set()
        return {normalized[index : index + 3] for index in range(len(normalized) - 2)}

    @classmethod
    def _fuzzy_token_similarity(cls, term: str, candidate: str) -> float:
        left = str(term or "").strip().lower()
        right = str(candidate or "").strip().lower()
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            overlap_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if overlap_ratio >= 0.7:
                return 0.95 * overlap_ratio
        ratio = difflib.SequenceMatcher(None, left, right).ratio()
        trigram_left = cls._char_trigrams(left)
        trigram_right = cls._char_trigrams(right)
        trigram_union = trigram_left | trigram_right
        trigram_score = (len(trigram_left & trigram_right) / len(trigram_union)) if trigram_union else 0.0
        return max(ratio, trigram_score)

    @classmethod
    def _fuzzy_lexical_bonus(cls, search_terms: list[str], lowered_text: str, metadata: dict) -> float:
        if not search_terms:
            return 0.0
        candidate_tokens = cls._lexical_tokens(lowered_text)
        bonus = 0.0
        for term in search_terms:
            if term in lowered_text:
                bonus += 0.2
                continue
            best_score = max((cls._fuzzy_token_similarity(term, token) for token in candidate_tokens), default=0.0)
            if best_score >= 0.84:
                bonus += 0.16 * best_score
        if metadata.get("summary"):
            summary_text = str(metadata.get("summary") or "").lower()
            for term in search_terms:
                if term in summary_text:
                    bonus += 0.1
                    continue
                best_score = max((cls._fuzzy_token_similarity(term, token) for token in cls._lexical_tokens(summary_text)), default=0.0)
                if best_score >= 0.84:
                    bonus += 0.08 * best_score
        return bonus

    @staticmethod
    def _phrase_alias_bonus(query_lower: str, metadata: dict) -> float:
        """Exact multi-word alias hit in the query, scored outside the cap.

        Sibling sections can be near-identical in embedding space — every
        Appendix A wage ladder reads "X hourly wage rates effective ..." — so
        the capped bag-of-words tiebreaker cannot lift "HEAD CLERK" over
        "HEAD BAKER" for "how much does a head clerk make". A verbatim
        multi-word phrase ("head clerk", "section 121") only ever matches the
        chunks that carry that exact alias, so unlike the per-term bonuses it
        cannot reorder unrelated content; it stays small and capped anyway.
        """
        bonus = 0.0
        seen: set[str] = set()
        for alias in [*(metadata.get("search_aliases") or []), metadata.get("section_label")]:
            phrase = str(alias or "").strip().lower()
            if len(phrase) < 6 or " " not in phrase or phrase in seen:
                continue
            seen.add(phrase)
            if phrase in query_lower:
                bonus += 0.08
                if bonus >= 0.1:
                    return 0.1
        return bonus

    @staticmethod
    def _structured_lexical_text(chunk_text: str, metadata: dict) -> str:
        alias_values = metadata.get("search_aliases") or []
        topic_values = metadata.get("topic_tags") or []
        cross_refs = metadata.get("cross_references") or []
        fields = [
            chunk_text,
            metadata.get("document_title"),
            metadata.get("summary"),
            metadata.get("article_title"),
            metadata.get("section_title"),
            metadata.get("section_num"),
            metadata.get("article_num"),
            " ".join(str(item) for item in alias_values if item),
            " ".join(str(item) for item in topic_values if item),
            " ".join(str(item) for item in cross_refs if item),
        ]
        return " ".join(str(value or "") for value in fields).lower()

    def ingest_document(
        self,
        db: Session,
        *,
        union_id: str,
        document_id: str,
        text: str,
        metadata: dict | None = None,
        pages: list[dict] | None = None,
        structured_sections: list[dict] | None = None,
        clear_existing: bool = True,
    ) -> int:
        metadata = dict(metadata or {})
        if clear_existing:
            existing = db.scalars(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id)).all()
            for row in existing:
                db.delete(row)

        chunk_rows: list[tuple[str, dict]] = []
        if structured_sections:
            for section in structured_sections:
                section_text = str(section.get("text") or "").strip()
                if not section_text:
                    continue
                # Strip provenance BEFORE splitting so every sub-chunk of the
                # section inherits the anchor; stripping afterwards would leave
                # the marker at the head of only the first piece and lose the
                # anchor for the rest.
                cleaned_text, provenance = extract_provenance(section_text)
                if not cleaned_text:
                    continue
                section_metadata = {
                    **metadata,
                    **{key: value for key, value in section.items() if key != "text"},
                    **provenance_metadata(provenance),
                    "structure_mode": "legal_structured",
                }
                # Structure extraction puts whole section bodies in
                # section_title. Keep the full text for lexical scoring, but
                # give the UI something short enough to read as a label.
                section_label = summarize_section_label(section_metadata.get("section_title") or "")
                if section_label:
                    section_metadata["section_label"] = section_label
                # Contract packs carry the printed page on the section itself
                # rather than in an inline PROV marker. Citations and the PDF
                # pane read source_page, so normalize onto that key.
                if not section_metadata.get("source_page"):
                    page_start = section_metadata.get("page_start")
                    if isinstance(page_start, int) and page_start > 0:
                        section_metadata["source_page"] = page_start
                # A "section" here can be a 50KB slab (the source books have
                # sections that big). One embedding cannot represent that much
                # text, and retrieval then cannot discriminate within it —
                # measured in production as a holiday-premium question
                # returning scheduling text. Split to the same budget every
                # other branch uses; siblings share the section metadata and
                # record their position so citations stay traceable.
                pieces = self.split_text(cleaned_text)
                total = len(pieces)
                for part_index, piece in enumerate(pieces):
                    piece_metadata = dict(section_metadata)
                    if total > 1:
                        piece_metadata["section_part"] = part_index + 1
                        piece_metadata["section_parts_total"] = total
                    chunk_rows.append((piece, piece_metadata))
        elif pages:
            for page in pages:
                page_text = str(page.get("text") or "").strip()
                if not page_text:
                    continue
                page_number = page.get("page_number")
                page_metadata = {
                    **metadata,
                    "page_number": int(page_number) if isinstance(page_number, int) or str(page_number).isdigit() else None,
                }
                for chunk_text in self._chunk_page(page_text):
                    chunk_rows.append((chunk_text, page_metadata))
        else:
            for chunk_text in self.split_text(text):
                chunk_rows.append((chunk_text, dict(metadata)))

        index = 0
        for raw_chunk_text, chunk_metadata in chunk_rows:
            # Strip machine annotations before the text is stored, embedded, or
            # shown. Doing it here covers every branch above, and means the
            # embedding is computed over what the member actually reads rather
            # than over marker noise.
            chunk_text, provenance = extract_provenance(raw_chunk_text)
            if not chunk_text:
                # The chunk was nothing but markers/page furniture.
                continue
            db.add(
                ChunkEmbedding(
                    union_id=union_id,
                    document_id=document_id,
                    chunk_index=index,
                    chunk_text=chunk_text,
                    metadata_json={
                        **chunk_metadata,
                        **provenance_metadata(provenance),
                        "content_length": len(chunk_text),
                        "embedding_backend": self.embedder.descriptor.backend,
                        "embedding_model": self.embedder.descriptor.model_name,
                    },
                    embedding=self.embedder.embed(chunk_text),
                )
            )
            index += 1
        db.flush()
        return index

    def search(
        self,
        db: Session,
        *,
        union_id: str,
        query: str,
        limit: int = 5,
        document_id: str | None = None,
        contract_id: str | None = None,
        contract_ids: list[str] | None = None,
        preferred_article_num: str | None = None,
        preferred_topic_tags: list[str] | None = None,
        member_safe_only: bool = True,
    ) -> list[RetrievedChunk]:
        stmt = select(Document).where(
            Document.union_id == union_id,
            Document.status == DocumentStatus.ACTIVE,
        )
        if contract_id:
            # Hard scope, not a preference: a meat-department member must never
            # be answered out of the clerks agreement. Documents with no
            # contract_id are excluded too — an unscoped document could be from
            # any book, and guessing is exactly the failure being prevented.
            stmt = stmt.where(Document.contract_id == contract_id)
        elif contract_ids:
            # Store-scoped steward: restrict to their store's contract set. Same
            # hard-scope reasoning — NULL-contract documents are excluded so the
            # steward never sees outside the allowlist.
            stmt = stmt.where(Document.contract_id.in_(list(contract_ids)))
        ready_documents = db.scalars(stmt).all()
        ready_document_ids = [
            document.id
            for document in ready_documents
            if bool(((document.metadata_json or {}).get("ready_for_query")))
            and (not member_safe_only or bool((document.metadata_json or {}).get("member_visible", True)))
        ]
        if not ready_document_ids:
            return []

        base_stmt = select(ChunkEmbedding).where(
            ChunkEmbedding.union_id == union_id,
            ChunkEmbedding.document_id.in_(ready_document_ids),
        )
        if document_id:
            base_stmt = base_stmt.where(ChunkEmbedding.document_id == document_id)

        search_terms = self._search_terms(query)
        candidate_limit = 800 if search_terms else 300
        rows = db.scalars(base_stmt.limit(candidate_limit)).all()

        query_lower = str(query or "").lower()
        query_embedding = self.embedder.embed(query)
        scored: list[RetrievedChunk] = []
        for row in rows:
            row_embedding = list(row.embedding) if row.embedding is not None else []
            similarity = sum(a * b for a, b in zip(query_embedding, row_embedding))
            metadata = dict(row.metadata_json or {})
            lowered = self._structured_lexical_text(str(row.chunk_text or ""), metadata)
            lexical_bonus = self._fuzzy_lexical_bonus(search_terms, lowered, metadata)
            if member_safe_only and bool(metadata.get("sensitive_data_risk")):
                lexical_bonus -= 0.22
            if member_safe_only and bool(metadata.get("prompt_injection_risk")):
                lexical_bonus -= 1.5
            if metadata.get("section_title"):
                lexical_bonus += 0.15 * sum(1 for term in search_terms if term in str(metadata.get("section_title") or "").lower())
            if metadata.get("article_title"):
                lexical_bonus += 0.1 * sum(1 for term in search_terms if term in str(metadata.get("article_title") or "").lower())
            if metadata.get("topic_tags"):
                lexical_bonus += 0.15 * sum(1 for term in search_terms if any(term in str(tag).lower() for tag in metadata.get("topic_tags") or []))
            if preferred_article_num and str(metadata.get("article_num") or "").strip() == str(preferred_article_num).strip():
                lexical_bonus += 0.45
            normalized_preferred_topics = [str(tag or "").strip().lower() for tag in (preferred_topic_tags or []) if str(tag or "").strip()]
            if normalized_preferred_topics and metadata.get("topic_tags"):
                row_topics = {str(tag or "").strip().lower() for tag in (metadata.get("topic_tags") or []) if str(tag or "").strip()}
                lexical_bonus += 0.2 * sum(1 for tag in normalized_preferred_topics if tag in row_topics)
            # The lexical signal is a tiebreaker, not the ranking. Measured on
            # the live index: pure embedding similarity put the correct
            # "Premium Pay for Holiday Work" section first at 0.79 vs 0.72 for
            # the runner-up, while uncapped term bonuses (+0.2 per matched
            # word) let an unrelated jury-duty chunk outscore it. Positive
            # bonus is capped; penalties (sensitive data, prompt injection)
            # stay uncapped because they exist to bury a chunk, not nudge it.
            if lexical_bonus > 0:
                lexical_bonus = min(lexical_bonus, 0.05)
            # Exact-phrase alias hits sit outside the cap; see the helper.
            phrase_bonus = self._phrase_alias_bonus(query_lower, metadata) if lexical_bonus > -0.1 else 0.0
            scored.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    content=row.chunk_text,
                    similarity=float(similarity + lexical_bonus + phrase_bonus),
                    metadata=metadata,
                )
            )
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[: max(1, int(limit))]

    def delete_document(self, db: Session, *, document_id: str) -> int:
        rows = db.scalars(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id)).all()
        deleted = len(rows)
        for row in rows:
            db.delete(row)
        db.flush()
        return deleted
