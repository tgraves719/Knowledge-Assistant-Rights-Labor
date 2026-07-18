from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.embeddings import DeterministicTextEmbedder
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus, Union
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.text_normalization import extract_provenance, provenance_metadata


PROV_LINE = (
    "PROV(contract_id=local7_safeway_pueblo_clerks_2022, anchor_id=a10_s25_p2, "
    "effective_version_id=effective_local7_safeway_moa_2025_07_05, "
    "sources=[base:SW+Pueblo+Clerks+2022.2025.pdf#p11])"
)


def test_extract_provenance_removes_marker_and_keeps_contract_language():
    text = f"{PROV_LINE}\n\nOvertime shall be paid at time and one-half after eight (8) hours."

    cleaned, provenance = extract_provenance(text)

    assert "PROV(" not in cleaned
    assert cleaned.startswith("Overtime shall be paid at time and one-half")
    assert provenance["anchor_id"] == "a10_s25_p2"
    assert provenance["contract_id"] == "local7_safeway_pueblo_clerks_2022"
    assert provenance["sources"] == ["base:SW+Pueblo+Clerks+2022.2025.pdf#p11"]


def test_provenance_metadata_keeps_the_anchor_for_citations():
    _, provenance = extract_provenance(f"{PROV_LINE}\n\nSome clause.")

    metadata = provenance_metadata(provenance)

    assert metadata["anchor_id"] == "a10_s25_p2"
    assert metadata["effective_version_id"] == "effective_local7_safeway_moa_2025_07_05"
    assert metadata["provenance_sources"] == ["base:SW+Pueblo+Clerks+2022.2025.pdf#p11"]


def test_extract_provenance_strips_running_page_furniture():
    text = (
        f"{PROV_LINE}\n\nVacation shall accrue monthly.\n\n"
        "9                    PUEBLO CLERKS\n                     2022-2025\n\n"
        "Employees may bid by seniority."
    )

    cleaned, _ = extract_provenance(text)

    assert "PUEBLO CLERKS" not in cleaned
    assert "Vacation shall accrue monthly." in cleaned
    assert "Employees may bid by seniority." in cleaned


def test_strips_page_furniture_stranded_mid_sentence():
    """Structured extraction collapses line breaks before text reaches us.

    By then the running header sits inside a sentence, so a line-anchored rule
    misses it and the member reads contract language interrupted by page
    numbers.
    """
    text = (
        "unless the average scheduled hours of all part-time employees is "
        "twenty-four (24) hours or less 9 PUEBLO CLERKS 2022-2025 --- "
        "10 PUEBLO CLERKS 2022-2025 for the involved workweek."
    )

    cleaned, _ = extract_provenance(text)

    assert "PUEBLO CLERKS" not in cleaned
    assert "2022-2025" not in cleaned
    assert cleaned.endswith("hours or less for the involved workweek.")


def test_extract_provenance_leaves_ordinary_contract_language_alone():
    """The furniture rule must not eat real clauses.

    Contract text routinely contains numbers, capitals and year ranges; a
    greedy rule would quietly delete terms members depend on.
    """
    text = (
        "Section 12. The term of this Agreement is 2022-2025 and covers all "
        "employees.\nARTICLE 5 WAGES apply to 40 hours per week.\n"
        "3 days of funeral leave shall be granted."
    )

    cleaned, provenance = extract_provenance(text)

    assert "The term of this Agreement is 2022-2025" in cleaned
    assert "ARTICLE 5 WAGES apply to 40 hours per week." in cleaned
    assert "3 days of funeral leave shall be granted." in cleaned
    assert provenance == {}


def test_multiple_markers_keep_the_owning_anchor_and_drop_the_rest():
    second = PROV_LINE.replace("a10_s25_p2", "a10_s25_p3")
    text = f"{PROV_LINE}\n\nFirst clause.\n\n{second}\n\nSecond clause."

    cleaned, provenance = extract_provenance(text)

    assert "PROV(" not in cleaned
    assert "First clause." in cleaned and "Second clause." in cleaned
    # The first marker owns the passage; the merged-in one must not overwrite it.
    assert provenance["anchor_id"] == "a10_s25_p2"


def test_ingest_stores_cleaned_text_and_skips_marker_only_chunks():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(engine)

    retrieval = TenantRetrievalService(embedder=DeterministicTextEmbedder(dimensions=768))

    with SessionLocal() as db:
        union = Union(slug="local-7", name="UFCW Local 7", union_local_id="local7")
        db.add(union)
        db.flush()
        document = Document(
            union_id=union.id,
            title="clerks.md",
            contract_id="clerks_2022",
            storage_key="local-7/clerks.md",
            content_type="text/markdown",
            bytes_size=10,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True},
        )
        db.add(document)
        db.flush()

        retrieval.ingest_document(
            db,
            union_id=union.id,
            document_id=document.id,
            text="",
            structured_sections=[
                {"text": f"{PROV_LINE}\n\nHoliday premium is one and one-half times."},
                {"text": PROV_LINE},  # marker only -- nothing a member could read
            ],
        )
        db.commit()

        chunks = db.scalars(
            select(ChunkEmbedding).where(ChunkEmbedding.document_id == document.id)
        ).all()

        assert len(chunks) == 1, "marker-only section should not become a chunk"
        assert "PROV(" not in chunks[0].chunk_text
        assert chunks[0].chunk_text.startswith("Holiday premium")
        assert chunks[0].metadata_json["anchor_id"] == "a10_s25_p2"
        # chunk_index must stay contiguous after the skip.
        assert chunks[0].chunk_index == 0
