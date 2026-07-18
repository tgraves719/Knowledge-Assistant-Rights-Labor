import math

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.embeddings import (
    DeterministicTextEmbedder,
    EmbeddingProviderError,
    GoogleTextEmbedder,
    build_text_embedder,
)
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus, Union
from backend.platform.retrieval import TenantRetrievalService


def test_build_text_embedder_defaults_to_deterministic():
    embedder = build_text_embedder(backend="deterministic", dimensions=384)
    assert isinstance(embedder, DeterministicTextEmbedder)
    assert embedder.descriptor.backend == "deterministic"
    assert embedder.descriptor.dimensions == 384


def test_build_text_embedder_supports_google_boundary():
    embedder = build_text_embedder(
        backend="google",
        dimensions=768,
        google_model_name="text-embedding-005",
        google_api_key="test-key",
    )
    assert isinstance(embedder, GoogleTextEmbedder)
    assert embedder.descriptor.backend == "google_text"
    assert embedder.descriptor.model_name == "text-embedding-005"


def test_retrieval_service_records_embedding_backend_metadata():
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
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        document = Document(
            union_id=union.id,
            title="notice.txt",
            storage_key="local-1/notice.txt",
            content_type="text/plain",
            bytes_size=42,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True},
        )
        db.add(union)
        db.flush()
        document.union_id = union.id
        db.add(document)
        db.flush()

        chunk_count = retrieval.ingest_document(
            db,
            union_id=union.id,
            document_id=document.id,
            text="Seniority rights govern bidding and vacation preference.",
        )
        db.commit()

        chunk = db.scalar(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document.id))
        assert chunk_count >= 1
        assert chunk is not None
        assert chunk.metadata_json["embedding_backend"] == "deterministic"
        assert chunk.metadata_json["embedding_model"] == "hashing_v1"


def test_retrieval_service_preserves_page_number_metadata():
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
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        document = Document(
            union_id=union.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=42,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True},
        )
        db.add(union)
        db.flush()
        document.union_id = union.id
        db.add(document)
        db.flush()

        retrieval.ingest_document(
            db,
            union_id=union.id,
            document_id=document.id,
            text="Page one text\n\nPage two text",
            pages=[
                {"page_number": 1, "text": "Page one text"},
                {"page_number": 2, "text": "Page two text"},
            ],
        )
        db.commit()

        chunks = db.scalars(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document.id).order_by(ChunkEmbedding.chunk_index.asc())).all()
        assert [chunk.metadata_json.get("page_number") for chunk in chunks] == [1, 2]


class _FakeEmbeddings:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, values):
        self.embeddings = [_FakeEmbeddings(values)]


class _FakeModels:
    def __init__(self, values, recorder):
        self._values = values
        self._recorder = recorder

    def embed_content(self, *, model, contents, config):
        self._recorder.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(self._values)


class _FakeClient:
    def __init__(self, values, recorder):
        self.models = _FakeModels(values, recorder)


def _google_embedder(monkeypatch, values, *, dimensions=768):
    pytest.importorskip("google.genai")
    from backend.platform import embeddings as embeddings_module

    calls = []
    embedder = embeddings_module.GoogleTextEmbedder(
        model_name="gemini-embedding-001", api_key="test-key", dimensions=dimensions
    )
    monkeypatch.setattr(embedder, "_get_client", lambda: _FakeClient(values, calls))
    return embedder, calls


def test_google_embedder_returns_normalized_vector_of_expected_width(monkeypatch):
    embedder, calls = _google_embedder(monkeypatch, [3.0] + [0.0] * 767)

    vector = embedder.embed("Overtime after eight hours.")

    assert len(vector) == 768
    # Normalized, so the magnitude is 1 regardless of the provider's scale.
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9
    assert calls[0]["model"] == "gemini-embedding-001"
    assert calls[0]["config"].output_dimensionality == 768


def test_google_embedder_rejects_width_mismatch(monkeypatch):
    """A short vector must fail loudly rather than be padded.

    Silently reshaping would corrupt the index in a way that only shows up as
    degraded retrieval quality, long after the ingest that caused it.
    """
    embedder, _ = _google_embedder(monkeypatch, [1.0] * 384)

    with pytest.raises(EmbeddingProviderError) as excinfo:
        embedder.embed("Seniority governs vacation preference.")

    assert "384" in str(excinfo.value) and "768" in str(excinfo.value)


def test_google_embedder_short_circuits_empty_text(monkeypatch):
    embedder, calls = _google_embedder(monkeypatch, [1.0] * 768)

    vector = embedder.embed("   ")

    assert vector == [0.0] * 768
    assert calls == []


def test_google_embedder_requires_an_api_key():
    embedder = GoogleTextEmbedder(model_name="gemini-embedding-001", api_key="", dimensions=768)

    with pytest.raises(EmbeddingProviderError) as excinfo:
        embedder.embed("anything")

    assert "KARL_GOOGLE_EMBEDDING_API_KEY" in str(excinfo.value)


def _seed_two_contracts(db):
    union = Union(slug="local-7", name="UFCW Local 7", union_local_id="local7")
    db.add(union)
    db.flush()
    docs = {}
    for contract_id, title, text in (
        ("clerks_2022", "clerks.md", "Courtesy clerks receive a meal period after five hours."),
        ("meat_2022", "meat.md", "Meat cutters receive a meal period after five hours."),
    ):
        document = Document(
            union_id=union.id,
            title=title,
            contract_id=contract_id,
            storage_key=f"local-7/{title}",
            content_type="text/markdown",
            bytes_size=len(text),
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True, "member_visible": True},
        )
        db.add(document)
        db.flush()
        docs[contract_id] = (document, text)
    return union, docs


def test_search_never_crosses_contract_boundaries():
    """A meat question must not be answerable from the clerks book.

    This is the pilot's core safety property: the two Pueblo agreements cover
    different bargaining units, and presenting one unit's terms to the other --
    with a citation, which reads as verified -- is the failure this project
    exists to prevent.
    """
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
        union, docs = _seed_two_contracts(db)
        for contract_id, (document, text) in docs.items():
            retrieval.ingest_document(db, union_id=union.id, document_id=document.id, text=text)
        db.commit()

        for scope in ("clerks_2022", "meat_2022"):
            hits = retrieval.search(
                db, union_id=union.id, query="meal period after five hours", contract_id=scope, limit=10
            )
            assert hits, f"expected hits within {scope}"
            returned = {
                db.get(Document, hit.document_id).contract_id for hit in hits
            }
            assert returned == {scope}, f"{scope} query leaked into {returned}"

        # Unscoped queries still see everything, so nothing is silently hidden
        # from an admin or a union with a single contract.
        unscoped = retrieval.search(
            db, union_id=union.id, query="meal period after five hours", limit=10
        )
        assert len({db.get(Document, hit.document_id).contract_id for hit in unscoped}) == 2


def test_search_excludes_unscoped_documents_when_a_contract_is_requested():
    """A document with no contract_id must not leak into a scoped query.

    It could belong to any book; including it would reintroduce exactly the
    cross-contract bleed the scope is there to stop.
    """
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
        union, docs = _seed_two_contracts(db)
        loose = Document(
            union_id=union.id,
            title="unfiled_notice.md",
            contract_id=None,
            storage_key="local-7/unfiled_notice.md",
            content_type="text/markdown",
            bytes_size=40,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True, "member_visible": True},
        )
        db.add(loose)
        db.flush()
        retrieval.ingest_document(
            db, union_id=union.id, document_id=loose.id, text="Meal period after five hours applies generally."
        )
        for contract_id, (document, text) in docs.items():
            retrieval.ingest_document(db, union_id=union.id, document_id=document.id, text=text)
        db.commit()

        hits = retrieval.search(
            db, union_id=union.id, query="meal period after five hours", contract_id="meat_2022", limit=10
        )
        assert hits
        assert all(db.get(Document, hit.document_id).contract_id == "meat_2022" for hit in hits)
