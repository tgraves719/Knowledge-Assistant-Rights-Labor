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
