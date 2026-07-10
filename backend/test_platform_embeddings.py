from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.embeddings import DeterministicTextEmbedder, GoogleTextEmbedder, build_text_embedder
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

    retrieval = TenantRetrievalService(embedder=DeterministicTextEmbedder(dimensions=384))

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

    retrieval = TenantRetrievalService(embedder=DeterministicTextEmbedder(dimensions=384))

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
