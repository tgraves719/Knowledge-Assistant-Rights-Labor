"""Regression guard for message non-retention (privacy/governance merge gate).

`ChatHistoryStore.persist_turn` must write **zero** `Message` rows when a union has
`message_retention_enabled = False` (the default), and must persist the user+assistant pair
when retention is explicitly enabled. This pins the privacy-by-default behavior at
`backend/platform/chat_history.py` so it cannot silently regress.
"""

from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.chat_history import ChatHistoryStore
from backend.platform.db import Base
from backend.platform.models import Chat, Message, Union


def _store():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return ChatHistoryStore(SessionLocal), SessionLocal


def _seed_union(SessionLocal, *, slug: str, retention_enabled: bool) -> str:
    with SessionLocal() as db:
        union = Union(slug=slug, name=slug, union_local_id=slug, message_retention_enabled=retention_enabled)
        db.add(union)
        db.commit()
        return union.id


def _message_count(SessionLocal) -> int:
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(Message))


def test_persist_turn_writes_nothing_when_retention_disabled():
    store, SessionLocal = _store()
    _seed_union(SessionLocal, slug="local-noretain", retention_enabled=False)

    binding = store.bind_session(session_id="s-noretain", union_local_id="local-noretain")
    assert binding.message_retention_enabled is False

    store.persist_turn(session_id="s-noretain", question="private question", answer="private answer")

    assert _message_count(SessionLocal) == 0
    with SessionLocal() as db:
        assert db.scalar(select(Chat).where(Chat.session_id == "s-noretain")) is None


def test_persist_turn_writes_pair_when_retention_enabled():
    store, SessionLocal = _store()
    _seed_union(SessionLocal, slug="local-retain", retention_enabled=True)

    binding = store.bind_session(session_id="s-retain", union_local_id="local-retain")
    assert binding.message_retention_enabled is True

    store.persist_turn(session_id="s-retain", question="kept question", answer="kept answer")

    assert _message_count(SessionLocal) == 2
    with SessionLocal() as db:
        roles = db.scalars(
            select(Message.role).join(Chat, Message.chat_id == Chat.id).where(Chat.session_id == "s-retain").order_by(Message.created_at.asc())
        ).all()
        assert roles == ["user", "assistant"]
