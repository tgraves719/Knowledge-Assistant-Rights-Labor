"""Persistent chat history helpers for union-scoped sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.platform.auth import AuthContext, get_current_auth_context
from backend.platform.db import apply_request_context, apply_service_bootstrap_context
from backend.platform.models import Chat, Message, Union


@dataclass
class SessionBinding:
    session_id: str
    union_id: str | None = None
    user_id: str | None = None
    message_retention_enabled: bool = False


class ChatHistoryStore:
    def __init__(self, session_factory: sessionmaker[Session] | None):
        self.session_factory = session_factory
        self._bindings: dict[str, SessionBinding] = {}

    def _resolve_auth_context(self, binding: SessionBinding | None) -> AuthContext | None:
        auth = get_current_auth_context()
        if auth is not None:
            return auth
        if binding is None:
            return None
        return AuthContext(
            user_id=binding.user_id,
            email=None,
            full_name=None,
            role="user",
            union_id=binding.union_id,
            union_slug=None,
            source="chat_history",
            is_authenticated=bool(binding.user_id or binding.union_id),
        )

    def bind_session(
        self,
        *,
        session_id: str,
        union_local_id: str | None = None,
        union_id: str | None = None,
        user_id: str | None = None,
        message_retention_enabled: bool = False,
    ) -> SessionBinding:
        binding = self._bindings.get(session_id) or SessionBinding(session_id=session_id)
        binding.user_id = user_id or binding.user_id
        binding.message_retention_enabled = bool(message_retention_enabled)

        if self.session_factory is not None and (union_id or union_local_id):
            with self.session_factory() as db:
                apply_service_bootstrap_context(db)
                resolved_union_id = union_id
                if resolved_union_id is None and union_local_id:
                    union = db.scalar(
                        select(Union).where((Union.union_local_id == union_local_id) | (Union.slug == union_local_id))
                    )
                    resolved_union_id = union.id if union else None
                    if union is not None:
                        binding.message_retention_enabled = bool(union.message_retention_enabled)
                if resolved_union_id:
                    binding.union_id = resolved_union_id
                apply_request_context(db, self._resolve_auth_context(binding))
        else:
            binding.union_id = union_id or binding.union_id

        self._bindings[session_id] = binding
        return binding

    def load_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        binding = self._bindings.get(session_id)
        if self.session_factory is None or binding is None or not binding.union_id:
            return []
        with self.session_factory() as db:
            apply_request_context(db, self._resolve_auth_context(binding))
            chat = db.scalar(
                select(Chat).where(Chat.session_id == session_id, Chat.union_id == binding.union_id)
            )
            if chat is None:
                return []
            messages = db.scalars(
                select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at.asc()).limit(limit)
            ).all()
            return [
                {
                    "role": item.role,
                    "content": item.content,
                    "metadata": item.metadata_json,
                    "created_at": item.created_at,
                }
                for item in messages
            ]

    def persist_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> None:
        binding = self._bindings.get(session_id)
        if (
            self.session_factory is None
            or binding is None
            or not binding.union_id
            or not binding.message_retention_enabled
        ):
            return

        with self.session_factory() as db:
            apply_request_context(db, self._resolve_auth_context(binding))
            chat = db.scalar(
                select(Chat).where(Chat.session_id == session_id, Chat.union_id == binding.union_id)
            )
            if chat is None:
                chat = Chat(union_id=binding.union_id, user_id=binding.user_id, session_id=session_id)
                db.add(chat)
                db.flush()
            chat.updated_at = datetime.utcnow()
            db.add(
                Message(
                    union_id=binding.union_id,
                    chat_id=chat.id,
                    role="user",
                    content=question,
                    metadata_json=metadata or {},
                )
            )
            db.add(
                Message(
                    union_id=binding.union_id,
                    chat_id=chat.id,
                    role="assistant",
                    content=answer,
                    metadata_json=metadata or {},
                )
            )
            db.commit()
