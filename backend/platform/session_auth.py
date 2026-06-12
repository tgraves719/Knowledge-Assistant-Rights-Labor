"""Server-managed browser session authentication."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.platform.models import AuthSession, Role, SessionType, Union, UnionMembership, User


@dataclass
class SessionResolution:
    user: User | None
    union: Union | None
    role: str | None
    session: AuthSession | None
    clear_cookie: bool = False

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None and self.role is not None


class SessionAuthService:
    def __init__(
        self,
        *,
        secret_key: str,
        cookie_name: str,
        member_idle_seconds: int,
        union_admin_idle_seconds: int,
        super_admin_idle_seconds: int,
    ) -> None:
        self.cookie_name = str(cookie_name or "karl_session")
        self._secret_key = str(secret_key).encode("utf-8")
        self._member_idle_seconds = max(3600, int(member_idle_seconds))
        self._union_admin_idle_seconds = max(3600, int(union_admin_idle_seconds))
        self._super_admin_idle_seconds = max(300, int(super_admin_idle_seconds))

    def _hash_secret(self, secret: str) -> str:
        return hmac.new(self._secret_key, str(secret).encode("utf-8"), hashlib.sha256).hexdigest()

    def _session_idle_seconds(self, session_type: SessionType) -> int:
        if session_type == SessionType.SUPER_ADMIN:
            return self._super_admin_idle_seconds
        if session_type == SessionType.UNION_ADMIN:
            return self._union_admin_idle_seconds
        return self._member_idle_seconds

    def _derive_session_type(self, role: str) -> SessionType:
        if role == Role.SUPER_ADMIN.value:
            return SessionType.SUPER_ADMIN
        if role in {Role.UNION_ADMIN.value, Role.STEWARD_ADMIN.value}:
            return SessionType.UNION_ADMIN
        return SessionType.MEMBER

    def create_session(
        self,
        db: Session,
        *,
        user: User,
        union: Union | None,
        role: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[AuthSession, str]:
        session_type = self._derive_session_type(role)
        secret = secrets.token_urlsafe(48)
        now = datetime.utcnow()
        auth_session = AuthSession(
            user_id=user.id,
            union_id=union.id if union is not None else None,
            session_secret_hash=self._hash_secret(secret),
            session_type=session_type,
            ip_address=(str(ip_address or "").strip() or None),
            user_agent=(str(user_agent or "").strip() or None),
            created_at=now,
            last_seen_at=now,
            expires_at=now + timedelta(seconds=self._session_idle_seconds(session_type)),
        )
        db.add(auth_session)
        db.flush()
        return auth_session, secret

    def revoke_session(self, db: Session, session_id: str | None) -> None:
        if not session_id:
            return
        auth_session = db.get(AuthSession, session_id)
        if auth_session is None or auth_session.revoked_at is not None:
            return
        auth_session.revoked_at = datetime.utcnow()
        db.flush()

    def touch_session(
        self,
        db: Session,
        *,
        session_id: str | None,
        min_interval_seconds: int = 300,
    ) -> bool:
        if not session_id:
            return False
        auth_session = db.get(AuthSession, session_id)
        if auth_session is None or auth_session.revoked_at is not None:
            return False
        now = datetime.utcnow()
        if auth_session.expires_at <= now:
            return False
        elapsed = (now - auth_session.last_seen_at).total_seconds() if auth_session.last_seen_at else None
        if elapsed is not None and elapsed < max(30, int(min_interval_seconds)):
            return False
        auth_session.last_seen_at = now
        auth_session.expires_at = now + timedelta(seconds=self._session_idle_seconds(auth_session.session_type))
        db.flush()
        return True

    def resolve(self, db: Session | None, *, session_secret: str | None, requested_union_slug: str | None = None) -> SessionResolution:
        if db is None or not str(session_secret or "").strip():
            return SessionResolution(user=None, union=None, role=None, session=None, clear_cookie=False)

        auth_session = db.scalar(
            select(AuthSession).where(AuthSession.session_secret_hash == self._hash_secret(session_secret.strip()))
        )
        if auth_session is None:
            return SessionResolution(user=None, union=None, role=None, session=None, clear_cookie=True)

        now = datetime.utcnow()
        if auth_session.revoked_at is not None or auth_session.expires_at <= now:
            if auth_session.revoked_at is None:
                auth_session.revoked_at = now
                db.flush()
            return SessionResolution(user=None, union=None, role=None, session=auth_session, clear_cookie=True)

        user = db.get(User, auth_session.user_id)
        if user is None or not user.is_active:
            auth_session.revoked_at = now
            db.flush()
            return SessionResolution(user=None, union=None, role=None, session=auth_session, clear_cookie=True)

        requested_union = None
        if requested_union_slug:
            requested_union = db.scalar(
                select(Union).where(or_(Union.slug == requested_union_slug, Union.union_local_id == requested_union_slug))
            )

        role: str | None = None
        active_union: Union | None = None
        if auth_session.session_type == SessionType.SUPER_ADMIN:
            role = Role.SUPER_ADMIN.value
            active_union = requested_union
        else:
            if auth_session.union_id is None:
                auth_session.revoked_at = now
                db.flush()
                return SessionResolution(user=None, union=None, role=None, session=auth_session, clear_cookie=True)
            active_union = db.get(Union, auth_session.union_id)
            if active_union is None or not active_union.is_active:
                auth_session.revoked_at = now
                db.flush()
                return SessionResolution(user=None, union=None, role=None, session=auth_session, clear_cookie=True)
            if requested_union is not None and requested_union.id != active_union.id:
                role = Role.USER.value
            membership = db.scalar(
                select(UnionMembership).where(
                    UnionMembership.user_id == user.id,
                    UnionMembership.union_id == active_union.id,
                    UnionMembership.is_active.is_(True),
                )
            )
            if membership is None:
                auth_session.revoked_at = now
                db.flush()
                return SessionResolution(user=None, union=None, role=None, session=auth_session, clear_cookie=True)
            role = membership.role.value

        return SessionResolution(user=user, union=active_union, role=role, session=auth_session, clear_cookie=False)
