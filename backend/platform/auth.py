"""Authentication adapter and request identity resolution."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.platform.local_auth import LocalAuthService
from backend.platform.session_auth import SessionAuthService
from backend.platform.models import Role, Union, UnionMembership, User
from backend.platform.settings import PlatformSettings


_current_auth_context: ContextVar["AuthContext | None"] = ContextVar(
    "current_auth_context",
    default=None,
)


@dataclass
class AuthContext:
    user_id: str | None
    email: str | None
    full_name: str | None
    role: str
    union_id: str | None
    union_slug: str | None
    source: str
    is_authenticated: bool = False
    session_id: str | None = None
    session_type: str | None = None
    clear_session_cookie: bool = False

    @property
    def is_super_admin(self) -> bool:
        return self.role == Role.SUPER_ADMIN.value


def set_current_auth_context(auth: "AuthContext | None") -> Token:
    return _current_auth_context.set(auth)


def reset_current_auth_context(token: Token) -> None:
    _current_auth_context.reset(token)


def get_current_auth_context() -> "AuthContext | None":
    return _current_auth_context.get()


class HeaderAuthAdapter:
    """Header-driven identity adapter while the IdP choice remains open."""

    def __init__(
        self,
        settings: PlatformSettings,
        *,
        local_auth: LocalAuthService | None = None,
        session_auth: SessionAuthService | None = None,
    ):
        self._bootstrap_super_admin_emails = {
            item.strip().lower()
            for item in settings.bootstrap_super_admin_emails
            if item.strip()
        }
        self._local_auth = local_auth
        self._session_auth = session_auth

    def resolve(
        self,
        *,
        db: Session | None,
        session_cookie: str | None,
        authorization: str | None,
        external_auth_id: str | None,
        email: str | None,
        full_name: str | None,
        requested_role: str | None,
        union_slug: str | None,
    ) -> AuthContext:
        token_user_id = None
        token_union_slug = None
        clear_session_cookie = False
        if self._session_auth is not None and session_cookie:
            session_resolution = self._session_auth.resolve(db, session_secret=session_cookie, requested_union_slug=union_slug)
            clear_session_cookie = session_resolution.clear_cookie
            if session_resolution.is_authenticated:
                return AuthContext(
                    user_id=session_resolution.user.id,
                    email=session_resolution.user.email,
                    full_name=session_resolution.user.full_name,
                    role=session_resolution.role or Role.USER.value,
                    union_id=session_resolution.union.id if session_resolution.union else None,
                    union_slug=session_resolution.union.slug if session_resolution.union else union_slug,
                    source="session",
                    is_authenticated=True,
                    session_id=session_resolution.session.id if session_resolution.session else None,
                    session_type=session_resolution.session.session_type.value if session_resolution.session else None,
                    clear_session_cookie=False,
                )
        if self._local_auth is not None and authorization:
            scheme, _, token = str(authorization).partition(" ")
            if scheme.strip().lower() == "bearer" and token.strip():
                payload = self._local_auth.decode_token(token.strip())
                if payload:
                    token_user_id = str(payload.get("sub") or "").strip() or None
                    token_union_slug = str(payload.get("union_slug") or "").strip() or None
                    requested_role = None
                    external_auth_id = None
                    if token_user_id:
                        email = None
                        full_name = None
        union_slug = token_union_slug or union_slug
        requested_role_value = str(requested_role or Role.USER.value).strip().lower()
        if requested_role_value not in {member.value for member in Role}:
            requested_role_value = Role.USER.value
        role = requested_role_value
        normalized_email = str(email or "").strip().lower()
        is_bootstrap_super_admin = bool(
            normalized_email and normalized_email in self._bootstrap_super_admin_emails
        )
        if is_bootstrap_super_admin:
            role = Role.SUPER_ADMIN.value

        if db is None or (not token_user_id and not external_auth_id and not email and not union_slug):
            return AuthContext(
                user_id=token_user_id,
                email=email,
                full_name=full_name,
                role=role,
                union_id=None,
                union_slug=union_slug,
                source="local_token" if token_user_id else "header",
                is_authenticated=bool(token_user_id or email or external_auth_id),
                clear_session_cookie=clear_session_cookie,
            )

        user = None
        if token_user_id:
            user = db.get(User, token_user_id)
        if user is None and external_auth_id:
            user = db.scalar(select(User).where(User.external_auth_id == external_auth_id))
        if user is None and email:
            user = db.scalar(select(User).where(User.email == email))

        union = None
        if union_slug:
            union = db.scalar(
                select(Union).where((Union.slug == union_slug) | (Union.union_local_id == union_slug))
            )

        resolved_role = Role.USER.value
        union_id = None
        if is_bootstrap_super_admin:
            resolved_role = Role.SUPER_ADMIN.value
            union_id = union.id if union else None
        elif user and union:
            membership = db.scalar(
                select(UnionMembership).where(
                    UnionMembership.user_id == user.id,
                    UnionMembership.union_id == union.id,
                    UnionMembership.is_active.is_(True),
                )
            )
            if membership:
                resolved_role = membership.role.value
                union_id = union.id
        elif user and requested_role_value == Role.SUPER_ADMIN.value:
            resolved_role = Role.USER.value

        if user and user.email and not email:
            email = user.email
        if user and user.full_name and not full_name:
            full_name = user.full_name

        return AuthContext(
            user_id=user.id if user else None,
            email=email,
            full_name=full_name,
            role=resolved_role,
            union_id=union_id,
            union_slug=union.slug if union else union_slug,
            source="local_token" if token_user_id else "header",
            is_authenticated=bool(user or token_user_id or email or external_auth_id),
            clear_session_cookie=clear_session_cookie,
        )
