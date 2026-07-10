"""FastAPI dependencies for the production platform layer."""

from __future__ import annotations

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.platform.auth import AuthContext
from backend.platform.models import Role


def get_container(request: Request):
    return request.app.state.platform


def get_db(request: Request) -> Session | None:
    return getattr(request.state, "db", None)


def get_auth_context(request: Request) -> AuthContext:
    auth = getattr(request.state, "auth_context", None)
    if auth is None:
        return AuthContext(
            user_id=None,
            email=None,
            full_name=None,
            role=Role.USER.value,
            union_id=None,
            union_slug=None,
            source="anonymous",
            is_authenticated=False,
        )
    return auth


def require_roles(*roles: str):
    def _dependency(request: Request) -> AuthContext:
        auth = get_auth_context(request)
        if not auth.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if auth.role not in roles and not auth.is_super_admin:
            raise HTTPException(status_code=403, detail="Insufficient permissions.")
        return auth

    return _dependency

