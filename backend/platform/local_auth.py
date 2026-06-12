"""Local username/password auth for demo and fallback environments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.platform.models import LocalAuthCredential, Union, UnionMembership, User


class LocalAuthError(RuntimeError):
    pass


class LocalAuthService:
    def __init__(self, *, secret_key: str, token_ttl_seconds: int = 43_200):
        self._secret_key = str(secret_key).encode("utf-8")
        self._token_ttl_seconds = max(300, int(token_ttl_seconds))

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padded = str(value) + "=" * (-len(str(value)) % 4)
        return base64.urlsafe_b64decode(padded.encode("utf-8"))

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
        return LocalAuthService._b64encode(digest)

    def create_or_update_credential(
        self,
        db: Session,
        *,
        user: User,
        username: str,
        password: str,
    ) -> LocalAuthCredential:
        normalized_username = str(username or "").strip().lower()
        if not normalized_username or not password:
            raise LocalAuthError("Username and password are required.")
        credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.user_id == user.id))
        if credential is None:
            credential = LocalAuthCredential(user_id=user.id, username=normalized_username, password_hash="", password_salt="")
            db.add(credential)
        salt = secrets.token_bytes(16)
        credential.username = normalized_username
        credential.password_salt = self._b64encode(salt)
        credential.password_hash = self._hash_password(password, salt)
        credential.is_active = True
        db.flush()
        return credential

    def authenticate_credentials(
        self,
        db: Session,
        *,
        username: str,
        password: str,
    ) -> User:
        normalized_username = str(username or "").strip().lower()
        credential = db.scalar(
            select(LocalAuthCredential).where(
                LocalAuthCredential.username == normalized_username,
                LocalAuthCredential.is_active.is_(True),
            )
        )
        if credential is None:
            raise LocalAuthError("Invalid username or password.")
        expected = self._hash_password(password, self._b64decode(credential.password_salt))
        if not hmac.compare_digest(expected, credential.password_hash):
            raise LocalAuthError("Invalid username or password.")
        user = db.get(User, credential.user_id)
        if user is None or not user.is_active:
            raise LocalAuthError("User is inactive.")
        return user

    def authenticate(
        self,
        db: Session,
        *,
        username: str,
        password: str,
        union_slug: str | None = None,
    ) -> tuple[User, Union | None]:
        user = self.authenticate_credentials(db, username=username, password=password)

        memberships = db.scalars(
            select(UnionMembership).where(
                UnionMembership.user_id == user.id,
                UnionMembership.is_active.is_(True),
            )
        ).all()
        if not memberships:
            raise LocalAuthError("User has no active union membership.")

        if union_slug:
            union = db.scalar(select(Union).where((Union.slug == union_slug) | (Union.union_local_id == union_slug)))
            if union is None or not any(membership.union_id == union.id for membership in memberships):
                raise LocalAuthError("Union is not available for this user.")
            return user, union

        if len(memberships) == 1:
            union = db.get(Union, memberships[0].union_id)
            return user, union

        raise LocalAuthError("Union selection is required for multi-union users.")

    def issue_token(self, *, user_id: str, union_slug: str | None) -> str:
        payload = {
            "sub": str(user_id),
            "union_slug": str(union_slug or ""),
            "exp": int(time.time()) + self._token_ttl_seconds,
        }
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self._secret_key, payload_bytes, hashlib.sha256).digest()
        return f"{self._b64encode(payload_bytes)}.{self._b64encode(signature)}"

    def decode_token(self, token: str) -> dict | None:
        try:
            payload_b64, signature_b64 = str(token or "").split(".", 1)
            payload_bytes = self._b64decode(payload_b64)
            expected_sig = hmac.new(self._secret_key, payload_bytes, hashlib.sha256).digest()
            if not hmac.compare_digest(expected_sig, self._b64decode(signature_b64)):
                return None
            payload = json.loads(payload_bytes.decode("utf-8"))
            if int(payload.get("exp") or 0) < int(time.time()):
                return None
            return payload
        except Exception:
            return None
