"""Production-oriented settings for multi-tenant KARL deployments."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _derive_fernet_key(seed: str) -> str:
    raw = seed.encode("utf-8")
    return base64.urlsafe_b64encode(raw.ljust(32, b"0")[:32]).decode("utf-8")


def _discover_liteparse_executable(project_root: Path) -> str:
    explicit = os.getenv("KARL_LITEPARSE_EXECUTABLE", "").strip()
    if explicit:
        return explicit
    vendored = project_root / "vendor" / "liteparse" / "dist" / "src" / "index.js"
    if vendored.exists():
        return f"node {vendored}"
    return ""


@dataclass(frozen=True)
class PlatformSettings:
    project_root: Path
    postgres_url: str
    auto_create_tables: bool
    apply_rls_policies: bool
    allowed_origins: list[str]
    request_rate_limit_per_minute: int
    login_rate_limit_per_minute: int
    query_token_limit: int
    hard_cap_default_requests_per_day: int
    hard_cap_default_tokens_per_day: int
    hard_cap_default_cost_usd_per_day: float
    local_storage_root: Path
    storage_backend: str
    document_parser_backend: str
    liteparse_executable: str
    embedding_backend: str
    embedding_dimensions: int
    google_embedding_model: str
    google_embedding_api_key: str
    inference_request_timeout_seconds: int
    legacy_contract_pipeline_enabled: bool
    inline_parse_max_bytes: int
    ocr_auto_retry_enabled: bool
    ocr_auto_retry_max_attempts: int
    local_auth_token_ttl_seconds: int
    secret_encryption_key: str
    sentinel_email_from: str
    sentinel_email_enabled: bool
    bootstrap_super_admin_emails: list[str]
    session_cookie_name: str = "karl_session"
    member_session_idle_seconds: int = 604800
    union_admin_session_idle_seconds: int = 259200
    super_admin_session_idle_seconds: int = 3600

    @property
    def db_enabled(self) -> bool:
        return bool(self.postgres_url.strip())


def get_platform_settings() -> PlatformSettings:
    project_root = Path(__file__).resolve().parents[2]
    encryption_seed = os.getenv("KARL_SECRET_ENCRYPTION_KEY", "development-only-change-me")
    return PlatformSettings(
        project_root=project_root,
        postgres_url=os.getenv("KARL_POSTGRES_URL", "").strip(),
        auto_create_tables=_as_bool(os.getenv("KARL_AUTO_CREATE_TABLES"), default=False),
        apply_rls_policies=_as_bool(os.getenv("KARL_APPLY_RLS_POLICIES"), default=False),
        allowed_origins=_split_csv(os.getenv("KARL_ALLOWED_ORIGINS")) or ["*"],
        request_rate_limit_per_minute=int(os.getenv("KARL_REQUEST_RATE_LIMIT_PER_MINUTE", "60")),
        login_rate_limit_per_minute=int(os.getenv("KARL_LOGIN_RATE_LIMIT_PER_MINUTE", "10")),
        query_token_limit=int(os.getenv("KARL_QUERY_TOKEN_LIMIT", "4000")),
        hard_cap_default_requests_per_day=int(os.getenv("KARL_DEFAULT_REQUESTS_PER_DAY", "500")),
        hard_cap_default_tokens_per_day=int(os.getenv("KARL_DEFAULT_TOKENS_PER_DAY", "250000")),
        hard_cap_default_cost_usd_per_day=float(os.getenv("KARL_DEFAULT_COST_USD_PER_DAY", "25")),
        local_storage_root=Path(os.getenv("KARL_STORAGE_ROOT", str(project_root / "var" / "storage"))),
        storage_backend=os.getenv("KARL_STORAGE_BACKEND", "local").strip().lower() or "local",
        document_parser_backend=os.getenv("KARL_DOCUMENT_PARSER_BACKEND", "auto").strip().lower() or "auto",
        liteparse_executable=_discover_liteparse_executable(project_root),
        embedding_backend=os.getenv("KARL_EMBEDDING_BACKEND", "deterministic").strip().lower() or "deterministic",
        embedding_dimensions=max(32, int(os.getenv("KARL_EMBEDDING_DIMENSIONS", "384"))),
        google_embedding_model=os.getenv("KARL_GOOGLE_EMBEDDING_MODEL", "text-embedding-004").strip() or "text-embedding-004",
        google_embedding_api_key=os.getenv("KARL_GOOGLE_EMBEDDING_API_KEY", "").strip(),
        inference_request_timeout_seconds=max(3, int(os.getenv("KARL_INFERENCE_REQUEST_TIMEOUT_SECONDS", "15"))),
        legacy_contract_pipeline_enabled=_as_bool(os.getenv("KARL_LEGACY_CONTRACT_PIPELINE_ENABLED"), default=False),
        inline_parse_max_bytes=int(os.getenv("KARL_INLINE_PARSE_MAX_BYTES", "1000000")),
        ocr_auto_retry_enabled=_as_bool(os.getenv("KARL_OCR_AUTO_RETRY_ENABLED"), default=True),
        ocr_auto_retry_max_attempts=max(0, int(os.getenv("KARL_OCR_AUTO_RETRY_MAX_ATTEMPTS", "1"))),
        local_auth_token_ttl_seconds=max(300, int(os.getenv("KARL_LOCAL_AUTH_TOKEN_TTL_SECONDS", "43200"))),
        secret_encryption_key=_derive_fernet_key(encryption_seed),
        sentinel_email_from=os.getenv("KARL_SENTINEL_EMAIL_FROM", "sentinel@localhost"),
        sentinel_email_enabled=_as_bool(os.getenv("KARL_SENTINEL_EMAIL_ENABLED"), default=False),
        bootstrap_super_admin_emails=_split_csv(os.getenv("KARL_BOOTSTRAP_SUPER_ADMIN_EMAILS")),
        session_cookie_name=os.getenv("KARL_SESSION_COOKIE_NAME", "karl_session").strip() or "karl_session",
        member_session_idle_seconds=max(3600, int(os.getenv("KARL_MEMBER_SESSION_IDLE_SECONDS", "604800"))),
        union_admin_session_idle_seconds=max(3600, int(os.getenv("KARL_UNION_ADMIN_SESSION_IDLE_SECONDS", "259200"))),
        super_admin_session_idle_seconds=max(300, int(os.getenv("KARL_SUPER_ADMIN_SESSION_IDLE_SECONDS", "3600"))),
    )
