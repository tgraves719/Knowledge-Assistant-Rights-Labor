"""Application service container for production platform components."""

from __future__ import annotations

from functools import cached_property

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.platform.auth import HeaderAuthAdapter
from backend.platform.chat_history import ChatHistoryStore
from backend.platform.crypto import SecretCipher
from backend.platform.db import apply_rls_policies, create_all, create_session_factory, session_scope
from backend.platform.embeddings import build_text_embedder
from backend.platform.guardrails import GuardrailService
from backend.platform.ingestion import IngestionService
from backend.platform.local_auth import LocalAuthService
from backend.platform.parsing import LiteParseDocumentParser, ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.session_auth import SessionAuthService
from backend.platform.sentinel import SentinelService
from backend.platform.settings import PlatformSettings, get_platform_settings
from backend.platform.storage import LocalDiskStorage
from backend.platform.telemetry import TelemetryService


class ServiceContainer:
    """Lightweight bootstrap shell with lazy heavy services."""

    def __init__(
        self,
        *,
        settings: PlatformSettings,
        engine: Engine | None,
        session_factory: sessionmaker[Session] | None,
        auth_adapter: HeaderAuthAdapter | None = None,
        guardrails: GuardrailService | None = None,
        quotas: QuotaService | None = None,
        sentinel: SentinelService | None = None,
        secret_cipher: SecretCipher | None = None,
        storage: LocalDiskStorage | None = None,
        chat_history: ChatHistoryStore | None = None,
        retrieval: TenantRetrievalService | None = None,
        document_parsers: ParserRegistry | None = None,
        ingestion: IngestionService | None = None,
        local_auth: LocalAuthService | None = None,
        session_auth: SessionAuthService | None = None,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.session_factory = session_factory
        self._auth_adapter_override = auth_adapter
        self._guardrails_override = guardrails
        self._quotas_override = quotas
        self._sentinel_override = sentinel
        self._secret_cipher_override = secret_cipher
        self._storage_override = storage
        self._chat_history_override = chat_history
        self._retrieval_override = retrieval
        self._document_parsers_override = document_parsers
        self._ingestion_override = ingestion
        self._local_auth_override = local_auth
        self._session_auth_override = session_auth
        self._telemetry_override = telemetry

    @cached_property
    def local_auth(self) -> LocalAuthService:
        if self._local_auth_override is not None:
            return self._local_auth_override
        return LocalAuthService(
            secret_key=self.settings.secret_encryption_key,
            token_ttl_seconds=self.settings.local_auth_token_ttl_seconds,
        )

    @cached_property
    def auth_adapter(self) -> HeaderAuthAdapter:
        if self._auth_adapter_override is not None:
            return self._auth_adapter_override
        return HeaderAuthAdapter(self.settings, local_auth=self.local_auth, session_auth=self.session_auth)

    @cached_property
    def session_auth(self) -> SessionAuthService:
        if self._session_auth_override is not None:
            return self._session_auth_override
        return SessionAuthService(
            secret_key=self.settings.secret_encryption_key,
            cookie_name=self.settings.session_cookie_name,
            member_idle_seconds=self.settings.member_session_idle_seconds,
            union_admin_idle_seconds=self.settings.union_admin_session_idle_seconds,
            super_admin_idle_seconds=self.settings.super_admin_session_idle_seconds,
        )

    @cached_property
    def guardrails(self) -> GuardrailService:
        if self._guardrails_override is not None:
            return self._guardrails_override
        return GuardrailService(token_limit=self.settings.query_token_limit)

    @cached_property
    def quotas(self) -> QuotaService:
        if self._quotas_override is not None:
            return self._quotas_override
        return QuotaService(self.settings)

    @cached_property
    def sentinel(self) -> SentinelService:
        if self._sentinel_override is not None:
            return self._sentinel_override
        return SentinelService(self.settings)

    @cached_property
    def secret_cipher(self) -> SecretCipher:
        if self._secret_cipher_override is not None:
            return self._secret_cipher_override
        return SecretCipher(self.settings.secret_encryption_key)

    @cached_property
    def storage(self) -> LocalDiskStorage:
        if self._storage_override is not None:
            return self._storage_override
        return LocalDiskStorage(self.settings.local_storage_root)

    @cached_property
    def chat_history(self) -> ChatHistoryStore:
        if self._chat_history_override is not None:
            return self._chat_history_override
        return ChatHistoryStore(self.session_factory)

    @cached_property
    def retrieval(self) -> TenantRetrievalService:
        if self._retrieval_override is not None:
            return self._retrieval_override
        return TenantRetrievalService(
            embedding_dimensions=self.settings.embedding_dimensions,
            embedder=build_text_embedder(
                backend=self.settings.embedding_backend,
                dimensions=self.settings.embedding_dimensions,
                google_model_name=self.settings.google_embedding_model,
                google_api_key=self.settings.google_embedding_api_key,
            ),
        )

    @cached_property
    def document_parsers(self) -> ParserRegistry:
        if self._document_parsers_override is not None:
            return self._document_parsers_override
        parser_backend = self.settings.document_parser_backend
        parser_candidates = []
        if parser_backend in {"auto", "plain_text", "text"}:
            parser_candidates.append(PlainTextDocumentParser())
        if parser_backend in {"auto", "liteparse"}:
            parser_candidates.append(LiteParseDocumentParser(self.settings.liteparse_executable))
        return ParserRegistry(parser_candidates)

    @cached_property
    def ingestion(self) -> IngestionService:
        if self._ingestion_override is not None:
            return self._ingestion_override
        return IngestionService(
            storage=self.storage,
            retrieval=self.retrieval,
            parsers=self.document_parsers,
            guardrails=self.guardrails,
            sentinel=self.sentinel,
            inline_parse_max_bytes=self.settings.inline_parse_max_bytes,
            ocr_auto_retry_enabled=self.settings.ocr_auto_retry_enabled,
            ocr_auto_retry_max_attempts=self.settings.ocr_auto_retry_max_attempts,
        )

    @cached_property
    def telemetry(self) -> TelemetryService:
        if self._telemetry_override is not None:
            return self._telemetry_override
        return TelemetryService(self.settings)


def build_service_container() -> ServiceContainer:
    settings = get_platform_settings()
    engine, session_factory = create_session_factory(settings)
    if engine is not None and settings.auto_create_tables:
        create_all(engine)
        if settings.apply_rls_policies:
            apply_rls_policies(engine)

    return ServiceContainer(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
    )


def get_session(container: ServiceContainer):
    return session_scope(container.session_factory)
