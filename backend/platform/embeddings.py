"""Embedding provider abstractions for tenant document retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)


def _normalize_vector(values: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude <= 0:
        return values
    return [value / magnitude for value in values]


@dataclass(frozen=True)
class EmbedderDescriptor:
    backend: str
    dimensions: int
    model_name: str | None = None


class TextEmbedder:
    """Interface for text embedding providers."""

    @property
    def descriptor(self) -> EmbedderDescriptor:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class DeterministicTextEmbedder(TextEmbedder):
    """Cheap, offline-safe hashing embedder for tenant document search."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = max(32, int(dimensions))

    @property
    def descriptor(self) -> EmbedderDescriptor:
        return EmbedderDescriptor(
            backend="deterministic",
            dimensions=self.dimensions,
            model_name="hashing_v1",
        )

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(str(text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize_vector(vector)


class EmbeddingProviderError(RuntimeError):
    """Raised when the embedding provider cannot produce a usable vector."""


class GoogleTextEmbedder(TextEmbedder):
    """Google text embeddings via the google-genai SDK.

    Vectors are requested at exactly ``dimensions`` using the API's
    output_dimensionality parameter, so the returned width always matches the
    pgvector column. text-embedding-004 is natively 768; asking for fewer
    truncates a Matryoshka representation, which then needs renormalizing —
    the API does not renormalize truncated output for you.
    """

    # Retries cover transient 429/5xx only. Failing loudly beats returning a
    # zero or random vector: a silently wrong embedding is indistinguishable
    # from a working one until retrieval quality quietly collapses.
    _MAX_ATTEMPTS = 4
    _BACKOFF_SECONDS = (1.0, 3.0, 8.0)

    def __init__(self, *, model_name: str, api_key: str | None = None, dimensions: int = 768):
        self.model_name = str(model_name or "").strip() or "text-embedding-004"
        self.api_key = str(api_key or "").strip()
        self.dimensions = max(32, int(dimensions))
        self._client = None

    @property
    def descriptor(self) -> EmbedderDescriptor:
        return EmbedderDescriptor(
            backend="google_text",
            dimensions=self.dimensions,
            model_name=self.model_name,
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise EmbeddingProviderError(
                "KARL_GOOGLE_EMBEDDING_API_KEY is not set; cannot embed with backend 'google'."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise EmbeddingProviderError(
                "google-genai is not installed; required for embedding backend 'google'."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        payload = str(text or "").strip()
        if not payload:
            # An all-zero vector is a legitimate "no content" answer here and
            # keeps chunk indexes aligned; normalizing it would divide by zero.
            return [0.0] * self.dimensions

        client = self._get_client()
        from google.genai import types as genai_types

        last_error: Exception | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=payload,
                    config=genai_types.EmbedContentConfig(
                        output_dimensionality=self.dimensions
                    ),
                )
            except Exception as exc:
                last_error = exc
                if attempt < len(self._BACKOFF_SECONDS) and _is_retryable(exc):
                    time.sleep(self._BACKOFF_SECONDS[attempt])
                    continue
                raise EmbeddingProviderError(
                    f"Google embedding request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            values = _extract_embedding_values(response)
            if len(values) != self.dimensions:
                # Width drift silently corrupts the index, so refuse it rather
                # than padding or truncating behind the caller's back.
                raise EmbeddingProviderError(
                    f"Google embedding returned {len(values)} dimensions, expected {self.dimensions}. "
                    "Check KARL_EMBEDDING_DIMENSIONS against the model and the pgvector column width."
                )
            return _normalize_vector(values)

        raise EmbeddingProviderError(
            f"Google embedding request failed: {last_error}"
        )


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "resource_exhausted", "timeout", "deadline", "unavailable", "500", "503")
    )


def _extract_embedding_values(response) -> list[float]:
    """Pull the float list out of an embed_content response.

    The SDK has moved this around between versions, so accept the shapes it
    has used rather than pinning to one and breaking on upgrade.
    """
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None) or getattr(first, "value", None)
        if values:
            return [float(v) for v in values]
    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None) or embedding
        if values:
            return [float(v) for v in values]
    raise EmbeddingProviderError(
        f"Could not read embedding values from response of type {type(response).__name__}."
    )


def build_text_embedder(
    *,
    backend: str,
    dimensions: int,
    google_model_name: str | None = None,
    google_api_key: str | None = None,
) -> TextEmbedder:
    normalized = str(backend or "deterministic").strip().lower() or "deterministic"
    if normalized in {"deterministic", "local", "hash"}:
        return DeterministicTextEmbedder(dimensions=dimensions)
    if normalized in {"google", "google_text"}:
        return GoogleTextEmbedder(
            model_name=google_model_name or "text-embedding-004",
            api_key=google_api_key,
            dimensions=dimensions,
        )
    raise ValueError(f"Unsupported embedding backend '{backend}'.")
