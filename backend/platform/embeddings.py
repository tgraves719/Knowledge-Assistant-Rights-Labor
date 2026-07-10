"""Embedding provider abstractions for tenant document retrieval."""

from __future__ import annotations

import hashlib
import math
import re
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


class GoogleTextEmbedder(TextEmbedder):
    """Reserved adapter boundary for future Google text embeddings."""

    def __init__(self, *, model_name: str, api_key: str | None = None, dimensions: int = 768):
        self.model_name = str(model_name or "").strip() or "text-embedding-004"
        self.api_key = str(api_key or "").strip()
        self.dimensions = max(32, int(dimensions))

    @property
    def descriptor(self) -> EmbedderDescriptor:
        return EmbedderDescriptor(
            backend="google_text",
            dimensions=self.dimensions,
            model_name=self.model_name,
        )

    def embed(self, text: str) -> list[float]:
        raise RuntimeError(
            "Google text embeddings are not wired into runtime calls yet. "
            "Use the embedder abstraction now; add the provider client later."
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
