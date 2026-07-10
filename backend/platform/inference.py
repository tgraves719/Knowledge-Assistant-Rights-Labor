"""Union-aware inference configuration loading."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select

from backend.platform.auth import get_current_auth_context
from backend.platform.db import apply_request_context
from backend.platform.models import ProviderConfig, Union


@dataclass
class InferenceConfig:
    provider_name: str
    model_name: str
    api_key: str
    base_url: str | None
    config: dict


async def test_inference_config(config: "InferenceConfig", *, timeout_seconds: int) -> dict:
    started = time.perf_counter()

    async def _run_with_timeout(callable_obj, *args, **kwargs):
        return await asyncio.wait_for(
            asyncio.to_thread(callable_obj, *args, **kwargs),
            timeout=timeout_seconds,
        )

    provider_name = str(config.provider_name or "").strip().lower()
    model_name = str(config.model_name or "").strip()
    runtime_config = dict(config.config or {})
    try:
        if provider_name in {"openrouter", "openai", "openai_compatible"}:
            from openai import OpenAI

            base_url = config.base_url
            if provider_name == "openrouter" and not base_url:
                base_url = "https://openrouter.ai/api/v1"
            default_headers = {}
            http_referer = str(runtime_config.get("http_referer") or runtime_config.get("referer") or "").strip()
            x_title = str(runtime_config.get("x_title") or runtime_config.get("title") or "").strip()
            if http_referer:
                default_headers["HTTP-Referer"] = http_referer
            if x_title:
                default_headers["X-Title"] = x_title
            client = OpenAI(
                api_key=config.api_key,
                base_url=base_url,
                timeout=timeout_seconds,
                default_headers=default_headers or None,
            )
            response = await _run_with_timeout(
                client.chat.completions.create,
                model=model_name,
                messages=[
                    {"role": "system", "content": "Reply with exactly OK."},
                    {"role": "user", "content": "Reply with exactly OK."},
                ],
                max_tokens=8,
                temperature=0,
            )
            content = ""
            if getattr(response, "choices", None):
                content = str(response.choices[0].message.content or "").strip()
            return {
                "ok": True,
                "provider_name": provider_name,
                "model_name": model_name,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "preview": content[:120],
            }
        return {
            "ok": False,
            "provider_name": provider_name,
            "model_name": model_name,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_type": "unsupported_provider",
            "error_message": f"Provider test is not implemented for {provider_name or 'unknown provider'}.",
        }
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "provider_name": provider_name,
            "model_name": model_name,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_type": "timeout",
            "error_message": f"Provider request exceeded {timeout_seconds} seconds.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider_name": provider_name,
            "model_name": model_name,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc) or exc.__class__.__name__,
        }


def load_union_inference_config(container, union_local_id: str | None) -> Optional[InferenceConfig]:
    if container is None or container.session_factory is None or not union_local_id:
        return None

    with container.session_factory() as db:
        apply_request_context(db, get_current_auth_context())
        union = db.scalar(
            select(Union).where((Union.union_local_id == union_local_id) | (Union.slug == union_local_id))
        )
        if union is None:
            return None

        provider = db.scalar(
            select(ProviderConfig).where(
                ProviderConfig.union_id == union.id,
                ProviderConfig.is_active.is_(True),
            )
        )
        if provider is None:
            return None

        config = dict(provider.config_json or {})
        return InferenceConfig(
            provider_name=str(provider.provider_name or "").strip().lower(),
            model_name=str(provider.model_name or "").strip(),
            api_key=container.secret_cipher.decrypt(provider.encrypted_api_key),
            base_url=str(config.get("base_url") or "").strip() or None,
            config=config,
        )
