"""Middleware for auth context, DB sessions, security headers, and query governance."""

from __future__ import annotations

import json
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.platform.auth import AuthContext, reset_current_auth_context, set_current_auth_context
from backend.platform.db import apply_request_context, apply_service_bootstrap_context
from backend.platform.models import Role


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        frame_embed_path = (
            request.url.path.startswith("/embed/member-frame/")
            or (request.url.path.startswith("/u/") and request.url.path.endswith("/app"))
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        if frame_embed_path:
            if "X-Frame-Options" in response.headers:
                del response.headers["X-Frame-Options"]
            response.headers["Content-Security-Policy"] = "frame-ancestors *;"
        else:
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Cache-Control"] = "no-store"
        return response


class PlatformContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        container = request.app.state.platform
        requested_tenant_slug = (
            request.headers.get("X-Tenant-Slug")
            or request.headers.get("X-Union-Slug")
            or request.headers.get("X-Union-Local-Id")
        )
        request.state.tenant_slug = requested_tenant_slug
        session = container.session_factory() if container.session_factory else None
        request.state.db = session
        try:
            apply_service_bootstrap_context(session)
            auth = container.auth_adapter.resolve(
                db=session,
                session_cookie=request.cookies.get(container.session_auth.cookie_name),
                authorization=request.headers.get("Authorization"),
                external_auth_id=request.headers.get("X-Auth-User-Id"),
                email=request.headers.get("X-Auth-Email"),
                full_name=request.headers.get("X-Auth-Name"),
                requested_role=request.headers.get("X-Auth-Role"),
                union_slug=requested_tenant_slug,
            )
            if (
                requested_tenant_slug
                and auth.is_authenticated
                and not auth.is_super_admin
                and auth.union_slug
                and auth.union_slug != requested_tenant_slug
            ):
                response = JSONResponse(status_code=403, content={"detail": "Tenant route mismatch."})
                if auth.clear_session_cookie:
                    response.delete_cookie(container.session_auth.cookie_name, path="/")
                return response
            apply_request_context(session, auth)
            request.state.auth_context = auth
            auth_token = set_current_auth_context(auth)
            response = await call_next(request)
            if session is not None:
                session.commit()
            if (
                auth.is_authenticated
                and auth.session_id
                and container.session_factory is not None
                and request.url.path not in {"/api/auth/session/login", "/api/auth/session/logout"}
            ):
                try:
                    with container.session_factory() as touch_session:
                        container.session_auth.touch_session(
                            touch_session,
                            session_id=auth.session_id,
                        )
                        touch_session.commit()
                except Exception:
                    pass
            if auth.clear_session_cookie:
                response.delete_cookie(container.session_auth.cookie_name, path="/")
            return response
        except Exception:
            if session is not None:
                session.rollback()
            raise
        finally:
            if "auth_token" in locals():
                reset_current_auth_context(auth_token)
            if session is not None:
                session.close()


class QueryGovernanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path != "/api/query":
            return await call_next(request)

        container = request.app.state.platform
        transient_session = None
        db = getattr(request.state, "db", None)
        if db is None and container.session_factory:
            transient_session = container.session_factory()
            db = transient_session
            request.state.db = db
        auth: AuthContext = getattr(request.state, "auth_context", None)
        if auth is None:
            apply_service_bootstrap_context(db)
            auth = container.auth_adapter.resolve(
                db=db,
                session_cookie=request.cookies.get(container.session_auth.cookie_name),
                authorization=request.headers.get("Authorization"),
                external_auth_id=request.headers.get("X-Auth-User-Id"),
                email=request.headers.get("X-Auth-Email"),
                full_name=request.headers.get("X-Auth-Name"),
                requested_role=request.headers.get("X-Auth-Role"),
                union_slug=getattr(request.state, "tenant_slug", None),
            )
            apply_request_context(db, auth)
            request.state.auth_context = auth

        try:
            body = await request.body()
            payload = {}
            if body:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    payload = {}

            question = str(payload.get("question") or "")
            estimated_tokens = max(1, len(question.split()))

            guardrail_decision = container.guardrails.scan_prompt(question)
            if not guardrail_decision.allowed:
                container.sentinel.record_event(
                    db,
                    auth,
                    event_type="prompt_blocked",
                    details={"reasons": guardrail_decision.reasons, "risk_score": guardrail_decision.risk_score},
                )
                if transient_session is not None:
                    transient_session.commit()
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "Prompt blocked by guardrails.",
                        "reasons": guardrail_decision.reasons,
                    },
                )

            quota_decision = container.quotas.check_query(db, auth, estimated_tokens)
            if not quota_decision.allowed:
                container.sentinel.record_event(
                    db,
                    auth,
                    event_type="quota_exceeded",
                    details={"reason": quota_decision.reason, "usage": quota_decision.usage_snapshot or {}},
                )
                if transient_session is not None:
                    transient_session.commit()
                return JSONResponse(status_code=429, content={"detail": quota_decision.reason})

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)
            request.state.db = db
            request.state.auth_context = auth

            started = time.perf_counter()
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            if quota_decision.warn:
                container.sentinel.record_event(
                    db,
                    auth,
                    event_type="quota_warning",
                    details={"usage": quota_decision.usage_snapshot or {}, "elapsed_ms": elapsed_ms},
                )

            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            decoded = response_body.decode("utf-8", errors="ignore")
            sanitized_payload = None
            try:
                parsed_payload = json.loads(decoded)
            except Exception:
                parsed_payload = None
            if isinstance(parsed_payload, dict) and "answer" in parsed_payload:
                answer_scan = container.guardrails.redact_sensitive_text(str(parsed_payload.get("answer") or ""))
                if answer_scan.reasons:
                    parsed_payload["answer"] = answer_scan.sanitized_text
                    warning = parsed_payload.get("provider_warning") or {}
                    merged_reasons = sorted(set((warning.get("reasons") or []) + list(answer_scan.reasons or [])))
                    parsed_payload["provider_warning"] = {
                        **warning,
                        "type": "redaction",
                        "reasons": merged_reasons,
                        "message": "Sensitive details were hidden before the answer was returned.",
                    }
                sources = parsed_payload.get("sources")
                if isinstance(sources, list):
                    for source in sources:
                        if not isinstance(source, dict):
                            continue
                        excerpt_scan = container.guardrails.redact_sensitive_text(str(source.get("excerpt") or ""))
                        if excerpt_scan.reasons:
                            source["excerpt"] = excerpt_scan.sanitized_text
                            source["safety_redacted"] = True
                parsed_payload.setdefault("provider_warning", parsed_payload.get("provider_warning"))
                sanitized_payload = parsed_payload
                decoded = json.dumps(parsed_payload)
                response_body = decoded.encode("utf-8")

            output_guard = container.guardrails.scan_output(decoded)
            token_count = max(estimated_tokens, len(decoded.split()))
            estimated_cost = round((token_count / 1000) * 0.01, 6)
            container.quotas.record_usage(
                db,
                auth,
                route=request.url.path,
                token_count=token_count,
                estimated_cost_usd=estimated_cost,
                metadata={"elapsed_ms": elapsed_ms, "status_code": response.status_code},
            )
            if not output_guard.allowed:
                container.sentinel.record_event(
                    db,
                    auth,
                    event_type="output_flagged",
                    details={"reasons": output_guard.reasons, "risk_score": output_guard.risk_score},
                )
                if isinstance(sanitized_payload, dict) and "answer" in sanitized_payload:
                    sanitized_payload["answer"] = (
                        "Karl hid part of the answer because it appeared to contain sensitive information. "
                        "Please ask a union admin or superadmin to review the source document if you need more detail."
                    )
                    sanitized_payload["sources"] = []
                    warning = sanitized_payload.get("provider_warning") or {}
                    sanitized_payload["provider_warning"] = {
                        **warning,
                        "type": "guardrail",
                        "reasons": sorted(set((warning.get("reasons") or []) + list(output_guard.reasons or []))),
                        "message": "Karl returned a safer response because the original answer did not pass output safety checks.",
                    }
                    response_body = json.dumps(sanitized_payload).encode("utf-8")

            headers = dict(response.headers)
            headers.pop("content-length", None)
            headers.pop("Content-Length", None)
            if quota_decision.warn:
                headers["X-KARL-Quota-Warning"] = "true"
            if transient_session is not None:
                transient_session.commit()
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )
        except Exception:
            if transient_session is not None:
                transient_session.rollback()
            raise
        finally:
            if transient_session is not None:
                transient_session.close()
