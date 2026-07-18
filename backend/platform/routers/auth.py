"""Local auth endpoints for demo and fallback environments."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from backend.platform.auth import AuthContext
from backend.platform.deps import get_auth_context, get_container, get_db
from backend.platform.local_auth import LocalAuthError
from backend.platform.models import (
    InviteCode,
    LocalAuthCredential,
    MemberTrackingChoiceMode,
    Role,
    TrackingPreference,
    Union,
    UnionMembership,
    User,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LocalLoginRequest(BaseModel):
    username: str
    password: str
    union_slug: str | None = None


class SessionRegisterRequest(BaseModel):
    username: str
    password: str
    email: str
    full_name: str
    union_slug: str | None = None


class SessionPreferenceUpdateRequest(BaseModel):
    tracking_preference: str


class SessionJoinRequest(BaseModel):
    code: str
    username: str
    password: str
    email: str
    full_name: str


def _resolve_login_target(db, container, *, user, union_slug: str | None):
    memberships = db.scalars(
        select(UnionMembership).where(
            UnionMembership.user_id == user.id,
            UnionMembership.is_active.is_(True),
        )
    ).all()
    membership_by_union_id = {membership.union_id: membership for membership in memberships}
    normalized_email = str(user.email or "").strip().lower()
    is_bootstrap_super_admin = normalized_email in {
        item.strip().lower() for item in container.settings.bootstrap_super_admin_emails if item.strip()
    }

    if union_slug:
        union = db.scalar(select(Union).where((Union.slug == union_slug) | (Union.union_local_id == union_slug)))
        if union is None:
            raise LocalAuthError("Union is not available for this user.")
        membership = membership_by_union_id.get(union.id)
        if membership is None and not is_bootstrap_super_admin:
            raise LocalAuthError("Union is not available for this user.")
        role = Role.SUPER_ADMIN.value if is_bootstrap_super_admin else membership.role.value
        return union, role

    if is_bootstrap_super_admin:
        return None, Role.SUPER_ADMIN.value

    super_admin_membership = next((membership for membership in memberships if membership.role == Role.SUPER_ADMIN), None)
    if super_admin_membership is not None:
        return None, Role.SUPER_ADMIN.value

    if len(memberships) == 1:
        membership = memberships[0]
        union = db.get(Union, membership.union_id)
        if union is None:
            raise LocalAuthError("Union is not available for this user.")
        return union, membership.role.value

    raise LocalAuthError("Union selection is required for multi-union users.")


def _clear_session_cookie(response: Response, container) -> None:
    response.delete_cookie(container.session_auth.cookie_name, path="/")


def _set_session_cookie(response: Response, request: Request, container, *, session_secret: str) -> None:
    response.set_cookie(
        key=container.session_auth.cookie_name,
        value=session_secret,
        httponly=True,
        samesite="lax",
        secure=bool(request.url.scheme == "https"),
        path="/",
    )


@router.post("/local/login")
def login_local(payload: LocalLoginRequest, request: Request, x_tenant_slug: str | None = Header(default=None)):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    effective_union_slug = payload.union_slug or x_tenant_slug
    try:
        user, union = container.local_auth.authenticate(
            db,
            username=payload.username,
            password=payload.password,
            union_slug=effective_union_slug,
        )
    except LocalAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    membership = None
    if union is not None:
        membership = db.scalar(
            select(UnionMembership).where(
                UnionMembership.user_id == user.id,
                UnionMembership.union_id == union.id,
                UnionMembership.is_active.is_(True),
            )
        )
    role = membership.role.value if membership is not None else Role.USER.value
    token = container.local_auth.issue_token(user_id=user.id, union_slug=union.slug if union else None)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": role,
            "union_id": union.id if union else None,
            "union_slug": union.slug if union else None,
        },
    }


@router.post("/session/login")
def login_session(
    payload: LocalLoginRequest,
    request: Request,
    response: Response,
    x_tenant_slug: str | None = Header(default=None),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    effective_union_slug = payload.union_slug or x_tenant_slug
    try:
        user = container.local_auth.authenticate_credentials(
            db,
            username=payload.username,
            password=payload.password,
        )
        union, role = _resolve_login_target(db, container, user=user, union_slug=effective_union_slug)
        _, session_secret = container.session_auth.create_session(
            db,
            user=user,
            union=union,
            role=role,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except LocalAuthError as exc:
        failed_union = db.scalar(select(Union).where((Union.slug == effective_union_slug) | (Union.union_local_id == effective_union_slug))) if effective_union_slug else None
        container.telemetry.record_event(
            db,
            None,
            category="bug_journey",
            event_type="session_login_failed",
            route="/api/auth/session/login",
            metadata={
                "reason": str(exc),
                "requested_union_slug": effective_union_slug,
            },
            union_id=failed_union.id if failed_union is not None else None,
            is_member=False,
        )
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    _set_session_cookie(response, request, container, session_secret=session_secret)
    container.telemetry.record_event(
        db,
        AuthContext(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=role,
            union_id=union.id if union else None,
            union_slug=union.slug if union else None,
            source="session_login",
            is_authenticated=True,
        ),
        category="bug_journey",
        event_type="session_login_success",
        route="/api/auth/session/login",
        metadata={"user_role": role},
        is_member=role == Role.USER.value,
    )
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": role,
            "union_id": union.id if union else None,
            "union_slug": union.slug if union else None,
        },
    }


@router.post("/session/register")
def register_session(
    payload: SessionRegisterRequest,
    request: Request,
    response: Response,
    x_tenant_slug: str | None = Header(default=None),
):
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    effective_union_slug = (payload.union_slug or x_tenant_slug or "").strip()
    if not effective_union_slug:
        raise HTTPException(status_code=400, detail="A union workspace is required to create an account.")

    union = db.scalar(select(Union).where((Union.slug == effective_union_slug) | (Union.union_local_id == effective_union_slug)))
    if union is None or not union.is_active:
        raise HTTPException(status_code=404, detail="Union workspace not found.")

    username = str(payload.username or "").strip().lower()
    email = str(payload.email or "").strip().lower()
    full_name = str(payload.full_name or "").strip()
    password = str(payload.password or "")

    if not username or not email or not full_name or not password:
        raise HTTPException(status_code=400, detail="Full name, email, username, and password are required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing_credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.username == username))
    if existing_credential is not None:
        raise HTTPException(status_code=409, detail="That username is already in use.")

    existing_user = db.scalar(select(User).where(User.email == email))
    if existing_user is not None:
        existing_membership = db.scalar(
            select(UnionMembership).where(
                UnionMembership.user_id == existing_user.id,
                UnionMembership.union_id == union.id,
            )
        )
        if existing_membership is not None:
            raise HTTPException(status_code=409, detail="An account already exists for that email in this union.")
        user = existing_user
        user.full_name = full_name
        user.is_active = True
    else:
        user = User(email=email, full_name=full_name, is_active=True)
        db.add(user)
        db.flush()

    membership = db.scalar(
        select(UnionMembership).where(
            UnionMembership.user_id == user.id,
            UnionMembership.union_id == union.id,
        )
    )
    if membership is None:
        membership = UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER, is_active=True)
        db.add(membership)
    else:
        membership.role = Role.USER
        membership.is_active = True

    try:
        container.local_auth.create_or_update_credential(
            db,
            user=user,
            username=username,
            password=password,
        )
        _, session_secret = container.session_auth.create_session(
            db,
            user=user,
            union=union,
            role=Role.USER.value,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except LocalAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _set_session_cookie(response, request, container, session_secret=session_secret)
    container.telemetry.record_event(
        db,
        AuthContext(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=Role.USER.value,
            union_id=union.id,
            union_slug=union.slug,
            source="session_register",
            is_authenticated=True,
        ),
        category="bug_journey",
        event_type="session_register_success",
        route="/api/auth/session/register",
        metadata={"user_role": Role.USER.value},
        is_member=True,
    )
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": Role.USER.value,
            "union_id": union.id,
            "union_slug": union.slug,
        },
    }


@router.post("/session/logout")
def logout_session(request: Request, response: Response):
    db = get_db(request)
    container = get_container(request)
    auth = get_auth_context(request)
    if db is not None and auth.is_authenticated:
        container.telemetry.record_event(
            db,
            auth,
            category="bug_journey",
            event_type="session_logout",
            route="/api/auth/session/logout",
            is_member=auth.role == Role.USER.value,
        )
    if db is not None and auth.session_id:
        container.session_auth.revoke_session(db, auth.session_id)
    _clear_session_cookie(response, container)
    return {"ok": True}


@router.get("/session/me")
def session_me(request: Request, response: Response):
    auth: AuthContext = get_auth_context(request)
    db = get_db(request)
    if auth.clear_session_cookie:
        _clear_session_cookie(response, get_container(request))
    tracking = get_container(request).telemetry.bootstrap_summary(
        db,
        union_id=auth.union_id,
        user_id=auth.user_id,
        is_member=auth.role == Role.USER.value,
    )
    return {
        "authenticated": auth.is_authenticated,
        "user_id": auth.user_id,
        "email": auth.email,
        "full_name": auth.full_name,
        "role": auth.role,
        "union_id": auth.union_id,
        "union_slug": auth.union_slug,
        "source": auth.source,
        "tracking": tracking,
    }


@router.get("/session/preferences")
def get_session_preferences(request: Request):
    auth: AuthContext = get_auth_context(request)
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
    db = get_db(request)
    container = get_container(request)
    tracking = container.telemetry.resolve_policy(
        db,
        union_id=auth.union_id,
        user_id=auth.user_id,
        is_member=auth.role == Role.USER.value,
    )
    return {
        "tracking_preference": tracking.member_preference,
        "effective_policy": tracking.to_summary(),
        "member_choice_available": tracking.member_choice_mode != MemberTrackingChoiceMode.NONE.value,
    }


@router.put("/session/preferences")
def update_session_preferences(payload: SessionPreferenceUpdateRequest, request: Request):
    auth: AuthContext = get_auth_context(request)
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    effective = container.telemetry.resolve_policy(
        db,
        union_id=auth.union_id,
        user_id=auth.user_id,
        is_member=auth.role == Role.USER.value,
    )
    if effective.member_choice_mode == MemberTrackingChoiceMode.NONE.value:
        raise HTTPException(status_code=403, detail="Tracking preference changes are not available for this workspace.")
    preference = str(payload.tracking_preference or "").strip().lower()
    allowed = {TrackingPreference.SYSTEM_DEFAULT.value, TrackingPreference.BUG_ONLY.value, TrackingPreference.FULL.value}
    if effective.member_choice_mode == MemberTrackingChoiceMode.FULL_OPT_OUT.value:
        allowed.add(TrackingPreference.OFF.value)
    if preference not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported tracking preference.")
    row = container.telemetry.set_member_preference(
        db,
        user_id=auth.user_id,
        union_id=auth.union_id,
        preference=preference,
    )
    return {"tracking_preference": row.preference.value}


@router.get("/me")
def auth_me(request: Request):
    auth: AuthContext = get_auth_context(request)
    return {
        "authenticated": auth.is_authenticated,
        "user_id": auth.user_id,
        "email": auth.email,
        "full_name": auth.full_name,
        "role": auth.role,
        "union_id": auth.union_id,
        "union_slug": auth.union_slug,
        "source": auth.source,
    }


def _resolve_active_invite(db, code: str) -> tuple[InviteCode, Union]:
    """Resolve a join code to an active invite + union, or raise a join-flow HTTP error.

    404 for unknown/retired workspaces (indistinguishable from a bad code on purpose);
    410 for codes that once worked but are now closed (revoked, expired, or full).
    """
    normalized = str(code or "").strip()
    if not normalized:
        raise HTTPException(status_code=404, detail="This join code is not recognized.")
    invite = db.scalar(select(InviteCode).where(InviteCode.code == normalized))
    if invite is None:
        raise HTTPException(status_code=404, detail="This join code is not recognized.")
    if invite.revoked_at is not None:
        raise HTTPException(status_code=410, detail="This join code has been deactivated.")
    if invite.expires_at is not None and invite.expires_at <= datetime.utcnow():
        raise HTTPException(status_code=410, detail="This join code has expired.")
    if invite.max_uses is not None and invite.use_count >= invite.max_uses:
        raise HTTPException(status_code=410, detail="This join code has reached its member limit.")
    union = db.get(Union, invite.union_id)
    if union is None or not union.is_active:
        raise HTTPException(status_code=404, detail="This join code is not recognized.")
    return invite, union


@router.get("/join/{code}")
def join_code_info(code: str, request: Request):
    """Public pre-enrollment check for a QR join code (no auth required)."""
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    invite, union = _resolve_active_invite(db, code)
    return {
        "valid": True,
        "code": invite.code,
        "label": invite.label,
        "contract_id": invite.contract_id,
        "union": {"slug": union.slug, "name": union.name},
    }


@router.post("/session/join")
def join_session(payload: SessionJoinRequest, request: Request, response: Response):
    """Enroll a member via a QR invite code and start their session.

    Mirrors /session/register, but the union is resolved from the invite (never
    from client-supplied tenant fields) and the resulting session is attributed
    to the invite for per-placement usage visibility.
    """
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    invite, union = _resolve_active_invite(db, payload.code)

    username = str(payload.username or "").strip().lower()
    email = str(payload.email or "").strip().lower()
    full_name = str(payload.full_name or "").strip()
    password = str(payload.password or "")

    if not username or not email or not full_name or not password:
        raise HTTPException(status_code=400, detail="Full name, email, username, and password are required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing_credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.username == username))
    if existing_credential is not None:
        raise HTTPException(status_code=409, detail="That username is already in use.")

    existing_user = db.scalar(select(User).where(User.email == email))
    if existing_user is not None:
        existing_membership = db.scalar(
            select(UnionMembership).where(
                UnionMembership.user_id == existing_user.id,
                UnionMembership.union_id == union.id,
            )
        )
        if existing_membership is not None:
            raise HTTPException(status_code=409, detail="An account already exists for that email in this union.")
        user = existing_user
        user.full_name = full_name
        user.is_active = True
    else:
        user = User(email=email, full_name=full_name, is_active=True)
        db.add(user)
        db.flush()

    membership = db.scalar(
        select(UnionMembership).where(
            UnionMembership.user_id == user.id,
            UnionMembership.union_id == union.id,
        )
    )
    if membership is None:
        membership = UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER, is_active=True)
        db.add(membership)
    else:
        membership.role = Role.USER
        membership.is_active = True

    try:
        container.local_auth.create_or_update_credential(
            db,
            user=user,
            username=username,
            password=password,
        )
        auth_session, session_secret = container.session_auth.create_session(
            db,
            user=user,
            union=union,
            role=Role.USER.value,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except LocalAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    auth_session.invite_code_id = invite.id
    invite.use_count = int(invite.use_count or 0) + 1
    invite.updated_at = datetime.utcnow()

    _set_session_cookie(response, request, container, session_secret=session_secret)
    container.telemetry.record_event(
        db,
        AuthContext(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=Role.USER.value,
            union_id=union.id,
            union_slug=union.slug,
            source="session_join",
            is_authenticated=True,
        ),
        category="bug_journey",
        event_type="session_join_success",
        route="/api/auth/session/join",
        metadata={
            "user_role": Role.USER.value,
            "invite_code_id": invite.id,
            "invite_code": invite.code,
            "invite_label": invite.label,
        },
        is_member=True,
    )
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": Role.USER.value,
            "union_id": union.id,
            "union_slug": union.slug,
        },
        "invite": {"code": invite.code, "label": invite.label, "contract_id": invite.contract_id},
    }


class SessionGuestJoinRequest(BaseModel):
    code: str


@router.post("/session/join-guest")
def join_session_guest(payload: SessionGuestJoinRequest, request: Request, response: Response):
    """Zero-friction QR enrollment: scanning a valid code starts a member session directly.

    No credentials are collected — the invite code itself is the admission ticket. Each join
    creates a lightweight guest identity so quotas, tracking preferences, and telemetry keep
    working per-member, and the session is attributed to the invite so a misused QR placement
    can be shut off (and optionally disconnected) individually from the admin console.
    """
    db = get_db(request)
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")
    container = get_container(request)
    invite, union = _resolve_active_invite(db, payload.code)

    import uuid as _uuid

    guest_tag = _uuid.uuid4().hex[:12]
    user = User(
        email=f"guest-{guest_tag}@join.karl.invalid",
        full_name="Union member",
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER, is_active=True))

    auth_session, session_secret = container.session_auth.create_session(
        db,
        user=user,
        union=union,
        role=Role.USER.value,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
    )
    auth_session.invite_code_id = invite.id
    invite.use_count = int(invite.use_count or 0) + 1
    invite.updated_at = datetime.utcnow()

    _set_session_cookie(response, request, container, session_secret=session_secret)
    container.telemetry.record_event(
        db,
        AuthContext(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=Role.USER.value,
            union_id=union.id,
            union_slug=union.slug,
            source="session_join_guest",
            is_authenticated=True,
        ),
        category="bug_journey",
        event_type="session_join_guest_success",
        route="/api/auth/session/join-guest",
        metadata={
            "user_role": Role.USER.value,
            "invite_code_id": invite.id,
            "invite_code": invite.code,
            "invite_label": invite.label,
        },
        is_member=True,
    )
    return {
        "authenticated": True,
        "guest": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": Role.USER.value,
            "union_id": union.id,
            "union_slug": union.slug,
        },
        "invite": {"code": invite.code, "label": invite.label, "contract_id": invite.contract_id},
    }
