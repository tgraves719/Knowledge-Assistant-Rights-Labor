"""Tracking policy resolution and governed telemetry persistence."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.platform.auth import AuthContext
from backend.platform.models import (
    RawQueryRecord,
    RawQueryStorageMode,
    TelemetryEvent,
    TrackingMode,
    TrackingPolicy,
    TrackingPreference,
    TrackingPrivacyMode,
    MemberTrackingChoiceMode,
    Union,
    UserTrackingPreference,
)
from backend.platform.settings import PlatformSettings


@dataclass
class EffectiveTrackingPolicy:
    source: str
    tracking_mode: str
    privacy_mode: str
    member_choice_mode: str
    raw_query_storage_mode: str
    default_member_preference: str
    allow_union_override: bool
    union_override_enabled: bool = False
    member_preference: str = TrackingPreference.SYSTEM_DEFAULT.value

    def allows_category(self, category: str) -> bool:
        normalized = str(category or "").strip().lower()
        if self.tracking_mode == TrackingMode.NONE.value:
            return False
        if self.tracking_mode == TrackingMode.BOTH.value:
            return normalized in {"bug_journey", "usage_ux"}
        if self.tracking_mode == TrackingMode.BUG_AND_JOURNEY.value:
            return normalized == "bug_journey"
        if self.tracking_mode == TrackingMode.USAGE_AND_UX.value:
            return normalized == "usage_ux"
        return False

    def effective_member_level(self) -> str:
        pref = str(self.member_preference or TrackingPreference.SYSTEM_DEFAULT.value)
        default_pref = str(self.default_member_preference or TrackingPreference.BUG_ONLY.value)
        if pref == TrackingPreference.SYSTEM_DEFAULT.value:
            pref = default_pref
        return pref

    def allows_member_category(self, category: str) -> bool:
        if not self.allows_category(category):
            return False
        level = self.effective_member_level()
        if level == TrackingPreference.OFF.value:
            return False
        if level == TrackingPreference.BUG_ONLY.value:
            return str(category or "").strip().lower() == "bug_journey"
        return True

    def raw_query_enabled(self) -> bool:
        return self.raw_query_storage_mode != RawQueryStorageMode.DISABLED.value

    def raw_query_identified(self) -> bool:
        return self.raw_query_storage_mode == RawQueryStorageMode.ENABLED_IDENTIFIED.value

    def is_anonymized(self) -> bool:
        return self.privacy_mode == TrackingPrivacyMode.ANONYMIZED.value

    def to_summary(self) -> dict:
        return {
            "source": self.source,
            "tracking_mode": self.tracking_mode,
            "privacy_mode": self.privacy_mode,
            "member_choice_mode": self.member_choice_mode,
            "raw_query_storage_mode": self.raw_query_storage_mode,
            "default_member_preference": self.default_member_preference,
            "allow_union_override": self.allow_union_override,
            "union_override_enabled": self.union_override_enabled,
            "member_preference": self.member_preference,
        }


class TelemetryService:
    def __init__(self, settings: PlatformSettings):
        self.settings = settings
        self._hmac_key = settings.secret_encryption_key.encode("utf-8")

    def default_policy_values(self) -> dict:
        return {
            "tracking_mode": TrackingMode.BUG_AND_JOURNEY.value,
            "privacy_mode": TrackingPrivacyMode.ANONYMIZED.value,
            "member_choice_mode": MemberTrackingChoiceMode.BUG_ONLY_OR_FULL.value,
            "raw_query_storage_mode": RawQueryStorageMode.DISABLED.value,
            "default_member_preference": TrackingPreference.BUG_ONLY.value,
            "allow_union_override": True,
        }

    def get_or_create_global_policy(self, db: Session) -> TrackingPolicy:
        policy = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id.is_(None)))
        if policy is not None:
            return policy
        defaults = self.default_policy_values()
        policy = TrackingPolicy(
            union_id=None,
            tracking_mode=TrackingMode(defaults["tracking_mode"]),
            privacy_mode=TrackingPrivacyMode(defaults["privacy_mode"]),
            member_choice_mode=MemberTrackingChoiceMode(defaults["member_choice_mode"]),
            raw_query_storage_mode=RawQueryStorageMode(defaults["raw_query_storage_mode"]),
            default_member_preference=TrackingPreference(defaults["default_member_preference"]),
            allow_union_override=bool(defaults["allow_union_override"]),
        )
        db.add(policy)
        db.flush()
        return policy

    def resolve_policy(self, db: Session | None, *, union_id: str | None = None, user_id: str | None = None, is_member: bool = False) -> EffectiveTrackingPolicy:
        defaults = self.default_policy_values()
        if db is None:
            return EffectiveTrackingPolicy(source="default", member_preference=TrackingPreference.SYSTEM_DEFAULT.value, union_override_enabled=False, **defaults)

        global_policy = self.get_or_create_global_policy(db)
        effective = EffectiveTrackingPolicy(
            source="global",
            tracking_mode=global_policy.tracking_mode.value,
            privacy_mode=global_policy.privacy_mode.value,
            member_choice_mode=global_policy.member_choice_mode.value,
            raw_query_storage_mode=global_policy.raw_query_storage_mode.value,
            default_member_preference=global_policy.default_member_preference.value,
            allow_union_override=bool(global_policy.allow_union_override),
            member_preference=TrackingPreference.SYSTEM_DEFAULT.value,
            union_override_enabled=False,
        )

        if union_id and effective.allow_union_override:
            union_policy = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id == union_id))
            if union_policy is not None:
                effective = EffectiveTrackingPolicy(
                    source="union_override",
                    tracking_mode=union_policy.tracking_mode.value,
                    privacy_mode=union_policy.privacy_mode.value,
                    member_choice_mode=union_policy.member_choice_mode.value,
                    raw_query_storage_mode=union_policy.raw_query_storage_mode.value,
                    default_member_preference=union_policy.default_member_preference.value,
                    allow_union_override=bool(union_policy.allow_union_override),
                    member_preference=TrackingPreference.SYSTEM_DEFAULT.value,
                    union_override_enabled=True,
                )

        if is_member and user_id and effective.member_choice_mode != MemberTrackingChoiceMode.NONE.value:
            pref = db.scalar(
                select(UserTrackingPreference).where(
                    UserTrackingPreference.user_id == user_id,
                    UserTrackingPreference.union_id == union_id,
                )
            )
            if pref is not None:
                effective.member_preference = pref.preference.value
        return effective

    def serialize_policy(self, policy: TrackingPolicy | None) -> dict:
        defaults = self.default_policy_values()
        if policy is None:
            return {"scope": "default", **defaults}
        return {
            "scope": "union_override" if policy.union_id else "global",
            "tracking_mode": policy.tracking_mode.value,
            "privacy_mode": policy.privacy_mode.value,
            "member_choice_mode": policy.member_choice_mode.value,
            "raw_query_storage_mode": policy.raw_query_storage_mode.value,
            "default_member_preference": policy.default_member_preference.value,
            "allow_union_override": bool(policy.allow_union_override),
            "union_id": policy.union_id,
        }

    def update_policy(self, db: Session, *, union_id: str | None, payload: dict) -> TrackingPolicy:
        policy = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id == union_id))
        if policy is None:
            defaults = self.default_policy_values()
            policy = TrackingPolicy(
                union_id=union_id,
                tracking_mode=TrackingMode(defaults["tracking_mode"]),
                privacy_mode=TrackingPrivacyMode(defaults["privacy_mode"]),
                member_choice_mode=MemberTrackingChoiceMode(defaults["member_choice_mode"]),
                raw_query_storage_mode=RawQueryStorageMode(defaults["raw_query_storage_mode"]),
                default_member_preference=TrackingPreference(defaults["default_member_preference"]),
                allow_union_override=bool(defaults["allow_union_override"]),
            )
            db.add(policy)
            db.flush()

        if "tracking_mode" in payload:
            policy.tracking_mode = TrackingMode(str(payload["tracking_mode"]).strip().lower())
        if "privacy_mode" in payload:
            policy.privacy_mode = TrackingPrivacyMode(str(payload["privacy_mode"]).strip().lower())
        if "member_choice_mode" in payload:
            policy.member_choice_mode = MemberTrackingChoiceMode(str(payload["member_choice_mode"]).strip().lower())
        if "raw_query_storage_mode" in payload:
            policy.raw_query_storage_mode = RawQueryStorageMode(str(payload["raw_query_storage_mode"]).strip().lower())
        if "default_member_preference" in payload:
            policy.default_member_preference = TrackingPreference(str(payload["default_member_preference"]).strip().lower())
        if "allow_union_override" in payload and union_id is None:
            policy.allow_union_override = bool(payload["allow_union_override"])
        policy.updated_at = datetime.utcnow()
        db.flush()
        return policy

    def clear_union_override(self, db: Session, *, union_id: str) -> None:
        policy = db.scalar(select(TrackingPolicy).where(TrackingPolicy.union_id == union_id))
        if policy is not None:
            db.delete(policy)
            db.flush()

    def set_member_preference(self, db: Session, *, user_id: str, union_id: str | None, preference: str) -> UserTrackingPreference:
        row = db.scalar(select(UserTrackingPreference).where(UserTrackingPreference.user_id == user_id, UserTrackingPreference.union_id == union_id))
        if row is None:
            row = UserTrackingPreference(user_id=user_id, union_id=union_id, preference=TrackingPreference(preference))
            db.add(row)
        else:
            row.preference = TrackingPreference(preference)
            row.updated_at = datetime.utcnow()
        db.flush()
        return row

    def anonymized_user_key(self, *, user_id: str | None, union_id: str | None, session_id: str | None) -> str | None:
        seed = json.dumps({"user_id": user_id, "union_id": union_id, "session_id": session_id}, sort_keys=True)
        if seed == '{"session_id": null, "union_id": null, "user_id": null}':
            return None
        return hmac.new(self._hmac_key, seed.encode("utf-8"), hashlib.sha256).hexdigest()

    def _event_user_fields(self, policy: EffectiveTrackingPolicy, *, user_id: str | None, union_id: str | None, session_id: str | None) -> tuple[str | None, str | None]:
        anon_key = self.anonymized_user_key(user_id=user_id, union_id=union_id, session_id=session_id)
        if policy.is_anonymized():
            return None, anon_key
        return user_id, anon_key

    def record_event(
        self,
        db: Session | None,
        auth: AuthContext | None,
        *,
        category: str,
        event_type: str,
        route: str | None = None,
        metadata: dict | None = None,
        session_id: str | None = None,
        union_id: str | None = None,
        is_member: bool = False,
    ) -> None:
        if db is None:
            return
        effective_union_id = union_id or getattr(auth, "union_id", None)
        effective_user_id = getattr(auth, "user_id", None)
        policy = self.resolve_policy(db, union_id=effective_union_id, user_id=effective_user_id, is_member=is_member)
        if is_member:
            allowed = policy.allows_member_category(category)
        else:
            allowed = policy.allows_category(category)
        if not allowed:
            return

        stored_user_id, anon_key = self._event_user_fields(
            policy,
            user_id=effective_user_id,
            union_id=effective_union_id,
            session_id=session_id,
        )
        db.add(
            TelemetryEvent(
                union_id=effective_union_id,
                user_id=stored_user_id,
                session_id=session_id,
                route=route,
                category=str(category),
                event_type=str(event_type),
                anonymized_user_key=anon_key,
                metadata_json={
                    **(metadata or {}),
                    "policy": policy.to_summary(),
                },
            )
        )

    def record_query(
        self,
        db: Session | None,
        auth: AuthContext | None,
        *,
        question_text: str,
        answer_text: str,
        route: str,
        session_id: str | None,
        provider_name: str | None,
        metadata: dict | None = None,
        is_member: bool = True,
    ) -> None:
        if db is None:
            return
        effective_union_id = getattr(auth, "union_id", None)
        effective_user_id = getattr(auth, "user_id", None)
        policy = self.resolve_policy(db, union_id=effective_union_id, user_id=effective_user_id, is_member=is_member)

        if policy.allows_member_category("usage_ux"):
            self.record_event(
                db,
                auth,
                category="usage_ux",
                event_type="query_completed",
                route=route,
                metadata={
                    **(metadata or {}),
                    "question_length": len(str(question_text or "")),
                    "answer_length": len(str(answer_text or "")),
                    "provider_name": provider_name,
                },
                session_id=session_id,
                is_member=is_member,
            )

        if not policy.raw_query_enabled():
            return
        stored_user_id, anon_key = self._event_user_fields(
            policy,
            user_id=effective_user_id if policy.raw_query_identified() else None,
            union_id=effective_union_id,
            session_id=session_id,
        )
        db.add(
            RawQueryRecord(
                union_id=effective_union_id,
                user_id=stored_user_id if policy.raw_query_identified() else None,
                session_id=session_id,
                route=route,
                anonymized_user_key=anon_key,
                question_text=str(question_text or "") if policy.raw_query_enabled() else None,
                answer_text=str(answer_text or "") if policy.raw_query_enabled() else None,
                provider_name=provider_name,
                metadata_json={
                    **(metadata or {}),
                    "policy": policy.to_summary(),
                },
            )
        )

    def bootstrap_summary(self, db: Session | None, *, union_id: str | None, user_id: str | None = None, is_member: bool = False) -> dict:
        return self.resolve_policy(db, union_id=union_id, user_id=user_id, is_member=is_member).to_summary()
