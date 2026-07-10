from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.platform.local_auth import LocalAuthService
from backend.platform.models import Role, Union, UnionMembership, User
from backend.platform.settings import get_platform_settings


def main() -> None:
    database_url = os.getenv("KARL_POSTGRES_URL", "").strip()
    if not database_url:
        raise SystemExit("KARL_POSTGRES_URL is required.")

    username = os.getenv("KARL_SUPERADMIN_USERNAME", "karl_superadmin").strip() or "karl_superadmin"
    password = os.getenv("KARL_SUPERADMIN_PASSWORD", "demo_password").strip() or "demo_password"
    email = os.getenv("KARL_SUPERADMIN_EMAIL", "karl_superadmin@example.com").strip() or "karl_superadmin@example.com"
    full_name = os.getenv("KARL_SUPERADMIN_NAME", "Karl Superadmin").strip() or "Karl Superadmin"

    settings = get_platform_settings()
    engine = create_engine(database_url, future=True)
    local_auth = LocalAuthService(
        secret_key=settings.secret_encryption_key,
        token_ttl_seconds=settings.local_auth_token_ttl_seconds,
    )

    with Session(engine, future=True) as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, full_name=full_name, is_active=True)
            db.add(user)
            db.flush()
        else:
            user.full_name = full_name
            user.is_active = True

        union = db.scalar(select(Union).order_by(Union.created_at.asc()))
        if union is not None:
            membership = db.scalar(
                select(UnionMembership).where(
                    UnionMembership.user_id == user.id,
                    UnionMembership.union_id == union.id,
                )
            )
            if membership is None:
                db.add(
                    UnionMembership(
                        union_id=union.id,
                        user_id=user.id,
                        role=Role.SUPER_ADMIN,
                        is_active=True,
                    )
                )
            else:
                membership.role = Role.SUPER_ADMIN
                membership.is_active = True

        local_auth.create_or_update_credential(db, user=user, username=username, password=password)
        db.commit()
        print(
            {
                "user_id": user.id,
                "username": username,
                "password": password,
                "email": email,
                "union_id": union.id if union is not None else None,
            }
        )


if __name__ == "__main__":
    main()
