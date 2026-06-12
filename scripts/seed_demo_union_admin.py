#!/usr/bin/env python3
"""Seed a demo union admin with local auth credentials."""

from __future__ import annotations

import argparse

from sqlalchemy import select

from backend.platform.models import Role, Union, UnionMembership, User
from backend.platform.service_container import build_service_container


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a demo union admin with local username/password auth.")
    parser.add_argument("--union-slug", default="demo-local")
    parser.add_argument("--union-name", default="Demo Local")
    parser.add_argument("--union-local-id", default="demo-local")
    parser.add_argument("--username", default="union_demo")
    parser.add_argument("--password", default="demo_password")
    parser.add_argument("--email", default="union_demo@example.com")
    parser.add_argument("--full-name", default="Union Demo Admin")
    args = parser.parse_args()

    container = build_service_container()
    if container.session_factory is None:
        raise SystemExit("Database is not configured. Set KARL_POSTGRES_URL first.")

    with container.session_factory() as db:
        union = db.scalar(select(Union).where((Union.slug == args.union_slug) | (Union.union_local_id == args.union_local_id)))
        if union is None:
            union = Union(slug=args.union_slug, name=args.union_name, union_local_id=args.union_local_id)
            db.add(union)
            db.flush()

        user = db.scalar(select(User).where(User.email == args.email))
        if user is None:
            user = User(email=args.email, full_name=args.full_name)
            db.add(user)
            db.flush()

        membership = db.scalar(
            select(UnionMembership).where(
                UnionMembership.union_id == union.id,
                UnionMembership.user_id == user.id,
            )
        )
        if membership is None:
            membership = UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN)
            db.add(membership)
        else:
            membership.role = Role.UNION_ADMIN
            membership.is_active = True

        container.local_auth.create_or_update_credential(
            db,
            user=user,
            username=args.username,
            password=args.password,
        )
        db.commit()

        print(
            {
                "union_id": union.id,
                "union_slug": union.slug,
                "user_id": user.id,
                "username": args.username,
                "email": user.email,
                "role": Role.UNION_ADMIN.value,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
