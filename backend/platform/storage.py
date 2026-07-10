"""Storage abstraction for tenant-scoped document files."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StoredObject:
    key: str
    path: Path
    bytes_size: int


class LocalDiskStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, union_slug: str, filename: str, payload: bytes) -> StoredObject:
        safe_union = union_slug.replace("/", "_")
        target_dir = self.root / safe_union
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        target.write_bytes(payload)
        return StoredObject(key=f"{safe_union}/{filename}", path=target, bytes_size=len(payload))

    def save_json(self, union_slug: str, relative_path: str, payload: dict) -> StoredObject:
        safe_union = union_slug.replace("/", "_")
        target = self.root / safe_union / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        target.write_bytes(encoded)
        return StoredObject(
            key=f"{safe_union}/{relative_path}",
            path=target,
            bytes_size=len(encoded),
        )

    def open(self, key: str) -> Path:
        return self.root / key

    def delete(self, key: str) -> None:
        target = self.root / key
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            return
        if target.exists():
            target.unlink()

    def delete_prefix(self, prefix: str) -> None:
        target = self.root / prefix
        if target.is_file():
            target.unlink()
            return
        if not target.exists():
            return
        shutil.rmtree(target, ignore_errors=True)


class S3CompatibleStorage:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("S3-compatible storage wiring is planned after the local backend foundation.")
