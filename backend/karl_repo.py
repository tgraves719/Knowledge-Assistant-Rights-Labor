"""Read-only explorer over the KARL git repository.

Powers the "About KARL" document explorer in the app. Only files that git
tracks are exposed, so anything in .gitignore (secrets, .env, build caches) is
excluded automatically, and the git allowlist also blocks path traversal.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Cap the size of a single file we will read into memory / ship to the browser.
_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

_MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown"}

# Extensions we are confident are binary; short-circuits null-byte sniffing.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".pdf",
    ".zip", ".gz", ".tar", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".wav", ".pyc", ".so", ".dll", ".exe",
    ".faiss", ".npy", ".npz", ".bin",
}


def _run_git_ls_files() -> list[str]:
    """Return git-tracked paths (forward-slashed, relative to repo root)."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return paths


@lru_cache(maxsize=1)
def _tracked_paths() -> tuple[str, ...]:
    """Cached, sorted tuple of git-tracked relative paths.

    Falls back to a filesystem walk if git is unavailable so the explorer still
    works in environments without a git binary.
    """
    try:
        paths = _run_git_ls_files()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        paths = []
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
        for candidate in _REPO_ROOT.rglob("*"):
            if not candidate.is_file():
                continue
            rel_parts = candidate.relative_to(_REPO_ROOT).parts
            if any(part in skip_dirs for part in rel_parts):
                continue
            paths.append("/".join(rel_parts))
    return tuple(sorted(set(paths)))


def _classify(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _BINARY_SUFFIXES:
        return "binary"
    return "text"


def list_repo_tree() -> dict:
    """Return the full list of explorable (git-tracked) files."""
    files = [
        {"path": path, "type": _classify(path)}
        for path in _tracked_paths()
    ]
    return {
        "root": _REPO_ROOT.name,
        "count": len(files),
        "files": files,
    }


def _resolve_tracked(rel_path: str) -> Path:
    """Resolve a request path, ensuring it is a git-tracked file in the repo."""
    normalized = str(rel_path or "").strip().replace("\\", "/").lstrip("/")
    if not normalized or normalized not in set(_tracked_paths()):
        raise KeyError(rel_path)
    absolute = (_REPO_ROOT / normalized).resolve()
    # Defense in depth: never let a resolved path escape the repo root.
    if _REPO_ROOT not in absolute.parents and absolute != _REPO_ROOT:
        raise KeyError(rel_path)
    if not absolute.is_file():
        raise FileNotFoundError(normalized)
    return absolute


def get_repo_file(rel_path: str) -> dict:
    """Return the content of a single git-tracked file for viewing."""
    absolute = _resolve_tracked(rel_path)
    normalized = str(rel_path).strip().replace("\\", "/").lstrip("/")
    file_type = _classify(normalized)
    size = absolute.stat().st_size

    if file_type == "binary":
        return {
            "path": normalized,
            "type": "binary",
            "size": size,
            "truncated": False,
            "content": "",
        }

    raw = absolute.read_bytes()
    truncated = False
    if len(raw) > _MAX_FILE_BYTES:
        raw = raw[:_MAX_FILE_BYTES]
        truncated = True

    # Sniff for binary content that slipped past the extension allowlist.
    if b"\x00" in raw:
        return {
            "path": normalized,
            "type": "binary",
            "size": size,
            "truncated": False,
            "content": "",
        }

    content = raw.decode("utf-8", errors="replace")
    return {
        "path": normalized,
        "type": file_type,
        "size": size,
        "truncated": truncated,
        "content": content,
    }
