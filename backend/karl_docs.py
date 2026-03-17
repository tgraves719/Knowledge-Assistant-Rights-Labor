from __future__ import annotations

from pathlib import Path

KARL_VERSION = "0.8.110"
KARL_RELEASE_CHANNEL = "dev"

_REPO_ROOT = Path(__file__).resolve().parents[1]

_KARL_DOCS = {
    "update_log": {
        "title": "Update Log",
        "path": "UPDATE_LOG.md",
    },
    "readme": {
        "title": "README",
        "path": "README.md",
    },
    "real_user_error_correction_guide": {
        "title": "Real User Error Correction Guide",
        "path": "docs/REAL_USER_ERROR_CORRECTION_GUIDE.md",
    },
    "release_readiness": {
        "title": "Release 0.9.0 Readiness",
        "path": "RELEASE_0_9_0_READINESS.md",
    },
    "deployment_policy": {
        "title": "Deployment Policy",
        "path": "legal/DEPLOYMENT-POLICY.md",
    },
    "release_gates": {
        "title": "Release Gates",
        "path": "legal/RELEASE-GATES.md",
    },
    "governance_charter": {
        "title": "Governance Charter",
        "path": "legal/GOVERNANCE-CHARTER.md",
    },
}


def list_karl_documents() -> list[dict]:
    documents: list[dict] = []
    for doc_id, meta in _KARL_DOCS.items():
        documents.append(
            {
                "id": doc_id,
                "title": str(meta["title"]),
                "path": str(meta["path"]),
            }
        )
    return documents


def get_karl_info() -> dict:
    return {
        "version": KARL_VERSION,
        "release_channel": KARL_RELEASE_CHANNEL,
        "documents": list_karl_documents(),
    }


def get_karl_document(doc_id: str) -> dict:
    key = str(doc_id or "").strip().lower()
    if key not in _KARL_DOCS:
        raise KeyError(key)
    meta = _KARL_DOCS[key]
    relative_path = Path(str(meta["path"]))
    absolute_path = (_REPO_ROOT / relative_path).resolve()
    if not absolute_path.is_file():
        raise FileNotFoundError(str(relative_path))
    return {
        "id": key,
        "title": str(meta["title"]),
        "path": str(relative_path).replace("\\", "/"),
        "content": absolute_path.read_text(encoding="utf-8"),
    }
