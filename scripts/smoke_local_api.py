"""Cross-platform local API smoke for KARL release hardening."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_json(url: str, *, timeout: int = 20) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post_json(url: str, payload: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_for_health(base_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{base_url}/api/health"
    while time.time() < deadline:
        try:
            _get_json(health_url, timeout=2)
            return
        except Exception:
            time.sleep(0.75)
    raise RuntimeError(f"Server did not become healthy within {timeout_sec} seconds.")


def run_smoke(base_url: str) -> dict[str, Any]:
    contracts = _get_json(f"{base_url}/api/contracts")
    contract_rows = list(contracts.get("contracts") or [])
    if not contract_rows:
        raise RuntimeError("/api/contracts returned no contracts.")

    default_contract_id = str(contracts.get("default_contract_id") or "")
    selected = next(
        (row for row in contract_rows if str(row.get("contract_id") or "") == default_contract_id),
        contract_rows[0],
    )
    contract_id = str(selected.get("contract_id") or "")
    if not contract_id:
        raise RuntimeError("Selected contract row missing contract_id.")

    quoted_contract_id = urllib.parse.quote(contract_id)
    health = _get_json(f"{base_url}/api/health?contract_id={quoted_contract_id}")
    history = _get_json(f"{base_url}/api/contract-history?contract_id={quoted_contract_id}")
    browse = _get_json(f"{base_url}/api/contract-browse?contract_id={quoted_contract_id}")
    groups = list(browse.get("groups") or [])
    if not groups:
        raise RuntimeError("/api/contract-browse returned no groups.")

    first_item: dict[str, Any] | None = None
    for group in groups:
        items = list((group or {}).get("items") or [])
        if items:
            first_item = items[0]
            break
    if not first_item:
        raise RuntimeError("/api/contract-browse returned groups but no items.")

    if str(first_item.get("kind") or "") == "article" and first_item.get("article_num") is not None:
        article_num = str(first_item.get("article_num"))
        _get_json(f"{base_url}/api/article/{article_num}?contract_id={quoted_contract_id}")
        _get_json(f"{base_url}/api/article/{article_num}?contract_id={quoted_contract_id}&source_view=base")
        _get_json(f"{base_url}/api/pdf-location?contract_id={quoted_contract_id}&article_num={urllib.parse.quote(article_num)}")
    else:
        kind = urllib.parse.quote(str(first_item.get("kind") or ""))
        key = urllib.parse.quote(str(first_item.get("key") or ""))
        _get_json(f"{base_url}/api/contract-browse-item?contract_id={quoted_contract_id}&kind={kind}&key={key}")
        _get_json(f"{base_url}/api/contract-browse-item?contract_id={quoted_contract_id}&kind={kind}&key={key}&source_view=base")
        _get_json(f"{base_url}/api/pdf-location?contract_id={quoted_contract_id}&browse_kind={kind}&browse_key={key}")

    query = _post_json(
        f"{base_url}/api/query",
        {
            "question": "What article covers discipline?",
            "union_local_id": str(selected.get("union_local_id") or ""),
            "contract_id": contract_id,
            "contract_version": str(selected.get("contract_version") or ""),
            "hours_worked": 0,
            "months_employed": 0,
        },
    )
    if str(query.get("contract_id") or "") != contract_id:
        raise RuntimeError("/api/query returned missing or mismatched contract_id.")

    return {
        "contract_id": contract_id,
        "union_local_id": str(selected.get("union_local_id") or ""),
        "contract_version": str(selected.get("contract_version") or ""),
        "health_status": health.get("status"),
        "effective_version_id": history.get("effective_version_id"),
        "patch_count": history.get("patch_count"),
        "query_contract_id": query.get("contract_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform KARL local API smoke.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--python-exe", default=str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-sec", type=int, default=60)
    parser.add_argument("--start-server", action="store_true")
    args = parser.parse_args()

    server_process: subprocess.Popen[str] | None = None
    try:
        if args.start_server:
            server_process = subprocess.Popen(
                [
                    args.python_exe,
                    "-m",
                    "uvicorn",
                    "backend.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ],
                cwd=str(PROJECT_ROOT),
            )
            _wait_for_health(args.base_url, args.startup_timeout_sec)

        summary = run_smoke(args.base_url)
        print("KARL local smoke: PASS")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print("KARL local smoke: FAIL")
        print(str(exc))
        return 1
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except Exception:
                server_process.kill()
                server_process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
