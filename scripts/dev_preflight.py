"""KARL local environment preflight ("doctor") checks.

Purpose:
- Fail fast on obvious local setup issues (Python/version/path/deps/data/ports)
- Provide one deterministic report for Windows/native setups before long installs
- Support post-install verification against a specific interpreter (e.g. .venv)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
RECOMMENDED_PYTHON_MIN = (3, 10)
RECOMMENDED_PYTHON_MAX_EXCLUSIVE = (3, 14)


@dataclass
class CheckResult:
    code: str
    status: str  # ok, warn, error, info
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _ok(code: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(code=code, status="ok", message=message, details=details)


def _warn(code: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(code=code, status="warn", message=message, details=details)


def _err(code: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(code=code, status="error", message=message, details=details)


def _info(code: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(code=code, status="info", message=message, details=details)


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 15) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return int(proc.returncode), (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _resolve_default_venv_python() -> Path:
    if os.name == "nt":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def _python_version_for_interpreter(python_exe: str) -> tuple[Optional[tuple[int, int, int]], Optional[str]]:
    try:
        rc, out, err = _run(
            [
                python_exe,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
            ]
        )
    except Exception as e:
        return None, f"failed to execute interpreter: {e}"
    if rc != 0:
        return None, err or out or f"python exited with code {rc}"
    try:
        parts = [int(p) for p in out.strip().split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return (parts[0], parts[1], parts[2]), None
    except Exception:
        return None, f"unable to parse version string: {out!r}"


def _check_repo_structure() -> list[CheckResult]:
    checks: list[CheckResult] = []
    required = [
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "backend" / "api.py",
        REPO_ROOT / "frontend" / "modular" / "index.html",
        REPO_ROOT / "data" / "manifests",
    ]
    for path in required:
        if path.exists():
            checks.append(_ok("repo.path", f"Found {path.relative_to(REPO_ROOT)}"))
        else:
            checks.append(_err("repo.path", f"Missing {path.relative_to(REPO_ROOT)}"))
    return checks


def _check_tools(profile: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for tool in ("git",):
        tool_path = shutil.which(tool)
        if tool_path:
            checks.append(_ok("tool.git", "git available", path=tool_path))
        else:
            checks.append(_warn("tool.git", "git not found on PATH"))

    for tool in ("node", "npm"):
        tool_path = shutil.which(tool)
        if tool_path:
            checks.append(_ok(f"tool.{tool}", f"{tool} available", path=tool_path))
        else:
            msg = f"{tool} not found on PATH"
            if profile in {"full", "mobile"}:
                checks.append(_warn(f"tool.{tool}", msg))
            else:
                checks.append(_info(f"tool.{tool}", msg))
    return checks


def _check_python_interpreter(python_exe: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not shutil.which(python_exe) and not Path(python_exe).exists():
        return [_err("python.exec", f"Python interpreter not found: {python_exe}")]

    version, err = _python_version_for_interpreter(python_exe)
    if err or not version:
        return [_err("python.exec", f"Unable to run interpreter: {python_exe}", error=err or "unknown")]

    major, minor, patch = version
    checks.append(_ok("python.exec", "Python interpreter runnable", executable=python_exe, version=f"{major}.{minor}.{patch}"))
    if (major, minor) < RECOMMENDED_PYTHON_MIN:
        checks.append(
            _err(
                "python.version",
                f"Python {major}.{minor}.{patch} is too old for recommended KARL setup",
                recommended_min="3.10",
            )
        )
    elif (major, minor) >= RECOMMENDED_PYTHON_MAX_EXCLUSIVE:
        checks.append(
            _warn(
                "python.version",
                f"Python {major}.{minor}.{patch} is newer than validated baseline range",
                validated_range="3.10-3.13",
            )
        )
    else:
        checks.append(_ok("python.version", f"Python {major}.{minor}.{patch} is in validated range (3.10-3.13)"))

    rc, out, err2 = _run([python_exe, "-m", "pip", "--version"])
    if rc == 0:
        checks.append(_ok("python.pip", "pip available for interpreter", output=out))
    else:
        checks.append(_err("python.pip", "pip unavailable for interpreter", error=err2 or out))
    return checks


def _check_python_imports(python_exe: str, check_heavy_imports: bool) -> list[CheckResult]:
    checks: list[CheckResult] = []
    modules = ["fastapi", "uvicorn", "pydantic", "httpx", "dotenv"]
    heavy = ["chromadb", "sentence_transformers"]
    if check_heavy_imports:
        modules.extend(heavy)
    code = (
        "import importlib, json; "
        f"mods={json.dumps(modules)}; "
        "out={}; "
        "errs={}; "
        "import traceback; "
        "\nfor m in mods:\n"
        "  try:\n"
        "    importlib.import_module(m)\n"
        "    out[m]=True\n"
        "  except Exception as e:\n"
        "    out[m]=False\n"
        "    errs[m]=str(e)\n"
        "print(json.dumps({'ok': out, 'errors': errs}, sort_keys=True))"
    )
    try:
        rc, out, err = _run([python_exe, "-c", code], cwd=REPO_ROOT, timeout=90)
    except Exception as e:
        return [_err("python.imports", "Import check failed to run", error=str(e))]
    if rc != 0:
        return [_err("python.imports", "Import check subprocess failed", error=err or out)]
    try:
        payload = json.loads(out)
    except Exception:
        return [_err("python.imports", "Import check output parse failed", raw=out)]

    ok_map = payload.get("ok") or {}
    err_map = payload.get("errors") or {}
    for mod in modules:
        if ok_map.get(mod):
            checks.append(_ok("python.import", f"Import ok: {mod}"))
        else:
            checks.append(_warn("python.import", f"Import failed: {mod}", error=str(err_map.get(mod) or "")))
    return checks


def _check_data_artifacts(profile: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    manifests = sorted((REPO_ROOT / "data" / "manifests").glob("*.json"))
    chunk_files = sorted((REPO_ROOT / "data" / "chunks").glob("*.json"))
    effective_latest = sorted((REPO_ROOT / "data" / "contracts").glob("*/effective/latest.json"))

    if manifests:
        checks.append(_ok("data.manifests", f"Found {len(manifests)} contract manifest(s)"))
    else:
        checks.append(_err("data.manifests", "No manifests found in data/manifests"))

    if chunk_files:
        checks.append(_ok("data.chunks", f"Found {len(chunk_files)} chunk artifact(s)"))
    else:
        status = "warn" if profile in {"backend", "full", "demo", "eval"} else "info"
        fn = _warn if status == "warn" else _info
        checks.append(fn("data.chunks", "No chunk artifacts found; run onboarding/ingestion before querying"))

    if effective_latest:
        checks.append(_ok("data.effective", f"Found {len(effective_latest)} effective snapshot pointer(s)"))
    else:
        checks.append(_info("data.effective", "No effective/latest.json pointers found yet (MOA layer may not be materialized)"))
    return checks


def _check_env_file() -> list[CheckResult]:
    checks: list[CheckResult] = []
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        checks.append(_info("env.file", ".env not present (LLM answers optional; retrieval/evals can still run)"))
        return checks
    try:
        text = env_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        checks.append(_warn("env.file", ".env exists but could not be read", error=str(e)))
        return checks
    key_present = False
    for line in text.splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            key_present = bool(value)
            break
    if key_present:
        checks.append(_ok("env.gemini", "GEMINI_API_KEY present in .env"))
    else:
        checks.append(_info("env.gemini", "GEMINI_API_KEY not set in .env (LLM synthesis will be unavailable)"))
    return checks


def _check_port(host: str, port: int) -> list[CheckResult]:
    checks: list[CheckResult] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        in_use = sock.connect_ex((host, port)) == 0
    except Exception as e:
        checks.append(_warn("net.port", "Unable to probe local port", host=host, port=port, error=str(e)))
        return checks
    finally:
        try:
            sock.close()
        except Exception:
            pass

    if in_use:
        checks.append(_info("net.port", f"Port {host}:{port} is in use (may already be running KARL)", host=host, port=port))
    else:
        checks.append(_ok("net.port", f"Port {host}:{port} is free", host=host, port=port))
    return checks


def _check_windows_runtime() -> list[CheckResult]:
    if os.name != "nt":
        return [_info("windows.runtime", "Windows runtime checks skipped (non-Windows host)")]

    checks: list[CheckResult] = []
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    sys32 = system_root / "System32"
    vcrt = sys32 / "vcruntime140.dll"
    msvc = sys32 / "msvcp140.dll"
    if vcrt.exists() and msvc.exists():
        checks.append(_ok("windows.vcredist", "Visual C++ runtime DLLs found", vcruntime=str(vcrt), msvcp=str(msvc)))
    else:
        checks.append(
            _warn(
                "windows.vcredist",
                "Visual C++ runtime DLLs not found (some Python wheels/native libs may fail)",
                expected_files=[str(vcrt), str(msvc)],
            )
        )

    try:
        import winreg  # type: ignore

        uninstall_roots = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        found = []
        for hive, root in uninstall_roots:
            try:
                with winreg.OpenKey(hive, root) as key:
                    i = 0
                    while True:
                        try:
                            sub_name = winreg.EnumKey(key, i)
                            i += 1
                            with winreg.OpenKey(key, sub_name) as sub:
                                display_name, _ = winreg.QueryValueEx(sub, "DisplayName")
                                if "Visual C++" in str(display_name):
                                    found.append(str(display_name))
                        except OSError:
                            break
                        except Exception:
                            continue
            except Exception:
                continue
        if found:
            sample = sorted(set(found))[:5]
            checks.append(_ok("windows.vcredist.registry", "Visual C++ redistributable entries found", sample=sample))
        else:
            checks.append(_info("windows.vcredist.registry", "No Visual C++ redistributable entries found in registry scan"))
    except Exception as e:
        checks.append(_info("windows.vcredist.registry", "Registry scan skipped", error=str(e)))
    return checks


def build_report(
    *,
    profile: str,
    python_exe: str,
    check_imports: bool,
    check_heavy_imports: bool,
    host: str,
    port: int,
) -> dict[str, Any]:
    checks: list[CheckResult] = []
    checks.extend(_check_repo_structure())
    checks.extend(_check_tools(profile))
    checks.extend(_check_python_interpreter(python_exe))
    if check_imports:
        checks.extend(_check_python_imports(python_exe, check_heavy_imports))
    checks.extend(_check_data_artifacts(profile))
    checks.extend(_check_env_file())
    checks.extend(_check_port(host, port))
    checks.extend(_check_windows_runtime())

    counts = {"ok": 0, "warn": 0, "error": 0, "info": 0}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1

    report = {
        "schema_version": "karl_dev_preflight_v1",
        "repo_root": str(REPO_ROOT),
        "host": {
            "platform": platform.platform(),
            "python_current": sys.version.split()[0],
            "os_name": os.name,
        },
        "inputs": {
            "profile": profile,
            "python_exe": python_exe,
            "check_imports": bool(check_imports),
            "check_heavy_imports": bool(check_heavy_imports),
            "host": host,
            "port": int(port),
        },
        "checks": [asdict(c) for c in checks],
        "summary": {
            **counts,
            "pass": counts["error"] == 0,
        },
    }
    return report


def _print_human(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    print("=" * 72)
    print("KARL Dev Preflight (doctor)")
    print("=" * 72)
    print(f"Repo: {report.get('repo_root')}")
    print(f"Profile: {(report.get('inputs') or {}).get('profile')}")
    print(f"Python target: {(report.get('inputs') or {}).get('python_exe')}")
    print("-" * 72)
    for row in report.get("checks") or []:
        status = str(row.get("status") or "").upper()
        marker = {"OK": "[OK]", "WARN": "[!!]", "ERROR": "[XX]", "INFO": "[--]"}.get(status, "[--]")
        print(f"{marker} {row.get('code')}: {row.get('message')}")
    print("-" * 72)
    print(
        "Summary:",
        f"ok={int(summary.get('ok') or 0)}",
        f"warn={int(summary.get('warn') or 0)}",
        f"error={int(summary.get('error') or 0)}",
        f"info={int(summary.get('info') or 0)}",
    )
    print(f"Pass: {bool(summary.get('pass'))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="KARL local preflight checks (doctor).")
    parser.add_argument("--profile", default="backend", choices=["backend", "full", "ui-only", "eval", "demo", "mobile"])
    parser.add_argument("--python-exe", default=sys.executable or "python")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check-imports", action="store_true", help="Validate Python imports for installed dependencies.")
    parser.add_argument("--check-heavy-imports", action="store_true", help="Also check heavier imports (chromadb, sentence_transformers).")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any error findings.")
    parser.add_argument("--output", default="", help="Optional JSON file path for the report.")
    args = parser.parse_args()

    report = build_report(
        profile=str(args.profile),
        python_exe=str(args.python_exe),
        check_imports=bool(args.check_imports),
        check_heavy_imports=bool(args.check_heavy_imports),
        host=str(args.host),
        port=int(args.port),
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_human(report)

    if bool(args.strict) and not bool((report.get("summary") or {}).get("pass")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
