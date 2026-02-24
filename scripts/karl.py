"""Thin local developer wrapper for KARL setup workflows.

Provides a stable entrypoint for:
- doctor (preflight)
- setup (Windows bootstrap)
- smoke (local API + Contract-tab endpoint smoke checks)

This is intentionally a thin wrapper around existing scripts so we can evolve the
implementation without changing the user-facing command shape.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEV_PREFLIGHT = SCRIPTS_DIR / "dev_preflight.py"
BOOTSTRAP_WINDOWS = SCRIPTS_DIR / "bootstrap_windows.ps1"
SMOKE_LOCAL = SCRIPTS_DIR / "smoke_local.ps1"


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def _powershell_exe() -> str | None:
    for name in ("pwsh", "powershell"):
        if shutil.which(name):
            return name
    return None


def _doctor_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(DEV_PREFLIGHT)]
    if args.profile:
        cmd.extend(["--profile", args.profile])
    if args.python_exe:
        cmd.extend(["--python-exe", args.python_exe])
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port is not None:
        cmd.extend(["--port", str(args.port)])
    if args.check_imports:
        cmd.append("--check-imports")
    if args.check_heavy_imports:
        cmd.append("--check-heavy-imports")
    if args.json:
        cmd.append("--json")
    if args.strict:
        cmd.append("--strict")
    if args.output:
        cmd.extend(["--output", args.output])
    return cmd


def _setup_command(args: argparse.Namespace) -> list[str]:
    ps = _powershell_exe()
    if not ps:
        raise RuntimeError("PowerShell not found (expected 'pwsh' or 'powershell').")
    cmd = [ps, "-ExecutionPolicy", "Bypass", "-File", str(BOOTSTRAP_WINDOWS)]
    if args.profile:
        cmd.extend(["-Profile", args.profile])
    if args.venv_path:
        cmd.extend(["-VenvPath", args.venv_path])
    if args.recreate_venv:
        cmd.append("-RecreateVenv")
    if args.skip_preflight:
        cmd.append("-SkipPreflight")
    if args.run_smoke:
        cmd.append("-RunSmoke")
    if args.start_server_for_smoke:
        cmd.append("-StartServerForSmoke")
    if args.heavy_import_check:
        cmd.append("-HeavyImportCheck")
    if args.python_preference:
        cmd.extend(["-PythonPreference", args.python_preference])
    if args.port is not None:
        cmd.extend(["-Port", str(args.port)])
    return cmd


def _smoke_command(args: argparse.Namespace) -> list[str]:
    ps = _powershell_exe()
    if not ps:
        raise RuntimeError("PowerShell not found (expected 'pwsh' or 'powershell').")
    cmd = [ps, "-ExecutionPolicy", "Bypass", "-File", str(SMOKE_LOCAL)]
    if args.base_url:
        cmd.extend(["-BaseUrl", args.base_url])
    if args.contract_id:
        cmd.extend(["-ContractId", args.contract_id])
    if args.start_server:
        cmd.append("-StartServer")
    if args.python_exe:
        cmd.extend(["-PythonExe", args.python_exe])
    if args.port is not None:
        cmd.extend(["-Port", str(args.port)])
    if args.startup_timeout_sec is not None:
        cmd.extend(["-StartupTimeoutSec", str(args.startup_timeout_sec)])
    if args.run_query_smoke:
        cmd.append("-RunQuerySmoke")
    if args.no_contract_tab_smoke:
        cmd.append("-RunContractTabSmoke:$false")
    if args.verbose_output:
        cmd.append("-VerboseOutput")
    return cmd


def _handle_doctor(args: argparse.Namespace) -> int:
    return _run(_doctor_command(args))


def _handle_setup(args: argparse.Namespace) -> int:
    if os.name != "nt":
        print("KARL setup wrapper is optimized for Windows native bootstrap.", file=sys.stderr)
        print("Use docker compose (`docker-compose.dev.yml`) or run `python scripts/dev_preflight.py` + profile pip installs.", file=sys.stderr)
    return _run(_setup_command(args))


def _handle_smoke(args: argparse.Namespace) -> int:
    return _run(_smoke_command(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KARL local developer helper CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="Run local preflight checks")
    doctor.add_argument("--profile", default="backend", choices=["backend", "full", "ui-only", "eval", "demo", "mobile"])
    doctor.add_argument("--python-exe", default=sys.executable)
    doctor.add_argument("--host", default="127.0.0.1")
    doctor.add_argument("--port", type=int, default=8000)
    doctor.add_argument("--check-imports", action="store_true")
    doctor.add_argument("--check-heavy-imports", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--output", default="")
    doctor.set_defaults(func=_handle_doctor)

    setup = sub.add_parser("setup", help="Run Windows bootstrap setup wrapper")
    setup.add_argument("--profile", default="backend", choices=["backend", "full", "ui-only", "eval", "demo", "ingest"])
    setup.add_argument("--venv-path", default=".venv")
    setup.add_argument("--recreate-venv", action="store_true")
    setup.add_argument("--skip-preflight", action="store_true")
    setup.add_argument("--run-smoke", action="store_true")
    setup.add_argument("--start-server-for-smoke", action="store_true")
    setup.add_argument("--heavy-import-check", action="store_true")
    setup.add_argument("--python-preference", default="")
    setup.add_argument("--port", type=int, default=8000)
    setup.set_defaults(func=_handle_setup)

    smoke = sub.add_parser("smoke", help="Run local API/Contract-tab smoke checks")
    smoke.add_argument("--base-url", default="http://127.0.0.1:8000")
    smoke.add_argument("--contract-id", default="")
    smoke.add_argument("--start-server", action="store_true")
    smoke.add_argument("--python-exe", default=str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    smoke.add_argument("--port", type=int, default=8000)
    smoke.add_argument("--startup-timeout-sec", type=int, default=60)
    smoke.add_argument("--run-query-smoke", action="store_true")
    smoke.add_argument("--no-contract-tab-smoke", action="store_true")
    smoke.add_argument("--verbose-output", action="store_true")
    smoke.set_defaults(func=_handle_smoke)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
