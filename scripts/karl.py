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
import socket
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEV_PREFLIGHT = SCRIPTS_DIR / "dev_preflight.py"
BOOTSTRAP_WINDOWS = SCRIPTS_DIR / "bootstrap_windows.ps1"
SMOKE_LOCAL = SCRIPTS_DIR / "smoke_local.ps1"


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def _venv_python() -> str:
    if os.name == "nt":
        return str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    return str(REPO_ROOT / ".venv" / "bin" / "python")


def _powershell_exe() -> str | None:
    for name in ("pwsh", "powershell"):
        if shutil.which(name):
            return name
    if os.name == "nt":
        fallbacks = [
            r"C:\Program Files\PowerShell\7\pwsh.exe",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        ]
        for path in fallbacks:
            if Path(path).exists():
                return path
    return None


def _docker_compose_prefix() -> list[str] | None:
    if shutil.which("docker"):
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


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


def _start_command(args: argparse.Namespace) -> list[str]:
    python_exe = args.python_exe or _venv_python()
    cmd = [python_exe, "-m", "uvicorn", "backend.api:app", "--host", args.host, "--port", str(args.port)]
    if args.reload:
        cmd.append("--reload")
    return cmd


def _docker_up_command(args: argparse.Namespace) -> list[str]:
    compose = _docker_compose_prefix()
    if not compose:
        raise RuntimeError("Docker Compose not found (expected `docker compose` or `docker-compose`).")
    cmd = compose + ["-f", "docker-compose.dev.yml", "up"]
    if args.build:
        cmd.append("--build")
    if args.detach:
        cmd.append("-d")
    return cmd


def _docker_down_command(args: argparse.Namespace) -> list[str]:
    compose = _docker_compose_prefix()
    if not compose:
        raise RuntimeError("Docker Compose not found (expected `docker compose` or `docker-compose`).")
    cmd = compose + ["-f", "docker-compose.dev.yml", "down"]
    if args.volumes:
        cmd.append("-v")
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


def _handle_start(args: argparse.Namespace) -> int:
    if not Path(args.python_exe or _venv_python()).exists():
        print("[ERROR] .venv Python not found. Run `python scripts/karl.py setup --profile backend` first.", file=sys.stderr)
        return 2
    if _port_in_use(args.host, args.port):
        print(
            f"[ERROR] Port {args.host}:{args.port} is already in use. Stop the existing server or pass --port <new-port>.",
            file=sys.stderr,
        )
        return 2
    return _run(_start_command(args))


def _handle_docker_up(args: argparse.Namespace) -> int:
    return _run(_docker_up_command(args))


def _handle_docker_down(args: argparse.Namespace) -> int:
    return _run(_docker_down_command(args))


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

    start = sub.add_parser("start", help="Start KARL API from local .venv")
    start.add_argument("--python-exe", default=_venv_python())
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--reload", action="store_true")
    start.set_defaults(func=_handle_start)

    docker_up = sub.add_parser("docker-up", help="Start KARL dev container stack")
    docker_up.add_argument("--build", action="store_true")
    docker_up.add_argument("--detach", action="store_true")
    docker_up.set_defaults(func=_handle_docker_up)

    docker_down = sub.add_parser("docker-down", help="Stop KARL dev container stack")
    docker_down.add_argument("--volumes", action="store_true")
    docker_down.set_defaults(func=_handle_docker_down)

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
