# KARL Local Setup (Windows) - Fast Path

This is the supported native Windows setup path for local KARL development/testing.

If you are doing ongoing engineering work, prefer the containerized path below and treat native Windows setup as the fallback.

The goal is to avoid:

- long dependency resolver churn without feedback
- wrong Python version/wheel selection
- hidden runtime failures (VC++ runtime, PATH, missing data artifacts)
- "it installed but the app still does not run" situations

Use the wrapper CLI in this doc first. Do not start with manual `pip install` unless you are debugging the setup scripts themselves.

## Strategic Direction (Scale)

KARL should be treated as an **API-first system with thin clients**:

- Web, iOS, and Android should share the same backend API contracts.
- MOA materialization, indexing, retrieval, and evals remain backend concerns.
- Client apps render current-effective text, provenance, history, and PDF navigation.

That means most product work should not require every contributor to compile native Python dependencies locally.

For engineering, we support:

1. Containerized/devcontainer path (recommended default for active engineering)
2. Native Windows path (this document)

## Preferred Engineering Path (Containerized)

### Option A: Docker Compose (backend runtime)

```powershell
docker compose -f docker-compose.dev.yml up --build
```

Then in another terminal:

```powershell
python scripts/karl.py smoke
```

Choose a lighter dependency profile during image build if needed:

```powershell
$env:KARL_REQ_PROFILE = "base"
docker compose -f docker-compose.dev.yml up --build
```

Dependency profiles:

- `base` (runtime/API)
- `ingest` (currently same as base)
- `eval` (base + eval dependencies)
- `full` (default)

### Option B: Dev Container (VS Code)

Use `Dev Containers: Reopen in Container` from VS Code.

The repo includes `.devcontainer/devcontainer.json` wired to `docker-compose.dev.yml`.

## What These Scripts Do

- `scripts/karl.py`
  - wrapper CLI for `doctor`, `setup`, `smoke`
- `scripts/dev_preflight.py` = `karl doctor` backend implementation
  - checks Python/toolchain, PATH, data artifacts, port availability, Windows VC++ runtime hints
- `scripts/bootstrap_windows.ps1` (`karl setup` implementation)
  - deterministic native setup bootstrap (`venv`, pip, profile-based requirements, preflight)
- `scripts/smoke_local.ps1` (`karl smoke` implementation)
  - API + Contract-tab backend smoke checks (`/api/contracts`, `/api/health`, `/api/contract-history`, `/api/contract-browse`, `/api/pdf-location`)

## Quick Start (Recommended)

From the repo root in PowerShell:

```powershell
python scripts/karl.py setup --profile backend
```

Then start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, run smoke checks:

```powershell
python scripts/karl.py smoke
```

## Profiles

`python scripts/karl.py setup` supports these profiles:

- `backend` (default): local backend/API development
- `full`: backend + general full-stack local work (same Python install path today)
- `eval`: local eval development (same Python install path today)
- `demo`: backend runtime + smoke checks (same Python install path today)
- `ingest`: ingestion/materialization-focused install profile (currently same Python install path as base)
- `ui-only`: skips Python dependency install (use when working only on frontend files/static UI)

Example:

```powershell
python scripts/karl.py setup --profile eval --heavy-import-check
```

## Dependency Profiles (Manual Installs)

If you are not using the bootstrap script, prefer profile installs over `requirements.txt`:

```powershell
# API/runtime
python -m pip install -r requirements/base.txt

# Ingestion/materialization
python -m pip install -r requirements/ingest.txt

# Eval work
python -m pip install -r requirements/eval.txt

# Full local engineering stack (backward-compatible default)
python -m pip install -r requirements/full.txt
```

## Doctor (Preflight) Usage

Run before setup (host-level) or after setup (`.venv`) to fail fast.

Host-level:

```powershell
python scripts/karl.py doctor --profile backend
```

Post-install strict check against the virtualenv:

```powershell
.\.venv\Scripts\python.exe scripts/dev_preflight.py --profile backend --check-imports --strict
```

Heavy import check (optional, slower):

```powershell
.\.venv\Scripts\python.exe scripts/dev_preflight.py --profile backend --check-imports --check-heavy-imports --strict
```

JSON output (useful for debugging machine setup):

```powershell
python scripts/karl.py doctor --json --output .\tmp\doctor_report.json
```

## Smoke Test Usage

Use when the server is already running:

```powershell
python scripts/karl.py smoke
```

Or let the script start/stop the server:

```powershell
python scripts/karl.py smoke --start-server
```

Add a lightweight `/api/query` check (optional, slower):

```powershell
python scripts/karl.py smoke --run-query-smoke
```

## Why Windows Sometimes Asks for "Visual Studio" / MSVC Runtime

Some Python packages in ML/search stacks use native extensions or depend on runtime DLLs.

You may need the **Microsoft Visual C++ 2015-2022 Redistributable (x64)** (runtime), even if you are not compiling code yourself.

This is different from full "Visual Studio Build Tools":

- **VC++ Redistributable**: runtime DLLs (`vcruntime140.dll`, `msvcp140.dll`) required by some wheels
- **Build Tools**: only needed when pip cannot find a wheel and tries to compile from source

The bootstrap script prefers binary wheels, and the doctor script warns when common runtime DLLs are missing.

## Python Version Recommendation

Use Python `3.10`-`3.13`.

Practical recommendation:

- `3.11` is usually the smoothest for wheel availability.

If you have multiple versions installed, use:

```powershell
python scripts/karl.py setup --python-preference py
```

## API Key / `.env`

`.env` is optional for retrieval and many evals.

- Without `GEMINI_API_KEY`, KARL can still run retrieval and deterministic features.
- LLM synthesis quality/features will be reduced or unavailable.

`doctor` checks for `.env` and reports it without printing secrets.

## Common Failure Modes (and Fast Fixes)

### 1. `pip install` fails with native/runtime errors

- Install Microsoft Visual C++ 2015-2022 Redistributable (x64)
- Re-run:

```powershell
.\.venv\Scripts\python.exe scripts/dev_preflight.py --check-imports --check-heavy-imports
```

### 2. Wrong Python version gets used

Run:

```powershell
python --version
py -0p
```

Then bootstrap with an explicit preference:

```powershell
python scripts/karl.py setup --python-preference py
```

### 3. Server starts but frontend/chat behaves oddly

Run smoke checks first:

```powershell
python scripts/karl.py smoke
```

If smoke passes, the issue is likely frontend/runtime state or browser cache, not core API availability.

### 5. PowerShell says the script is not digitally signed

Use the one-shot bypass flag (recommended for local repo scripts):

```powershell
python scripts/karl.py setup --profile backend
```

and

```powershell
python scripts/karl.py smoke
```

### 4. Contract/MOA features missing in UI

Smoke checks cover Contract-tab backend endpoints (`contract-history`, `contract-browse`, `pdf-location`).

If those pass, validate frontend build/runtime state next.

## Next Improvements (Planned)

- machine-readable setup lock manifest (Python/Node/dependency versions)
- one-command wrapper (`karl doctor`, `karl setup`, `karl smoke`)
- CI artifact to validate setup scripts on Windows runner
