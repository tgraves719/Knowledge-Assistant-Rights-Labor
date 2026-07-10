param(
    [ValidateSet("backend","full","ui-only","eval","demo","ingest")]
    [string]$Profile = "backend",
    [string]$VenvPath = ".venv",
    [switch]$RecreateVenv,
    [switch]$SkipPreflight,
    [switch]$RunSmoke,
    [switch]$StartServerForSmoke,
    [switch]$HeavyImportCheck,
    [string]$PythonPreference = "",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ("[STEP] " + $Message)
}

function Resolve-RequirementsFile {
    param([string]$BootstrapProfile)
    switch ($BootstrapProfile) {
        "backend" { return "requirements/base.txt" }
        "demo"    { return "requirements/base.txt" }
        "ingest"  { return "requirements/ingest.txt" }
        "eval"    { return "requirements/eval.txt" }
        "full"    { return "requirements/full.txt" }
        default   { return "requirements/full.txt" }
    }
}

function Get-CmdTailArgs {
    param([object[]]$Cmd)
    if (-not $Cmd) { return @() }
    if ($Cmd.Count -le 1) { return @() }
    return @($Cmd[1..($Cmd.Count-1)])
}

function Resolve-PythonCommand {
    param([string]$Preference)
    if ($Preference) {
        # Be forgiving if a wrapped path was pasted from chat/terminal output.
        $Preference = ($Preference -replace "\s*`r?`n\s*", "").Trim()
        if ($Preference -eq "py") { return @("py", "-3.11") }
        return ,$Preference
    }

    $candidates = @(
        @("py", "-3.11"),
        @("py", "-3.12"),
        @("py", "-3.13"),
        @("python")
    )

    foreach ($candidate in $candidates) {
        try {
            $tail = Get-CmdTailArgs -Cmd $candidate
            & $candidate[0] @($tail | Where-Object { $_ }) -c "import sys;print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {
            continue
        }
    }
    throw "No usable Python interpreter found. Install Python 3.10-3.13 or use -PythonPreference."
}

function Get-VenvPythonPath {
    param([string]$PathValue)
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    return Join-Path $repoRoot (Join-Path $PathValue "Scripts\python.exe")
}

function Get-PythonVersionString {
    param([object[]]$Cmd)
    try {
        $tail = Get-CmdTailArgs -Cmd $Cmd
        $version = & $Cmd[0] @($tail | Where-Object { $_ }) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        if ($LASTEXITCODE -eq 0) { return ($version | Select-Object -First 1) }
    } catch {
    }
    return $null
}

function Invoke-Doctor {
    param(
        [string]$PythonExe,
        [switch]$CheckImports,
        [switch]$CheckHeavyImports,
        [switch]$Strict
    )
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $args = @("scripts/dev_preflight.py", "--profile", $Profile, "--python-exe", $PythonExe, "--port", "$Port")
    if ($CheckImports) { $args += "--check-imports" }
    if ($CheckHeavyImports) { $args += "--check-heavy-imports" }
    if ($Strict) { $args += "--strict" }
    & $PythonExe $args
    if ($LASTEXITCODE -ne 0) {
        throw "Preflight (doctor) failed."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "======================================================================"
Write-Host "KARL Windows Bootstrap"
Write-Host "======================================================================"
Write-Host ("Profile: " + $Profile)
Write-Host ("Repo:    " + $repoRoot)

if (-not $SkipPreflight) {
    Write-Step "Running repo preflight (host Python)"
    if ($PythonPreference) {
        $hostPythonCmd = @(Resolve-PythonCommand -Preference $PythonPreference)
    } else {
        $hostPythonCmd = @("python")
        try {
            python --version *> $null
            if ($LASTEXITCODE -ne 0) { $hostPythonCmd = @(Resolve-PythonCommand -Preference $PythonPreference) }
        } catch {
            $hostPythonCmd = @(Resolve-PythonCommand -Preference $PythonPreference)
        }
    }
    $hostTail = Get-CmdTailArgs -Cmd $hostPythonCmd
    & $hostPythonCmd[0] @($hostTail | Where-Object { $_ }) "scripts/dev_preflight.py" --profile $Profile --port $Port
}

if ($Profile -eq "ui-only") {
    Write-Host ""
    Write-Host "UI-only profile selected."
    Write-Host "No Python dependency install performed (frontend is served by backend/static files in this repo)."
    Write-Host "Use the backend profile if you need API/runtime locally."
    exit 0
}

$pythonCmd = @(Resolve-PythonCommand -Preference $PythonPreference)
Write-Step ("Using Python launcher: " + ($pythonCmd -join " "))
$requirementsFile = Resolve-RequirementsFile -BootstrapProfile $Profile
Write-Step ("Dependency profile file: " + $requirementsFile)

$venvPython = Get-VenvPythonPath -PathValue $VenvPath
if ($RecreateVenv -and (Test-Path $VenvPath)) {
    Write-Step ("Removing existing venv: " + $VenvPath)
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $venvPython)) {
    Write-Step ("Creating venv: " + $VenvPath)
    $pythonTail = Get-CmdTailArgs -Cmd $pythonCmd
    & $pythonCmd[0] @($pythonTail | Where-Object { $_ }) -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment." }
} else {
    $selectedVersion = Get-PythonVersionString -Cmd $pythonCmd
    $venvVersion = Get-PythonVersionString -Cmd @($venvPython)
    if ($selectedVersion -and $venvVersion -and (($selectedVersion -split '\.')[0..1] -join '.') -ne (($venvVersion -split '\.')[0..1] -join '.')) {
        throw "Existing venv uses Python $venvVersion but selected interpreter is $selectedVersion. Rerun with -RecreateVenv to rebuild $VenvPath."
    }
    Write-Step ("Reusing existing venv: " + $VenvPath)
}

if (-not (Test-Path $venvPython)) {
    throw "Expected venv Python not found after creation: $venvPython"
}

Write-Step "Upgrading pip/setuptools/wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip tooling. If you see VC++ runtime errors, install Microsoft Visual C++ 2015-2022 Redistributable (x64) and rerun."
}

Write-Step ("Installing Python dependencies from " + $requirementsFile + " (prefer binary wheels)")
& $venvPython -m pip install --prefer-binary -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Dependency install failed. Run scripts/dev_preflight.py and check VC++ runtime / Python version."
}

Write-Step "Running strict post-install doctor checks (.venv)"
Invoke-Doctor -PythonExe $venvPython -CheckImports -CheckHeavyImports:$HeavyImportCheck -Strict

if ($RunSmoke) {
    Write-Step "Running local smoke checks"
    $smokeArgs = @("-File", "scripts/smoke_local.ps1", "-PythonExe", $venvPython, "-Port", "$Port")
    if ($StartServerForSmoke) { $smokeArgs += "-StartServer" }
    & powershell @smokeArgs
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed." }
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host ("1. Run doctor:  " + $venvPython + " scripts/dev_preflight.py --profile " + $Profile + " --check-imports")
Write-Host ("2. Start API:   " + $venvPython + " -m uvicorn backend.api:app --host 127.0.0.1 --port " + $Port)
Write-Host ("3. Smoke test:  powershell -ExecutionPolicy Bypass -File scripts/smoke_local.ps1 -Port " + $Port)
Write-Host ("Installed deps profile: " + $requirementsFile)
Write-Host ""
Write-Host "If pip/build fails on Windows native installs:"
Write-Host "- Install Microsoft Visual C++ 2015-2022 Redistributable (x64)"
Write-Host "- Prefer Python 3.10-3.13 (3.11 often has the best wheel coverage)"
