param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ContractId = "",
    [switch]$StartServer,
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8000,
    [int]$StartupTimeoutSec = 60,
    [switch]$RunQuerySmoke,
    [switch]$RunContractTabSmoke = $true,
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ("[STEP] " + $Message)
}

function Invoke-KarlJson {
    param(
        [Parameter(Mandatory=$true)][string]$Method,
        [Parameter(Mandatory=$true)][string]$Url,
        [object]$Body = $null
    )
    if ($VerboseOutput) {
        Write-Host ("[HTTP] " + $Method + " " + $Url)
    }
    try {
        if ($null -eq $Body) {
            return Invoke-RestMethod -Method $Method -Uri $Url -TimeoutSec 20
        }
        $json = $Body | ConvertTo-Json -Depth 8
        return Invoke-RestMethod -Method $Method -Uri $Url -ContentType "application/json" -Body $json -TimeoutSec 45
    }
    catch {
        $statusCode = $null
        try { $statusCode = [int]$_.Exception.Response.StatusCode.value__ } catch {}
        $suffix = if ($statusCode) { " (HTTP $statusCode)" } else { "" }
        throw ("HTTP request failed: " + $Method + " " + $Url + $suffix + " :: " + $_.Exception.Message)
    }
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$TimeoutSec = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-KarlJson -Method GET -Url $Url
            return $true
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    return $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serverProcess = $null
$serverStdout = Join-Path $repoRoot "tmp_karl_smoke_server.out.log"
$serverStderr = Join-Path $repoRoot "tmp_karl_smoke_server.err.log"

try {
    if ($StartServer) {
        Write-Step "Starting local server via uvicorn"
        if (-not (Test-Path $PythonExe)) {
            throw "Python executable not found: $PythonExe"
        }
        if (Test-Path $serverStdout) { Remove-Item $serverStdout -Force }
        if (Test-Path $serverStderr) { Remove-Item $serverStderr -Force }

        $serverProcess = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList @("-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "$Port") `
            -WorkingDirectory $repoRoot `
            -RedirectStandardOutput $serverStdout `
            -RedirectStandardError $serverStderr `
            -PassThru

        if (-not (Wait-ForUrl -Url "$BaseUrl/api/health" -TimeoutSec $StartupTimeoutSec)) {
            $stdoutTail = if (Test-Path $serverStdout) { (Get-Content $serverStdout -Tail 60) -join "`n" } else { "" }
            $stderrTail = if (Test-Path $serverStderr) { (Get-Content $serverStderr -Tail 60) -join "`n" } else { "" }
            throw "Server did not become healthy within $StartupTimeoutSec seconds.`nSTDOUT:`n$stdoutTail`nSTDERR:`n$stderrTail"
        }
    }

    Write-Step "GET /api/contracts"
    $contracts = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/contracts"
    if (-not $contracts.contracts -or $contracts.contracts.Count -lt 1) {
        throw "/api/contracts returned no contracts"
    }

    $selectedContract = $null
    if ($ContractId) {
        $selectedContract = $contracts.contracts | Where-Object { $_.contract_id -eq $ContractId } | Select-Object -First 1
        if (-not $selectedContract) {
            throw "Requested contract_id '$ContractId' not found in /api/contracts"
        }
    } else {
        $defaultId = [string]$contracts.default_contract_id
        $selectedContract = $contracts.contracts | Where-Object { $_.contract_id -eq $defaultId } | Select-Object -First 1
        if (-not $selectedContract) {
            $selectedContract = $contracts.contracts | Select-Object -First 1
        }
    }
    $contractId = [string]$selectedContract.contract_id
    $unionLocalId = [string]$selectedContract.union_local_id
    $contractVersion = [string]$selectedContract.contract_version

    Write-Step "GET /api/health (global + contract-scoped)"
    $healthGlobal = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/health"
    $healthContract = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/health?contract_id=$([uri]::EscapeDataString($contractId))"
    if (-not $healthContract.active_contract_id) {
        throw "/api/health contract-scoped response missing active_contract_id"
    }

    if ($RunContractTabSmoke) {
        try {
            Write-Step "GET /api/contract-history"
            $history = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/contract-history?contract_id=$([uri]::EscapeDataString($contractId))"

            Write-Step "GET /api/contract-browse"
            $browse = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/contract-browse?contract_id=$([uri]::EscapeDataString($contractId))"
        }
        catch {
            $msg = [string]$_.Exception.Message
            if ($msg -match "/api/contract-(history|browse)" -and $msg -match "HTTP 404") {
                throw ($msg + " :: Contract-tab endpoints missing on the running server. Restart the API from the current repo branch/build.")
            }
            throw
        }
        if (-not $browse.groups -or $browse.groups.Count -lt 1) {
            throw "/api/contract-browse returned no groups"
        }

        $firstItem = $null
        foreach ($group in $browse.groups) {
            if ($group.items -and $group.items.Count -gt 0) {
                $firstItem = $group.items | Select-Object -First 1
                break
            }
        }
        if (-not $firstItem) {
            throw "/api/contract-browse returned groups but no items"
        }

        if ($firstItem.kind -eq "article" -and $firstItem.article_num) {
            Write-Step "GET /api/article/$($firstItem.article_num) (effective + base)"
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/article/$($firstItem.article_num)?contract_id=$([uri]::EscapeDataString($contractId))"
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/article/$($firstItem.article_num)?contract_id=$([uri]::EscapeDataString($contractId))&source_view=base"
            Write-Step "GET /api/pdf-location (article target)"
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/pdf-location?contract_id=$([uri]::EscapeDataString($contractId))&article_num=$($firstItem.article_num)"
        } else {
            Write-Step "GET /api/contract-browse-item (effective + base)"
            $kind = [uri]::EscapeDataString([string]$firstItem.kind)
            $key = [uri]::EscapeDataString([string]$firstItem.key)
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/contract-browse-item?contract_id=$([uri]::EscapeDataString($contractId))&kind=$kind&key=$key"
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/contract-browse-item?contract_id=$([uri]::EscapeDataString($contractId))&kind=$kind&key=$key&source_view=base"
            Write-Step "GET /api/pdf-location (browse target)"
            $null = Invoke-KarlJson -Method GET -Url "$BaseUrl/api/pdf-location?contract_id=$([uri]::EscapeDataString($contractId))&browse_kind=$kind&browse_key=$key"
        }
    }

    if ($RunQuerySmoke) {
        Write-Step "POST /api/query (minimal smoke)"
        $queryPayload = @{
            question = "What article covers discipline?"
            union_local_id = $unionLocalId
            contract_id = $contractId
            contract_version = $contractVersion
            hours_worked = 0
            months_employed = 0
        }
        $queryResponse = Invoke-KarlJson -Method POST -Url "$BaseUrl/api/query" -Body $queryPayload
        if (-not $queryResponse.contract_id) {
            throw "/api/query smoke response missing contract_id"
        }
    }

    Write-Host ""
    Write-Host "KARL local smoke: PASS"
    Write-Host ("- contract_id: " + $contractId)
    Write-Host ("- union_local_id: " + $unionLocalId)
    Write-Host ("- contract_version: " + $contractVersion)
    Write-Host ("- /api/health status: " + [string]$healthContract.status)
    if ($RunContractTabSmoke -and $history) {
        Write-Host ("- effective_version_id: " + [string]$history.effective_version_id)
        Write-Host ("- patches: " + [string]$history.patch_count)
    }
    exit 0
}
catch {
    Write-Host ""
    Write-Host "KARL local smoke: FAIL"
    Write-Host $_.Exception.Message
    exit 1
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Write-Step "Stopping server process started by smoke script"
        try {
            Stop-Process -Id $serverProcess.Id -Force -ErrorAction Stop
        } catch {
            Write-Host ("[WARN] Failed to stop server process " + $serverProcess.Id)
        }
    }
}
