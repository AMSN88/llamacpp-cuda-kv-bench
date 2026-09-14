# run_server_experiment.ps1
#
# Starts llama-server with a given argument set, waits for /health, optionally runs the
# python stress harness against it, then shuts the server down. All raw logs are kept.
#
# Paths are resolved relative to this repository:
#   <repo>/scripts/run_server_experiment.ps1   (this file)
#   <repo>/results/                            stress_results.csv, gpu_mem_*.csv, metrics_*.txt
#   <repo>/logs/                               llama_server_<tag>.log
#
# Only -BinDir and the model path inside -ServerArgs are machine specific.
#
# Example:
#   pwsh -File scripts/run_server_experiment.ps1 -Tag np10_c32768 `
#     -ServerArgs "-m E:\llama.cpp\models\Qwen-7B-Chat.Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 -np 10 --cont-batching --metrics"

param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$ServerArgs,
    [int]$Port = 8080,
    [int]$Concurrency = 10,
    [int]$PromptTokens = 8000,
    [int]$MaxTokens = 64,
    [int]$ReadyTimeoutSec = 300,
    [string]$BinDir = "E:\llama.cpp\build\bin\Release",
    [switch]$SkipStress,
    [switch]$Warmup
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resDir   = Join-Path $repoRoot "results"
$logDir   = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $resDir, $logDir | Out-Null

$exe      = Join-Path $BinDir "llama-server.exe"
$stressPy = Join-Path $PSScriptRoot "stress_llama_server.py"
$logMain  = Join-Path $logDir "llama_server_$Tag.log"
$logOut   = Join-Path $logDir "_tmp_stdout_$Tag.log"
$logErr   = Join-Path $logDir "_tmp_stderr_$Tag.log"
$baseUrl  = "http://127.0.0.1:$Port"

if (-not (Test-Path $exe))      { throw "llama-server.exe not found at $exe (pass -BinDir)" }
if (-not (Test-Path $stressPy)) { throw "stress_llama_server.py not found at $stressPy" }

"=== llama-server experiment: $Tag ===" | Out-File $logMain -Encoding utf8
"started : $(Get-Date -Format o)" | Out-File $logMain -Append -Encoding utf8
"exe     : $exe" | Out-File $logMain -Append -Encoding utf8
"args    : $ServerArgs" | Out-File $logMain -Append -Encoding utf8
"cwd     : $BinDir" | Out-File $logMain -Append -Encoding utf8
"" | Out-File $logMain -Append -Encoding utf8

# clean any stale server from a previous run
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

$proc = Start-Process -FilePath $exe -ArgumentList $ServerArgs `
        -WorkingDirectory $BinDir -PassThru -NoNewWindow `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr

Write-Output "launched pid=$($proc.Id)"

# ---- wait for /health -----------------------------------------------------
$ready  = $false
$dead   = $false
$t0     = Get-Date
while (((Get-Date) -lt $t0.AddSeconds($ReadyTimeoutSec))) {
    if ($proc.HasExited) { $dead = $true; break }
    try {
        $r = Invoke-WebRequest -Uri "$baseUrl/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        try {
            # some builds answer 503 while loading; any HTTP answer means the process is alive
            $r2 = Invoke-WebRequest -Uri "$baseUrl/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($r2) { $ready = $true; break }
        } catch { }
    }
    Start-Sleep -Milliseconds 500
}

$elapsed = ((Get-Date) - $t0).TotalSeconds
if ($dead) {
    "SERVER_EXITED during startup after $([math]::Round($elapsed,1))s, exitcode=$($proc.ExitCode)" | Out-File $logMain -Append -Encoding utf8
    Write-Output "SERVER_EXITED after $([math]::Round($elapsed,1))s exitcode=$($proc.ExitCode)"
} elseif ($ready) {
    "SERVER_READY after $([math]::Round($elapsed,1))s" | Out-File $logMain -Append -Encoding utf8
    Write-Output "SERVER_READY after $([math]::Round($elapsed,1))s"
} else {
    "SERVER_NOT_READY after ${ReadyTimeoutSec}s timeout" | Out-File $logMain -Append -Encoding utf8
    Write-Output "SERVER_NOT_READY (timeout)"
}

# ---- stress ---------------------------------------------------------------
$stressExit = "skipped"
if ($ready -and -not $SkipStress) {
    "=== stress run ===" | Out-File $logMain -Append -Encoding utf8
    $py = @(
        $stressPy,
        "--base-url", $baseUrl,
        "--concurrency", "$Concurrency",
        "--target-prompt-tokens", "$PromptTokens",
        "--max-tokens", "$MaxTokens",
        "--tag", $Tag,
        "--out-csv", (Join-Path $resDir "stress_results.csv"),
        "--gpu-csv", (Join-Path $resDir "gpu_mem_$Tag.csv"),
        "--gpu-interval", "0.1",
        "--timeout", "1800",
        "--metrics-json", (Join-Path $resDir "metrics_$Tag.txt")
    )
    if ($Warmup) { $py += "--warmup" }
    $stressOut = & python @py 2>&1
    $stressExit = $LASTEXITCODE
    $stressOut | Out-File $logMain -Append -Encoding utf8
    Write-Output "stress exit=$stressExit"
    $stressOut | Select-Object -Last 20 | ForEach-Object { Write-Output $_ }
}

# ---- shutdown -------------------------------------------------------------
Start-Sleep -Milliseconds 500
if (-not $proc.HasExited) {
    taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
    Start-Sleep -Seconds 1
}
"=== server stopped: $(Get-Date -Format o) ===" | Out-File $logMain -Append -Encoding utf8

# ---- merge stdout/stderr into the main log --------------------------------
"" | Out-File $logMain -Append -Encoding utf8
"=== SERVER STDOUT ===" | Out-File $logMain -Append -Encoding utf8
if (Test-Path $logOut) { Get-Content $logOut | Out-File $logMain -Append -Encoding utf8 }
"=== SERVER STDERR ===" | Out-File $logMain -Append -Encoding utf8
if (Test-Path $logErr) { Get-Content $logErr | Out-File $logMain -Append -Encoding utf8 }
Remove-Item $logOut, $logErr -ErrorAction SilentlyContinue

Write-Output "log -> $logMain"
