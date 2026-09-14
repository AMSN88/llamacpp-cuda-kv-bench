# run_bench_final.ps1
#
# Final llama-bench suite: Q4_K_M vs the locally requantized Q8_0.
#
# Two offload levels are used:
#   -ngl 99 : "all layers on GPU". Q8_0 does NOT actually fit in 8 GiB here
#             (7195 MiB model + KV + compute > 8188 MiB), so its numbers are taken
#             under memory pressure / WDDM shared-memory backing.
#   -ngl 24 : a level where BOTH models fit with headroom -> controlled comparison.
#
# Every invocation is sampled with nvidia-smi at 100 ms so the VRAM ceiling is visible.
# Outputs: <repo>/results/llama_bench_q4_q8.csv, <repo>/results/gpu_mem_bench_*.csv
#          <repo>/logs/llama_bench.log

param(
    [string]$Csv      = "",
    [string]$Log      = "",
    [string]$BinDir   = "E:\llama.cpp\build\bin\Release",
    [string]$ModelDir = "E:\llama.cpp\models"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$resDir   = Join-Path $repoRoot "results"
New-Item -ItemType Directory -Force -Path $resDir | Out-Null

if (-not $Csv) { $Csv = Join-Path $resDir "llama_bench_q4_q8.csv" }
if (-not $Log) { $Log = Join-Path $repoRoot "logs\llama_bench.log" }

$bench = Join-Path $BinDir "llama-bench.exe"
if (-not (Test-Path $bench)) { throw "llama-bench.exe not found at $bench (pass -BinDir)" }

$models = [ordered]@{
    "q4" = Join-Path $ModelDir "Qwen-7B-Chat.Q4_K_M.gguf"
    "q8" = Join-Path $ModelDir "Qwen-7B-Chat.Q8_0.gguf"
}

"=== llama-bench final suite $(Get-Date -Format o) ===" | Out-File $Log -Encoding utf8

$configs = @(
    @{ m = "q4"; ngl = 99; p = 512 },
    @{ m = "q4"; ngl = 99; p = 4096 },
    @{ m = "q8"; ngl = 99; p = 512 },
    @{ m = "q8"; ngl = 99; p = 4096 },
    @{ m = "q4"; ngl = 24; p = 512 },
    @{ m = "q4"; ngl = 24; p = 4096 },
    @{ m = "q8"; ngl = 24; p = 512 },
    @{ m = "q8"; ngl = 24; p = 4096 }
)

$csvRows = New-Object System.Collections.ArrayList

foreach ($c in $configs) {
    $tag = "$($c.m)_ngl$($c.ngl)_p$($c.p)"
    $mp  = $models[$c.m]
    $gpu = Join-Path $resDir "gpu_mem_bench_$tag.csv"
    Write-Output "########## $tag ##########"

    if (-not (Test-Path $mp)) {
        Write-Output "  SKIP: model missing at $mp"
        "RESULT: $tag SKIPPED (model missing: $mp)" | Out-File $Log -Append -Encoding utf8
        continue
    }

    # sampler
    "timestamp,memory.used,memory.total,utilization.gpu" | Out-File $gpu -Encoding utf8
    $job = Start-Job -ScriptBlock {
        param($gpu, $seconds)
        $end = (Get-Date).AddSeconds($seconds)
        while ((Get-Date) -lt $end) {
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null |
                Out-File $gpu -Append -Encoding utf8
            Start-Sleep -Milliseconds 100
        }
    } -ArgumentList $gpu, 240
    Start-Sleep -Milliseconds 700

    $out  = & $bench -m $mp -p $c.p -n 128 -ngl $c.ngl -r 5 -o csv 2>&1
    $code = $LASTEXITCODE

    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -Force -ErrorAction SilentlyContinue

    "--- $tag : llama-bench -m $mp -p $($c.p) -n 128 -ngl $($c.ngl) -r 5 -o csv" | Out-File $Log -Append -Encoding utf8
    $out | Out-File $Log -Append -Encoding utf8
    "--- exit=$code" | Out-File $Log -Append -Encoding utf8

    $dataRows = $out | Where-Object { $_ -match '^\s*"555' }
    if ($dataRows.Count -eq 0) {
        Write-Output "  NO DATA (exit=$code) - see $Log"
        "RESULT: $tag FAILED (exit=$code)" | Out-File $Log -Append -Encoding utf8
    } else {
        foreach ($r in $dataRows) { [void]$csvRows.Add($r) }
        foreach ($r in $dataRows) {
            $f = $r.Split(',')
            Write-Output ("  n_prompt={0} n_gen={1} tps={2}" -f $f[32], $f[33], $f[38])
        }
    }

    $used = (Get-Content $gpu | Select-Object -Skip 1 | Where-Object { $_ -match '\S' } |
             ForEach-Object { [double]($_ -split ',')[1] })
    if ($used.Count -gt 0) {
        Write-Output ("  VRAM peak = {0} MiB (samples={1})" -f ($used | Measure-Object -Maximum).Maximum, $used.Count)
    }
}

$header = "build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts"
$header | Out-File $Csv -Encoding utf8
$csvRows | Out-File $Csv -Append -Encoding utf8

Write-Output "BENCH_DONE rows=$($csvRows.Count) -> $Csv"
