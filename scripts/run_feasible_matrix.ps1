# run_feasible_matrix.ps1
#
# Sequential driver for the stress experiments that fit in 8 GiB of VRAM.
# Every run keeps its own <repo>/logs/llama_server_<tag>.log and
# <repo>/results/gpu_mem_<tag>.csv, and appends to <repo>/results/stress_results.csv.
#
# All runs use -ngl 99 -ctk q8_0 -ctv q8_0 -fa on because f16 KV does not fit.
# Total KV cells are matched between the -np 1 / -np 10 and unified / non-unified pairs:
#   -c 4096 -np 10  (non-unified) -> n_ctx_seq = pad(4096/10,256) = 512, 10 streams -> 5120 cells
#   -c 5120 -np 10 --kv-unified    -> n_ctx_seq = 5120,            1 stream  -> 5120 cells

param(
    [string]$Model  = "E:\llama.cpp\models\Qwen-7B-Chat.Q4_K_M.gguf",
    [string]$BinDir = "E:\llama.cpp\build\bin\Release"
)

$ErrorActionPreference = "Continue"
$runner = Join-Path $PSScriptRoot "run_server_experiment.ps1"

if (-not (Test-Path $Model)) { throw "model not found at $Model (pass -Model)" }

function Run-One {
    param([string]$Tag, [string]$Extra, [int]$Concurrency, [int]$PromptTokens, [int]$MaxTokens)

    $common = "-m $Model --host 127.0.0.1 --port 8080 -ngl 99 --cont-batching --metrics -ctk q8_0 -ctv q8_0 -fa on"
    $args_  = "$common $Extra"

    Write-Output "############ $Tag ############"
    Write-Output "args: $args_"
    & $runner -Tag $Tag -Port 8080 -Concurrency $Concurrency `
              -PromptTokens $PromptTokens -MaxTokens $MaxTokens `
              -ReadyTimeoutSec 300 -ServerArgs $args_ -BinDir $BinDir
    Write-Output "############ $Tag done ############"
    Start-Sleep -Seconds 3
}

# 1) long-context single stream: 1 slot, 8k ctx, ~6k token prompt, 1 client
Run-One -Tag "np1_c8192_ctx6k" -Extra "-c 8192 -np 1" -Concurrency 1 -PromptTokens 6000 -MaxTokens 64

# 2) -np 1 with 10 concurrent clients -> server must queue (1 slot only)
Run-One -Tag "np1_c4096_c10"   -Extra "-c 4096 -np 1" -Concurrency 10 -PromptTokens 384 -MaxTokens 64

# 3) -np 10, non-unified KV: 10 fixed streams of 512 cells = 5120 cells total
Run-One -Tag "np10_c4096_c10"  -Extra "-c 4096 -np 10" -Concurrency 10 -PromptTokens 384 -MaxTokens 64

# 4) -np 10, unified KV: one shared pool of 5120 cells = same total as run 3
Run-One -Tag "np10_c5120_kvu_c10" -Extra "-c 5120 -np 10 --kv-unified" -Concurrency 10 -PromptTokens 384 -MaxTokens 64

# 5) negative test: unified pool smaller than the aggregate demand
#    10 clients x ~471 prompt tokens = ~4710 > 4096 cells -> expect failures / stalling
Run-One -Tag "np10_c4096_kvu_overflow" -Extra "-c 4096 -np 10 --kv-unified" -Concurrency 10 -PromptTokens 384 -MaxTokens 64

Write-Output "MATRIX_DONE"
