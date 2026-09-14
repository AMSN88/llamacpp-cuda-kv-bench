# llama.cpp CUDA on an 8 GiB Laptop GPU: KV Cache Source Study, llama-server Concurrency Stress, llama-bench

A measurement record of llama.cpp's CUDA backend on one **8 GiB** laptop GPU (NVIDIA RTX 4060 Laptop): build, source reading, concurrency stress testing and benchmarking.

English | [中文完整报告](README.md)

**Scope note:** this is a **condensed English version**. The complete report - full per-request tables, source-line indices and every caveat - is the Chinese `README.md`; this file keeps all key figures and conclusions while omitting most intermediate detail. All figures come from the raw logs, CSV files and `/metrics` snapshots included in this repository.

## Key findings

1. **llama.cpp has no PagedAttention.** A whole-tree search for `paged|PagedAttention|paged_attention|block_table` matches only `vendor/miniaudio/miniaudio.h` (an audio ring buffer, unrelated to attention), and there is no block table in the vLLM sense. The design is a contiguous per-layer KV tensor plus cell-slot reuse driven by a ring-cursor allocator.
2. **`-c 32768 -np 10` is physically infeasible on an 8 GiB card.** Qwen-7B's KV costs **512 KiB/token (f16)**, so 32768 tokens require **16384 MiB** of pure KV. A real `cudaMalloc failed: out of memory` log is included, and the number is predictable by exact source arithmetic.
3. **VRAM stayed essentially flat during the stress runs (only 6 to 8 MiB of variation).** llama.cpp allocates the entire KV cache once at context creation (`llama-kv-cache.cpp:274-293`) and only reuses slots afterwards, so no VRAM is requested at request time.

## Environment and versions

| Item | Value |
|---|---|
| llama.cpp | commit `555881ebc8b0fc0402b30e09258a32a7bfd13c52`, build `10121`, 2026-07-24 |
| GPU | NVIDIA GeForce RTX 4060 Laptop, **8188 MiB**, compute capability 8.9, VMM yes; idle desktop baseline about 739 MiB |
| CUDA / driver | CUDA 12.8 (nvcc V12.8.61) / driver 610.47 (UMD 13.3) |
| Compiler / build | MSVC 19.44.35228.0 for x64, cmake 3.31.6-msvc6 (Visual Studio 2022, generator `Visual Studio 17 2022`), CUDA arch `89-real`, ggml 0.17.0 |
| CPU / RAM / Python / git | AMD Ryzen 7 7840H, 16 logical cores / 15.19 GB; Python 3.9.13 (no aiohttp, the stress script uses requests + ThreadPoolExecutor); git 2.48.1 |
| Models | `Qwen-7B-Chat.Q4_K_M.gguf`, 4,899,217,600 B (**4.56 GiB**/4667 MiB), 7,721,324,544 params; Q8_0 produced locally by `llama-quantize` requantization of Q4_K_M, 8,211,787,968 B (**7.65 GiB**/7826 MiB) |

Model structure, taken from the GGUF metadata dump in `llama_quantize_q8.log` (not inferred): `qwen.context_length = 32768`, `block_count = 32`, `embedding_length = 4096`, `attention.head_count = 32`, `rope.dimension_count = 128`, tensor names `blk.N.attn_qkv.weight` (fused Q/K/V); there is no `attention.head_count_kv` key, hence no GQA (`head_count_kv == head_count == 32`).

> This repository contains **no GGUF model files and no llama.cpp source code**. Clone and build llama.cpp from https://github.com/ggml-org/llama.cpp (MIT License) to reproduce anything here.

## Repository layout

```
.
|-- README.md, README.en.md      # full report (Chinese); this condensed English version
|-- LICENSE                      # MIT, with third-party notice
|-- docs/llama_kv_cache_notes.md # KV cache source notes with per-item line-number index
|-- scripts/                     # stress_llama_server.py, stress_summarize.py, gpu_mem_analyze.py, bench_analyze.py,
|                                # run_server_experiment.ps1, run_feasible_matrix.ps1, run_bench_final.ps1
|-- results/, logs/              # numeric artifacts (CSV / summary MD / metrics snapshots); raw logs (build, server, bench, quantize)
```

`scripts/*.py` read and write `<repo>/results` by default and regenerate the summary tables; `scripts/*.ps1` take `-BinDir` / `-ModelDir` to point at a local llama.cpp build.

## Build steps

`E:\llama.cpp` was already a clean git checkout at the target commit (`git status -sb` showed no changes, HEAD was `555881ebc`), so origin and revision were verified instead of re-cloning, and the real build was executed:

```powershell
nvidia-smi ; nvcc --version ; cmake --version ; python --version   # -> <repo>/logs/env_raw.log
git -C E:\llama.cpp remote -v ; git -C E:\llama.cpp rev-parse HEAD   # 555881ebc8b0fc0402b30e09258a32a7bfd13c52
git -C E:\llama.cpp status -sb ; cmake -S E:\llama.cpp -B E:\llama.cpp\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build E:\llama.cpp\build --config Release --parallel 16
```

Windows/MSVC generators have no `nproc`, so `--parallel 16` (= `$env:NUMBER_OF_PROCESSORS`) replaces `-j$(nproc)`. Key build-log lines (`llama_build.log`): `Using CMAKE_CUDA_ARCHITECTURES=89-real`, `Including CUDA backend`, `ggml version: 0.17.0`, `ggml commit: 555881ebc`, ending in `BUILD_EXIT=0`. The resulting binaries report `version: 10121 (555881ebc)`, `built with MSVC 19.44.35228.0 for x64` and `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB): Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes, VRAM: 8187 MiB`. This build exposes `-kvu/--kv-unified` (and `-no-kvu/--no-kv-unified`), `-np/--parallel` and `-cb/--cont-batching`, so the unified/non-unified comparison was run as requested.

## KV cache architecture notes (from the source study)

The KV cache is physically one contiguous 3D tensor per layer, `[n_embd_k_gqa, kv_size, n_stream]` (`src/llama-kv-cache.cpp:231-232`). Cell metadata lives in `src/llama-kv-cells.h:32`: `pos[]` (`-1` means empty), `shift[]`, and `seq` as a `std::bitset<LLAMA_MAX_SEQ>`. Free-slot search is `find_slot()` (`src/llama-kv-cache.cpp:894`), a ring cursor plus linear scan rather than a block table. Sequence sharing is `seq_add()` (`llama-kv-cells.h:309`, `llama-kv-cache.cpp:459-488`): a cell is reference-shared by several sequences through the bitset, with **no copy-on-write**. Attention reads scan the contiguous KV densely in `nbatch_fa` chunks (`ggml/src/ggml-cuda/fattn-tile.cuh:954-977`), masking unrelated positions to `-inf`. `find_slot()`'s availability test (`src/llama-kv-cache.cpp:1038-1057`) has only two paths - the cell is empty, or it is held by a single sequence whose position was evicted by the SWA window; causal overwrite reuse within the same sequence is explicitly disabled by a comment at `:1043-1047`.

`--kv-unified` controls `n_stream` (`src/llama-kv-cache.cpp:82`, `src/llama-context.cpp:286-300`): unified keeps `n_stream = 1` while non-unified uses `n_stream = n_seq_max`, and `n_ctx_seq` is the cache's `kv_size` (`src/llama-model.cpp:2138`). With unified each sequence may use the full `n_ctx`; without it `n_ctx` is divided statically by `n_seq_max` and padded to a multiple of 256. Measurement confirms the design difference does not change total VRAM:

| Launch arguments | `n_ctx_seq` | `n_seq_max`/`n_stream` | cells | KV total (measured log) |
|---|---|---|---|---|
| `-c 4096 -np 10` | 512 | 10 / 10 | 512 | **1360.00 MiB** (K 680 + V 680) |
| `-c 5120 -np 10 --kv-unified` | 5120 | 10 / 1 | 5120 | **1360.00 MiB** (K 680 + V 680) |
| `-c 4096 -np 1` | 4096 | 1 / 1 | 4096 | **1088.00 MiB** (K 544 + V 544) |
| `-c 8192 -np 1` | 8192 | 1 / 1 | 8192 | **2176.00 MiB** (K 1088 + V 1088) |

A second easily-missed detail (`tools/server/server.cpp:146-151`): `kv_unified` is switched on automatically only when `-np` is omitted (`n_parallel is set to auto, using n_parallel = 4 and kv_unified = true`); passing `-np 10` explicitly leaves `kv_unified` at its default `false` (`common/common.h:574`), giving the statically partitioned cache. The following is a feature-level contrast, not a same-named-component contrast:

| Dimension | vLLM PagedAttention | llama.cpp |
|---|---|---|
| Physical storage | block pool of fixed-size blocks, physically scattered | one contiguous tensor per stream |
| Logical-to-physical mapping | **block table** | **no mapping table**; `llama_kv_cells.pos[]` + `slot_info.idxs[]` |
| Allocation timing | lazy per block, grows on demand | **full `kv_size` allocated at context creation** |
| Free-slot search | block-level free list | ring cursor + linear scan (`find_slot`) |
| Internal fragmentation | yes (tail block of each sequence) | no intra-block waste; non-unified mode wastes idle partitions |
| Prefix sharing | block sharing + COW | bitset reference sharing, **no COW** |
| Attention read | gather by block table | dense linear scan plus masking |

vLLM trades paging for elasticity and utilization; llama.cpp trades one contiguous large block for implementation simplicity and kernel friendliness, at the cost of allocating all KV up front - exactly why 32k context necessarily OOMs on an 8 GiB card.

## VRAM measurements and what they can and cannot prove

The original configuration `-m Qwen-7B-Chat.Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 -np 10 --cont-batching --metrics` was executed and failed (`llama_server_np10_c32768.log`, `llama_server_np10_c32768_kvu.log`):

```
llama_context: n_ctx is not divisible by n_seq_max - rounding down to 33280
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16640.00 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 17448304640
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

Adding `--kv-unified` changes the failure to `allocating 16384.00 MiB on device 0: cudaMalloc failed: out of memory`. Both numbers match source arithmetic exactly, derivable two independent ways: KV per token (f16) = `2 (K,V) x 32 layers x 4096 (n_embd_k_gqa) x 2 B` = **512 KiB**; non-unified `32768/10 = 3276 -> pad 3328`, `n_ctx = 33280`, `33280 x 512 KiB = 16640 MiB`; unified `32768 x 512 KiB = 16384 MiB`. Because f16 KV does not fit, the stress matrix uses `-ctk q8_0 -ctv q8_0` (q8_0 = 1.0625 B/element, i.e. **272 KiB/token**), and the measured KV sizes match the arithmetic bit for bit:

| Configuration | Formula | Measured KV |
|---|---|---|
| `-c 4096 -np 1` | 4096 x 272 KiB | 1088 MiB |
| `-c 8192 -np 1` | 8192 x 272 KiB | 2176 MiB |
| `-c 4096 -np 10` | pad(4096/10,256)=512, 512 x 10 = 5120 cells | 1360 MiB |
| `-c 5120 -np 10 --kv-unified` | 5120 x 1 = 5120 cells | 1360 MiB |

Load-time VRAM accounting from the `-lv 5` probe logs (CUDA0):

| Configuration | model buffer | KV buffer | compute buffer | Total |
|---|---|---|---|---|
| `-c 4096 -np 10` | 4332.75 MiB | 1360.00 MiB | 137.47 MiB | **5830 MiB** |
| `-c 5120 -np 10 --kv-unified` | 4332.75 MiB | 1360.00 MiB | 142.04 MiB | **5835 MiB** |
| `-c 8192 -np 1` | 4332.75 MiB | 2176.00 MiB | 192.09 MiB | **6701 MiB** |

VRAM during the stress runs (nvidia-smi, 100 ms sampling; `gpu_mem_*.csv`, `results/gpu_mem_summary.md`):

| run | samples | baseline MiB | peak MiB | mean MiB | peak-baseline | max GPU util |
|---|---|---|---|---|---|---|
| np1_c8192_ctx6k | 34 | 7540 | 7548 | 7546 | **+8** | 100% |
| np1_c4096_c10 | 110 | 6384 | 6390 | 6390 | **+6** | 100% |
| np10_c4096_c10 | 25 | 6668 | 6676 | 6675 | **+8** | 100% |
| np10_c5120_kvu_c10 | 35 | 6674 | 6682 | 6681 | **+8** | 100% |
| np10_c4096_kvu_overflow | 20 | 6384 | 6392 | 6390 | **+8** | 100% |

Across the whole stress window VRAM is essentially flat, varying by **6 to 8 MiB (<0.15%)**, which agrees with the source: KV and compute buffers are allocated once at context creation (`llama-kv-cache.cpp:274-293`) and only slots inside fixed arrays are reused afterwards. Measured baselines sit 74 to 100 MiB above the computed values, explained by the CUDA context, display output and alignment overhead (for example `-c 8192 -np 1`: 6701 + 739 desktop = 7440 computed, 7540 measured).

What these measurements cannot prove: `nvidia-smi` reports process-level or device-level used bytes only. It **cannot show allocator-internal fragmentation** and cannot attribute memory to an owner, so no conclusion about "VRAM fragmentation" can rest on nvidia-smi alone. What they do support: llama.cpp performs **no dynamic KV allocation at run time** (source plus the flat measured curve), so there is no run-time KV fragmentation growth at the llama.cpp level; the 32k failure is **capacity**, not fragmentation, and its required size (16640 MiB > 8188 MiB) is exactly predictable and independent of allocation order. The real llama.cpp-side waste is **out-of-partition waste in non-unified mode**: with `-np 10` each slot owns a fixed 512 cells and an idle slot cannot be used by the others (`try_clear_idle_slots()` returns immediately in non-unified mode, `tools/server/server-context.cpp:1648-1650`) - a logical waste, invisible to nvidia-smi and inferable only from logs such as `llama_kv_cache: size = ... (512 cells, 32 layers, 10/10 seqs)`. WDDM additionally allows over-commitment, so "the allocation succeeded" does not mean "it fits" - another state nvidia-smi cannot show.

## llama-server concurrency stress results

All runs used `-ngl 99 -ctk q8_0 -ctv q8_0 -fa on --cont-batching --metrics` (f16 KV does not fit). Ten clients were released simultaneously through a `threading.Barrier`; each request carried about 471 prompt tokens (about 6071 for the `-np 1 -c 8192` run) with `max_tokens=64` and `cache_prompt=false`, and asked the model to count from 1 to 60 to guarantee a long decode phase. TTFT = first streamed content token minus request start; TPOT = (last token - first token)/(output tokens - 1). Scripts: `stress_llama_server.py`, driven by `run_server_experiment.ps1` and `run_feasible_matrix.ps1`.

Client-side results (`stress_results.csv`, also in `results/stress_summary.md`):

| tag | configuration | clients | prompt tok | output tok | ok/fail | wall s | TTFT min/median/mean/max ms | TPOT min/mean/max ms | agg out tok/s |
|---|---|---|---|---|---|---|---|---|---|
| `np1_c8192_ctx6k` | `-c 8192 -np 1` non-unified | 1 | 6071 | 64 | 1/0 | 5.42 | 3549 / 3549 / 3549 / 3549 | 29.6 / 29.6 / 29.6 | 11.8 |
| `np1_c4096_c10` | `-c 4096 -np 1` non-unified | 10 | 471 | 64 | 10/0 | 16.73 | 308 / 7798 / 7800 / 15306 | 22.4 / 22.5 / 22.6 | 38.3 |
| `np10_c4096_c10` | `-c 4096 -np 10` non-unified | 10 | 471 | 41 | 10/0 | 3.80 | 986 / 1976 / 1647 / 2307 | 37.3 / 52.8 / 68.6 | **108.0** |
| `np10_c5120_kvu_c10` | `-c 5120 -np 10 --kv-unified` | 10 | 471 | 42 | 10/0 | 5.05 | 1028 / 2197 / 1817 / 2636 | 59.7 / 78.1 / 96.0 | 83.2 |
| `np10_c4096_kvu_overflow` | `-c 4096 -np 10 --kv-unified` (pool too small) | 10 | 471 | 1.4 | **8/2** | 2.28 | 249 / 1242 / 1377 / 2145 | 587 / 628 / 791 | 6.1 |

Server-side `/metrics`, counted by llama-server itself (`metrics_*.txt`):

| tag | prefill tokens | prefill s | prefill tok/s | decoded tokens | decode s | decode tok/s | `llama_decode()` calls | max n_tokens | busy slots per decode |
|---|---|---|---|---|---|---|---|---|---|
| `np1_c8192_ctx6k` | 6071 | 3.519 | 1725.2 | 64 | 1.867 | 34.3 | 66 | 6134 | 1.000 |
| `np1_c4096_c10` | 4710 | 2.241 | 2101.7 | 640 | 14.175 | 45.1 | 640 | 534 | 1.000 |
| `np10_c4096_c10` | 4710 | 10.456 | 450.5 | 410 | 21.116 | 19.4 | 43 | 511 | **9.721** |
| `np10_c5120_kvu_c10` | 4710 | 12.934 | 364.2 | 0 | 0.000 | 0.0 | 63 | 513 | **9.921** |
| `np10_c4096_kvu_overflow` | 3768 | 7.271 | 518.2 | 0 | 0.000 | 0.0 | 47 | 473 | 9.809 |

The two `--kv-unified` runs report `tokens_predicted_total` / `predicted_seconds_total` of 0 even though the clients did receive 40 to 43 tokens; this is an inconsistency between the metrics counters and the unified path in this build (or a snapshot-timing artifact), so **the server-side decode numbers for those two runs are not quoted** and their decode performance is taken from the client CSV only.

1. **`-np 1` versus `-np 10`: concurrency cuts wall-clock by 4.4x but lowers per-request efficiency.** With a single slot the ten requests queue and TTFT forms an exact arithmetic progression, 308 -> 1972 -> 3632 -> 5294 -> 6962 -> 8634 -> 10297 -> 11965 -> 13632 -> 15306 ms (step about 1670 ms, i.e. one 471-token prefill plus a 64-token decode), and `n_busy_slots_per_decode = 1.000` confirms strictly serial execution. With ten slots TTFT drops to 986-2307 ms and wall-clock falls from 16.73 s to 3.80 s with `n_busy_slots_per_decode = 9.721`, i.e. real continuous batching. The cost is TPOT rising from 22.5 ms to 52.8 ms (2.35x) because ten sequences share one GPU, but aggregate throughput rises from 38.3 to **108.0 tok/s (2.8x)**, so concurrency is a net win.
2. **A counter-intuitive but real effect: aggregate prefill throughput collapses under ten-way concurrency.** Server-side prefill is **2101.7 tok/s** for serial `-np 1` versus **450.5 tok/s** for concurrent `-np 10` (4.7x lower) for the same 4710 tokens of work. Source-consistent explanations: in non-unified mode the ubatch must be split per stream (`llama-kv-cache.cpp:965-970`; `init_batch()` uses `balloc.split_equal(n_ubatch, true, 0)` rather than `split_simple`, `:709`), which packs less efficiently; `kq_mask` becomes 4D `[n_kv, n_tokens/n_stream, 1, n_stream]` (`llama-graph.cpp:33-38`) so per-stream `n_kv` is maintained separately and attention work is multiplied by the stream count; and prefill and decode interleave inside the same `llama_decode()` (the 43 decode calls contain both). Wall-clock still favors concurrency (3.80 s versus 16.73 s), so this concerns throughput efficiency, not whether to use concurrency.
3. **`--kv-unified` on/off at the same 1360 MiB of KV: unified is measurably slower.** Per-slot context 512 versus 5120; mean TTFT 1646 ms versus 1817 ms; **mean TPOT 52.8 ms versus 78.1 ms (+48%)**; aggregate output **108.0 tok/s versus 83.2 tok/s (-23%)**; wall-clock 3.80 s versus 5.05 s. Source-backed explanation: under unified mode all ten sequences share one stream and `get_n_kv()` (`llama-kv-cache.cpp:1227-1241`) takes that stream's `used_max_p1()`, so every sequence's attention spans the whole used range of the shared pool (about 4710+ cells) instead of its own roughly 471 cells; the FA kernel scans contiguous KV in `nbatch_fa` chunks (`fattn-tile.cuh:954-977`) and scan length drives the cost.
4. **Shared-pool overflow (a negative control showing aggregate capacity is a hard constraint).** With `-c 4096 -np 10 --kv-unified` the pool holds only 4096 cells while the ten requests need 4710 together, so **2 of the 10 requests received no content tokens at all** (HTTP 200 with 0 tokens), the rest produced only 1 to 3 tokens, and TPOT inflated to 587-791 ms. Its 2.28 s wall-clock is the "fastest" only because it is failing and clearing slots. The unified pool does not raise the VRAM ceiling; it replaces a per-slot hard limit with a shared global limit. The root cause of infeasible 10-way long-context on this machine is **total KV bytes = concurrency x context length x 272 KiB (q8_0)**. The budget here is 8188 MiB total - 739 MiB desktop - 4332.75 MiB weights - about 140 MiB compute, leaving roughly **2976 MiB for KV**, which divided by 272 KiB/token gives about **11,200 cells** as this machine's hard KV capacity (the 5120 cells / 1360 MiB used here is half of that, kept as margin).
5. **Why 10 concurrent requests at 8k-16k context are infeasible, quantitatively:**

| Target | KV needed (q8_0, 272 KiB/token) | Versus about 7449 MiB usable on 8 GiB (after desktop) |
|---|---|---|
| 10 x 8k = 80k tokens | 80,000 x 272 KiB = **21,250 MiB** | 2.85x over |
| 10 x 16k = 160k tokens | **42,500 MiB** | 5.7x over |
| 10 x 32k = 320k tokens (the original task request, f16) | 320,000 x 512 KiB = **160,000 MiB** | 21.5x over |

Even quantizing KV to q4_0 (0.5625 B/element = 144 KiB/token) needs 11,250 MiB and still exceeds the card. On 8 GiB, "10 concurrent requests" and "long context" are mutually exclusive; what this repository provides instead are two curves - single-request long context (6071 tokens, TTFT 3549 ms) and 10-way short context (471 tokens, 108 tok/s aggregate).

## llama-bench results: Q4_K_M versus Q8_0

No Q8_0 model existed locally, so `llama-quantize` produced one from the Q4_K_M file (nothing was downloaded):

```
llama-quantize --allow-requantize Qwen-7B-Chat.Q4_K_M.gguf Qwen-7B-Chat.Q8_0.gguf Q8_0 16
model size  =  4666.59 MiB (5.07 BPW) ; quant size = 7825.70 MiB (8.50 BPW) ; quantize time = 18149.93 ms
```

This is a **requantized** file. Its size and speed are genuinely those of Q8_0 (8.50 BPW versus 5.07 BPW), so the throughput comparison is valid, but its **quality is that of Q4_K_M** and it cannot be used to judge real Q8_0 output quality. All runs below used `-r 5`, `-n 128`, f16 KV and `flash_attn = auto`; raw data in `results/llama_bench_q4_q8.csv`, source table `results/llama_bench_q4_q8.md`:

| Model | Size MiB | `-ngl` | test | tokens/s | stddev | VRAM peak MiB | TTFT / TPOT (converted) |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 4667 | 99 | pp512 | **2177.65** | 164.71 | 5497 | TTFT about 235.1 ms |
| Q4_K_M | 4667 | 99 | tg128 | **46.03** | 0.37 | 5497 | TPOT about 21.73 ms |
| Q4_K_M | 4667 | 99 | pp4096 | **2073.39** | 17.65 | 7339 | TTFT about 1975.5 ms |
| Q4_K_M | 4667 | 99 | tg128 | 46.41 | 0.04 | 7339 | TPOT about 21.55 ms |
| Q8_0 | 7826 | 99 | pp512 | 274.96 | 8.28 | **7778** | TTFT about 1862.1 ms |
| Q8_0 | 7826 | 99 | tg128 | 14.49 | 0.10 | **7778** | TPOT about 68.99 ms |
| Q8_0 | 7826 | 99 | pp4096 | 203.68 | 0.92 | **7779** | TTFT about 20110.4 ms |
| Q8_0 | 7826 | 99 | tg128 | 15.25 | 1.26 | **7779** | TPOT about 65.59 ms |
| Q4_K_M | 4667 | 24 | pp512 | 1342.39 | 110.05 | 4354 | TTFT about 381.4 ms |
| Q4_K_M | 4667 | 24 | tg128 | 24.12 | 0.32 | 4354 | TPOT about 41.46 ms |
| Q4_K_M | 4667 | 24 | pp4096 | 1233.30 | 29.08 | 5638 | TTFT about 3321.2 ms |
| Q4_K_M | 4667 | 24 | tg128 | 24.17 | 0.14 | 5638 | TPOT about 41.37 ms |
| Q8_0 | 7826 | 24 | pp512 | 1014.38 | 167.08 | 6458 | TTFT about 504.7 ms |
| Q8_0 | 7826 | 24 | tg128 | 14.50 | 0.37 | 6458 | TPOT about 68.98 ms |
| Q8_0 | 7826 | 24 | pp4096 | 909.96 | 13.15 | 7720 | TTFT about 4501.3 ms |
| Q8_0 | 7826 | 24 | tg128 | 14.81 | 0.07 | 7720 | TPOT about 67.54 ms |

Conversion: `TTFT_pp<N>_ms ~= 1000 * N / pp<N>_tps` and `TPOT_tg<N>_ms ~= 1000 / tg<N>_tps`. These are **baselines, not measurements**: `TTFT_pp<N>` assumes the whole prompt is prefilled in one pass at the measured rate, so it is a **lower bound** - the streaming stress harness measured TTFT = 3549 ms for a 6071-token prompt against about 2.93 s extrapolated, and is the authoritative TTFT/TPOT source. `TPOT_tg128` is the marginal decode cost at depth 0, while the same model measured TPOT = **29.6 ms** at about 6k context versus a **21.7 ms** baseline (a 36% difference). llama-bench is single-stream and bypasses HTTP, so it never exercises concurrency or queueing.

Q8_0 at `-ngl 99` does not actually fit in 8 GiB, which must be stated: llama-bench returned exit 0, but the probe log shows `load_tensors: CUDA0 model buffer size = 7195.12 MiB`, `llama_kv_cache: CUDA0 KV buffer size = 2048.00 MiB`, `sched_reserve: CUDA0 compute buffer size = 84.51 MiB` and `W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort`. That is `7195.12 + 2048 + 84.51 = 9327.6 MiB > 8188 MiB`. It still runs only because Windows/WDDM allows CUDA allocations to exceed physical VRAM with the overflow backed by shared system memory (host RAM 15.19 GB). The consequence is measurable: VRAM peaked at **7778 MiB** (near the 8188 MiB limit) where Q4_K_M used **5497 MiB**, and Q8_0 prefill fell **6 to 8 times** (274.96 versus 2177.65 tok/s), far beyond what a 1.68x larger model explains. Hence the controlled comparison with both models at `-ngl 24` (peak VRAM 4354 / 6458 MiB at p512, both with headroom):

| `-ngl 24` controlled comparison | Q4_K_M | Q8_0 | Q8_0 / Q4_K_M |
|---|---|---|---|
| pp512 tok/s | 1342.39 | 1014.38 | 0.76x |
| tg128 tok/s | 24.12 | 14.50 | 0.60x |
| VRAM peak (p512) | 4354 MiB | 6458 MiB | 1.48x |
| Model size | 4667 MiB | 7826 MiB | 1.68x |

Q8_0 therefore spends 1.68x the VRAM to become 1.32x slower at prefill and 1.66x slower at decode, and is impractical on an 8 GiB laptop GPU - it does not even fit at `-ngl 99`.

## Reproduction commands

```powershell
# $Repo = this repository root ; $LLAMA = llama.cpp repository root (clone and build it yourself)
$Repo  = "F:\dsh\downloads\llamacpp-cuda-kv-report"
$LLAMA = "E:\llama.cpp"
$Bin   = "$LLAMA\build\bin\Release"
$Model = "$LLAMA\models\Qwen-7B-Chat.Q4_K_M.gguf"

nvidia-smi ; nvcc --version ; python --version ; git --version              # 0) environment
cmake -S $LLAMA -B $LLAMA\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # 1) build the CUDA build
cmake --build $LLAMA\build --config Release --parallel 16
# 2) reproduce the 32k OOM (fails after roughly 15 seconds)
& $Repo\scripts\run_server_experiment.ps1 -Tag "np10_c32768" -SkipStress -ReadyTimeoutSec 240 `
  -BinDir $Bin -ServerArgs "-m $Model --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 -np 10 --cont-batching --metrics"
# 3) reproduce the full feasible stress matrix (roughly 3 minutes)
& $Repo\scripts\run_feasible_matrix.ps1 -Model $Model -BinDir $Bin
# 4) reproduce the KV buffer logs (-lv 5 is required to see KV buffer size / n_ctx_seq)
& "$Bin\llama-server.exe" -m $Model --port 8081 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on -lv 5 -c 5120 -np 10 --kv-unified
# 5) reproduce the Q8_0 requantization and all bench runs (roughly 10 minutes)
& "$Bin\llama-quantize.exe" --allow-requantize $Model "$LLAMA\models\Qwen-7B-Chat.Q8_0.gguf" Q8_0 16
& $Repo\scripts\run_bench_final.ps1 -BinDir $Bin -ModelDir "$LLAMA\models"
# 6) regenerate the summaries (the scripts default to <repo>/results, no arguments needed)
python $Repo\scripts\stress_summarize.py ; python $Repo\scripts\gpu_mem_analyze.py ; python $Repo\scripts\bench_analyze.py
```

## File inventory

53 files, about 0.88 MB, all text (report / scripts / CSV / logs), with no binaries and no model weights. `stress_results.csv` holds 42 rows of per-request raw data, `llama_bench_q4_q8.csv` 16 rows, and `llama_server_probe_*.log` holds the `-lv 5` KV/model buffer sizes that are the raw evidence for every KV arithmetic claim above.

| Path | Contents |
|---|---|
| `README.md`, `README.en.md` | Full report (Chinese); this condensed English version |
| `docs/llama_kv_cache_notes.md` | Source notes: structs and fields, slot/seq/cell management, paging semantics, `--kv-unified`, PagedAttention contrast, per-item line index |
| `scripts/` (7 files), `results/` (23 files), `logs/` (17 files) | scripts: `stress_llama_server.py`, `stress_summarize.py`, `gpu_mem_analyze.py`, `bench_analyze.py`, `run_server_experiment.ps1`, `run_feasible_matrix.ps1`, `run_bench_final.ps1`; results: `stress_results.csv` (42 rows), `stress_summary.md`, `llama_bench_q4_q8.csv` (16 rows)/`.md`, `gpu_mem_summary.md`, `gpu_mem_*.csv` (13), `metrics_*.txt` (5); logs: `llama_server_np10_c32768*.log` (both 32k OOM runs), `llama_server_probe_q8_ngl99.log`, `llama_build.log`, `llama_quantize_q8.log`, `llama_bench.log` and the remaining runs |

## Limitations / what is NOT proven

1. **Not executed or not possible.** A fresh `git clone` was not performed (the working tree was already clean at the target commit, so origin and revision were verified instead). `-c 32768 -np 10` stress testing is **blocked as infeasible**: `cudaMalloc failed: out of memory`, 16640 MiB of KV required against 8188 MiB available. 10 concurrent requests at 8k-16k context are likewise infeasible, needing 21-42 GB of KV (q8_0). No screenshots or plotted VRAM curves were produced; the VRAM story is given as raw 100 ms samples in `gpu_mem_*.csv` plus a summary table. No official upstream Q8_0 model was used; the local requantization above was used instead. The server-side decode counters for the two `--kv-unified` runs read 0 and are therefore not quoted.
2. **The stress prompt is not "long context".** Constrained by 8 GiB, the 10-concurrency groups used only 471 tokens per request; long context was exercised only as a single request with 6071 tokens.
3. **Single run per configuration, no averaging.** Each stress configuration was run once (10 requests) with no repetition and no mean, so its numbers carry thermal and scheduling noise; the llama-bench portion used `-r 5` and does include repetition.
4. **VRAM fragmentation is not established.** nvidia-smi cannot show allocator-internal fragmentation. What is shown is that llama.cpp performs no run-time KV allocation and therefore exhibits no run-time KV fragmentation growth at its own level; fragmentation at the driver level is neither shown nor claimed.
5. **Converted TTFT/TPOT are baselines, not measurements.** The llama-bench conversion is a lower bound, and the TPOT baseline is a depth-0 figure that real workloads exceed (36% higher at about 6k context).
6. **Q8_0 quality cannot be assessed here**, since the file is a requantization of Q4_K_M: its size and speed are real Q8_0, its accuracy is not.
7. **Q8_0 at `-ngl 99` is memory-pressure bound** and is not directly comparable to Q4_K_M at the same `-ngl`; use the `-ngl 24` pair for a controlled comparison, where Q8_0's pp4096 already approaches the limit at a 7720 MiB peak.
8. **Desktop interference.** About 739 MiB was held by desktop processes, including a wallpaper engine, during the runs. The VRAM baselines are stated explicitly, but GPU-utilization figures are affected by it.

## Third-party attribution and license

This repository is released under the MIT License; see `LICENSE`. As the third-party notice in `LICENSE` states: this repository contains measurement results, notes and tooling *about* llama.cpp (https://github.com/ggml-org/llama.cpp), which is distributed under the MIT License. No llama.cpp source code is redistributed here; the notes cite file paths and line numbers from a specific upstream commit instead. Clone and build llama.cpp from its own repository to reproduce the measurements.
