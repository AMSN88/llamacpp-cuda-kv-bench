# llama-bench: Qwen-7B-Chat Q4_K_M vs Q8_0 (RTX 4060 Laptop 8 GiB, CUDA)

- build: `555881ebc` (build 10121)
- gpu: `NVIDIA GeForce RTX 4060 Laptop GPU`
- cpu: `AMD Ryzen 7 7840H w/ Radeon 780M Graphics`
- raw data: `llama_bench_q4_q8.csv`; transcript: `llama_bench.log`
- all runs: `-r 5`, `-n 128`, `type_k = type_v = f16`, `flash_attn = auto`
- Q8_0 is a **local requantization** of the Q4_K_M file (see section 4)

## 1. Raw llama-bench throughput

| model | type | size MiB | n_gpu_layers | type_k/type_v | test | avg tokens/s | stddev | VRAM peak MiB |
|---|---|---|---|---|---|---|---|---|
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 99 | f16/f16 | pp512 | 2177.65 | 164.71 | 5497 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 99 | f16/f16 | tg128 | 46.03 | 0.37 | 5497 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 99 | f16/f16 | pp4096 | 2073.39 | 17.65 | 7339 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 99 | f16/f16 | tg128 | 46.41 | 0.04 | 7339 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 99 | f16/f16 | pp512 | 274.96 | 8.28 | 7778 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 99 | f16/f16 | tg128 | 14.49 | 0.10 | 7778 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 99 | f16/f16 | pp4096 | 203.68 | 0.92 | 7779 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 99 | f16/f16 | tg128 | 15.25 | 1.26 | 7779 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 24 | f16/f16 | pp512 | 1342.39 | 110.05 | 4354 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 24 | f16/f16 | tg128 | 24.12 | 0.32 | 4354 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 24 | f16/f16 | pp4096 | 1233.30 | 29.08 | 5638 |
| Qwen-7B-Chat.Q4_K_M.gguf | qwen 7B Q4_K - Medium | 4667 | 24 | f16/f16 | tg128 | 24.17 | 0.14 | 5638 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 24 | f16/f16 | pp512 | 1014.38 | 167.08 | 6458 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 24 | f16/f16 | tg128 | 14.50 | 0.37 | 6458 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 24 | f16/f16 | pp4096 | 909.96 | 13.15 | 7720 |
| Qwen-7B-Chat.Q8_0.gguf | qwen 7B Q8_0 | 7826 | 24 | f16/f16 | tg128 | 14.81 | 0.07 | 7720 |

## 2. Converted TTFT / TPOT baselines

```
TTFT_pp<N>_ms ~= 1000 * N / pp<N>_tps
TPOT_tg<N>_ms ~= 1000 / tg<N>_tps
```

| model | n_gpu_layers | test | tokens/s | TTFT (ms) | TPOT (ms/token) |
|---|---|---|---|---|---|
| Qwen-7B-Chat.Q4_K_M.gguf | 99 | pp512 | 2177.65 | 235.1 | - |
| Qwen-7B-Chat.Q4_K_M.gguf | 99 | tg128 | 46.03 | - | 21.726 |
| Qwen-7B-Chat.Q4_K_M.gguf | 99 | pp4096 | 2073.39 | 1975.5 | - |
| Qwen-7B-Chat.Q4_K_M.gguf | 99 | tg128 | 46.41 | - | 21.549 |
| Qwen-7B-Chat.Q8_0.gguf | 99 | pp512 | 274.96 | 1862.1 | - |
| Qwen-7B-Chat.Q8_0.gguf | 99 | tg128 | 14.49 | - | 68.990 |
| Qwen-7B-Chat.Q8_0.gguf | 99 | pp4096 | 203.68 | 20110.4 | - |
| Qwen-7B-Chat.Q8_0.gguf | 99 | tg128 | 15.25 | - | 65.589 |
| Qwen-7B-Chat.Q4_K_M.gguf | 24 | pp512 | 1342.39 | 381.4 | - |
| Qwen-7B-Chat.Q4_K_M.gguf | 24 | tg128 | 24.12 | - | 41.460 |
| Qwen-7B-Chat.Q4_K_M.gguf | 24 | pp4096 | 1233.30 | 3321.2 | - |
| Qwen-7B-Chat.Q4_K_M.gguf | 24 | tg128 | 24.17 | - | 41.371 |
| Qwen-7B-Chat.Q8_0.gguf | 24 | pp512 | 1014.38 | 504.7 | - |
| Qwen-7B-Chat.Q8_0.gguf | 24 | tg128 | 14.50 | - | 68.980 |
| Qwen-7B-Chat.Q8_0.gguf | 24 | pp4096 | 909.96 | 4501.3 | - |
| Qwen-7B-Chat.Q8_0.gguf | 24 | tg128 | 14.81 | - | 67.542 |

## 3. Caveats (read before quoting any number)

1. `TTFT_pp<N>_ms` is `1000*N/pp_tps`, i.e. it assumes the whole prompt is
   prefilled in one go at the measured rate. It is a **baseline/lower bound**,
   not a measured first-token latency. The streaming stress harness
   (`stress_results.csv`) is the authoritative TTFT/TPOT source.
2. `TPOT_tg<N>_ms` is `1000/tg_tps` at depth 0 (empty context). Real TPOT grows
   with context because attention reads more KV. Measured evidence: the stress
   run at ~6k context gave TPOT = 29.6 ms while this baseline says 21.7 ms.
3. llama-bench is single-stream and bypasses HTTP/JSON; it does not exercise slots.
4. **Q8_0 at `-ngl 99` does not fit in 8 GiB VRAM.** llama.cpp reports
   `CUDA0 model buffer size = 7195.12 MiB` plus KV plus compute, i.e. > 8188 MiB,
   and warns `failed to fit params to free device memory: n_gpu_layers already set
   by user to 99, abort`. The run completes only because Windows/WDDM backs the
   overflow with shared system memory. Its throughput is therefore **memory-pressure
   bound** (VRAM peaked at ~7778 MiB) and is not comparable to Q4_K_M at `-ngl 99`
   (which peaked at 5497 MiB). Use the `-ngl 24` pair for the controlled comparison.
5. Q8_0 is a **requantization**: `llama-quantize --allow-requantize` on the Q4_K_M
   file. Its size/speed are genuinely Q8_0 (8.50 BPW vs 5.07 BPW) so the
   throughput comparison is valid, but its **quality is that of Q4_K_M**, not of a
   real Q8_0 release.
