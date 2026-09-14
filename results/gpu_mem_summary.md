# GPU memory samples (nvidia-smi, 100 ms interval)

Each file covers exactly the wall-clock window of the corresponding stress run
(sampler started just before the requests were released, stopped ~0.5 s after the
last response). `memory.used` includes the desktop compositor, so the same ~0.7 GiB
baseline is present in every run.

| run | samples | baseline MiB | peak MiB | mean MiB | peak-baseline MiB | max GPU util % |
|---|---|---|---|---|---|---|
| np10_c4096_c10 | 25 | 6668 | 6676 | 6675 | 8 | 100 |
| np10_c4096_kvu_overflow | 20 | 6384 | 6392 | 6390 | 8 | 100 |
| np10_c5120_kvu_c10 | 35 | 6674 | 6682 | 6681 | 8 | 100 |
| np1_c4096_c10 | 110 | 6384 | 6390 | 6390 | 6 | 100 |
| np1_c8192_ctx6k | 34 | 7540 | 7548 | 7546 | 8 | 100 |

## Per-run detail (first / peak / last sample)

- **np10_c4096_c10** (25 samples): first `2026/09/12 23:49:51.695` = 6668 MiB, peak `2026/09/12 23:49:52.370` = 6676 MiB, last `2026/09/12 23:49:56.224` = 6676 MiB
- **np10_c4096_kvu_overflow** (20 samples): first `2026/09/12 23:50:30.135` = 6384 MiB, peak `2026/09/12 23:50:32.556` = 6392 MiB, last `2026/09/12 23:50:33.212` = 6392 MiB
- **np10_c5120_kvu_c10** (35 samples): first `2026/09/12 23:50:15.513` = 6674 MiB, peak `2026/09/12 23:50:18.393` = 6682 MiB, last `2026/09/12 23:50:21.334` = 6682 MiB
- **np1_c4096_c10** (110 samples): first `2026/09/12 23:49:13.465` = 6384 MiB, peak `2026/09/12 23:49:13.790` = 6390 MiB, last `2026/09/12 23:49:30.937` = 6390 MiB
- **np1_c8192_ctx6k** (34 samples): first `2026/09/12 23:48:47.099` = 7540 MiB, peak `2026/09/12 23:48:50.620` = 7548 MiB, last `2026/09/12 23:48:53.251` = 7548 MiB

Note: nvidia-smi reports the process-wide device allocation granularity; it cannot
show allocator fragmentation or per-allocation ownership. See README_llama.md for
what can and cannot be concluded from these numbers.
