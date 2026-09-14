#!/usr/bin/env python3
"""
bench_analyze.py -- turn raw llama-bench CSV into the markdown deliverable.

llama-bench emits one row per test:
    n_prompt > 0, n_gen == 0  ->  prompt processing (prefill), `pp<N>`
    n_gen > 0,   n_prompt == 0 ->  token generation (decode),  `tg<N>`

Conversions (the caveat matters and is printed in the output):
    TTFT_pp<N>_ms ~= 1000 * N / pp<N>_tps
    TPOT_tg<N>_ms ~= 1000 / tg<N>_tps

VRAM peaks are read from the gpu_mem_bench_<model>_ngl<ngl>_p<prompt>.csv files that
run_bench_final.ps1 produced alongside each invocation.

usage: python bench_analyze.py <output_dir>
"""

import csv
import os
import sys


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def vram_peak(path):
    if not os.path.exists(path):
        return None
    peak = None
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["memory.used"])
            except (KeyError, TypeError, ValueError):
                continue
            peak = v if peak is None else max(peak, v)
    return peak


def main():
    # default: <repo>/results (this script lives in <repo>/scripts)
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_repo, "results")
    src = os.path.join(d, "llama_bench_q4_q8.csv")
    dst = os.path.join(d, "llama_bench_q4_q8.md")

    with open(src, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    recs = []
    for i, r in enumerate(rows):
        np_ = int(r.get("n_prompt") or 0)
        ng_ = int(r.get("n_gen") or 0)
        ts = fnum(r.get("avg_ts"))
        sd = fnum(r.get("stddev_ts"))
        if np_ > 0:
            test, n = "pp%d" % np_, np_
        elif ng_ > 0:
            test, n = "tg%d" % ng_, ng_
        else:
            continue
        short = "q4" if "Q4_K" in r.get("model_filename", "") else "q8"
        # llama-bench emits exactly two rows (pp then tg) per invocation, in order
        recs.append(dict(
            inv=i // 2,
            model=os.path.basename(r.get("model_filename", "?")),
            short=short,
            mtype=r.get("model_type", "?"),
            msize=int(r.get("model_size") or 0),
            ngl=int(r.get("n_gpu_layers") or 0),
            tk=r.get("type_k", "?"),
            tv=r.get("type_v", "?"),
            fa=r.get("flash_attn", "?"),
            nparams=int(r.get("model_n_params") or 0),
            test=test, n=n, ts=ts, sd=sd,
        ))

    # invocation index -> prompt size used by run_bench_final.ps1 (pp row gives it directly,
    # but the tg row of the same invocation carries n_prompt == 0)
    inv_prompt = {}
    for r in recs:
        if r["test"].startswith("pp"):
            inv_prompt[r["inv"]] = r["n"]

    vram_cache = {}

    def peak_for(r):
        key = (r["short"], r["ngl"], inv_prompt.get(r["inv"], 0))
        if key not in vram_cache:
            p = os.path.join(d, "gpu_mem_bench_%s_ngl%d_p%d.csv" % key)
            vram_cache[key] = vram_peak(p)
        return vram_cache[key]

    out = []
    out.append("# llama-bench: Qwen-7B-Chat Q4_K_M vs Q8_0 (RTX 4060 Laptop 8 GiB, CUDA)")
    out.append("")
    if recs:
        first = rows[0]
        out.append("- build: `%s` (build %s)" % (first.get("build_commit"), first.get("build_number")))
        out.append("- gpu: `%s`" % first.get("gpu_info"))
        out.append("- cpu: `%s`" % (first.get("cpu_info") or "").strip())
    out.append("- raw data: `llama_bench_q4_q8.csv`; transcript: `llama_bench.log`")
    out.append("- all runs: `-r 5`, `-n 128`, `type_k = type_v = f16`, `flash_attn = auto`")
    out.append("- Q8_0 is a **local requantization** of the Q4_K_M file (see section 4)")
    out.append("")
    out.append("## 1. Raw llama-bench throughput")
    out.append("")
    out.append("| model | type | size MiB | n_gpu_layers | type_k/type_v | test | avg tokens/s | stddev | VRAM peak MiB |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        peak = peak_for(r)
        out.append("| %s | %s | %.0f | %d | %s/%s | %s | %.2f | %.2f | %s |"
                   % (r["model"], r["mtype"], r["msize"] / 1024.0 / 1024.0, r["ngl"],
                      r["tk"], r["tv"], r["test"], r["ts"], r["sd"],
                      ("%.0f" % peak) if peak else "-"))

    out.append("")
    out.append("## 2. Converted TTFT / TPOT baselines")
    out.append("")
    out.append("```")
    out.append("TTFT_pp<N>_ms ~= 1000 * N / pp<N>_tps")
    out.append("TPOT_tg<N>_ms ~= 1000 / tg<N>_tps")
    out.append("```")
    out.append("")
    out.append("| model | n_gpu_layers | test | tokens/s | TTFT (ms) | TPOT (ms/token) |")
    out.append("|---|---|---|---|---|---|")
    for r in recs:
        if r["test"].startswith("pp"):
            conv = "%.1f" % (1000.0 * r["n"] / r["ts"]) if r["ts"] else "-"
            tpot = "-"
        else:
            conv = "-"
            tpot = "%.3f" % (1000.0 / r["ts"]) if r["ts"] else "-"
        out.append("| %s | %d | %s | %.2f | %s | %s |"
                   % (r["model"], r["ngl"], r["test"], r["ts"], conv, tpot))

    out.append("")
    out.append("## 3. Caveats (read before quoting any number)")
    out.append("")
    out.append("1. `TTFT_pp<N>_ms` is `1000*N/pp_tps`, i.e. it assumes the whole prompt is")
    out.append("   prefilled in one go at the measured rate. It is a **baseline/lower bound**,")
    out.append("   not a measured first-token latency. The streaming stress harness")
    out.append("   (`stress_results.csv`) is the authoritative TTFT/TPOT source.")
    out.append("2. `TPOT_tg<N>_ms` is `1000/tg_tps` at depth 0 (empty context). Real TPOT grows")
    out.append("   with context because attention reads more KV. Measured evidence: the stress")
    out.append("   run at ~6k context gave TPOT = 29.6 ms while this baseline says 21.7 ms.")
    out.append("3. llama-bench is single-stream and bypasses HTTP/JSON; it does not exercise slots.")
    out.append("4. **Q8_0 at `-ngl 99` does not fit in 8 GiB VRAM.** llama.cpp reports")
    out.append("   `CUDA0 model buffer size = 7195.12 MiB` plus KV plus compute, i.e. > 8188 MiB,")
    out.append("   and warns `failed to fit params to free device memory: n_gpu_layers already set")
    out.append("   by user to 99, abort`. The run completes only because Windows/WDDM backs the")
    out.append("   overflow with shared system memory. Its throughput is therefore **memory-pressure")
    out.append("   bound** (VRAM peaked at ~7778 MiB) and is not comparable to Q4_K_M at `-ngl 99`")
    out.append("   (which peaked at 5497 MiB). Use the `-ngl 24` pair for the controlled comparison.")
    out.append("5. Q8_0 is a **requantization**: `llama-quantize --allow-requantize` on the Q4_K_M")
    out.append("   file. Its size/speed are genuinely Q8_0 (8.50 BPW vs 5.07 BPW) so the")
    out.append("   throughput comparison is valid, but its **quality is that of Q4_K_M**, not of a")
    out.append("   real Q8_0 release.")

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote %s (%d rows)" % (dst, len(recs)))


if __name__ == "__main__":
    main()
