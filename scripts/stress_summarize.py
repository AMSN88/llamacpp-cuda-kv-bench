#!/usr/bin/env python3
"""
stress_summarize.py -- combine stress_results.csv with the per-run server /metrics
snapshots into the markdown tables used by README_llama.md.

usage: python stress_summarize.py <output_dir>
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def read_metrics(path):
    vals = {}
    if not os.path.exists(path):
        return vals
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    vals[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return vals


def main():
    # default: <repo>/results (this script lives in <repo>/scripts)
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_repo, "results")
    csv_path = os.path.join(d, "stress_results.csv")
    dst = os.path.join(d, "stress_summary.md")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["run_tag"]].append(r)

    CONFIG = {
        "np1_c8192_ctx6k":        "-c 8192 -np 1 (non-unified), 1 client",
        "np1_c4096_c10":          "-c 4096 -np 1 (non-unified), 10 clients",
        "np10_c4096_c10":         "-c 4096 -np 10 (non-unified), 10 clients",
        "np10_c5120_kvu_c10":     "-c 5120 -np 10 --kv-unified, 10 clients",
        "np10_c4096_kvu_overflow": "-c 4096 -np 10 --kv-unified (pool too small), 10 clients",
    }

    out = []
    out.append("# llama-server concurrent stress test - summary")
    out.append("")
    out.append("Client-side numbers come from `stress_results.csv`; server-side numbers come from")
    out.append("the `/metrics` snapshot taken right after each run (`metrics_<tag>.txt`).")
    out.append("All runs: Qwen-7B-Chat Q4_K_M, `-ngl 99 -ctk q8_0 -ctv q8_0 -fa on --cont-batching`.")
    out.append("")
    out.append("TTFT = first streamed content token - request start. "
               "TPOT = (last token - first token)/(tokens-1).")
    out.append("")
    out.append("## Client-side")
    out.append("")
    out.append("| tag | config | clients | prompt tok | completion tok | ok/fail | wall s | TTFT min/median/mean/max ms | TPOT min/mean/max ms | agg out tok/s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")

    order = ["np1_c8192_ctx6k", "np1_c4096_c10", "np10_c4096_c10",
             "np10_c5120_kvu_c10", "np10_c4096_kvu_overflow"]

    for tag in order:
        rs = by_tag.get(tag)
        if not rs:
            continue
        ok = [r for r in rs if not r["error"]]
        fail = len(rs) - len(ok)
        ttft = [float(r["ttft_ms"]) for r in ok if r["ttft_ms"]]
        tpot = [float(r["tpot_ms"]) for r in ok if r["tpot_ms"]]
        comp = [int(r["completion_tokens"]) for r in rs if r["completion_tokens"]]
        prompt = [int(r["prompt_tokens"]) for r in rs if r["prompt_tokens"]]
        total_ms = [float(r["total_ms"]) for r in rs if r["total_ms"]]
        wall = max(total_ms) / 1000.0 if total_ms else 0.0
        agg = (sum(comp) / wall) if wall else 0.0

        out.append("| %s | %s | %d | %s | %.1f | %d/%d | %.2f | %s | %s | %.1f |"
                   % (tag,
                      CONFIG.get(tag, ""),
                      len(rs),
                      ("%d" % (sum(prompt) // len(prompt))) if prompt else "n/a",
                      (sum(comp) / len(comp)) if comp else 0.0,
                      len(ok), fail,
                      wall,
                      ("%.0f / %.0f / %.0f / %.0f"
                       % (min(ttft), median(ttft), sum(ttft) / len(ttft), max(ttft))) if ttft else "n/a",
                      ("%.1f / %.1f / %.1f"
                       % (min(tpot), sum(tpot) / len(tpot), max(tpot))) if tpot else "n/a",
                      agg))

    out.append("")
    out.append("## Server-side `/metrics` (measured by llama-server itself)")
    out.append("")
    out.append("| tag | prompt tokens | prefill s | prefill tok/s | predicted tokens | decode s | decode tok/s | llama_decode() calls | max n_tokens | busy slots/decode |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag in order:
        m = read_metrics(os.path.join(d, "metrics_%s.txt" % tag))
        if not m:
            continue
        g = m.get
        out.append("| %s | %.0f | %.3f | %.1f | %.0f | %.3f | %.1f | %.0f | %.0f | %.3f |"
                   % (tag,
                      g("llamacpp:prompt_tokens_total", 0),
                      g("llamacpp:prompt_seconds_total", 0),
                      g("llamacpp:prompt_tokens_seconds", 0),
                      g("llamacpp:tokens_predicted_total", 0),
                      g("llamacpp:tokens_predicted_seconds_total", 0),
                      g("llamacpp:predicted_tokens_seconds", 0),
                      g("llamacpp:n_decode_total", 0),
                      g("llamacpp:n_tokens_max", 0),
                      g("llamacpp:n_busy_slots_per_decode", 0)))

    out.append("")
    out.append("## Per-request detail")
    out.append("")
    for tag in order:
        rs = by_tag.get(tag)
        if not rs:
            continue
        out.append("### %s" % tag)
        out.append("")
        out.append("| req | http | TTFT ms | TPOT ms | total ms | out tok | error |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda x: int(x["request_id"])):
            out.append("| %s | %s | %s | %s | %s | %s | %s |"
                       % (r["request_id"], r["http_status"],
                          r["ttft_ms"] or "-", r["tpot_ms"] or "-",
                          r["total_ms"] or "-", r["completion_tokens"] or "-",
                          r["error"] or ""))
        out.append("")

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[:60]))
    print("...")
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
