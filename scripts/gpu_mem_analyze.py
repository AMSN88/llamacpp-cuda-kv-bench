#!/usr/bin/env python3
"""
gpu_mem_analyze.py -- summarize the nvidia-smi samples taken during each stress run.

Reads every gpu_mem_*.csv in the output dir and prints / writes a summary table:
    samples, baseline MiB, max MiB, mean MiB, delta vs baseline, max GPU util.

usage: python gpu_mem_analyze.py <dir> <output.md>
"""

import csv
import glob
import os
import sys


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                rows.append((
                    r["timestamp"].strip(),
                    float(r["memory.used"]),
                    float(r["memory.total"]),
                    float(r["utilization.gpu"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def main():
    # default: <repo>/results (this script lives in <repo>/scripts)
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_repo, "results")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "gpu_mem_summary.md")

    files = sorted(f for f in glob.glob(os.path.join(d, "gpu_mem_*.csv"))
                   if not os.path.basename(f).startswith("gpu_mem_bench_"))
    if not files:
        sys.exit("no gpu_mem_*.csv found in %s" % d)

    out = []
    out.append("# GPU memory samples (nvidia-smi, 100 ms interval)")
    out.append("")
    out.append("Each file covers exactly the wall-clock window of the corresponding stress run")
    out.append("(sampler started just before the requests were released, stopped ~0.5 s after the")
    out.append("last response). `memory.used` includes the desktop compositor, so the same ~0.7 GiB")
    out.append("baseline is present in every run.")
    out.append("")
    out.append("| run | samples | baseline MiB | peak MiB | mean MiB | peak-baseline MiB | max GPU util % |")
    out.append("|---|---|---|---|---|---|---|")

    summary = {}
    for f in files:
        tag = os.path.basename(f)[len("gpu_mem_"):-len(".csv")]
        rows = load(f)
        if not rows:
            out.append("| %s | 0 | - | - | - | - | - |" % tag)
            continue
        used = [r[1] for r in rows]
        util = [r[3] for r in rows]
        base = used[0]
        peak = max(used)
        mean = sum(used) / len(used)
        summary[tag] = dict(samples=len(rows), base=base, peak=peak, mean=mean,
                            delta=peak - base, util=max(util))
        out.append("| %s | %d | %.0f | %.0f | %.0f | %.0f | %.0f |"
                   % (tag, len(rows), base, peak, mean, peak - base, max(util)))

    out.append("")
    out.append("## Per-run detail (first / peak / last sample)")
    out.append("")
    for f in files:
        tag = os.path.basename(f)[len("gpu_mem_"):-len(".csv")]
        rows = load(f)
        if not rows:
            continue
        peak_i = max(range(len(rows)), key=lambda i: rows[i][1])
        out.append("- **%s** (%d samples): first `%s` = %.0f MiB, peak `%s` = %.0f MiB, "
                   "last `%s` = %.0f MiB"
                   % (tag, len(rows), rows[0][0], rows[0][1],
                      rows[peak_i][0], rows[peak_i][1],
                      rows[-1][0], rows[-1][1]))
    out.append("")
    out.append("Note: nvidia-smi reports the process-wide device allocation granularity; it cannot")
    out.append("show allocator fragmentation or per-allocation ownership. See README_llama.md for")
    out.append("what can and cannot be concluded from these numbers.")

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    print("\n".join(out))


if __name__ == "__main__":
    main()
