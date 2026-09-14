#!/usr/bin/env python3
"""
stress_llama_server.py -- concurrent long-context stress test for llama-server.

Records, per request:
    request_start, first_token_time, last_token_time, prompt tokens, completion
    tokens, TTFT, TPOT, total wall time, HTTP status and error.

Also samples nvidia-smi every --gpu-interval seconds for the whole measured
window and writes gpu_mem_*.csv.

Uses requests + ThreadPoolExecutor (aiohttp is not installed in this env).

Examples
--------
python stress_llama_server.py --concurrency 10 --target-prompt-tokens 8000 \
    --max-tokens 64 --tag np10_f16 --out-csv E:\\output\\stress_results.csv \
    --gpu-csv E:\\output\\gpu_mem_np10_f16.csv

TTFT/TPOT definitions (must stay consistent with the report):
    TTFT_ms  = (first_token_time - request_start) * 1000
    TPOT_ms  = (last_token_time - first_token_time) * 1000 / (completion_tokens - 1)
               (only when completion_tokens >= 2, else empty)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:
    sys.exit("this script needs the 'requests' package (pip install requests)")

# Filler text used to build long prompts. Kept deliberately boring and
# repetitive, with a per-request marker so the requests are distinguishable.
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the diligent engineer "
    "records throughput, latency and memory usage of the inference server. "
)

CSV_HEADER = [
    "run_tag",
    "request_id",
    "http_status",
    "request_start_epoch",
    "first_token_epoch",
    "last_token_epoch",
    "prompt_tokens",
    "completion_tokens",
    "ttft_ms",
    "tpot_ms",
    "total_ms",
    "stream_chunks",
    "error",
]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------
# GPU sampler
# --------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """Polls nvidia-smi and appends raw csv rows (no header) to gpu_csv."""

    QUERY = "timestamp,memory.used,memory.total,utilization.gpu"

    def __init__(self, gpu_csv, interval):
        super().__init__(daemon=True)
        self.gpu_csv = gpu_csv
        self.interval = interval
        # NOTE: do not call this ``_stop`` -- threading.Thread already defines a
        # method with that name and shadowing it breaks join().
        self._stop_evt = threading.Event()
        self.samples = 0
        self.errors = 0

    def run(self):
        with open(self.gpu_csv, "w", newline="", encoding="utf-8") as fh:
            fh.write("timestamp,memory.used,memory.total,utilization.gpu\n")
            fh.flush()
            while not self._stop_evt.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=" + self.QUERY,
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    for line in out.stdout.strip().splitlines():
                        if line.strip():
                            fh.write(line.strip() + "\n")
                            self.samples += 1
                    fh.flush()
                except Exception as exc:  # keep sampling, record the failure
                    self.errors += 1
                    if self.errors <= 3:
                        log("gpu sampler error: %r" % (exc,))
                self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------
def calibrate_chars_per_token(session, base_url, model):
    """Ask /tokenize how many characters map to how many tokens.

    Returns chars-per-token. Falls back to a sane default on any error.
    """
    probe = FILLER_SENTENCE * 20
    try:
        r = session.post(base_url + "/tokenize",
                         json={"content": probe, "model": model}, timeout=30)
        r.raise_for_status()
        n = len(r.json().get("tokens", []))
        if n > 0:
            cpt = len(probe) / float(n)
            log("calibration: %d chars -> %d tokens (%.3f chars/token)"
                % (len(probe), n, cpt))
            return cpt
    except Exception as exc:
        log("calibration failed (%r), using default ratio" % (exc,))
    return 3.6


def build_prompt(target_tokens, chars_per_token, request_id):
    """Repeat filler until we reach roughly target_tokens worth of text."""
    marker = "\n[request_id=%d]\n" % request_id
    target_chars = max(64, int(target_tokens * chars_per_token))
    parts = [marker]
    acc = len(marker)
    while acc < target_chars:
        parts.append(FILLER_SENTENCE)
        acc += len(FILLER_SENTENCE)
    parts.append(
        "\n[request_id=%d] Ignore all the filler above. Now count out loud from 1 to 60, "
        "separated by single spaces, and output nothing else.\n" % request_id
    )
    return "".join(parts), acc


# --------------------------------------------------------------------------
# one streaming request
# --------------------------------------------------------------------------
def do_request(req_id, args, barrier, prompt, results, metrics_out):
    row = {k: "" for k in CSV_HEADER}
    row["run_tag"] = args.tag
    row["request_id"] = req_id

    session = requests.Session()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }

    first_tok = None
    last_tok = None
    chunks = 0
    usage = None
    t_start = None

    try:
        # synchronise so that all workers hit the server at the same moment
        barrier.wait(timeout=120)
        t_start = time.time()
        row["request_start_epoch"] = "%.6f" % t_start

        with session.post(args.base_url + args.endpoint, json=payload,
                          stream=True, timeout=args.timeout) as resp:
            row["http_status"] = resp.status_code
            if resp.status_code != 200:
                body = resp.text[:500]
                row["error"] = "HTTP_%d: %s" % (resp.status_code, body.replace("\n", " "))
                results.append(row)
                return row

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        now = time.time()
                        if first_tok is None:
                            first_tok = now
                            row["first_token_epoch"] = "%.6f" % now
                        last_tok = now
                        chunks += 1

        t_end = time.time()
        if last_tok is not None:
            row["last_token_epoch"] = "%.6f" % last_tok

        row["stream_chunks"] = chunks
        if usage:
            row["prompt_tokens"] = usage.get("prompt_tokens", "")
            row["completion_tokens"] = usage.get("completion_tokens", "")
        if not row["completion_tokens"]:
            row["completion_tokens"] = chunks

        if first_tok is not None:
            row["ttft_ms"] = "%.2f" % ((first_tok - t_start) * 1000.0)
        n_out = row["completion_tokens"]
        try:
            n_out = int(n_out)
        except (TypeError, ValueError):
            n_out = 0
        if first_tok is not None and last_tok is not None and n_out >= 2:
            row["tpot_ms"] = "%.3f" % ((last_tok - first_tok) * 1000.0 / (n_out - 1))
        row["total_ms"] = "%.2f" % ((t_end - t_start) * 1000.0)
        if first_tok is None:
            row["error"] = row["error"] or "NO_CONTENT_TOKENS"
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc).replace("\n", " ")[:400])
        if t_start is not None and not row["total_ms"]:
            row["total_ms"] = "%.2f" % ((time.time() - t_start) * 1000.0)
    finally:
        session.close()

    results.append(row)
    return row


# --------------------------------------------------------------------------
def snapshot_metrics(args, out_path):
    try:
        r = requests.get(args.base_url + "/metrics", timeout=20)
        if r.status_code == 200:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(r.text)
            log("wrote %s (%d bytes)" % (out_path, len(r.text)))
        else:
            log("/metrics returned %d" % r.status_code)
    except Exception as exc:
        log("could not read /metrics: %r" % (exc,))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--target-prompt-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", default="run")
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-csv", default=os.path.join(_repo, "results", "stress_results.csv"))
    ap.add_argument("--gpu-csv", default="")
    ap.add_argument("--gpu-interval", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--warmup", action="store_true",
                    help="send one small request before the measured run")
    ap.add_argument("--metrics-json", default="",
                    help="path to dump the /metrics snapshot taken after the run")
    args = ap.parse_args()

    log("target=%s endpoint=%s concurrency=%d target_prompt_tokens=%d max_tokens=%d"
        % (args.base_url, args.endpoint, args.concurrency,
           args.target_prompt_tokens, args.max_tokens))

    session = requests.Session()
    try:
        props = session.get(args.base_url + "/props", timeout=30).json()
        log("/props: model_path=%s n_ctx=%s"
            % (props.get("model_path"), props.get("default_generation_settings", {}).get("n_ctx")))
    except Exception as exc:
        log("could not read /props: %r" % (exc,))

    if args.warmup:
        log("warmup request ...")
        try:
            session.post(args.base_url + args.endpoint,
                         json={"model": args.model,
                               "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 4, "temperature": 0.0},
                         timeout=args.timeout)
            log("warmup done")
        except Exception as exc:
            log("warmup failed: %r" % (exc,))

    cpt = calibrate_chars_per_token(session, args.base_url, args.model)

    prompts = []
    for i in range(args.concurrency):
        p, nchars = build_prompt(args.target_prompt_tokens, cpt, i)
        prompts.append(p)
    log("built %d prompts of ~%d chars each" % (len(prompts), len(prompts[0])))

    sampler = None
    if args.gpu_csv:
        sampler = GpuSampler(args.gpu_csv, args.gpu_interval)
        sampler.start()
        log("gpu sampler started -> %s (interval %.0f ms)"
            % (args.gpu_csv, args.gpu_interval * 1000))

    barrier = threading.Barrier(args.concurrency + 1)
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(do_request, i, args, barrier, prompts[i], results, None)
                   for i in range(args.concurrency)]
        time.sleep(0.3)
        barrier.wait(timeout=120)          # release all workers together
        for f in futures:
            f.result()
    wall = time.time() - t0
    log("all requests finished in %.2f s" % wall)

    if sampler:
        try:
            time.sleep(0.5)                # capture the tail of the memory curve
            sampler.stop()
            log("gpu sampler stopped: %d samples, %d errors"
                % (sampler.samples, sampler.errors))
        except Exception as exc:
            log("gpu sampler stop failed: %r" % (exc,))

    if args.metrics_json:
        try:
            snapshot_metrics(args, args.metrics_json)
        except Exception as exc:
            log("metrics snapshot failed: %r" % (exc,))

    # write / append csv -- never let an error above lose the measurements
    new_file = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        for row in sorted(results, key=lambda r: r["request_id"]):
            w.writerow(row)
    log("wrote %d rows -> %s" % (len(results), args.out_csv))

    # console summary
    ok = [r for r in results if not r["error"]]
    print("-" * 78)
    print("tag=%s  concurrency=%d  wall=%.2fs  ok=%d  failed=%d"
          % (args.tag, args.concurrency, wall, len(ok), len(results) - len(ok)))
    if ok:
        ttfts = [float(r["ttft_ms"]) for r in ok if r["ttft_ms"]]
        tpots = [float(r["tpot_ms"]) for r in ok if r["tpot_ms"]]
        ptok = [int(r["prompt_tokens"]) for r in ok if r["prompt_tokens"]]
        ctok = [int(r["completion_tokens"]) for r in ok if r["completion_tokens"]]
        if ttfts:
            print("TTFT  ms: min=%.1f mean=%.1f max=%.1f"
                  % (min(ttfts), sum(ttfts) / len(ttfts), max(ttfts)))
        if tpots:
            print("TPOT  ms: min=%.3f mean=%.3f max=%.3f"
                  % (min(tpots), sum(tpots) / len(tpots), max(tpots)))
        if ptok:
            print("prompt tokens: mean=%.0f  (n=%d)" % (sum(ptok) / len(ptok), len(ptok)))
        if ctok:
            print("completion tokens: mean=%.1f" % (sum(ctok) / len(ctok)))
        agg = sum(ctok) / wall if ctok else 0.0
        print("aggregate output throughput = %.2f tok/s" % agg)
    for r in results:
        if r["error"]:
            print("  req %s FAILED: %s" % (r["request_id"], r["error"]))
    print("-" * 78)


if __name__ == "__main__":
    main()
