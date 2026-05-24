"""
bench_serve_prefill.py
──────────────────────
Hit a running vLLM serve with prompts of varying lengths and isolate the
prefill time. Used to validate WMMA integration end-to-end.

Method:
  For each target prompt length L in {16, 64, 128, 256, 512, 1024}:
    1. Send 2 warmup requests at L (max_tokens=1) — first one may pay
       Triton autotune; second one warms HTTP / CUDAGraph / etc.
    2. Send N_MEAS requests at L (max_tokens=1) and average.
  Wall-clock at max_tokens=1 ≈ prefill + 1 decode + network overhead.
  Network overhead is ~1-2 ms locally; decode is ~10 ms; the rest is prefill.

Usage:
  python3 bench_serve_prefill.py \\
      --url http://localhost:8001/v1/completions \\
      --model /mnt/models/Qwen3.6-27B-FP8
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


PROMPT_LENGTHS = [16, 64, 128, 256, 512, 1024]
N_WARMUP = 2
N_MEAS   = 3

FILLER_UNIT = "the "
SEED_PROMPT = "Hello world. "


def make_prompt(target_tokens: int) -> str:
    """Build a prompt approximating target_tokens BPE tokens."""
    if target_tokens <= 4:
        return SEED_PROMPT
    n_filler = max(0, target_tokens - 4)
    return SEED_PROMPT + (FILLER_UNIT * n_filler)


def time_one(url, model, prompt, max_tokens, timeout):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    usage = body.get("usage", {})
    return {
        "elapsed": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=PROMPT_LENGTHS)
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--meas", type=int, default=N_MEAS)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds — autotune can take 5-10 min on a cold shape")
    args = ap.parse_args()

    print(f"endpoint: {args.url}")
    print(f"model:    {args.model}")
    print(f"lengths:  {args.lengths}")
    print(f"warmup:   {args.warmup}    meas: {args.meas}\n")
    print(f"{'target L':>8}  {'actual L':>8}  {'min ms':>9}  {'med ms':>9}  {'max ms':>9}  {'stddev':>8}")
    print("-" * 70)

    results = []
    for target in args.lengths:
        prompt = make_prompt(target)
        # Warmup
        for w in range(args.warmup):
            r = time_one(args.url, args.model, prompt, 1, args.timeout)
            wmark = "(cold)" if w == 0 else ""
            print(f"  warmup L={target} → actual_pt={r['prompt_tokens']:>5} "
                  f"elapsed={r['elapsed']*1000:.1f} ms {wmark}", flush=True)
        # Measurement
        ms_list = []
        for _ in range(args.meas):
            r = time_one(args.url, args.model, prompt, 1, args.timeout)
            ms_list.append(r["elapsed"] * 1000.0)
        actual_pt = r["prompt_tokens"]
        med = statistics.median(ms_list)
        mn  = min(ms_list); mx = max(ms_list)
        std = statistics.stdev(ms_list) if len(ms_list) > 1 else 0.0
        print(f"{target:>8}  {actual_pt:>8}  {mn:>8.1f}   {med:>8.1f}   {mx:>8.1f}   {std:>7.1f}", flush=True)
        results.append({"target": target, "actual_pt": actual_pt,
                        "min_ms": mn, "med_ms": med, "max_ms": mx, "std_ms": std})

    print("\n=== Summary (median ms per prompt length) ===")
    for r in results:
        print(f"  L_target={r['target']:>5}  actual={r['actual_pt']:>5}  med={r['med_ms']:>7.1f} ms")


if __name__ == "__main__":
    main()
