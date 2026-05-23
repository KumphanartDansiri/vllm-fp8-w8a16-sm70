"""
perf_harness.py
───────────────
Drive a running vLLM server (OpenAI-compatible) with a fixed prompt at
varying max_tokens; measure prefill seconds + decode tok/s.

Usage:
  python3 perf_harness.py \
      --url http://localhost:8001/v1/completions \
      --model /mnt/models/Qwen3.6-27B-FP8 \
      --label fp8

Examples for the comparison matrix GPT proposed:
  # 27B-FP8 TP=4 (this branch's serve_fp8_v100.py path), debug off
  python3 perf_harness.py --url http://localhost:8001/v1/completions \
      --model /mnt/models/Qwen3.6-27B-FP8 --label fp8 --csv /tmp/perf-fp8.csv

  # 27B-GPTQ-Int4 TP=4 (aiagent's serve on port 8000)
  python3 perf_harness.py --url http://localhost:8000/v1/completions \
      --model /mnt/models/Qwen3.6-27B-GPTQ-Int4 --label gptq \
      --csv /tmp/perf-gptq.csv

Captures:
  prompt_tokens, max_tokens (= completion_tokens for finish_reason=length),
  wall-clock seconds total, prefill seconds (estimated as wall - decode_s),
  decode_tok_s (= completion_tokens / decode_s),
  TTFT estimate via streaming (optional, --stream).
"""
import argparse
import csv
import json
import sys
import time
import urllib.request

DEFAULT_PROMPT = (
    "Write a concise technical summary of Rayleigh scattering and why "
    "sunsets are red:"
)
DEFAULT_MAX_TOKENS = [10, 50, 120, 300]


def time_completion(url: str, model: str, prompt: str, max_tokens: int,
                    timeout: float = 1800.0) -> dict:
    """One non-streaming request. Return wall-clock seconds + usage."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    t1 = time.perf_counter()

    j = json.loads(body)
    usage = j.get("usage", {})
    return {
        "wall_s": t1 - t0,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "finish_reason": j["choices"][0].get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="OpenAI completions endpoint, e.g. http://localhost:8001/v1/completions")
    ap.add_argument("--model", required=True,
                    help="Model name matching what the server advertises")
    ap.add_argument("--label", required=True,
                    help="Label for this run in the output (e.g. fp8, gptq, fp16)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, nargs="+", default=DEFAULT_MAX_TOKENS,
                    help="List of max_tokens to sweep (default: 10 50 120 300)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Number of warmup requests at smallest max_tokens (default: 1)")
    ap.add_argument("--csv", default=None,
                    help="Optional CSV output path")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="Per-request timeout seconds (default: 30 min for slow prefill)")
    args = ap.parse_args()

    # One warmup at smallest max_tokens to avoid first-call cold paths.
    if args.warmup > 0:
        print(f"# Warmup: {args.warmup} call(s) at max_tokens={args.max_tokens[0]}",
              file=sys.stderr)
        for _ in range(args.warmup):
            try:
                time_completion(args.url, args.model, args.prompt,
                                args.max_tokens[0], timeout=args.timeout)
            except Exception as e:
                print(f"# Warmup error: {e}", file=sys.stderr)

    results = []
    print(f"{'label':<8} {'max_tok':>7} {'p_tok':>6} {'c_tok':>6} "
          f"{'wall_s':>8} {'prefill_s_est':>14} {'decode_tok_s':>13} {'finish':<8}")
    print("-" * 80)
    for mt in args.max_tokens:
        try:
            r = time_completion(args.url, args.model, args.prompt, mt,
                                timeout=args.timeout)
        except Exception as e:
            print(f"{args.label:<8} {mt:>7} ERROR: {e}")
            continue

        # Estimate prefill seconds as: total wall - (completion_tokens / decode_tok_s).
        # But we don't know decode_tok_s without 2 data points. Use the SHORTEST
        # max_tokens result to anchor prefill, then derive decode rate from longer.
        # For now just compute a simple decode rate assuming prefill is small fraction.
        # We'll back-solve in the post-pass.
        r["max_tokens"] = mt
        results.append(r)
        print(f"{args.label:<8} {mt:>7} {r['prompt_tokens']:>6} "
              f"{r['completion_tokens']:>6} {r['wall_s']:>8.2f} {'-':>14} "
              f"{r['completion_tokens'] / r['wall_s']:>13.3f} "
              f"{r['finish_reason']:<8}")

    # Post-pass: estimate prefill_s by linear fit.
    # wall_s(mt) ≈ prefill_s + mt / decode_tok_s
    # Two points are enough; use the smallest and largest.
    if len(results) >= 2:
        r_lo, r_hi = results[0], results[-1]
        dt = r_hi["wall_s"] - r_lo["wall_s"]
        dn = r_hi["completion_tokens"] - r_lo["completion_tokens"]
        if dn > 0:
            decode_s_per_tok = dt / dn
            decode_tok_s = 1.0 / decode_s_per_tok if decode_s_per_tok > 0 else float("nan")
            # prefill = wall - decode_time
            for r in results:
                r["decode_tok_s_fit"] = decode_tok_s
                r["prefill_s_est"] = (r["wall_s"]
                                       - r["completion_tokens"] * decode_s_per_tok)
            print()
            print("# Linear fit (wall = prefill + max_tokens / decode_tok_s):")
            print(f"#   prefill_s estimated per run; decode_tok_s = {decode_tok_s:.3f}")
            print()
            print(f"{'label':<8} {'max_tok':>7} {'wall_s':>8} {'prefill_s_est':>14} {'decode_tok_s':>13}")
            for r in results:
                print(f"{args.label:<8} {r['max_tokens']:>7} {r['wall_s']:>8.2f} "
                      f"{r['prefill_s_est']:>14.2f} {r['decode_tok_s_fit']:>13.3f}")

    # CSV output
    if args.csv and results:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "max_tokens", "prompt_tokens", "completion_tokens",
                        "wall_s", "decode_tok_s_fit", "prefill_s_est", "finish_reason"])
            for r in results:
                w.writerow([args.label, r["max_tokens"], r["prompt_tokens"],
                            r["completion_tokens"], r["wall_s"],
                            r.get("decode_tok_s_fit", ""),
                            r.get("prefill_s_est", ""),
                            r["finish_reason"]])
        print(f"# Wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
