"""
prewarm.py
──────────
Send a small matrix of (prompt_length, max_tokens) requests to a running vLLM
server right after startup, so the per-shape Triton/FLA autotune cost is paid
ONCE before real users see latency.

The first request at each new prompt-token-count triggers Triton kernel
autotune for that shape, which can take 5-15 minutes on V100 with the FLA
Mamba prefill kernel. Subsequent requests at the same shape are fast.

Run this after `Application startup complete` appears in the server log,
before opening the server to real traffic.

Usage:
  python3 prewarm.py --url http://localhost:8001/v1/completions \\
      --model /mnt/models/Qwen3.6-27B-FP8

Defaults to 4 common prompt lengths × 1 short max_tokens each = 4 requests.
Total cost: 4 × (autotune_cold_time + ~few_seconds_of_decode), roughly
20-60 minutes depending on shape coverage. Run once, then steady-state.
"""
import argparse
import json
import sys
import time
import urllib.request

# Common prompt-token target counts. Pick values that span the expected
# real-world traffic. Each ENTRY triggers one fresh Triton autotune cycle
# on first invocation; thereafter cached.
DEFAULT_PROMPT_LENGTHS = [16, 64, 256, 1024]

# Pad with filler text to reach target token counts. "the" tokenizes to one
# token in most BPE vocabularies; "a " similar. Picking content-free
# repetitive text keeps the model from doing anything special.
FILLER_UNIT = "the "
SEED_PROMPT = "Hello world. "


def make_prompt_approx(target_tokens: int) -> str:
    """Build a prompt that tokenizes to APPROXIMATELY target_tokens tokens.

    Rough heuristic: each FILLER_UNIT ≈ 1 BPE token. Exact tokenization
    isn't important — we just need a variety of prompt lengths to trigger
    the autotune cache populator across shape buckets vLLM creates.
    """
    if target_tokens <= 4:
        return SEED_PROMPT
    n_filler = max(0, target_tokens - 4)
    return SEED_PROMPT + FILLER_UNIT * n_filler


def warmup_one(url: str, model: str, prompt: str, max_tokens: int,
               timeout: float = 1800.0) -> dict:
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-lengths", type=int, nargs="+",
                    default=DEFAULT_PROMPT_LENGTHS,
                    help="Approximate prompt-token counts to warm. Default: 16 64 256 1024")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="Max generation tokens per warmup call (default: 8 — small to keep cost down)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="Per-request timeout (default: 30 min — first cold autotune can be slow)")
    args = ap.parse_args()

    print(f"# Pre-warming {args.url}")
    print(f"# Model: {args.model}")
    print(f"# Prompt-length buckets: {args.prompt_lengths}")
    print(f"# max_tokens per warmup: {args.max_tokens}")
    print(f"# Total: {len(args.prompt_lengths)} requests; "
          f"expect 5-15 min per cold shape on first run")
    print()

    total_t0 = time.perf_counter()
    for i, plen in enumerate(args.prompt_lengths):
        prompt = make_prompt_approx(plen)
        print(f"[{i+1}/{len(args.prompt_lengths)}] prompt~{plen} tokens "
              f"(actual chars={len(prompt)}) — sending...",
              flush=True)
        try:
            r = warmup_one(args.url, args.model, prompt, args.max_tokens,
                           timeout=args.timeout)
            print(f"    done in {r['wall_s']:.1f}s "
                  f"(p_tok={r['prompt_tokens']}, c_tok={r['completion_tokens']})",
                  flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
    total = time.perf_counter() - total_t0
    print(f"\n# Pre-warm complete in {total:.1f}s ({total/60:.1f} min).")
    print(f"# Server should now respond fast for prompts in any of these "
          f"shape buckets: {args.prompt_lengths}")


if __name__ == "__main__":
    main()
