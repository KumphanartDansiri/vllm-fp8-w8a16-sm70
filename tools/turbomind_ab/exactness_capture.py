#!/usr/bin/env python3
"""Capture greedy (temp=0) generations + per-token logprobs from a running vLLM server,
for the TurboMind-vs-ours SERVING EXACTNESS check (correctness, not perf).

Runs on the HOST against the published port. Writes a JSON per backend; exactness_compare.py
diffs two captures. Uses chat/completions with logprobs+top_logprobs so we can measure, at the
first token that diverges, how close the tie was in the reference model (tiny Δlogp = benign
numerical noise; large gap = a systematic kernel/logit problem worth inspecting).
"""
import argparse
import json
import sys
import urllib.request

# Deterministic, varied prompts: factual, reasoning, code, list, explanation, a trick question.
PROMPTS = [
    "What is the capital of France, and name three famous landmarks located there?",
    "Explain in two sentences why the sky appears blue during the day.",
    "Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed).",
    "List the first eight prime numbers in order.",
    "Summarize the theory of evolution by natural selection in exactly three sentences.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain briefly.",
    "Translate 'Good morning, how are you today?' into French, Spanish, and German.",
    "What are the SI base units for length, mass, time, and electric current?",
]


def capture(port, served, max_tokens, top):
    out = []
    for p in PROMPTS:
        body = json.dumps({
            "model": served, "temperature": 0, "max_tokens": max_tokens, "seed": 0,
            "logprobs": True, "top_logprobs": top, "stream": False,
            "messages": [{"role": "user", "content": p}],
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
        ch = r["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        toks = []
        for c in content:
            toks.append({
                "t": c["token"], "lp": c["logprob"],
                "top": [[d["token"], d["logprob"]] for d in (c.get("top_logprobs") or [])],
            })
        out.append({"prompt": p, "text": ch["message"]["content"], "tokens": toks})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    data = capture(a.port, a.served, a.max_tokens, a.top)
    json.dump(data, open(a.out, "w"))
    have_lp = sum(1 for d in data if d["tokens"])
    print(f"[capture] wrote {a.out}: {len(data)} prompts, {have_lp} with token-logprobs",
          flush=True)
    sys.exit(0)
