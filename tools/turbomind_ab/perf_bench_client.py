#!/usr/bin/env python3
"""Concurrency perf driver for the FP8 TurboMind-vs-ours serving comparison.

Fires N concurrent streaming chat requests (same prompt, temp=0, ignore_eos, fixed max_tokens)
and reports: per-user decode tok/s (mean over requests), AGGREGATE decode tok/s (all tokens over
the concurrent wall span), and mean TTFT. Emits one machine-readable line for the shell to parse.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPT = ("Write a detailed, multi-section essay on the history, geography, economy, and "
          "culture of France. Use clear subsections with headings and develop each at length.")


def one_request(port, served, max_tokens, results, idx):
    body = json.dumps({
        "model": served, "stream": True, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    tf = tl = None
    toks = 0
    usage = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except Exception:
                    continue
                u = j.get("usage")
                if u and u.get("completion_tokens"):
                    usage = int(u["completion_tokens"])
                c = j.get("choices") or []
                delta = c[0]["delta"].get("content") if c else None
                if delta:
                    now = time.time()
                    if tf is None:
                        tf = now
                    tl = now
                    toks += 1
    except Exception as e:
        results[idx] = {"err": type(e).__name__}
        return
    n = usage or toks
    results[idx] = {
        "tok": n,
        "ttft": (tf - t0) if tf else float("nan"),
        "decode": (n - 1) / (tl - tf) if (tf and tl and tl > tf and n > 1) else float("nan"),
        "t0": t0, "tl": tl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    results = [None] * a.conc
    threads = [threading.Thread(target=one_request,
                                args=(a.port, a.served, a.max_tokens, results, i))
               for i in range(a.conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in results if r and "decode" in r]
    errs = a.conc - len(ok)
    if not ok:
        print(f"CONC={a.conc}\tokreq=0/{a.conc}\tperuser_toks=nan\tagg_toks=nan\tttft_s=nan\terrs={errs}")
        return
    per_user = sum(r["decode"] for r in ok) / len(ok)
    total_tok = sum(r["tok"] for r in ok)
    tmin = min(r["t0"] for r in ok)
    tmax = max(r["tl"] for r in ok)
    agg = total_tok / (tmax - tmin) if tmax > tmin else float("nan")
    mean_ttft = sum(r["ttft"] for r in ok) / len(ok)
    print(f"CONC={a.conc}\tokreq={len(ok)}/{a.conc}\tperuser_toks={per_user:.2f}"
          f"\tagg_toks={agg:.2f}\tttft_s={mean_ttft:.2f}\terrs={errs}")


if __name__ == "__main__":
    main()
