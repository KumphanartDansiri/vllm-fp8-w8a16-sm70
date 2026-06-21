#!/usr/bin/env python3
"""Performance Experiment v2 measurement client (see docs/PERF_EXPERIMENT_V2.md).

--phase main   : correctness battery (5 tests @ C1) + decode C1..C8 (Test-4) + TTFT short/long (FA off)
--phase ttft_fa: TTFT-long only (run against a FA-on serve)

Appends CSV rows (model,prec,engine,tp,metric,c,value,unit,quality_status,exactness) and
writes each test's verbatim output to <out>/tests/. quality_status in {pass,suspect,fail};
auto-gates are first-pass only — borderline cases are 'suspect' for human review.
"""
from __future__ import annotations
import argparse, hashlib, json, re, statistics, threading, time, urllib.request


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--port"); p.add_argument("--served"); p.add_argument("--out"); p.add_argument("--csv")
    p.add_argument("--model"); p.add_argument("--prec"); p.add_argument("--engine"); p.add_argument("--tp")
    p.add_argument("--users", default="1 2 4 8"); p.add_argument("--gentok", type=int, default=256)
    p.add_argument("--nrun", type=int, default=5); p.add_argument("--ttft-reps", type=int, default=3)
    p.add_argument("--phase", default="main")
    return p.parse_args()


def stream(port, served, prompt, max_tokens, ignore_eos, temperature=0.0):
    """One streamed request. Returns (text, ttft_s, decode_tps, completion_tokens, prompt_tokens)."""
    body = {"model": served, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    if ignore_eos:
        body["ignore_eos"] = True
    data = json.dumps(body).encode()
    rq = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=data,
                                headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; n = 0; ch = []; ct = 0; pt = 0
    with urllib.request.urlopen(rq, timeout=2400) as r:
        for raw in r:
            L = raw.decode("utf-8", "ignore").strip()
            if not L.startswith("data:"):
                continue
            d = L[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            u = j.get("usage")
            if u:
                ct = int(u.get("completion_tokens") or ct); pt = int(u.get("prompt_tokens") or pt)
            c = j.get("choices") or []
            dl = c[0]["delta"].get("content") if c else None
            if dl:
                t = time.time()
                if tf is None:
                    tf = t
                tl = t; n += 1; ch.append(dl)
    text = "".join(ch).strip()
    ttft = (tf - t0) if tf else float("nan")
    dt = (tl - tf) if (tf and tl and n > 1) else float("nan")
    mt = ct or n
    tps = ((mt - 1) / dt) if (dt and dt > 0) else float("nan")
    return text, ttft, tps, mt, pt


def rep_ratio(s):
    w = s.split()
    return (max((w.count(x) for x in set(w)), default=0) / len(w)) if w else 1.0


# ---- correctness battery ---------------------------------------------------------
ASPECTS = ["history", "geography", "economy", "culture", "cuisine", "politics", "science", "art"]
LONGFORM = "Write a detailed, multi-paragraph explanation of the {topic} of France."
# max_tokens are generous so REASONING models (Qwen3.6 etc.) that emit a long
# "thinking" preamble still reach the actual answer within budget — the gates look
# for the answer pattern anywhere in the output. (longform stays 256: it's the perf
# prompt and only needs coherence, not a specific answer.)
TESTS = [
    ("t1_exactness", "List the first 12 prime numbers as a comma-separated line, nothing else.", 512, "exact"),
    ("t2_reasoning", "A train goes 60 km in 1.5 hours, then 90 km in 2 hours. What is its average "
                     "speed for the whole trip in km/h? Show your steps.", 1024, "reason"),
    ("t3_code", "Write a Python function is_palindrome(s) that returns True if s is a palindrome "
                "ignoring case and spaces. Include a short docstring.", 1536, "code"),
    ("t4_longform", LONGFORM.format(topic="history, geography, and culture"), 256, "long"),
    ("t5_json", 'Return exactly 3 popular programming languages as a JSON array of objects, each with '
                'keys "name" and "year". Output only the JSON, no prose.', 1536, "json"),
]


def gate(kind, text):
    """-> quality_status in {pass,suspect,fail} with a short reason.

    'fail' is reserved for DEFINITE garbage (empty / high repetition) — only those
    withhold a cell's speed rows. Category checks (function present, JSON parses,
    item count, length) are first-pass heuristics: a miss is 'suspect' (flag for
    human review, e.g. a coherent reasoning model that didn't emit parseable JSON
    within budget), not 'fail'. Semantic correctness is never auto-judged."""
    if not text:
        return "fail", "empty"
    sus = []
    # Repetition is a PROSE garbage heuristic only — structured output (code/JSON)
    # legitimately repeats tokens (keys, keywords) and would false-trip it.
    if kind in ("exact", "reason", "long"):
        rr = rep_ratio(text)
        if rr >= 0.5:
            return "fail", f"repetition={rr:.2f}"
        if rr >= 0.35:
            sus.append(f"rep={rr:.2f}")
    if kind == "code" and "def " not in text:
        sus.append("no function def")
    if kind == "json":
        m = re.search(r"\[.*\]", text, re.S)
        try:
            arr = json.loads(m.group(0)) if m else json.loads(text)
            if not (isinstance(arr, list) and len(arr) == 3):
                sus.append(f"json_len={len(arr) if isinstance(arr, list) else 'NA'}")
        except Exception:
            sus.append("json unparseable")
    if kind == "long" and len(text.split()) < 20:   # essay should be long; concise reason answers are fine
        sus.append("too_short")
    return ("suspect", ";".join(sus)) if sus else ("pass", "ok")


def battery(args, csvf):
    worst = "pass"; order = {"pass": 0, "suspect": 1, "fail": 2}
    exact_label = "n/a"
    for tid, prompt, mt, kind in TESTS:
        if kind == "exact":
            outs = []
            for i in range(5):
                txt, *_ = stream(args.port, args.served, prompt, mt, ignore_eos=False)
                outs.append(txt)
            hashes = {hashlib.sha1(o.encode()).hexdigest() for o in outs}
            g, reason = gate("exact", outs[0])
            if g == "fail":
                exact_label = "Fail"
            elif len(hashes) == 1:
                exact_label = "Exact"
            else:
                exact_label = "Stable"
            open(f"{args.out}/tests/{tid}.txt", "w").write(
                f"[{exact_label}] {len(hashes)} distinct/5 reps; gate={g} ({reason})\n\n" + "\n---\n".join(outs))
            qs = "pass" if exact_label in ("Exact", "Stable") else "fail"
            print(f"[battery] {tid}: {exact_label} ({len(hashes)} distinct/5) gate={qs}")
        else:
            txt, *_ = stream(args.port, args.served, prompt, mt, ignore_eos=False)
            qs, reason = gate(kind, txt)
            open(f"{args.out}/tests/{tid}.txt", "w").write(f"[{qs}] {reason}\n\n{txt}")
            print(f"[battery] {tid}: {qs} ({reason})  \"{re.sub(chr(92)+'s+',' ',txt)[:90]}\"")
        worst = qs if order[qs] > order[worst] else worst
        csvf.write(f"{args.model},{args.prec},{args.engine},{args.tp},correctness_{tid},1,,,{qs},{exact_label}\n")
    return worst, exact_label


# ---- decode sweep (Test-4 long-form, topic-rotated) ------------------------------
def measure_decode(port, served, gentok, nu):
    res = [None] * nu
    def one(u):
        prompt = LONGFORM.format(topic=ASPECTS[u % len(ASPECTS)])
        try:
            _, _, tps, *_ = stream(port, served, prompt, gentok, ignore_eos=True)
            res[u] = tps
        except Exception:
            res[u] = float("nan")
    ts = [threading.Thread(target=one, args=(u,)) for u in range(nu)]
    for t in ts: t.start()
    for t in ts: t.join()
    ok = [x for x in res if x == x]
    return (sum(ok) / len(ok), sum(ok)) if ok else (float("nan"), float("nan"))


# ---- TTFT (reference-doc padded prompts) -----------------------------------------
DOC = ("The Rhone is a major European river. It rises in the Swiss Alps and flows through Lake "
       "Geneva and southeastern France before reaching the Mediterranean. Its valley has shaped "
       "trade, agriculture, and settlement for two thousand years. ")
def ttft_prompt(approx_tokens, uniq=""):
    target_chars = approx_tokens * 5   # ~5 chars/token measured on this tokenizer/doc
    body = (DOC * (target_chars // len(DOC) + 1))[:target_chars]
    pre = f"[doc-id {uniq}] " if uniq else ""   # unique prefix -> guaranteed cache MISS (cold)
    return (f"{pre}Here is a long reference document:\n{body}\n\nTask: Write a detailed multi-paragraph "
            "explanation of the main themes, facts, and implications.")


def ttft_cold(port, served, approx_tokens, uniq):
    """COLD: single send of a UNIQUE prompt -> no prefix-cache hit -> full prefill."""
    _, ttft, _, _, pt = stream(port, served, ttft_prompt(approx_tokens, uniq=uniq), 64, ignore_eos=False)
    return ttft, pt


def ttft_min(port, served, approx_tokens, reps):
    prompt = ttft_prompt(approx_tokens)
    vals = []; ptoks = 0
    for i in range(reps):
        _, ttft, _, _, pt = stream(port, served, prompt, 64, ignore_eos=False)
        ptoks = pt or ptoks
        if i > 0 and ttft == ttft:   # drop first (warmup)
            vals.append(ttft)
    return (min(vals) if vals else float("nan")), ptoks


def main():
    a = parse(); csvf = open(a.csv, "a")
    if a.phase == "main":
        worst, exact = battery(a, csvf)
        print(f"[battery] CELL quality_status={worst} exactness={exact}")
        for u in [int(x) for x in a.users.split()]:
            pus, ags = [], []
            for r in range(a.nrun):
                pu, ag = measure_decode(a.port, a.served, a.gentok, u)
                if pu == pu:
                    pus.append(pu); ags.append(ag)
                print(f"[decode] C{u} rep{r+1}: per_user={pu:.2f} agg={ag:.2f}")
            mpu = statistics.median(pus) if pus else float("nan")
            mag = statistics.median(ags) if ags else float("nan")
            print(f"[decode] C{u} MEDIAN(n={len(pus)}): per_user={mpu:.2f} agg={mag:.2f}")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},decode_per_user,{u},{mpu:.2f},tok/s,{worst},{exact}\n")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},decode_aggregate,{u},{mag:.2f},tok/s,{worst},{exact}\n")
        for nm, tok in [("ttft_short", 2048), ("ttft_long", 24576)]:
            v, pt = ttft_min(a.port, a.served, tok, a.ttft_reps)
            print(f"[ttft] {nm} (~{tok} tok, actual_prompt={pt}): {v:.2f}s")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},{nm},1,{v:.2f},s,{worst},{exact}\n")
    elif a.phase == "ttftboth":
        # ONE serve, chunked-prefill ON + prefix-caching ON (the production standard).
        # cold = unique prompt (cache miss, full prefill); warm = repeated prompt (cache hit).
        import os
        uq = str(os.getpid())
        for label, tok in [("short", 2048), ("long", 24576)]:
            c, pc = ttft_cold(a.port, a.served, tok, f"{uq}{label}")
            print(f"[ttft] ttft_{label}_cold (~{tok} tok, actual={pc}): {c:.3f}s")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},ttft_{label}_cold,1,{c:.3f},s,pass,n/a\n")
            w, pw = ttft_min(a.port, a.served, tok, a.ttft_reps)   # repeated -> cache hit = warm
            print(f"[ttft] ttft_{label}_warm (~{tok} tok, actual={pw}): {w:.3f}s")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},ttft_{label}_warm,1,{w:.3f},s,pass,n/a\n")
    elif a.phase == "ttftwarm":
        # WARM TTFT: serve has prefix-caching ON. ttft_min sends the prompt ttft_reps
        # times and drops the first -> rep 0 is the cold populate, reps 1+ are cache
        # HITS, so the min is the warm (cached-prefix) TTFT. Pairs with the cold table.
        for nm, tok in [("ttft_short_warm", 2048), ("ttft_long_warm", 24576)]:
            v, pt = ttft_min(a.port, a.served, tok, a.ttft_reps)
            print(f"[ttft] {nm} (~{tok} tok, actual_prompt={pt}): {v:.3f}s")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},{nm},1,{v:.3f},s,pass,n/a\n")
    elif a.phase == "ttftonly":
        # TTFT-only reconcile run (decode is prefix-caching-invariant and already valid;
        # this re-measures TTFT under the fixed config: prefix-caching OFF, so the
        # repeated-prompt min is real cold prefill, not a cache hit).
        for nm, tok in [("ttft_short", 2048), ("ttft_long", 24576)]:
            v, pt = ttft_min(a.port, a.served, tok, a.ttft_reps)
            print(f"[ttft] {nm} (~{tok} tok, actual_prompt={pt}): {v:.2f}s")
            csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},{nm},1,{v:.2f},s,pass,n/a\n")
    elif a.phase == "ttft_fa":
        import os
        v, pt = ttft_cold(a.port, a.served, 24576, str(os.getpid()) + "fa")   # cold FA (unique prompt)
        print(f"[ttft] ttft_long_cold_fa (~24576 tok, actual={pt}): {v:.2f}s")
        csvf.write(f"{a.model},{a.prec},{a.engine},{a.tp},ttft_long_cold_fa,1,{v:.2f},s,pass,n/a\n")
    csvf.close()


if __name__ == "__main__":
    main()
