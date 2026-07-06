#!/usr/bin/env python3
"""Compare two exactness captures (ours vs turbomind): token agreement, first divergence,
and — the key correctness signal — the reference model's Δlogp at the first divergence.

Decision rule (Codex): exact match, or divergence only at numerically tiny / logit-tie
differences -> the extension is serving-correct, proceed. Early SYSTEMATIC divergence
(tm's token far down ours' distribution) -> inspect logits/kernel path.

Usage: exactness_compare.py <ref(ours).json> <cand(turbomind).json>
"""
import json
import sys


def load(p):
    return json.load(open(p))


def main(ref_path, cand_path):
    ref, cand = load(ref_path), load(cand_path)
    n = min(len(ref), len(cand))
    tot_tok = tot_match = exact_prompts = 0
    worst_gap = 0.0
    print(f"ref (ours)     = {ref_path}")
    print(f"cand(turbomind)= {cand_path}\n")
    print(f"{'#':>2} {'Ntok':>5} {'agree':>10} {'1stDiv':>7}  divergence detail")
    print("-" * 96)
    for i in range(n):
        ot, tt = ref[i]["tokens"], cand[i]["tokens"]
        # Fall back to text compare if a backend returned no per-token logprobs.
        if not ot or not tt:
            same = ref[i]["text"] == cand[i]["text"]
            exact_prompts += same
            print(f"{i:>2} {'--':>5} {'text-only':>10} {'-':>7}  "
                  f"{'IDENTICAL text' if same else 'TEXT DIFFERS (no logprobs)'}")
            continue
        m = min(len(ot), len(tt))
        match = 0
        firstdiv = -1
        for j in range(m):
            if ot[j]["t"] == tt[j]["t"]:
                match += 1
            elif firstdiv < 0:
                firstdiv = j
        tot_tok += m
        tot_match += match
        exact = (firstdiv < 0 and len(ot) == len(tt))
        exact_prompts += exact
        detail = "EXACT" if exact else ""
        if firstdiv >= 0:
            o_top = {k: v for k, v in ot[firstdiv]["top"]}
            o_choice, t_choice = ot[firstdiv]["t"], tt[firstdiv]["t"]
            o_lp_o = o_top.get(o_choice, ot[firstdiv]["lp"])
            o_lp_t = o_top.get(t_choice)
            if o_lp_t is not None:
                gap = o_lp_o - o_lp_t
                worst_gap = max(worst_gap, gap)
                tie = "TIE~" if gap < 0.05 else ("close" if gap < 0.5 else "SYSTEMATIC")
                detail = (f"@{firstdiv} ours:{o_choice!r}({o_lp_o:.3f}) "
                          f"tm:{t_choice!r}({o_lp_t:.3f}) Δlogp={gap:.4f} [{tie}]")
            else:
                detail = (f"@{firstdiv} ours:{o_choice!r} tm:{t_choice!r} "
                          f"(tm token NOT in ours top-{len(o_top)} => SYSTEMATIC?)")
        print(f"{i:>2} {m:>5} {match:>4}/{m:<5} {firstdiv:>7}  {detail}")
    print("-" * 96)
    if tot_tok:
        print(f"TOKEN AGREEMENT : {tot_match}/{tot_tok} = {100*tot_match/tot_tok:.2f}%")
    print(f"EXACT PROMPTS   : {exact_prompts}/{n}")
    print(f"WORST TIE Δlogp : {worst_gap:.4f}  "
          f"({'all divergences are logit-ties/close => BENIGN' if worst_gap < 0.5 else 'has a SYSTEMATIC divergence => INSPECT'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
