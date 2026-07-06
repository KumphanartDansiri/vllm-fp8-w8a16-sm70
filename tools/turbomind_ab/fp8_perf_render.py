#!/usr/bin/env python3
"""Render the FP8 perf CSV as TurboMind-vs-ours: per (model, tp, mode, conc), per-user + aggregate
tok/s for each backend and the turbomind/ours speedup — the buyer-facing "how much faster?" answer.
"""
import csv
import sys
from collections import defaultdict


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main(path):
    rows = list(csv.DictReader(open(path)))
    d = defaultdict(dict)
    for r in rows:
        d[(r["model"], r["tp"], r["mode"], r["conc"])][r["backend"]] = r
    print(f"{'model':<7}{'tp':>3}{'mode':>10}{'C':>3} | "
          f"{'per-user tok/s':^20} | {'aggregate tok/s':^20} | {'peak MiB':^13}")
    print(f"{'':<7}{'':>3}{'':>10}{'':>3} | {'ours':>6}{'tm':>7}{'x':>7} | "
          f"{'ours':>6}{'tm':>7}{'x':>7} | {'ours':>6}{'tm':>7}")
    print("-" * 86)
    for k in sorted(d, key=lambda z: (z[0], int(z[1]), z[2], int(z[3]))):
        o, t = d[k].get("ours"), d[k].get("turbomind")
        if not o or not t:
            continue
        m, tp, mode, c = k
        pu_o, pu_t = f(o["peruser_toks"]), f(t["peruser_toks"])
        ag_o, ag_t = f(o["agg_toks"]), f(t["agg_toks"])
        sx_pu = pu_t / pu_o if pu_o else float("nan")
        sx_ag = ag_t / ag_o if ag_o else float("nan")
        print(f"{m:<7}{tp:>3}{mode:>10}{c:>3} | "
              f"{pu_o:>6.1f}{pu_t:>7.1f}{sx_pu:>6.2f}x | "
              f"{ag_o:>6.1f}{ag_t:>7.1f}{sx_ag:>6.2f}x | "
              f"{o['peakMiB']:>6}{t['peakMiB']:>7}")
    print("-" * 86)
    print("x = turbomind / ours (>1 => turbomind faster). per-user = single-stream feel; "
          "aggregate = total server throughput.")


if __name__ == "__main__":
    main(sys.argv[1])
