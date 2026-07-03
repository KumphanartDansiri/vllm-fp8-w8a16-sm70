#!/usr/bin/env python3
"""
Stage-C grouped-MoE A/B :: comparator.

Joins results_moe_ours_<model>.csv + results_moe_1catai_<model>.csv on (regime, tpe) and
emits the 5-timing comparison + an e2e verdict for the decode envelope.

Both sides are cos-gated in their own bench; this only reports configs where BOTH gates
passed. Ratios >1 mean OURS is faster.
"""
import argparse, csv
from pathlib import Path


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("model") or "").startswith("#") or not r.get("tpe"):
                continue
            if r["regime"] not in ("hot1", "hot8", "spread"):
                continue
            rows[(r["regime"], int(r["tpe"]))] = r
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()
    d = Path(args.dir)
    ours = load(d / f"results_moe_ours_{args.model}.csv")
    cat = load(d / f"results_moe_1catai_{args.model}.csv")

    order = {"hot1": 0, "hot8": 1, "spread": 2}
    keys = sorted(set(ours) & set(cat), key=lambda k: (order.get(k[0], 9), k[1]))

    print(f"\n=== Stage-C grouped-MoE A/B :: {args.model} ===")
    print("(ms; ratio>1 => OURS faster. ours coalesced has route=unperm=0.)\n")
    hdr = (f"{'regime':7s} {'tpe':>3s} {'R':>5s} | "
           f"{'1cat_route':>10s} {'1cat_kern':>9s} {'1cat_unp':>8s} {'1cat_e2e':>8s} | "
           f"{'our_kern':>8s} {'our_e2e':>7s} {'our_tiled':>9s} | "
           f"{'e2e_x':>6s} {'kern_x':>6s}")
    print(hdr); print("-" * len(hdr))
    wins = 0; tot = 0
    for k in keys:
        o, c = ours[k], cat[k]
        if o["gate"] != "ok" or c["gate"] != "ok":
            print(f"{k[0]:7s} {k[1]:>3d}  SKIP (gate: ours={o['gate']} 1catai={c['gate']})")
            continue
        R = int(o["R"])
        c_route, c_kern = fnum(c["route_ms"]), fnum(c["kernel_ms"])
        c_unp, c_e2e = fnum(c["unperm_ms"]), fnum(c["e2e_ms"])
        o_kern, o_e2e = fnum(o["coalesced_kernel_ms"]), fnum(o["coalesced_e2e_ms"])
        o_tiled = fnum(o["tiled_kernel_ms"])
        e2e_x = c_e2e / o_e2e if o_e2e > 0 else float("nan")
        kern_x = c_kern / o_tiled if o_tiled and o_tiled == o_tiled and o_tiled > 0 else float("nan")
        tot += 1; wins += (e2e_x > 1)
        print(f"{k[0]:7s} {k[1]:>3d} {R:>5d} | "
              f"{c_route:>10.3f} {c_kern:>9.3f} {c_unp:>8.3f} {c_e2e:>8.3f} | "
              f"{o_kern:>8.3f} {o_e2e:>7.3f} {o_tiled:>9.3f} | "
              f"{e2e_x:>6.2f} {kern_x:>6.2f}")

    print("-" * len(hdr))
    print(f"e2e: OURS faster in {wins}/{tot} configs.")
    print("\nInterpretation:")
    print("  * e2e_x = 1catai_e2e / ours_coalesced_e2e  (skip permute/unpermute vs pay it)")
    print("  * kern_x = 1catai_kernel / ours_tiled_kernel (sorted-grouped kernel algorithm only)")
    print("  * decode envelope = hot1/hot8 rows (few active experts, per-expert M=tpe)")


if __name__ == "__main__":
    main()
