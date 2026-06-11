#!/usr/bin/env python3
"""CH1 report: turn ch1_reliability_bench manifest.csv + saved outputs into the
publishable tables. Two axes (the noise-envelope framework from the MTP harness):

  Axis-1 SELF-STABILITY: per cell, Q1's repeated runs -> how many distinct outputs?
      1 distinct  -> Exact (run-to-run deterministic at temp=0)
      >1 + coherent -> Stable (FP-nondeterminism; expected on MoE/FP8)
  Axis-2 FAITHFULNESS: per model, FP8 vs its comparator (FP16 gold for mid models;
      Int4 for 122B = SPEED-only, NOT a faithfulness gold). For Q2..Q5, greedy-token
      agreement (longest common token prefix / total) — NOT bit-exact (different
      numerics never match token-for-token; that's expected, report the agreement).

Decode tables come straight from the manifest. Labels: Exact / Stable / Coherent / Fail.
Usage: python3 tools/ch1_report.py /tmp/v100_ch1/manifest.csv
"""
import sys, csv, os, statistics as st
from collections import defaultdict

REP_PROSE, REP_CODE = 0.40, 0.60  # coherence ceilings (code legitimately repeats more)

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def read_text(p):
    try:
        return open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""

def token_agree(a, b):
    """Greedy-prefix agreement: fraction of the shorter whitespace-token sequence that
    matches from the start. Cheap proxy for 'how long do two greedy decodes agree'."""
    ta, tb = a.split(), b.split()
    n = min(len(ta), len(tb))
    if n == 0:
        return 0.0
    m = 0
    for i in range(n):
        if ta[i] == tb[i]:
            m += 1
        else:
            break
    return m / n  # prefix-agreement fraction

def coherent(rep_score, qlabel):
    ceil = REP_CODE if qlabel == "code" else REP_PROSE
    return rep_score < ceil

def main():
    if len(sys.argv) < 2:
        print("usage: ch1_report.py <manifest.csv>"); sys.exit(1)
    rows = load(sys.argv[1])
    if not rows:
        print("empty manifest"); return

    # index: cell -> prec/group; (cell,qid,rep) -> row
    cells = {}
    by_cell_q = defaultdict(list)
    for r in rows:
        cells[r["model"]] = (r["prec"], r["group"])
        by_cell_q[(r["model"], r["qid"])].append(r)

    print("="*78)
    print("CH1 — AXIS 1: SELF-STABILITY + DECODE (Q1 ×reps, temp=0)")
    print("="*78)
    print(f"{'cell':18} {'prec':5} {'decode mean±sd (min..max)':28} {'self-stability':24} label")
    for cell in sorted(cells):
        prec, group = cells[cell]
        q1 = [r for r in by_cell_q.get((cell, "1"), []) if r["tag"] == "OK"]
        if not q1:
            print(f"{cell:18} {prec:5} {'— (no OK Q1 runs)':28} {'FAIL':24} Fail"); continue
        dec = [float(r["decode_tps"]) for r in q1 if r["decode_tps"] not in ("nan","")]
        shas = [r["sha256_16"] for r in q1]
        uniq = len(set(shas))
        reps_ok = all(coherent(float(r["rep_score"]), "essay") for r in q1)
        if uniq == 1 and reps_ok:
            label = "Exact"
        elif reps_ok:
            label = "Stable"
        else:
            label = "Coherent?"  # got tokens but high repetition — eyeball the text
        dmean = st.mean(dec) if dec else float("nan")
        dsd = st.pstdev(dec) if len(dec) > 1 else 0.0
        dstr = f"{dmean:.2f}±{dsd:.2f} ({min(dec):.1f}..{max(dec):.1f})" if dec else "—"
        ss = f"{uniq} distinct / {len(shas)} runs"
        print(f"{cell:18} {prec:5} {dstr:28} {ss:24} {label}")

    print()
    print("="*78)
    print("CH1 — AXIS 2: FAITHFULNESS  FP8 vs comparator  (Q2..Q5, greedy-token agreement)")
    print("  gold=FP16 (mid models, real faithfulness) | Int4 for 122B = SPEED-only, NOT gold")
    print("="*78)
    groups = defaultdict(dict)  # group -> {prec: cell}
    for cell, (prec, group) in cells.items():
        groups[group][prec] = cell
    for group in sorted(groups):
        variants = groups[group]
        fp8 = variants.get("fp8")
        comp = variants.get("fp16") or variants.get("int4")
        comp_prec = "fp16" if "fp16" in variants else ("int4" if "int4" in variants else None)
        if not fp8 or not comp:
            print(f"[{group}] incomplete pair (have {list(variants)}) — skip Axis-2"); continue
        gold_note = "FP16 gold" if comp_prec == "fp16" else "Int4 (speed-only, not gold)"
        print(f"\n[{group}]  FP8 vs {comp_prec.upper()} ({gold_note})")
        print(f"  {'Q':3} {'label':10} {'agree%':7} {'FP8 coh':8} {'cmp coh':8}")
        for qid in ("2", "3", "4", "5"):
            f8 = [r for r in by_cell_q.get((fp8, qid), []) if r["tag"] == "OK"]
            cm = [r for r in by_cell_q.get((comp, qid), []) if r["tag"] == "OK"]
            if not f8 or not cm:
                print(f"  {qid:3} {'(missing)':10}"); continue
            f8r, cmr = f8[0], cm[0]
            qlabel = f8r["qlabel"]
            ta = read_text(f8r["outfile"]); tb = read_text(cmr["outfile"])
            agree = token_agree(ta, tb)
            f8coh = coherent(float(f8r["rep_score"]), qlabel)
            cmcoh = coherent(float(cmr["rep_score"]), qlabel)
            if agree >= 0.999:
                lab = "Exact"
            elif f8coh and cmcoh:
                lab = "Stable" if comp_prec == "fp16" else "Coherent"  # Int4 isn't a gold → Coherent not Stable
            else:
                lab = "Fail"
            print(f"  {qid:3} {lab:10} {agree*100:6.1f}% {str(f8coh):8} {str(cmcoh):8}  ({qlabel})")
    print()
    print("NOTE: Exact across precisions is NOT expected (FP8≠FP16 numerics). FP16-gold pairs top out at")
    print("'Stable' (coherent, drift within FP noise); Int4 pairs = 'Coherent' (no gold). Read the saved")
    print("*_q*_run*.txt for side-by-side. Decode numbers: see Axis-1 table + manifest.csv.")

if __name__ == "__main__":
    main()
