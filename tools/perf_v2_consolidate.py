#!/usr/bin/env python3
"""Consolidate perf_v2 cell results into one uniform matrix (results/perf_v2_COMBINED.csv).

Merge policy (the "same conditions" reconcile):
  - decode + correctness/quality : newest dir per cell that HAS decode rows (banked;
    decode is prefix-caching-invariant so the fleet runs stay valid).
  - TTFT short/long              : newest dir per cell that HAS ttft rows (the
    prefix-caching-OFF reconcile dir for GLM/Gemma; the fleet dir for Qwen, which
    was already prefix-caching-off).
  - ttft_long_fa                 : kept ONLY for ENGINE=021 + plugin (fp8/fp16) +
    FA-eligible model (q27b/q35b/q122b/glm). FA is invalid on 0.19 (.so ABI) and
    int4 (stock vLLM) -> those no-op rows are dropped.
  - g26b:fp8:019                 : MISSING (stock vLLM 0.19 gemma4.py KeyError on MoE
    expert weights; use the 0.21 number).
Run: python3 tools/perf_v2_consolidate.py
"""
import glob, csv, os

CELLS = [(m, p, e) for e in ("021", "019") for (m, p) in [
    ("q27b", "fp8"), ("q27b", "fp16"), ("q35b", "fp8"), ("q35b", "fp16"),
    ("q122b", "fp8"), ("q122b", "int4"), ("glm", "fp8"),
    ("g31b", "fp8"), ("g31b", "fp16"), ("g26b", "fp8"), ("g26b", "fp16")]]
FA_ELIG = {"q27b", "q35b", "q122b", "glm"}


def dirs(m, p, e):
    return sorted(glob.glob(f"results/perf_v2_{m}_{p}_{e}_*"), reverse=True)


def rows(d):
    f = os.path.join(d, "rows.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def latest_with(m, p, e, metric):
    for d in dirs(m, p, e):
        rs = rows(d)
        if any(r["metric"] == metric or r["metric"].startswith(metric) for r in rs):
            return d, rs
    return None, []


def val(rs, metric, c=None):
    for r in rs:
        if r["metric"] == metric and (c is None or r["c"] == str(c)):
            return r["value"]
    return None


def main():
    out = open("results/perf_v2_COMBINED.csv", "w")
    out.write("model,prec,engine,tp,quality,exactness,dC1,dC2,dC4,dC8,aggC8,"
              "ttft_long_cold_mono,ttft_long_cold_chunk,ttft_long_warm,ttft_long_cold_fa,"
              "ttft_short_cold,ttft_short_warm\n")
    for m, p, e in CELLS:
        dd, drs = latest_with(m, p, e, "decode_per_user")
        bd, brs = latest_with(m, p, e, "ttft_long_cold")   # TTFT_BOTH (chunked-on) dir
        md, mrs = latest_with(m, p, e, "ttft_long")        # monolithic (chunked-off) dir
        if not dd and not bd and not md:
            out.write(f"{m},{p},{e},,MISSING,,,,,,,,,,,,\n")
            continue
        tp = next((r["tp"] for r in drs), "") or next((r["tp"] for r in brs), "")
        q = next((r["quality_status"] for r in drs if r["metric"].startswith("decode")), "")
        ex = next((r["exactness"] for r in drs if r["metric"].startswith("decode")), "")
        dpu = [val(drs, "decode_per_user", c) for c in (1, 2, 4, 8)]
        aggC8 = val(drs, "decode_aggregate", 8)
        cold_mono = val(mrs, "ttft_long")                  # chunked-OFF monolithic cold
        cold_chunk = val(brs, "ttft_long_cold")            # chunked-ON cold
        warm = val(brs, "ttft_long_warm")
        cfa = val(brs, "ttft_long_cold_fa") if (e == "021" and p in ("fp8", "fp16") and m in FA_ELIG) else ""
        sc, sw = val(brs, "ttft_short_cold"), val(brs, "ttft_short_warm")
        out.write(f"{m},{p},{e},{tp},{q},{ex},{dpu[0]},{dpu[1]},{dpu[2]},{dpu[3]},{aggC8},"
                  f"{cold_mono},{cold_chunk},{warm},{cfa},{sc},{sw}\n")
    out.close()
    print("wrote results/perf_v2_COMBINED.csv")


if __name__ == "__main__":
    main()
