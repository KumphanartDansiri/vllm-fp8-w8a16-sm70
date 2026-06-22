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
  - glm47:fp16                    : MLA (Glm4MoeLite), BF16/fp16 ONLY (no official FP8).
    Runs on BOTH engines via the env-gated V100 MLA prefill hook (021 dedicated prefill
    backend / 019 inline MLACommonImpl base) + MLA decode cudagraph. Not FA-eligible
    (MLA, not MHA/GQA) -> cold_fa stays blank.
  - q27b4:fp8                     : q27b FP8 pinned to TP4 (same TP as q27b-fp16) for a
    PURE precision delta. The canonical q27b:fp8 stays at its TP2 min-fit (the
    half-the-GPUs deployment point). Both kept so the TP2-vs-TP4 trade is visible
    (is "FP8 at 1/2 the FP16 TP" worth it?). Result dirs are perf_v2_q27b4_fp8_*
    (renamed from a q27b TP4 run by tools/perf_v2_q27b4_tp4.sh).
  - q35b2/g31b2/g26b2:fp8 (021)  : the TP2 "half-the-GPUs" indicator, benchmarked at a
    REDUCED max-model-len (MAXLEN=8192) since a full 32k KV won't fit at TP2 (fp8@TP2 has the
    same per-GPU weight bytes as fp16@TP4 but HALF the GPUs to shard KV across; est. max ctx
    g31b ~14k, g26b ~22k). So these rows are DECODE throughput on HALF the GPUs at SHORT
    context (long-TTFT is N/A at MAXLEN=8192). 021-only. q27b is the EXCEPTION (small enough
    to keep 32k at TP2 -> its canonical fp8 row); q122b can't fit at TP2 at all. Dirs
    perf_v2_<m>2_fp8_021_* (renamed by tools/perf_v2_fp8_tp2_indicator.sh).
Run: python3 tools/perf_v2_consolidate.py
"""
import glob, csv, os

CELLS = [(m, p, e) for e in ("021", "019") for (m, p) in [
    ("q27b", "fp8"), ("q27b", "fp16"), ("q27b4", "fp8"), ("q35b", "fp8"), ("q35b", "fp16"),
    ("q122b", "fp8"), ("q122b", "int4"), ("glm", "fp8"),
    ("g31b", "fp8"), ("g31b", "fp16"), ("g26b", "fp8"), ("g26b", "fp16"),
    ("glm47", "fp16")]]
CELLS += [(m, "fp8", "021") for m in ("q35b2", "g31b2", "g26b2")]  # TP2 half-GPU indicator, 021-only, reduced MAXLEN
FA_ELIG = {"q27b", "q35b", "q122b", "glm"}


def dirs(m, p, e):
    return sorted(glob.glob(f"results/perf_v2_{m}_{p}_{e}_*"), reverse=True)


def rows(d):
    f = os.path.join(d, "rows.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def latest_with(m, p, e, metric, exact=False):
    for d in dirs(m, p, e):
        rs = rows(d)
        hit = any(r["metric"] == metric for r in rs) if exact else \
            any(r["metric"] == metric or r["metric"].startswith(metric) for r in rs)
        if hit:
            return d, rs
    return None, []


def val(rs, metric, c=None):
    for r in rs:
        if r["metric"] == metric and (c is None or r["c"] == str(c)):
            return r["value"]
    return None


def main():
    # Last 3 cols are PROVENANCE: the raw per-cell dir each value group was pulled from, so
    # the SSOT build (build_matrix_from_results.py) can record result_path without re-doing
    # the reconciliation (which dir wins for decode vs mono-TTFT vs cold/warm-TTFT).
    HDR = ["model", "prec", "engine", "tp", "quality", "exactness",
           "dC1", "dC2", "dC4", "dC8", "aggC8",
           "ttft_long_cold_mono", "ttft_long_cold_chunk", "ttft_long_warm", "ttft_long_cold_fa",
           "ttft_short_cold", "ttft_short_warm",
           "decode_src", "ttft_mono_src", "ttft_both_src"]
    out = open("results/perf_v2_COMBINED.csv", "w")
    out.write(",".join(HDR) + "\n")
    for m, p, e in CELLS:
        dd, drs = latest_with(m, p, e, "decode_per_user")
        bd, brs = latest_with(m, p, e, "ttft_long_cold")          # TTFT_BOTH (chunked-on) dir
        md, mrs = latest_with(m, p, e, "ttft_long", exact=True)   # monolithic (chunked-off) dir
        if not dd and not bd and not md:
            out.write(",".join([m, p, e, "", "MISSING"] + [""] * (len(HDR) - 5)) + "\n")
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
        row = [m, p, e, tp, q, ex, dpu[0], dpu[1], dpu[2], dpu[3], aggC8,
               cold_mono, cold_chunk, warm, cfa, sc, sw, dd or "", md or "", bd or ""]
        out.write(",".join("" if x is None else str(x) for x in row) + "\n")
    out.close()
    print("wrote results/perf_v2_COMBINED.csv")


if __name__ == "__main__":
    main()
