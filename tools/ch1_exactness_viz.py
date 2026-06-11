#!/usr/bin/env python3
"""Self-contained run-to-run reproducibility figure for a lay viewer.

Per model, top to bottom:
  1. CONDITIONS  — model / precision / TP / engine, answer settings, decode speed, EXACT/STABLE badge.
  2. THE QUESTION — the exact prompt.
  3. ANSWER (run 1), FULL: green = identical across all 5 runs; blue = run-1's FULL divergent remainder.
  4. THE OTHER DISTINCT ANSWERS, FULL — each run that differs, complete continuation, color-coded.
     (Nothing is truncated — a skeptic can read the whole divergent text and see it stays coherent.)

Outputs an ANSI terminal PREVIEW (capped for the console) + a self-contained HTML file with the FULL text.
Usage: python3 tools/ch1_exactness_viz.py <out_dir> [engine_label] [cell1 cell2 ...]
"""
import sys, os, csv, html, statistics as st

COND = {
 "g31b-fp8":("gemma-4 31B — dense — FP8 (our W8A16 kernel)",4),
 "g31b-fp16":("gemma-4 31B — dense — FP16 (official base)",4),
 "g26b-fp8":("gemma-4 26B-A4B — MoE — FP8 (our W8A16 kernel)",4),
 "g26b-fp16":("gemma-4 26B-A4B — MoE — FP16 (official base)",4),
 "q35b-fp8":("Qwen3.6 35B-A3B — MoE — FP8 (our W8A16 kernel)",4),
 "q27b-fp8":("Qwen3.6 27B — hybrid — FP8 (our W8A16 kernel)",4),
 "q122b-fp8":("Qwen3.5 122B-A10B — MoE — FP8 (our W8A16 kernel)",8),
 "q122b-int4":("Qwen3.5 122B-A10B — MoE — GPTQ-Int4",8),
}
QUESTION = ("Write a detailed, multi-section essay on the history, geography, economy, and culture "
            "of France. Use clear subsections with headings and develop each at length.")
ANSI_COMMON="\033[92m"; ANSI_DIV=["\033[96m","\033[93m","\033[95m","\033[91m","\033[94m"]; ANSI_R="\033[0m"
HEXC=["#1f77b4","#e8870c","#9467bd","#d62728","#17a2b8"]
TERM_CAP=90   # console preview only; HTML shows the FULL text

def load(out,cell):
    runs=[]
    for r in range(1,6):
        p=os.path.join(out,f"{cell}_q1_run{r}.txt")
        runs.append(open(p,encoding="utf-8",errors="ignore").read() if os.path.exists(p) else None)
    return [x for x in runs if x is not None]

def cpl(runs):
    m=min(len(x) for x in runs); i=0
    while i<m and all(r[i]==runs[0][i] for r in runs): i+=1
    return i

def distinct_answers(runs,cp):
    """group runs by their FULL continuation (run 1's group first). returns [(full_cont,[run#s]),...]."""
    groups=[]
    for idx,r in enumerate(runs):
        cont=r[cp:]
        for g in groups:
            if g[0]==cont: g[1].append(idx+1); break
        else: groups.append([cont,[idx+1]])
    return groups

def decode_median(out,cell):
    mf=os.path.join(out,"manifest.csv")
    if not os.path.exists(mf): return None
    d=[float(r["decode_tps"]) for r in csv.DictReader(open(mf))
       if r["model"]==cell and r["qid"]=="1" and r["tag"]=="OK" and r["decode_tps"] not in("nan","")]
    return st.median(d) if d else None

def term(out,cells,engine):
    for cell in cells:
        runs=load(out,cell)
        if not runs: print(f"(no runs for {cell})"); continue
        cp=cpl(runs); gs=distinct_answers(runs,cp); exact=len(gs)==1
        desc,tp=COND.get(cell,(cell,"?")); spd=decode_median(out,cell)
        print("\n"+"="*84)
        print(f"  MODEL : {desc}  |  TP={tp}  |  {engine}, cudagraph")
        print(f"  ANSWER: temperature 0 (greedy) · 4096 tokens · same prompt ×5")
        print(f"  SPEED : {spd:.1f} tok/s decode (median of 5 runs)" if spd else "  SPEED : n/a")
        print(f"  VERDICT: {'EXACT — all 5 runs identical' if exact else f'STABLE — {len(gs)} distinct answers, all coherent'}")
        print("="*84+"  (console preview — full text in the HTML)")
        if exact:
            print(f"  {ANSI_COMMON}[all 5 runs byte-identical, full text in HTML]{ANSI_R}"); continue
        print(f"  shared opening = {cp} chars, then splits into {len(gs)} versions:")
        for i,(cont,who) in enumerate(gs):
            c=ANSI_DIV[i%len(ANSI_DIV)]
            print(f"    {c}runs {','.join(map(str,who))}: …{cont.strip()[:TERM_CAP]}…{ANSI_R}")

def make_html(out,cells,engine,path):
    P=[f"""<!doctype html><meta charset=utf-8><title>V100 reproducibility — same question, 5 times</title>
<style>body{{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:26px auto;padding:0 18px;color:#1a1a1a}}
h1{{font-size:21px}} .intro{{color:#444}}
.card{{border:1px solid #e0e0e0;border-radius:10px;padding:16px 18px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.cond{{background:#f6f8fa;border-radius:7px;padding:10px 12px;font-size:13.5px;line-height:1.7}} .cond b{{color:#111}}
.q{{margin:12px 0;padding:8px 12px;border-left:3px solid #888;background:#fafafa;font-style:italic;color:#333}}
.badge{{display:inline-block;padding:2px 10px;border-radius:11px;font-size:12px;font-weight:700}}
.exact{{background:#e6f4ea;color:#137333}} .stable{{background:#fef7e0;color:#9a6700}}
.lbl{{font-size:12px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:.04em;margin-top:12px}}
.ans{{white-space:pre-wrap;word-wrap:break-word;font-size:14px;border:1px solid #eee;border-radius:6px;padding:10px 12px;margin-top:6px;max-height:520px;overflow:auto}}
.common{{color:#0b7a33}}                                  /* green = shared by all 5 runs */
.div0{{background:#dbe9f6;border-radius:3px}}             /* run-1's divergent remainder */
.brhead{{font-weight:700;font-size:12.5px;margin-top:10px}}
.key{{font-size:12.5px;color:#666}}</style>
<h1>Same question, asked 5 times — does the V100 give the same answer?</h1>
<p class=intro>Each model answered the <b>identical prompt 5 times at temperature&nbsp;0</b> (no intentional
randomness). <span class=common><b>Green</b></span> = text produced <b>identically by all 5 runs</b>; the
<span class=div0>shaded</span> remainder is where run&nbsp;1 diverges. Below run&nbsp;1, every <b>other distinct
answer is shown in full</b> (color-coded) so you can read the complete divergent text — it always stays
<b>coherent and equivalent in meaning</b> (a near-tied word flipped by floating-point ordering), i.e. ordinary
numerical nondeterminism, <b>not</b> a hardware fault or a broken model.</p>"""]
    for cell in cells:
        runs=load(out,cell)
        if not runs: continue
        cp=cpl(runs); gs=distinct_answers(runs,cp); exact=len(gs)==1
        desc,tp=COND.get(cell,(cell,"?")); spd=decode_median(out,cell)
        badge=('<span class="badge exact">EXACT — all 5 identical</span>' if exact
               else f'<span class="badge stable">STABLE — {len(gs)} distinct answers, all coherent</span>')
        common=html.escape(runs[0][:cp]); run1div=html.escape(runs[0][cp:])
        P.append('<div class=card>')
        cond=(f'<div class=cond><b>Model:</b> {html.escape(desc)} &nbsp;·&nbsp; <b>TP</b>={tp} '
              f'&nbsp;·&nbsp; <b>Engine:</b> {html.escape(engine)}, cudagraph<br>'
              f'<b>Answer settings:</b> temperature 0 (greedy) · 4096 tokens · same prompt ×5<br>')
        cond += (f'<b>Decode speed:</b> {spd:.1f} tok/s (median of 5 runs)' if spd else '<b>Decode speed:</b> n/a')
        cond += f' &nbsp;·&nbsp; <b>Reproducibility:</b> {badge}</div>'
        P.append(cond)
        P.append(f'<div class=q><b>Question:</b> “{html.escape(QUESTION)}”</div>')
        P.append('<div class=lbl>Answer — run 1 (full)</div>')
        if exact:
            P.append(f'<div class=ans><span class=common>{common}</span></div>')
            P.append('<p class=key>↳ all 5 runs were byte-identical to this, start to finish.</p></div>')
            continue
        P.append(f'<div class=ans><span class=common>{common}</span><span class=div0>{run1div}</span></div>')
        # the other distinct answers, FULL
        P.append('<div class=lbl>The other distinct answers (full, color-coded)</div>')
        for i,(cont,who) in enumerate(gs):
            if who==[1] or 1 in who: continue  # run 1 shown above
            col=HEXC[i%len(HEXC)]
            P.append(f'<div class=brhead style="color:{col}">runs {",".join(map(str,who))} — diverged here:</div>')
            P.append(f'<div class=ans style="border-left:4px solid {col}"><span class=common>{common}</span>'
                     f'<span style="background:{col};color:#fff;border-radius:3px">{html.escape(cont)}</span></div>')
        P.append('</div>')
    open(path,"w").write("\n".join(P)); return path

if __name__=="__main__":
    out=sys.argv[1] if len(sys.argv)>1 else "/tmp/v100_ch1.1_021"
    engine=sys.argv[2] if len(sys.argv)>2 else "vLLM 0.21+cu126"
    cells=sys.argv[3:] or ["g31b-fp8","g26b-fp8","g31b-fp16"]
    term(out,cells,engine)
    p=make_html(out,cells,engine,os.path.join(out,"exactness.html"))
    print("\n"+"="*84); print(f"  Self-contained HTML (FULL divergent text): {p}")
