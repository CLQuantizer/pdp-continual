"""ACh-style mismatch veto on the margin gate (loop 19:40 hyp 2 / 20:25 hyp 1), scored offline.

Hasselmo 2006: acetylcholine signals mismatch between hippocampal recall and input; high ACh suppresses
recall-driven output and favours encoding. Here: per store class k, look at the store's *confident* claims
(store says k with relative margin >= 0.2). After 3 wrong of class k's last 50 confident claims, veto the
store's answers for class k until 20 consecutive confident claims of k agree with the label. State updates
after the answer (causal). Applied on top of mq10_s1 / mq10_s2 from gate_probe.py, on gate_probe's logged
arrivals (seeds 0-2 and held-out 3-5, bg 0.02). Only the veto half of the hypothesis is testable offline;
the prototype-lr half needs a store re-run and is tested only if the veto passes.
Pre-registered: pass = swap first 300 >= 76 with returns >= 64 and lifetime >= 96.0; kill = swap < 74.
"""
import json, os, sys
import numpy as np
import life as LF
import gate_probe as GP


def veto_mask(C, Y, S, SM, win=50, bad=3, clear=20, thr=0.2):
    hist = {k: [] for k in range(10)}
    veto = np.zeros(10, bool); streak = np.zeros(10, int)
    out = np.zeros(len(S), bool)
    for i in range(len(S)):
        s = int(S[i])
        if s < 0:
            continue
        out[i] = veto[s]
        if SM[i] >= thr:
            okc = s == Y[i]
            hist[s] = (hist[s] + [not okc])[-win:]
            if veto[s]:
                streak[s] = streak[s] + 1 if okc else 0
                if streak[s] >= clear:
                    veto[s] = False; streak[s] = 0; hist[s] = []
            elif sum(hist[s]) >= bad:
                veto[s] = True; streak[s] = 0
    return out


def score(r, w):
    R = np.array(r["rows"])
    T, C, Y, INS, RET, F, S, FFM, SM, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    low10 = FFM < GP.rolling_quantile(FFM, 0.1)
    V = veto_mask(C, Y, S, SM)
    gates = {}
    for nm, thr in (("mq10_s1", 0.1), ("mq10_s2", 0.2)):
        g = dis & prot & low10 & (SM >= thr)
        gates[nm] = g; gates[nm + "_veto"] = g & ~V
    gates["ff"] = np.zeros_like(dis)
    out = {}
    for g, use in gates.items():
        P = np.where(use, S, F); ok = P == Y
        first = lambda d, at: float(ok[np.flatnonzero(np.isin(C, d) & (T >= at))[:300]].mean())
        out[g] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), off=float(ok[INS == 0].mean()),
                      swap=first(list(w["swap"]), r["swap_at"]), novel=first(list(w["novel"]), r["novel_at"]),
                      fired=float(use.mean()))
    out["_veto"] = dict(share=float(V.mean()), on_swap=float(V[np.isin(S, list(w["swap"])) & (T >= r["swap_at"])].mean()),
                        pre_swap=float(V[T < r["swap_at"]].mean()))
    return out


def main():
    w = dict(LF.WORLD, bg=0.02)
    md = ["# ACh-style mismatch veto (`veto_probe.py`)\n", __doc__]
    allres = {}
    for src in ("results/gate_probe", "results/gate_probe_s345"):
        res = json.load(open(src + "/results.json"))["raw"]
        res = [r for r in res][:3]  # first world in each file is bg002
        sc = [score(r, w) for r in res]
        allres[src] = sc
        md += [f"\n## {src} seeds {[r['seed'] for r in res]}, mean ± std %\n",
               "| Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers % |", "|---|---|---|---|---|---|---|"]
        for g in [k for k in sc[0] if not k.startswith("_")]:
            m = lambda key: f"{np.mean([s[g][key] for s in sc]) * 100:.1f} ± {np.std([s[g][key] for s in sc]) * 100:.1f}"
            md.append(f"| {g} | {m('lifetime')} | {m('returns')} | {m('off')} | {m('swap')} | {m('novel')} | {m('fired')} |")
        v = lambda key: np.mean([s["_veto"][key] for s in sc]) * 100
        md.append(f"\nVeto active on {v('share'):.1f}% of arrivals; {v('pre_swap'):.1f}% before the swap; "
                  f"{v('on_swap'):.1f}% of post-swap arrivals where the store says 2 or 3.")
    os.makedirs("results/veto_probe", exist_ok=True)
    json.dump(allres, open("results/veto_probe/results.json", "w"))
    open("results/veto_probe/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
