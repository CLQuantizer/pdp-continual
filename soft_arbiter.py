"""Soft surprise: hierarchical (partially pooled) class evidence in the store-cortex arbiter
(loop 01:00 UTC+8 1 Nov, pre-registered before running).

Context: 23:00 and 00:00 closed every hard error-streak veto: each trades ~1.5 returns per swap point because
   84-96% of return-period blocks are store-right. Adams & MacKay 2007 (BOCPD): evidence of change should be weighed
   against the model's own predictive probability, not used as a switch. Here the bin table already predicts how
   often the store is right; a class-local run of errors should only move that prediction in proportion to how
   surprising it is.
H6: per store-predicted class c keep disagreement counts (store right a_c, FF right b_c, n_c). Answer with the store
   when (a_c + K*ps) / (n_c + K) > (b_c + K*pf) / (n_c + K), where ps, pf are the j+0.0b bin posteriors for this item
   (class counts shrunk toward the bin prior with strength K). Variants: 'h{K}' (class counts over the whole life),
   'he{K}' (class counts reset when class c returns after >= RET_GAP ticks, causal), 'hd{K}' (class counts with
   half-life 500 ticks). K in 2, 5, 10, 20. OR'd with mq30_s2 ('b').
Selection on s49 seeds 0-2: best swap first-300 among variants with lifetime >= j+0.0b - 0.02 and returns >=
   j+0.0b - 0.5. Prediction on held-out s49 3-5 and bg002 6-8: returns >= j+0.0b - 0.5, swap >= j+0.0b + 0.5,
   lifetime >= j+0.0b - 0.02; bg 0.2 lifetime >= j+0.0b - 0.02. Kill: any bar missed.
Mechanism prior: if class evidence helps only on swaps, large K should keep returns and still give some swap; if all
   K lie on the veto trade line (~1.5 returns per swap point), class-local reliability is closed as a route to swap
   speed in the arbiter and swap must come from the store's own remap.
Pure re-scoring of cached confirm_recency Z and FF logs (no new training).
Usage: OMP_NUM_THREADS=1 python3 soft_arbiter.py
"""
import json, os
import numpy as np
import life as LF
import confirm_recency as CR
from gate_probe import rolling_quantile
from arbiter_decay import arb as arb_decay

OD = "results/soft_arbiter"
KS = [2, 5, 10, 20]


def soft(dis, T, S, Y, fb, sb, s_ok, f_ok, nb, K, mode):
    a = np.zeros((nb, nb)); b = np.zeros((nb, nb)); n = np.zeros((nb, nb))
    ca = np.zeros(10); cb = np.zeros(10); cn = np.zeros(10); clast = np.zeros(10); last = {}
    use = np.zeros(len(dis), bool); moved = 0
    for i in range(len(T)):
        c = int(S[i])
        if dis[i] and c >= 0:
            if mode == "e" and c in last and T[i] - last[c] >= LF.RET_GAP:
                ca[c] = cb[c] = cn[c] = 0
            if mode == "d":
                f = 0.5 ** ((T[i] - clast[c]) / 500.0); ca[c] *= f; cb[c] *= f; cn[c] *= f; clast[c] = T[i]
            k = (fb[i], sb[i])
            ps, pf = (a[k] + 1) / (n[k] + 3), (b[k] + 1) / (n[k] + 3)
            use[i] = (ca[c] + K * ps) / (cn[c] + K) > (cb[c] + K * pf) / (cn[c] + K)
            moved += use[i] != (ps > pf)
            a[k] += s_ok[i]; b[k] += f_ok[i]; n[k] += 1
            ca[c] += s_ok[i]; cb[c] += f_ok[i]; cn[c] += 1
        last[int(Y[i])] = T[i]
    return use, moved


def score(wk, seed, Z, log, fe, se):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    cls = np.clip(S, 0, 9).astype(int); nb = len(fe) + 1
    routes = {"ff": np.zeros(len(T), bool), "j+0.0b": arb_decay(dis, T, fb, sb, cls, s_ok, f_ok, nb, 0, False) | base}
    diag = {}
    for K in KS:
        for tag, mode in (("h", ""), ("he", "e"), ("hd", "d")):
            u, mv = soft(dis, T, S, Y, fb, sb, s_ok, f_ok, nb, K, mode)
            routes[f"{tag}{K}b"] = u | base; diag[f"{tag}{K}b"] = dict(moved=int(mv))
    sw = log["swap_at"]; g = {}
    for k, use in routes.items():
        ok = np.where(use, S, F) == Y
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300] if w.get("swap") else np.array([], int)
        g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else float("nan"),
                    swap=float(ok[idx].mean()) if len(idx) else float("nan"), fired=float(use.mean()))
    return g, diag


def main():
    os.makedirs(OD, exist_ok=True)
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per, dg = {}, {}
    for k in logs:
        per[k], dg[k] = score(k[0], k[1], np.load(f"results/confirm_recency/Z_{k[0]}_{k[1]}.npy"), logs[k], fe, se)
    M = lambda cells, k, f: float(np.nanmean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]; ref = "j+0.0b"
    cands = [k for k in per[sel_cells[0]] if k[0] == "h"]
    ok = lambda cells, k: (M(cells, k, "lifetime") >= M(cells, ref, "lifetime") - 0.0002 and
                           M(cells, k, "returns") >= M(cells, ref, "returns") - 0.005)
    sel = max(cands, key=lambda k: M(sel_cells, k, "swap") if ok(sel_cells, k) else -1 + M(sel_cells, k, "swap") * 1e-3)
    groups = {"s49 selection 0-2": sel_cells, "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    md = ["# Soft surprise arbiter (`soft_arbiter.py`)\n", __doc__,
          f"\nSelected on s49 seeds 0-2: **{sel}** (passes selection bars: {ok(sel_cells, sel)})\n"]
    summary = {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean %\n",
               "| Route | Lifetime | Returns | Swap first 300 | Store answers | Swap vs j+0.0b | Returns vs j+0.0b | decisions moved |",
               "|---|---|---|---|---|---|---|---|"]
        for k in per[cells[0]]:
            mv = f"{np.mean([dg[c][k]['moved'] for c in cells]):.0f}" if k in dg[cells[0]] else ""
            md.append(f"| {k}{' (sel)' if k == sel else ''} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | "
                      f"{M(cells,k,'swap')*100:.1f} | {M(cells,k,'fired')*100:.2f} | {(M(cells,k,'swap')-M(cells,ref,'swap'))*100:+.1f} | "
                      f"{(M(cells,k,'returns')-M(cells,ref,'returns'))*100:+.1f} | {mv} |")
        summary[name] = {k: {f: M(cells, k, f) for f in ("lifetime", "returns", "swap", "fired")} for k in per[cells[0]]}
    verdict = {}
    for name in ("s49 held-out 3-5", "bg002 (swap 2<->3) seeds 6-8"):
        cs = groups[name]
        verdict[name] = dict(swap_gain=M(cs, sel, "swap") - M(cs, ref, "swap"),
                             life=M(cs, sel, "lifetime") - M(cs, ref, "lifetime"), ret=M(cs, sel, "returns") - M(cs, ref, "returns"))
    cs = groups["bg 0.2 seeds 0-5"]; verdict["bg02_life"] = M(cs, sel, "lifetime") - M(cs, ref, "lifetime")
    passed = all(v["swap_gain"] >= 0.005 and v["life"] >= -0.0002 and v["ret"] >= -0.005 for n, v in verdict.items() if n != "bg02_life") \
        and verdict["bg02_life"] >= -0.0002
    md.append(f"\n## Verdict H6 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "summary": summary,
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}, "diag": {f"{a}:{b}": v for (a, b), v in dg.items()}},
              open(os.path.join(OD, "results.json"), "w"))
    open(os.path.join(OD, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
