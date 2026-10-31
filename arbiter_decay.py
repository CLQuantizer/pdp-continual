"""Forgetting / class-keyed reliability counters for the store-cortex arbiter (loop 22:00 UTC+8 31 Oct,
pre-registered before running).

Context: 21:00 arbiter j+0.0b (joint FF-margin x store-margin Beta table, or mq30_s2) beats the c{s} frontier but
   costs ~1 pt of swap first-300 vs c3, because its counts are lifetime-long and global over classes.
H3 (neuron/gating, forgetting counters): counts decay with half-life HL ticks (exp on the tick gap since the bin's
   last update), so reliability tracks a swap. Variants 'dHL' (joint table, decayed) for HL in 500..8000.
H3b (class-keyed): the table is also keyed by the store's predicted class ('k', no decay; 'kHL' decayed), since a
   swap makes the store wrong for one or two classes only.
All with the mq30_s2 base ('b'), d = 0. Variant chosen on s49 seeds 0-2: best swap first-300 with lifetime >=
   j+0.0b - 0.02pt and returns >= j+0.0b - 0.5pt.
Prediction on held-out s49 3-5 and bg002 6-8: swap first-300 >= j+0.0b + 0.5 with lifetime >= j+0.0b - 0.02 and
   returns >= j+0.0b - 0.5; bg 0.2 lifetime >= j+0.0b - 0.02. Kill: swap gain < 0.5 on either held-out swap world.
Exact offline join on the cached confirm_recency runs (route changes only who answers).
Usage: OMP_NUM_THREADS=1 python3 arbiter_decay.py
"""
import json, os
import numpy as np
import life as LF
import confirm_recency as CR
from gate_probe import rolling_quantile

HLS = [500, 1000, 2000, 4000, 8000]


def arb(dis, T, fb, sb, cls, s_ok, f_ok, nb, hl, keyed):
    shape = (10 if keyed else 1, nb, nb)
    a = np.zeros(shape); b = np.zeros(shape); n = np.zeros(shape); last = np.zeros(shape)
    use = np.zeros(len(dis), bool)
    for i in np.flatnonzero(dis):
        k = (cls[i] if keyed else 0, fb[i], sb[i])
        if hl:
            f = 0.5 ** ((T[i] - last[k]) / hl); a[k] *= f; b[k] *= f; n[k] *= f; last[k] = T[i]
        use[i] = (a[k] + 1) / (n[k] + 3) > (b[k] + 1) / (n[k] + 3)
        a[k] += s_ok[i]; b[k] += f_ok[i]; n[k] += 1
    return use


def score(wk, seed, Z, log, fe, se):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    cls = np.clip(S, 0, 9).astype(int)
    routes = {"ff": np.zeros(len(T), bool), "c0.30": base | (dis & (SM >= 0.3))}
    nb = len(fe) + 1
    routes["j+0.0b"] = arb(dis, T, fb, sb, cls, s_ok, f_ok, nb, 0, False) | base
    routes["kb"] = arb(dis, T, fb, sb, cls, s_ok, f_ok, nb, 0, True) | base
    for hl in HLS:
        routes[f"d{hl}b"] = arb(dis, T, fb, sb, cls, s_ok, f_ok, nb, hl, False) | base
        routes[f"k{hl}b"] = arb(dis, T, fb, sb, cls, s_ok, f_ok, nb, hl, True) | base
    sw = log["swap_at"]
    g = {}
    for k, use in routes.items():
        ok = np.where(use, S, F) == Y
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300]
        g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else float("nan"),
                    swap=float(ok[idx].mean()), fired=float(use.mean()))
    return g


def main():
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    Zs = {(wk, s): np.load(f"results/confirm_recency/Z_{wk}_{s}.npy") for wk, (_, _, ss) in CR.WORLDS.items() for s in ss}
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per = {k: score(k[0], k[1], Z, logs[k], fe, se) for k, Z in Zs.items()}
    M = lambda cells, k, f: float(np.nanmean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]
    cands = [k for k in per[sel_cells[0]] if k[0] in "dk"]
    ref = "j+0.0b"
    ok = lambda cells, k: (M(cells, k, "lifetime") >= M(cells, ref, "lifetime") - 0.0002 and
                           M(cells, k, "returns") >= M(cells, ref, "returns") - 0.005)
    sel = max(cands, key=lambda k: M(sel_cells, k, "swap") if ok(sel_cells, k) else -1)
    groups = {"s49 selection 0-2": sel_cells, "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    md = ["# Forgetting and class-keyed arbiter counters (`arbiter_decay.py`)\n", __doc__,
          f"\nSelected on s49 seeds 0-2: **{sel}** (passes selection bars: {ok(sel_cells, sel)})\n"]
    summary = {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean %\n", "| Route | Lifetime | Returns | Swap first 300 | Store answers | Swap vs j+0.0b |",
               "|---|---|---|---|---|---|"]
        for k in per[cells[0]]:
            md.append(f"| {k}{' (sel)' if k == sel else ''} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | "
                      f"{M(cells,k,'swap')*100:.1f} | {M(cells,k,'fired')*100:.2f} | {(M(cells,k,'swap')-M(cells,ref,'swap'))*100:+.1f} |")
        summary[name] = {k: {f: M(cells, k, f) for f in ("lifetime", "returns", "swap", "fired")} for k in per[cells[0]]}
    od = "results/arbiter_decay"; os.makedirs(od, exist_ok=True)
    json.dump({"sel": sel, "summary": summary, "raw": {f"{a}:{b}": v for (a, b), v in per.items()}},
              open(os.path.join(od, "results.json"), "w"))
    open(os.path.join(od, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
