"""Uncertainty-based arbitration between store and cortex (loop 21:00 UTC+8 31 Oct, pre-registered before running).

Literature: Daw, Niv & Dayan 2005 (Nat Neurosci 8:1704) arbitrate between two controllers by each one's own
   running estimate of its reliability, not by a fixed confidence threshold on one of them. The 20:00 result says the
   store's own margin is the router's signal (c3 = mq30_s2 or store disagrees with relative margin >= 0.3).
H (outward move): an arbiter that learns online, from the labels that arrive anyway, P(store right) and P(FF right)
   on disagreements in a joint (FF margin bin x store margin bin) table (Beta(1,1) counts, causal: updated after
   answering) and lets the store answer when p_store > p_ff + d, beats the c{s} frontier.
   d and the variant (joint 'j' or store-margin-only 's') are chosen on s49 seeds 0-2: best returns with lifetime >=
   c3 - 0.02pt. Bin edges (quintiles of FF margin and store margin on disagreements) are fixed from s49 seeds 0-2.
   Prediction on held-out s49 3-5 and bg002 6-8: >= +0.5 returns above the c frontier at matched firing (best returns
   over c{s}, s in 0.10..0.80 step 0.05, with firing <= the arbiter's) and lifetime >= c3 - 0.02pt; on bg 0.2
   lifetime >= c3 - 0.02pt. Kill: < +0.5 on either held-out swap world, or lifetime below that bar.
Exact offline join: the route only changes who answers, not what either system learns.
Usage: OMP_NUM_THREADS=1 python3 arbiter.py
"""
import json, os
import numpy as np
import life as LF
import confirm_recency as CR
from gate_probe import rolling_quantile

CS = [round(x, 2) for x in np.arange(0.10, 0.801, 0.05)]
DS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


def arb(dis, fb, sb, s_ok, f_ok, d, nb, joint):
    a = np.ones((nb, nb)); b = np.ones((nb, nb))  # store-right, ff-right counts (+1 prior)
    n = np.full((nb, nb), 2.0)
    use = np.zeros(len(dis), bool)
    for i in np.flatnonzero(dis):
        k = (fb[i], sb[i]) if joint else (0, sb[i])
        ps, pf = a[k] / (n[k] + 1), b[k] / (n[k] + 1)
        use[i] = ps > pf + d
        a[k] += s_ok[i]; b[k] += f_ok[i]; n[k] += 1
    return use


def score(wk, seed, Z, log, fe, se):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    routes = {"ff": np.zeros(len(T), bool), "mq30_s2": base}
    for c in CS:
        routes[f"c{c:.2f}"] = base | (dis & (SM >= c))
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    for d in DS:
        for joint in (True, False):
            u = arb(dis, fb, sb, s_ok, f_ok, d, len(fe) + 1, joint)
            routes[f"{'j' if joint else 's'}{d:+.1f}"] = u
            routes[f"{'j' if joint else 's'}{d:+.1f}b"] = u | base
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
    fm, smv = [], []
    for s in (0, 1, 2):
        R = np.array(logs[("s49", s)]["rows"]); Z = Zs[("s49", s)]
        dis = (Z[:, 2] >= 0) & (Z[:, 2] != R[:, 5])
        fm.append(R[dis, 7]); smv.append(Z[dis, 3])
    fe = np.quantile(np.concatenate(fm), [0.2, 0.4, 0.6, 0.8]); se = np.quantile(np.concatenate(smv), [0.2, 0.4, 0.6, 0.8])
    per = {k: score(k[0], k[1], Z, logs[k], fe, se) for k, Z in Zs.items()}
    M = lambda cells, k, f: float(np.nanmean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]
    arbs = [k for k in per[sel_cells[0]] if k[0] in "js"]
    lb = M(sel_cells, "c0.30", "lifetime") - 0.0002
    sel = max(arbs, key=lambda k: M(sel_cells, k, "returns") if M(sel_cells, k, "lifetime") >= lb else -1)
    groups = {"s49 selection 0-2": sel_cells, "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    ck = [f"c{c:.2f}" for c in CS]

    def frontier(cells, fired):
        c = [M(cells, k, "returns") for k in ck if M(cells, k, "fired") <= fired + 1e-9]
        return max(c) if c else M(cells, "mq30_s2", "returns")
    md = ["# Uncertainty-based arbitration, store vs cortex (`arbiter.py`)\n", __doc__,
          f"\nFF-margin bin edges {np.round(fe, 3).tolist()}, store-margin edges {np.round(se, 3).tolist()} (s49 seeds 0-2)\n",
          f"Selected on s49 seeds 0-2: **{sel}**\n"]
    summary, verdict = {}, {}
    show = ["ff", "mq30_s2", "c0.20", "c0.25", "c0.30", "c0.40", sel, "j+0.0", "j+0.0b", "s+0.0b"]
    for name, cells in groups.items():
        md += [f"\n## {name}, mean %\n", "| Route | Lifetime | Returns | Swap first 300 | Store answers | c frontier returns at this firing | Above |",
               "|---|---|---|---|---|---|---|"]
        for k in dict.fromkeys(show):
            fr = frontier(cells, M(cells, k, "fired"))
            md.append(f"| {k} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | {M(cells,k,'swap')*100:.1f} | "
                      f"{M(cells,k,'fired')*100:.2f} | {fr*100:.1f} | {(M(cells,k,'returns')-fr)*100:+.1f} |")
        summary[name] = {k: dict(lifetime=M(cells, k, "lifetime"), returns=M(cells, k, "returns"), fired=M(cells, k, "fired"),
                                 above=M(cells, k, "returns") - frontier(cells, M(cells, k, "fired"))) for k in dict.fromkeys(show)}
        best = max(arbs, key=lambda k: M(cells, k, "lifetime"))
        md.append(f"\nHighest-lifetime arbiter here (oracle, not a test): {best} {M(cells,best,'lifetime')*100:.2f} / "
                  f"{M(cells,best,'returns')*100:.1f} / fired {M(cells,best,'fired')*100:.2f}; best c lifetime "
                  f"{max(M(cells,k,'lifetime') for k in ck)*100:.2f}")
    od = "results/arbiter"; os.makedirs(od, exist_ok=True)
    json.dump({"sel": sel, "edges": [fe.tolist(), se.tolist()], "summary": summary,
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}}, open(os.path.join(od, "results.json"), "w"))
    open(os.path.join(od, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
