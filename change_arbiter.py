"""Change-gated forgetting for the store-cortex arbiter (loop 23:00 UTC+8 31 Oct, pre-registered before running).

Context: 22:00 killed clock decay of the arbiter's Beta counts (about +1 swap for -2 returns). Behrens et al. 2007
   (Nat Neurosci 10:1214): the learning rate should follow estimated volatility, i.e. forget when a change is
   detected, not on a clock. Two causal change signals are available without touching what either system learns:
H4a (store remap resets the table): whenever the z5 store remaps a prototype (its own change detector), the joint
   table is reset to the prior ('rG') or its counts halved ('rH').
H4b (class-local surprise veto): per store-predicted class c, a streak counter of disagreements where the store was
   wrong and the FF right; while the streak >= m the arbiter may not hand class c to the store. Released by a
   disagreement where the store is right ('vm'), or also by any remap into or out of c ('wm'). The global table is
   never forgotten, so stable-period evidence for returns is kept. m in 2, 3, 5.
All OR'd with mq30_s2 ('b'), d = 0. Variant chosen on s49 seeds 0-2: best swap first-300 with lifetime >=
   j+0.0b - 0.02pt and returns >= j+0.0b - 0.5pt.
Prediction on held-out s49 3-5 and bg002 6-8: swap first-300 >= j+0.0b + 0.5 with lifetime >= j+0.0b - 0.02 and
   returns >= j+0.0b - 0.5; bg 0.2 lifetime >= j+0.0b - 0.02. Kill: swap gain < 0.5 on either held-out swap world
   or a bar missed. Mechanism prior (written before running): the store is wrong on swapped classes BEFORE it remaps
   (~75 ticks), so H4a acts too late and should fail; H4b acts inside that window.
Store-only replay of confirm_recency.run, adding the remap log; answers are asserted equal to the cached Z.
Usage: OMP_NUM_THREADS=1 python3 change_arbiter.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
import confirm_recency as CR
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist
from gate_probe import rolling_quantile
from arbiter_decay import arb as arb_decay

OD = "results/change_arbiter"
MS = [2, 3, 5]


def replay(job):
    wk, seed = job
    p = f"{OD}/remaps_{wk}_{seed}.npy"
    if os.path.exists(p):
        return wk, seed, np.load(p)
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, 16 * w["day"], seed, w)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=CR.K)
    Z = np.load(f"results/confirm_recency/Z_{wk}_{seed}.npy"); i = 0
    for t in range(16 * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        s, sm = RP.answer(st, x)
        assert Z[i, 0] == t and Z[i, 2] == s, (wk, seed, t); i += 1
        st.t = t; st.observe(x, y); st.add(x, y)
    L = np.array(st.log, float).reshape(-1, 3)
    np.save(p, L); print(f"{wk} seed={seed} remaps={len(L)}", flush=True)
    return wk, seed, L


def arb_reset(dis, T, fb, sb, s_ok, f_ok, nb, L, mode):
    a = np.zeros((nb, nb)); b = np.zeros((nb, nb)); n = np.zeros((nb, nb))
    use = np.zeros(len(dis), bool); ev = list(L[:, 0]); e = 0
    for i in np.flatnonzero(dis):
        while e < len(ev) and ev[e] < T[i]:
            f = 0.0 if mode == "G" else 0.5
            a *= f; b *= f; n *= f; e += 1
        k = (fb[i], sb[i])
        use[i] = a[k] > b[k]
        a[k] += s_ok[i]; b[k] += f_ok[i]; n[k] += 1
    return use


def veto(dis, T, S, s_ok, f_ok, L, m, remap_release):
    blk = np.zeros(len(dis), bool); streak = np.zeros(10); e = 0
    for i in np.flatnonzero(dis):
        while remap_release and e < len(L) and L[e, 0] < T[i]:
            streak[int(L[e, 1])] = 0; streak[int(L[e, 2])] = 0; e += 1
        c = int(S[i])
        blk[i] = streak[c] >= m
        if s_ok[i]:
            streak[c] = 0
        elif f_ok[i]:
            streak[c] += 1
    return blk


def score(wk, seed, Z, log, fe, se, L):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    cls = np.clip(S, 0, 9).astype(int); nb = len(fe) + 1
    j = arb_decay(dis, T, fb, sb, cls, s_ok, f_ok, nb, 0, False)
    routes = {"ff": np.zeros(len(T), bool), "c0.30": base | (dis & (SM >= 0.3)), "j+0.0b": j | base,
              "d2000b": arb_decay(dis, T, fb, sb, cls, s_ok, f_ok, nb, 2000, False) | base}
    for mode in "GH":
        routes[f"r{mode}b"] = arb_reset(dis, T, fb, sb, s_ok, f_ok, nb, L, mode) | base
    for m in MS:
        for tag, rr in (("v", False), ("w", True)):
            blk = veto(dis, T, S, s_ok, f_ok, L, m, rr)
            routes[f"{tag}{m}b"] = (j & ~blk) | base
            routes[f"{tag}{m}B"] = (j | base) & ~blk  # veto also silences the base route
    sw = log["swap_at"]; g = {}
    for k, use in routes.items():
        ok = np.where(use, S, F) == Y
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300]
        g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else float("nan"),
                    swap=float(ok[idx].mean()), fired=float(use.mean()))
    pre = float((L[:, 0] < sw).sum()) if len(L) else 0.0
    return g, dict(remaps=len(L), pre_swap=pre)


def main():
    os.makedirs(OD, exist_ok=True)
    jobs = [(wk, s) for wk, (_, _, ss) in CR.WORLDS.items() for s in ss]
    with Pool(min(8, len(jobs))) as pool:
        Ls = {(wk, s): L for wk, s, L in pool.map(replay, jobs)}
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per, rm = {}, {}
    for k in jobs:
        per[k], rm[k] = score(k[0], k[1], np.load(f"results/confirm_recency/Z_{k[0]}_{k[1]}.npy"), logs[k], fe, se, Ls[k])
    M = lambda cells, k, f: float(np.nanmean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]
    ref = "j+0.0b"
    cands = [k for k in per[sel_cells[0]] if k[0] in "rvw"]
    ok = lambda cells, k: (M(cells, k, "lifetime") >= M(cells, ref, "lifetime") - 0.0002 and
                           M(cells, k, "returns") >= M(cells, ref, "returns") - 0.005)
    sel = max(cands, key=lambda k: M(sel_cells, k, "swap") if ok(sel_cells, k) else -1)
    groups = {"s49 selection 0-2": sel_cells, "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    md = ["# Change-gated forgetting for the arbiter (`change_arbiter.py`)\n", __doc__,
          f"\nSelected on s49 seeds 0-2: **{sel}** (passes selection bars: {ok(sel_cells, sel)})\n"]
    summary, verdict = {}, {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean % (store remaps per run {np.mean([rm[c]['remaps'] for c in cells]):.1f}, "
               f"before swap {np.mean([rm[c]['pre_swap'] for c in cells]):.1f})\n",
               "| Route | Lifetime | Returns | Swap first 300 | Store answers | Swap vs j+0.0b |", "|---|---|---|---|---|---|"]
        for k in per[cells[0]]:
            md.append(f"| {k}{' (sel)' if k == sel else ''} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | "
                      f"{M(cells,k,'swap')*100:.1f} | {M(cells,k,'fired')*100:.2f} | {(M(cells,k,'swap')-M(cells,ref,'swap'))*100:+.1f} |")
        summary[name] = {k: {f: M(cells, k, f) for f in ("lifetime", "returns", "swap", "fired")} for k in per[cells[0]]}
    for name in ("s49 held-out 3-5", "bg002 (swap 2<->3) seeds 6-8"):
        cs = groups[name]
        verdict[name] = dict(swap_gain=M(cs, sel, "swap") - M(cs, ref, "swap"),
                             life=M(cs, sel, "lifetime") - M(cs, ref, "lifetime"), ret=M(cs, sel, "returns") - M(cs, ref, "returns"))
    cs = groups["bg 0.2 seeds 0-5"]; verdict["bg02_life"] = M(cs, sel, "lifetime") - M(cs, ref, "lifetime")
    passed = all(v["swap_gain"] >= 0.005 and v["life"] >= -0.0002 and v["ret"] >= -0.005 for n, v in verdict.items() if n != "bg02_life") \
        and verdict["bg02_life"] >= -0.0002
    md.append(f"\n## Verdict\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "summary": summary, "remaps": {f"{a}:{b}": v for (a, b), v in rm.items()},
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}}, open(os.path.join(OD, "results.json"), "w"))
    open(os.path.join(OD, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
