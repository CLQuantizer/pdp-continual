"""Recency-gate confirmation (loop 20:00 UTC+8 31 Oct, pre-registered before running; 19:00 hypotheses 1 and 2).

H1 (confirm, frozen): rt2000_s2 (store answers when it disagrees with FF, relative margin >= 0.2, and its label s or the
   FF label f returned from >= 2000 ticks away < 2000 ticks ago; plus mq30_s2) was selected on s49 seeds 0-2.
   Evaluate it unchanged on worlds it never saw: bg002 (swap 2<->3) seeds 6-8 (logs results/gate_probe_s678) and bg 0.2
   seeds 0-5 (logs results/gate_probe_bg02). Prediction: >= +1.0 returns above the mq frontier at matched firing
   (frontier = best returns over the mq{q}_s{s} sweep (and, after the control was added, c{s}) with firing <= the gate's firing), lifetime within 0.05 of
   mq30_s2. Kill: < +0.5 on either world.
Added after first pass (same hour, before reading the bg02 rerun): the selected trace dk3000_1_4 also raised lifetime on
   bg 0.2, where no class ever returns (age is infinite), so part of its gain is a recency-free route "disagree and
   margin >= 0.4". Control c{s} = mq30_s2 | (disagree & margin >= s), no recency; hyperbolic trace hy (Nassar et al.
   2010 run-length form) s(age) = s_lo + (s_hi - s_lo) age/(age + tau). H2 now passes only if dk beats the best
   c{s} at matched firing by >= +0.5 returns. Kill otherwise.
H2 (decaying trace): margin threshold s(age) = s_lo + (s_hi - s_lo) (1 - exp(-age/tau)), age = ticks since the more
   recent return of s or f, store answers when it disagrees and margin >= s(age) (plus mq30_s2). Variant chosen on s49
   seeds 0-2 (best returns at lifetime >= mq30_s2 - 0.05pt), compared with rt2000_s2 on held-out s49 3-5, bg002 6-8,
   bg02 0-5. Prediction: >= +0.5 returns over rt2000_s2 at lifetime within 0.02. Kill: no gain at any tau.
Usage: OMP_NUM_THREADS=1 python3 confirm_recency.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist
from gate_probe import rolling_quantile

K = "a25w250h2k2_c1.5_uinf_z5"
WORLDS = {"s49": (dict(bg=0.02, swap=(4, 9)), "results/gate_probe_s49", [0, 1, 2, 3, 4, 5]),
          "bg002": (dict(bg=0.02), "results/gate_probe_s678", [6, 7, 8]),
          "bg02": (dict(bg=0.2), "results/gate_probe_bg02", [0, 1, 2, 3, 4, 5])}
TAUS, SLO, SHI = (300, 1000, 3000), (0.1, 0.2), (0.4, 0.6, 1.0)
MQ = [(q, s) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7) for s in (0.0, 0.1, 0.2, 0.3)]


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **WORLDS[wk][0])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K)
    rows, last_l, rs_l = [], {}, {}
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        s, sm = RP.answer(st, x)
        rl = (t - rs_l[s]) if s >= 0 and rs_l.get(s) is not None else 10 ** 9
        rows.append((t, y, s, sm, rl))
        if y in last_l and t - last_l[y] >= LF.RET_GAP:
            rs_l[y] = t
        last_l[y] = t
        st.t = t; st.observe(x, y); st.add(x, y)
    print(f"{wk} seed={seed} done", flush=True)
    Z = np.array(rows, float)
    os.makedirs("results/confirm_recency", exist_ok=True)
    np.save(f"results/confirm_recency/Z_{wk}_{seed}.npy", Z)
    return wk, seed, Z


def score(wk, seed, Z, log):
    w = dict(LF.WORLD, **WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    assert np.array_equal(Z[:, 1], Y) and np.array_equal(Z[:, 0], T)
    S, SM, RLS = Z[:, 2], Z[:, 3], Z[:, 4]
    last_l, rs_l, RLF = {}, {}, np.full(len(T), 10 ** 9, float)
    for i in range(len(T)):
        f = int(F[i])
        if rs_l.get(f) is not None:
            RLF[i] = T[i] - rs_l[f]
        y = int(Y[i])
        if y in last_l and T[i] - last_l[y] >= LF.RET_GAP:
            rs_l[y] = T[i]
        last_l[y] = T[i]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    age = np.minimum(RLS, RLF)
    routes = {"ff": np.zeros(len(T), bool)}
    for q, s in MQ:
        routes[f"mq{int(q*100)}_s{int(s*10)}"] = dis & prot & (FFM < rolling_quantile(FFM, q)) & (SM >= s)
    base = routes["mq30_s2"]
    routes["rt2000_s2"] = base | (dis & (age < 2000) & (SM >= 0.2))
    for tau in TAUS:
        for lo in SLO:
            for hi in SHI:
                thr = lo + (hi - lo) * (1 - np.exp(-age / tau))
                routes[f"dk{tau}_{int(lo*10)}_{int(hi*10)}"] = base | (dis & (SM >= thr))
                hy = lo + (hi - lo) * age / (age + tau)
                routes[f"hy{tau}_{int(lo*10)}_{int(hi*10)}"] = base | (dis & (SM >= hy))
    for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        routes[f"c{int(c*10)}"] = base | (dis & (SM >= c))
        routes[f"p{int(c*10)}"] = dis & (SM >= c)  # store margin alone, no FF-margin route  # control: same disagreement route, no recency at all
    g = {}
    sw = log["swap_at"]
    for k, use in routes.items():
        ok = np.where(use, S, F) == Y
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300]
        g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), swap=float(ok[idx].mean()),
                    fired=float(use.mean()))
    return g


def main():
    load_mnist("mnist")
    logs = {}
    for wk, (_, d, seeds) in WORLDS.items():
        J = json.load(open(os.path.join(d, "results.json")))
        for r in J["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    jobs = [(wk, s) for wk, (_, _, ss) in WORLDS.items() for s in ss]
    cached = all(os.path.exists(f"results/confirm_recency/Z_{wk}_{s}.npy") for wk, s in jobs)
    if cached:
        res = [(wk, s, np.load(f"results/confirm_recency/Z_{wk}_{s}.npy")) for wk, s in jobs]
    else:
        with Pool(8) as p:
            res = p.map(run, jobs, chunksize=1)
    per = {(wk, s): score(wk, s, Z, logs[(wk, s)]) for wk, s, Z in res}
    M = lambda cells, k, f: float(np.mean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]
    dks = [k for k in per[sel_cells[0]] if k.startswith("dk")]
    lb = M(sel_cells, "mq30_s2", "lifetime") - 0.0005
    sel = max(dks, key=lambda k: M(sel_cells, k, "returns") if M(sel_cells, k, "lifetime") >= lb else -1)
    groups = {"s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)], "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)],
              "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)], "s49 selection 0-2": sel_cells}
    mqk = [f"mq{int(q*100)}_s{int(s*10)}" for q, s in MQ] + [f"c{c}" for c in (1, 2, 3, 4, 5, 6, 8)] + [f"p{c}" for c in (1, 2, 3, 4, 5, 6, 8)]

    def frontier(cells, fired):
        c = [M(cells, k, "returns") for k in mqk if M(cells, k, "fired") <= fired + 1e-9]
        return max(c) if c else M(cells, "ff", "returns")
    md = ["# Recency gate: confirmation off its selection world, and a decaying trace (`confirm_recency.py`)\n", __doc__,
          f"\nDecaying-trace variant selected on s49 seeds 0-2: **{sel}**\n"]
    summary = {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean %\n",
               "| Route | Lifetime | Returns | Swap first 300 | Store answers | mq frontier returns at this firing | Above frontier |",
               "|---|---|---|---|---|---|---|"]
        hys = [k for k in per[cells[0]] if k.startswith("hy")]
        hsel = max(hys, key=lambda k: M(sel_cells, k, "returns") if M(sel_cells, k, "lifetime") >= lb else -1)
        for k in ["ff", "mq30_s2", "mq30_s1", "c2", "c3", "c4", "c5", "p2", "p3", "p4", "rt2000_s2", sel, hsel]:
            fr = frontier(cells, M(cells, k, "fired"))
            md.append(f"| {k} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | {M(cells,k,'swap')*100:.1f} | "
                      f"{M(cells,k,'fired')*100:.2f} | {fr*100:.1f} | {(M(cells,k,'returns')-fr)*100:+.1f} |")
        summary[name] = {k: dict(lifetime=M(cells, k, "lifetime"), returns=M(cells, k, "returns"), fired=M(cells, k, "fired"),
                                 above=M(cells, k, "returns") - frontier(cells, M(cells, k, "fired")))
                         for k in ["mq30_s2", "rt2000_s2", sel, hsel, "c2", "c3", "c4", "p3"]}
        best_dk = max(dks, key=lambda k: M(cells, k, "returns") if M(cells, k, "lifetime") >= M(cells, "rt2000_s2", "lifetime") - 0.0002 else -1)
        md.append(f"\nBest decaying trace here at lifetime >= rt2000_s2 - 0.02pt (oracle pick, not a test): {best_dk} "
                  f"{M(cells,best_dk,'lifetime')*100:.2f} / {M(cells,best_dk,'returns')*100:.1f} / fired {M(cells,best_dk,'fired')*100:.2f}")
    od = "results/confirm_recency"; os.makedirs(od, exist_ok=True)
    json.dump({"sel": sel, "summary": summary, "raw": {f"{a}:{b}": v for (a, b), v in per.items()}}, open(os.path.join(od, "results.json"), "w"))
    open(os.path.join(od, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
