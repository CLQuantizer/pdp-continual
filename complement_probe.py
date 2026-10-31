"""Complement probe (loop 19:00 UTC+8 31 Oct, pre-registered before running; 18:00 hypothesis 1).

Krogh & Vedelsby 1995: the store helps only through disagreement where it is right. Measure that directly. Rerun the
s49 world (bg 0.02, swap 4<->9) with the default pixel z5 store (a25w250h2k2_c1.5_uinf_z5, no FF needed: the store does
not depend on the cortex) and join its answers to the logged FF answers in results/gate_probe_s49 (seeds 0-5).
Complement accuracy = P(store right | FF wrong). Diagnostic bins by ticks since the TRUE class's last return start
(a return = class back after >= 2000 ticks away). Gate uses only causal label-side features: ticks since the return
start of the store's answer label s (or FF label f), from labels already observed.
  H1 diag: complement accuracy in the first 300 ticks after a return >= 40% and < 15% elsewhere.
     Kill: ratio (first 300 / elsewhere) < 1.5.
  H1 gate: rt_T gate (store when it disagrees and label s or f returned < T ticks ago, plus mq30_s2) beats
     mq30_s2 by >= +2 returns at lifetime >= 96.0 (T chosen on seeds 0-2, confirmed on 3-5). Kill otherwise.
Usage: OMP_NUM_THREADS=1 python3 complement_probe.py
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist
from gate_probe import rolling_quantile

SEEDS = [0, 1, 2, 3, 4, 5]
WEX = dict(bg=0.02, swap=(4, 9))
K = "a25w250h2k2_c1.5_uinf_z5"
TS = (100, 200, 300, 500, 1000, 2000)
BINS = [(0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10 ** 9)]


def run(seed):
    w = dict(LF.WORLD, **WEX)
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K)
    rows = []
    last_c, rs_c, last_l, rs_l = {}, {}, {}, {}
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        if c not in last_c:
            rs_c[c] = None
        elif t - last_c[c] >= LF.RET_GAP:
            rs_c[c] = t
        last_c[c] = t
        s, sm = RP.answer(st, x)
        tc = t - rs_c[c] if rs_c.get(c) is not None else -1
        rl = lambda l: (t - rs_l[l]) if l >= 0 and rs_l.get(l) is not None else 10 ** 9
        rows.append((t, y, s, sm, tc, rl(s)))
        if y in last_l and t - last_l[y] >= LF.RET_GAP:
            rs_l[y] = t
        last_l[y] = t
        st.t = t; st.observe(x, y); st.add(x, y)
    print(f"seed={seed} done", flush=True)
    return seed, np.array(rows, float), {l: v for l, v in rs_l.items()}




def main():
    J = json.load(open("results/gate_probe_s49/results.json"))
    logs = {r["seed"]: r for r in J["raw"]}
    load_mnist("mnist")
    with Pool(len(SEEDS)) as p:
        res = p.map(run, SEEDS, chunksize=1)
    w = dict(LF.WORLD, **WEX)
    per = {}
    for seed, Z, _ in res:
        R = np.array(logs[seed]["rows"])
        T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
        assert np.array_equal(Z[:, 1], Y) and np.array_equal(Z[:, 0], T)
        S, SM, TC, RLS = Z[:, 2], Z[:, 3], Z[:, 4], Z[:, 5]
        # causal ticks since return start of FF label f (recomputed from the observed label stream)
        last_l, rs_l, RLF = {}, {}, np.full(len(T), 10 ** 9, float)
        for i in range(len(T)):
            f = int(F[i])
            if rs_l.get(f) is not None:
                RLF[i] = T[i] - rs_l[f]
            y = int(Y[i])
            if y in last_l and T[i] - last_l[y] >= LF.RET_GAP:
                rs_l[y] = T[i]
            last_l[y] = T[i]
        ffw = F != Y; dis = (S >= 0) & (S != F); prot = CF >= 1000
        d = {}
        for lo, hi in BINS:
            m = (TC >= lo) & (TC < hi)
            d[f"{lo}-{hi}"] = dict(n=int(m.sum()), ffw=float(ffw[m].mean()) if m.any() else None,
                                   comp=float((S[m & ffw] == Y[m & ffw]).mean()) if (m & ffw).any() else None,
                                   nffw=int((m & ffw).sum()),
                                   harm=float(dis[m & ~ffw].mean()) if (m & ~ffw).any() else None)
        m = TC < 0
        d["first-season"] = dict(n=int(m.sum()), ffw=float(ffw[m].mean()), nffw=int((m & ffw).sum()),
                                 comp=float((S[m & ffw] == Y[m & ffw]).mean()) if (m & ffw).any() else None, harm=float(dis[m & ~ffw].mean()))
        early, late = (TC >= 0) & (TC < 300), (TC >= 300)
        d["_ratio"] = dict(early=float((S[early & ffw] == Y[early & ffw]).mean()),
                           late=float((S[late & ffw] == Y[late & ffw]).mean()))
        lo30 = FFM < rolling_quantile(FFM, 0.3)
        base = dis & prot & lo30 & (SM >= 0.2)
        routes = {"ff": np.zeros(len(T), bool), "mq30_s2": base}
        for Tg in TS:
            rec = (RLS < Tg) | (RLF < Tg)
            routes[f"rt{Tg}"] = base | (dis & rec)
            for s in (0.1, 0.2, 0.3):
                routes[f"rt{Tg}_s{int(s*10)}"] = base | (dis & rec & (SM >= s))
        g = {}
        sw = logs[seed]["swap_at"]
        for k, use in routes.items():
            P = np.where(use, S, F); ok = P == Y
            idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300]
            g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), swap=float(ok[idx].mean()),
                        fired=float(use.mean()))
        per[seed] = dict(bins=d, gates=g)
    M = lambda ss, k, f: float(np.mean([per[s]["gates"][k][f] for s in ss]))
    gk = [k for k in per[0]["gates"] if k.startswith("rt")]
    sel = max(gk, key=lambda k: M([0, 1, 2], k, "returns") if M([0, 1, 2], k, "lifetime") >= M([0, 1, 2], "mq30_s2", "lifetime") - 0.0005 else -1)
    md = ["# Complement probe: where does the store fix FF errors? (`complement_probe.py`)\n", __doc__,
          "\n## Complement accuracy by ticks since the true class's return start (all seeds pooled per-seed mean)\n",
          "| Bin | Arrivals | FF wrong % | FF-wrong n | Store right on FF-wrong % | Store disagrees on FF-right % |", "|---|---|---|---|---|---|"]
    for b in [f"{lo}-{hi}" for lo, hi in BINS] + ["first-season"]:
        v = lambda f: [per[s]["bins"][b][f] for s in SEEDS if per[s]["bins"][b][f] is not None]
        fmt = lambda xs: f"{np.mean(xs)*100:.1f}" if xs else "-"
        md.append(f"| {b} | {np.mean(v('n')):.0f} | {fmt(v('ffw'))} | {np.mean(v('nffw')):.0f} | {fmt(v('comp'))} | {fmt(v('harm'))} |")
    e = np.mean([per[s]["bins"]["_ratio"]["early"] for s in SEEDS]); l = np.mean([per[s]["bins"]["_ratio"]["late"] for s in SEEDS])
    md.append(f"\nFirst 300 ticks after return: complement {e*100:.1f}%; >= 300 ticks: {l*100:.1f}%; ratio {e/max(l,1e-9):.2f} (kill < 1.5).\n")
    md.append(f"Gate selected on seeds 0-2 (best returns at lifetime >= mq30_s2): **{sel}**\n")
    for name, ss in (("selection seeds 0-2", [0, 1, 2]), ("held-out seeds 3-5", [3, 4, 5]), ("all seeds", SEEDS)):
        md += [f"\n## {name}, mean %\n", "| Route | Lifetime | Returns | Swap first 300 | Store answers |", "|---|---|---|---|---|"]
        for k in ["ff", "mq30_s2", sel] + [x for x in ("rt300", "rt300_s2", "rt1000_s2") if x != sel]:
            md.append(f"| {k} | {M(ss,k,'lifetime')*100:.2f} | {M(ss,k,'returns')*100:.1f} | {M(ss,k,'swap')*100:.1f} | {M(ss,k,'fired')*100:.2f} |")
    md += ["\n## Full gate sweep, all seeds (lifetime / returns / fired %)\n", "| Route | L/R/F |", "|---|---|"]
    for k in per[0]["gates"]:
        md.append(f"| {k} | {M(SEEDS,k,'lifetime')*100:.2f} / {M(SEEDS,k,'returns')*100:.1f} / {M(SEEDS,k,'fired')*100:.2f} |")
    od = "results/complement_probe"; os.makedirs(od, exist_ok=True)
    json.dump({"sel": sel, "raw": {str(k): v for k, v in per.items()}}, open(os.path.join(od, "results.json"), "w"))
    open(os.path.join(od, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
