"""Confidence-weighted store remap (loop 02:00 UTC+8 1 Nov, pre-registered before running).

Context: 23:00-01:00 closed class-local reliability in the router (hard or soft): at the arbiter a swap and a return
   look the same. Courville, Daw & Touretzky 2006: change can only be inferred from evidence that differs between
   "the world changed" and "I was wrong". Inside the store it does differ: on a swap the arriving label contradicts
   a prototype that is confidently and closely matched; ambiguous items near a class border contradict weakly.
   Lorden/CUSUM: detection delay ~ log(false-alarm interval) / per-observation information, so weighting each
   contradiction by how informative it is should cut first-remap latency at a fixed false-alarm rate.
H7: in the default store (a25w250h2k2_c1.5_uinf_z5) a contradiction that passes the radius and rival tests is
   appended to the prototype's history twice (counts as 2 toward the H=2 hot threshold) when it is confident:
   'm{q}' store relative margin on the item >= q (q in 0.3, 0.5); 'r{p}' distance <= p * prototype radius
   (p in 0.5, 0.7). Control 'z5' is the default store (must reproduce confirm_recency Z exactly).
Routing: every variant's store is scored with the j+0.0b arbiter (same bin edges as results/arbiter) OR'd with
   mq30_s2, on the cached FF logs (store is pixel-only and does not depend on the cortex, so this is exact).
Selection on s49 seeds 0-2: lowest mean first post-swap remap latency among variants with pre-swap false remaps
   <= z5's on s49 0-2. Prediction on held-out s49 3-5 and bg002 6-8: first-remap latency <= z5 - 10 ticks, swap first
   300 >= j+0.0b(z5) + 0.5, returns >= -0.5, lifetime >= -0.02; and on bg 0.2 0-5 and all held-out runs, pre-swap
   false remaps per run <= z5 + 0.2, bg 0.2 lifetime >= -0.02. Kill: any bar missed.
Usage: OMP_NUM_THREADS=1 python3 conf_remap.py
Exploratory neighbour sweep (post hoc, not a test): CW_OD=results/conf_remap_sweep CW_VARS=z5,m0.2,m0.25,m0.35,m0.4 python3 conf_remap.py
"""
import json, os, re
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import rival_s49 as RS
import presence_bg as PB
import confirm_recency as CR
from data import load_mnist
from gate_probe import rolling_quantile
from arbiter_decay import arb as arb_decay

OD = os.environ.get("CW_OD", "results/conf_remap")
BASEK = "a25w250h2k2_c1.5_uinf_z5"
VARS = os.environ.get("CW_VARS", "z5,m0.3,m0.5,r0.5,r0.7").split(",")


class CWStore(RS.ZStore):
    def __init__(self, *a, V="z5", **kw):
        self.mode, self.thr = (V[0], float(V[1:])) if V != "z5" else (None, None)
        self.dbl = 0
        super().__init__(*a, K=BASEK, **kw)

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        if not self.mu:
            return
        sm = RP.answer(self, x)[1] if self.mode == "m" else None
        s, j, d = self.nearest(x)
        r = self.rad[s][j]
        if s == y:
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
        inr = r is not None and d <= r
        if inr and self.rival(x, y, d):
            self.rivals += 1; inr = False
        self.hist[s][j].append((1, y) if inr else 2)
        if not inr:
            return
        if (self.mode == "m" and sm >= self.thr) or (self.mode == "r" and d <= self.thr * r):
            self.hist[s][j].append((1, y)); self.dbl += 1
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: CWStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in VARS}
    rows = {v: [] for v in VARS}
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for v, st in sts.items():
            s, sm = RP.answer(st, x); rows[v].append((t, y, s, sm))
            st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    stats = {}
    for v, st in sts.items():
        lat = [next((t - sw for t, s_, _ in st.log if t >= sw and s_ == c), None) for c in w.get("swap", ())] if sw is not None else []
        stats[v] = dict(remaps=len(st.log), pre_swap=sum(t < sw for t, _, _ in st.log) if sw is not None else len(st.log),
                        lat=lat, dbl=st.dbl, rivals=st.rivals)
    print(f"{wk} seed={seed} done {json.dumps({v: (s['pre_swap'], s['lat']) for v, s in stats.items()})}", flush=True)
    Z = {v: np.array(r, float) for v, r in rows.items()}
    np.savez(f"{OD}/Z_{wk}_{seed}.npz", **Z)
    return wk, seed, stats


def score(wk, Z, log, fe, se):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    assert np.array_equal(Z[:, 0], T) and np.array_equal(Z[:, 1], Y)
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    use = arb_decay(dis, T, fb, sb, np.clip(S, 0, 9).astype(int), s_ok, f_ok, len(fe) + 1, 0, False) | base
    ok = np.where(use, S, F) == Y
    sw = log["swap_at"]
    idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300] if w.get("swap") else np.array([], int)
    return dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else float("nan"),
                swap=float(ok[idx].mean()) if len(idx) else float("nan"), fired=float(use.mean()),
                store_only=float((S == Y).mean()))


def main():
    os.makedirs(OD, exist_ok=True)
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    jobs = sorted(logs)
    with Pool(min(8, len(jobs))) as p:
        res = p.map(run, jobs)
    stats = {(wk, s): st for wk, s, st in res}
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per = {}
    for k in jobs:
        Zs = np.load(f"{OD}/Z_{k[0]}_{k[1]}.npz")
        z0 = np.load(f"results/confirm_recency/Z_{k[0]}_{k[1]}.npy")
        assert np.array_equal(Zs["z5"][:, :4], z0[:, :4]), f"z5 control does not reproduce {k}"
        per[k] = {v: dict(score(k[0], Zs[v], logs[k], fe, se), **stats[k][v]) for v in VARS}
    def M(cells, v, f):
        xs = []
        for c in cells:
            val = per[c][v][f]
            if f == "lat":
                xs += [l if l is not None else 2000 for l in val]
            else:
                xs.append(val)
        return float(np.nanmean(xs)) if xs else float("nan")
    groups = {"s49 selection 0-2": [("s49", s) for s in (0, 1, 2)], "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    sel_cells = groups["s49 selection 0-2"]
    cands = [v for v in VARS if v != "z5"]
    okp = lambda v: M(sel_cells, v, "pre_swap") <= M(sel_cells, "z5", "pre_swap")
    sel = min(cands, key=lambda v: M(sel_cells, v, "lat") + (0 if okp(v) else 10 ** 6))
    md = ["# Confidence-weighted store remap (`conf_remap.py`)\n", __doc__,
          f"\nSelected on s49 seeds 0-2: **{sel}** (pre-swap bar met: {okp(sel)})\n"]
    summary = {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean (route = j+0.0b on that store)\n",
               "| Store | Lifetime % | Returns % | Swap first 300 % | Store answers % | Store-only % | First post-swap remap (ticks) | Pre-swap remaps / run | Doubled contradictions / run | Swap vs z5 | Returns vs z5 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in VARS:
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {M(cells,v,'lifetime')*100:.2f} | {M(cells,v,'returns')*100:.1f} | "
                      f"{M(cells,v,'swap')*100:.1f} | {M(cells,v,'fired')*100:.2f} | {M(cells,v,'store_only')*100:.1f} | {M(cells,v,'lat'):.0f} | "
                      f"{M(cells,v,'pre_swap'):.2f} | {M(cells,v,'dbl'):.0f} | {(M(cells,v,'swap')-M(cells,'z5','swap'))*100:+.1f} | "
                      f"{(M(cells,v,'returns')-M(cells,'z5','returns'))*100:+.1f} |")
        summary[name] = {v: {f: M(cells, v, f) for f in ("lifetime", "returns", "swap", "fired", "store_only", "lat", "pre_swap", "dbl")} for v in VARS}
    verdict = {}
    for name in ("s49 held-out 3-5", "bg002 (swap 2<->3) seeds 6-8"):
        cs = groups[name]
        verdict[name] = {f: M(cs, sel, f) - M(cs, "z5", f) for f in ("lat", "swap", "returns", "lifetime", "pre_swap")}
    cs = groups["bg 0.2 seeds 0-5"]
    verdict["bg02"] = {f: M(cs, sel, f) - M(cs, "z5", f) for f in ("lifetime", "pre_swap")}
    passed = all(v["lat"] <= -10 and v["swap"] >= 0.005 and v["returns"] >= -0.005 and v["lifetime"] >= -0.0002 and v["pre_swap"] <= 0.2
                 for n, v in verdict.items() if n != "bg02") and verdict["bg02"]["lifetime"] >= -0.0002 and verdict["bg02"]["pre_swap"] <= 0.2
    md.append(f"\n## Verdict H7 ({sel}; differences vs z5 store, both routed by j+0.0b)\n\n{json.dumps(verdict, indent=1)}\n\n"
              f"Prediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "summary": summary,
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}}, open(os.path.join(OD, "results.json"), "w"))
    open(os.path.join(OD, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
