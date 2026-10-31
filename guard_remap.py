"""Guarded confidence-weighted remap (loop 03:00 UTC+8 1 Nov, pre-registered before running).

Context: conf_remap.py (m0.3: a contradiction counts twice toward remap when the store's relative margin >= 0.3)
   cut first-remap latency 10-19 ticks and added 2.4-4.3 swap points with returns flat, but 3 of 21 runs
   (s49 seed 4, bg002a seed 4, bg002b seed 10) show a transient false remap pair (out and back) before the swap.
   With H=2, K=2 a doubled contradiction makes a prototype hot by itself, so a remap can rest on only 2 arrivals.
   Sequential-test view (Wald/Lorden): weighting raises per-item information but also the variance of the sum; a
   floor on the number of independent observations restores the false-alarm rate while keeping most of the gain.
H8: guard 'n{N}': a remap of class s to label y is allowed only if at least N distinct in-radius, rival-passing
   contradictions s->y arrived in the last W=250 ticks (z5 needs ~4 by construction; m0.3 can fire on 2).
   Variants: m0.3n3, m0.3n4, m0.35n3, m0.35n4, plus controls z5, m0.3, m0.35. Exploratory only (not selectable):
   m0.3f (double only when the cached FF cortex also predicts y), cached runs only.
Selection (cached FF-log runs, 21: s49 0-5, bg002 6-8, bg02 0-5, bg002a 3-5, bg002b 9-11): among n-variants with
   zero pre-swap remaps summed over all 21, lowest mean first post-swap remap latency.
Prediction: (a) fresh store-only seeds 12-17 in s49 (4<->9) and bg002 (2<->3), never run before: pre-swap remaps per
   run <= z5 + 0.2 and mean first-remap latency <= z5 - 8 ticks; (b) cached runs routed by j+0.0b, in each of s49 3-5,
   bg002 6-8, bg002a, bg002b: swap first 300 >= z5 + 0.5, returns >= z5 - 0.5, lifetime >= z5 - 0.02; bg02 lifetime
   >= z5 - 0.02. Kill: any bar missed.
Usage: OMP_NUM_THREADS=1 python3 guard_remap.py
"""
import json, os
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import presence_bg as PB
import confirm_recency as CR
import conf_remap as CW
from data import load_mnist

OD = "results/guard_remap"
NV = ["z5", "m0.3", "m0.35", "m0.3n3", "m0.3n4", "m0.35n3", "m0.35n4"]
CR.WORLDS.update({"bg002a": (dict(bg=0.02), "results/gate_probe_s345", [3, 4, 5]),
                  "bg002b": (dict(bg=0.02), "results/gate_probe_s91011", [9, 10, 11])})
FRESH = {"s49f": dict(CR.WORLDS["s49"][0]), "bg002f": dict(CR.WORLDS["bg002"][0])}
GW = 250


class GStore(CW.CWStore):
    def __init__(self, *a, V="z5", fmap=None, **kw):
        self.N, self.ffg = None, False
        base = V
        if "n" in V[1:]:
            base, n = V.split("n"); self.N = int(n)
        if V.endswith("f"):
            base, self.ffg = V[:-1], True
        self.fmap, self.cev, self.gblk = fmap, deque(), 0
        super().__init__(*a, V=base, **kw)

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        while self.cev and self.cev[0][0] < self.t - GW:
            self.cev.popleft()
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
        self.cev.append((self.t, s, y))
        if self.mode == "m" and sm >= self.thr and (not self.ffg or self.fmap.get(self.t) == y):
            self.hist[s][j].append((1, y)); self.dbl += 1
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            if self.N and sum(1 for _, a, b in self.cev if a == s and b == y) < self.N:
                self.gblk += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


def run(job):
    wk, seed, cached = job
    w = dict(LF.WORLD, **(CR.WORLDS[wk][0] if cached else FRESH[wk]))
    VARS = NV + (["m0.3f"] if cached else [])
    fmap = None
    if cached:
        for r in json.load(open(os.path.join(CR.WORLDS[wk][1], "results.json")))["raw"]:
            if r["seed"] == seed:
                R = np.array(r["rows"]); fmap = {int(t): int(f) for t, f in zip(R[:, 0], R[:, 5])}
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: GStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v, fmap=fmap) for v in VARS}
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
        stats[v] = dict(pre_swap=sum(t < sw for t, _, _ in st.log) if sw is not None else len(st.log), lat=lat,
                        dbl=st.dbl, gblk=st.gblk)
    print(f"{wk} seed={seed} {json.dumps({v: (s['pre_swap'], s['lat']) for v, s in stats.items()})}", flush=True)
    if cached:
        np.savez(f"{OD}/Z_{wk}_{seed}.npz", **{v: np.array(r, float) for v, r in rows.items()})
    return wk, seed, cached, stats


def main():
    os.makedirs(OD, exist_ok=True)
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    jobs = [(a, b, True) for a, b in sorted(logs)] + [(wk, s, False) for wk in FRESH for s in range(12, 18)]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per, fresh = {}, {}
    for wk, s, cached, st in res:
        if cached:
            Z = np.load(f"{OD}/Z_{wk}_{s}.npz")
            per[(wk, s)] = {v: dict(CW.score(wk, Z[v], logs[(wk, s)], fe, se), **st[v]) for v in st}
        else:
            fresh[(wk, s)] = st
    L = lambda ls: [l if l is not None else 2000 for l in ls]
    def M(D, cells, v, f):
        xs = []
        for c in cells:
            xs += L(D[c][v][f]) if f == "lat" else [D[c][v][f]]
        return float(np.nanmean(xs))
    allc = sorted(per)
    groups = {"s49 3-5": [("s49", s) for s in (3, 4, 5)], "bg002 6-8": [("bg002", s) for s in (6, 7, 8)],
              "bg002a 3-5": [("bg002a", s) for s in (3, 4, 5)], "bg002b 9-11": [("bg002b", s) for s in (9, 10, 11)],
              "s49 0-2 (old selection)": [("s49", s) for s in (0, 1, 2)], "bg02 0-5": [("bg02", s) for s in range(6)]}
    cand = [v for v in NV if "n" in v[1:]]
    tot = lambda v: sum(per[c][v]["pre_swap"] for c in allc)
    swc = [c for c in allc if c[0] != "bg02"]
    ok = [v for v in cand if tot(v) == 0]
    sel = min(ok, key=lambda v: M(per, swc, v, "lat")) if ok else None
    md = ["# Guarded confidence-weighted remap (`guard_remap.py`)\n", __doc__,
          f"\nSelected: **{sel}** (n-variants with zero pre-swap remaps over 21 cached runs: {ok})\n",
          "\n## Cached runs: pre-swap remaps summed over 21 runs, mean first remap over swap runs\n",
          "| Store | Pre-swap remaps (sum) | First remap (ticks) | Guard blocks / run | Doubled / run |", "|---|---|---|---|---|"]
    for v in NV + ["m0.3f"]:
        md.append(f"| {v} | {tot(v)} | {M(per, swc, v, 'lat'):.0f} | {M(per, allc, v, 'gblk'):.1f} | {M(per, allc, v, 'dbl'):.0f} |")
    for name, cells in groups.items():
        md += [f"\n## {name}, mean (route = j+0.0b on that store)\n",
               "| Store | Lifetime % | Returns % | Swap first 300 % | First remap | Pre-swap / run | Swap vs z5 | Returns vs z5 |",
               "|---|---|---|---|---|---|---|---|"]
        for v in NV + ["m0.3f"]:
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {M(per,cells,v,'lifetime')*100:.2f} | {M(per,cells,v,'returns')*100:.1f} | "
                      f"{M(per,cells,v,'swap')*100:.1f} | {M(per,cells,v,'lat'):.0f} | {M(per,cells,v,'pre_swap'):.2f} | "
                      f"{(M(per,cells,v,'swap')-M(per,cells,'z5','swap'))*100:+.1f} | {(M(per,cells,v,'returns')-M(per,cells,'z5','returns'))*100:+.1f} |")
    fc = sorted(fresh)
    md += ["\n## Fresh store-only seeds 12-17 (s49f 4<->9, bg002f 2<->3), never run before\n",
           "| Store | Pre-swap remaps / run | First remap (ticks) | s49f first remap | bg002f first remap |", "|---|---|---|---|---|"]
    for v in NV:
        md.append(f"| {v}{' (sel)' if v == sel else ''} | {M(fresh,fc,v,'pre_swap'):.2f} | {M(fresh,fc,v,'lat'):.0f} | "
                  f"{M(fresh,[c for c in fc if c[0]=='s49f'],v,'lat'):.0f} | {M(fresh,[c for c in fc if c[0]=='bg002f'],v,'lat'):.0f} |")
    md.append("\n## Per fresh seed (pre-swap remaps, latency per swapped class)\n")
    for c in fc:
        md.append(f"- {c[0]} seed {c[1]}: " + "; ".join(f"{v} {fresh[c][v]['pre_swap']} {fresh[c][v]['lat']}" for v in NV))
    verdict, passed = {}, sel is not None
    if sel:
        verdict["fresh"] = {f: M(fresh, fc, sel, f) - M(fresh, fc, "z5", f) for f in ("pre_swap", "lat")}
        passed &= verdict["fresh"]["pre_swap"] <= 0.2 and verdict["fresh"]["lat"] <= -8
        for g in ("s49 3-5", "bg002 6-8", "bg002a 3-5", "bg002b 9-11", "bg02 0-5"):
            d = {f: M(per, groups[g], sel, f) - M(per, groups[g], "z5", f) for f in ("swap", "returns", "lifetime")}
            verdict[g] = d
            passed &= d["lifetime"] >= -0.0002 and (g == "bg02 0-5" or (d["swap"] >= 0.005 and d["returns"] >= -0.005))
    md.append(f"\n## Verdict H8 ({sel} vs z5)\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": bool(passed), "verdict": verdict,
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}, "fresh": {f"{a}:{b}": v for (a, b), v in fresh.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
