"""Remap hysteresis for the confidence-weighted store (loop 04:00 UTC+8 1 Nov, pre-registered before running).

Context: conf_remap.py m0.3 (a contradiction counts twice toward remap when the store's relative margin >= 0.3) cuts
   first-remap latency ~19 ticks but false-remaps: 3.33 events/run on fresh seeds 12-17, and with a 4-contradiction
   floor (m0.3n4) one fresh stream (s49f seed 26) ping-pongs 4<->9 for the whole pre-swap period (68 events).
   Wills et al. 2005 (Science 308:873): CA3 place-cell maps switch abruptly but show hysteresis; the state you are in
   is sticky. A moved prototype in our store has a fresh hist deque, so moving it back costs exactly as little as
   moving it in, and doubled credit makes it cheaper still: the loop has no energy barrier.
H9: hysteresis 'h{T}': a prototype moved within the last T ticks (T=2000 or inf) gets no doubled credit and needs
   H+1=3 contradictions to become hot; 'b' variants apply this only to a move back to the label it came from.
   Variants: m0.3h2000, m0.3hinf, m0.3binf, m0.3n4hinf; controls z5, m0.3, m0.3n4.
Selection on dev fresh seeds 12-29 (s49f 4<->9, bg002f 2<->3; 36 runs incl. seed 26): among h/b variants, lowest
   pre-swap remaps per run, ties broken by lower mean first post-swap remap latency.
Prediction on held-out fresh seeds 30-47 (36 runs, never used): selected variant pre-swap remaps per run <= z5 + 0.2,
   no run with > 4 pre-swap events, and mean first-remap latency <= z5 - 8 ticks. Kill: any bar missed. (Null = the
   false remaps are first moves, not ping-pong, so the barrier must sit on the first move, not on the return.)
Usage: OMP_NUM_THREADS=1 python3 hyst_remap.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import presence_bg as PB
import guard_remap as G
from data import load_mnist

OD = "results/hyst_remap"
NV = ["z5", "m0.3", "m0.3n4", "m0.3h2000", "m0.3hinf", "m0.3binf", "m0.3n4hinf"]
CAND = [v for v in NV if "h" in v or "b" in v]


class HStore(G.GStore):
    def __init__(self, *a, V="z5", **kw):
        self.T, self.back = None, False
        for tag in ("h", "b"):
            if tag in V:
                V, T = V.split(tag); self.T = float(T); self.back = tag == "b"
        self.mv, self.keep, self.hblk = {}, [], 0
        super().__init__(*a, V=V, **kw)

    def move(self, s, j, y):
        super().move(s, j, y)
        dq = self.hist[y][-1] if self.hist[y] else None
        if dq is not None:
            self.mv[id(dq)] = (self.t, s); self.keep.append(dq)

    def sticky(self, s, i, y):
        if self.T is None:
            return False
        m = self.mv.get(id(self.hist[s][i]))
        if m is None or self.t - m[0] > self.T:
            return False
        return (not self.back) or m[1] == y

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        while self.cev and self.cev[0][0] < self.t - G.GW:
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
        if self.mode == "m" and sm >= self.thr:
            if self.sticky(s, j, y):
                self.hblk += 1
            else:
                self.hist[s][j].append((1, y)); self.dbl += 1
        hot = [i for i, h in enumerate(self.hist[s])
               if sum(e == (1, y) for e in h) >= self.H + (1 if self.sticky(s, i, y) else 0)]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            if self.N and sum(1 for _, a, b in self.cev if a == s and b == y) < self.N:
                self.gblk += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **G.FRESH[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: HStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in NV}
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for st in sts.values():
            st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    stats = {}
    for v, st in sts.items():
        lat = [next((t - sw for t, s_, _ in st.log if t >= sw and s_ == c), None) for c in w.get("swap", ())]
        pre = [(t, s_, y_) for t, s_, y_ in st.log if t < sw]
        stats[v] = dict(pre_swap=len(pre), lat=lat, first_pre=pre[:4], hblk=st.hblk, dbl=st.dbl)
    print(f"{wk} seed={seed} {json.dumps({v: (s['pre_swap'], s['lat']) for v, s in stats.items()})}", flush=True)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    jobs = [(wk, s) for wk in G.FRESH for s in range(12, 48)]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 30)
    L = lambda ls: [l if l is not None else 2000 for l in ls]
    def M(cells, v, f):
        xs = []
        for c in cells:
            xs += L(fr[c][v][f]) if f == "lat" else [fr[c][v][f]]
        return float(np.mean(xs))
    sel = min(CAND, key=lambda v: (M(dev, v, "pre_swap"), M(dev, v, "lat")))
    md = ["# Remap hysteresis (`hyst_remap.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29 (36 runs)", dev), ("Held-out fresh seeds 30-47 (36 runs)", test)):
        md += [f"\n## {name}\n", "| Store | Pre-swap remaps / run | Runs with pre-swap remaps | Max per run | First remap (ticks) | s49f | bg002f | Hysteresis blocks / run |",
               "|---|---|---|---|---|---|---|---|"]
        for v in NV:
            pr = [fr[c][v]["pre_swap"] for c in cells]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {np.mean(pr):.2f} | {sum(p > 0 for p in pr)} | {max(pr)} | {M(cells,v,'lat'):.1f} | "
                      f"{M([c for c in cells if c[0]=='s49f'],v,'lat'):.1f} | {M([c for c in cells if c[0]=='bg002f'],v,'lat'):.1f} | {M(cells,v,'hblk'):.1f} |")
    md.append("\n## Runs with any pre-swap remap (variant: count, first events (tick, from, to))\n")
    for c in sorted(fr):
        bad = {v: fr[c][v] for v in NV if fr[c][v]["pre_swap"]}
        if bad:
            md.append(f"- {c[0]} seed {c[1]}: " + "; ".join(f"{v} {b['pre_swap']} {b['first_pre']}" for v, b in bad.items()))
    tp = [fr[c][sel]["pre_swap"] for c in test]
    verdict = dict(pre_swap=M(test, sel, "pre_swap") - M(test, "z5", "pre_swap"), max_run=int(max(tp)),
                   lat=M(test, sel, "lat") - M(test, "z5", "lat"))
    passed = verdict["pre_swap"] <= 0.2 and verdict["max_run"] <= 4 and verdict["lat"] <= -8
    md.append(f"\n## Verdict H9 ({sel} vs z5 on held-out seeds 30-47)\n\n{json.dumps(verdict, indent=1)}\n\n"
              f"Prediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
