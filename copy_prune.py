"""Copy, then prune: let retrieval flicker, rewrite storage only once one label wins (loop 05:00 UTC+8 1 Nov,
pre-registered before running).

Context: hyst_remap.py (04:00) showed false remaps on the confident store m0.3 are first moves, not only ping-pong:
   hysteresis ends seed 26's 4<->9 oscillation (68 -> 22 events) but cannot stop the first false move, because every
   remap is a structural rewrite of a prototype's label. Jezek et al. 2011 (Nature 478:246): after a cue switch CA3
   flickers between the two stored maps on single theta cycles and settles; the flicker is in retrieval and neither
   map is rewritten by it.
H10: 'c{P}': when a remap would fire, do not move the hot prototypes. Add a copy of each under the new label y and
   keep the original (a twin pair). A nearest-prototype hit on either twin whose arriving label is s or y credits the
   twin with that label (a win). Retrieval answers with the twin that won most recently (tie: original). A twin that
   goes P ticks without a win is deleted and the survivor becomes an ordinary prototype. A committed remap is an
   original deleted; a committed FALSE remap is one deleted before the swap. Twins cannot be hot again while paired.
   Variants: m0.3c300, m0.3c600, m0.3n4c300, m0.3n4c600; controls z5, m0.3n4, m0.3n4hinf.
Metrics (store only): committed pre-swap remaps; tentative pre-swap copies; ans_lat = ticks after the swap until the
   store answers the new label on 3 consecutive swapped-class arrivals; swap300 = store accuracy on swapped-class
   arrivals in the first 300 ticks after the swap; pre_acc = store accuracy on all arrivals before the swap.
   s49f and bg002f with the same seed share their pre-swap stream, so false-remap bars count s49f streams.
Selection on dev fresh seeds 12-29 (incl. seed 26): among c variants, fewest committed pre-swap remaps per stream,
   ties by lower mean ans_lat (both worlds).
Prediction on held-out fresh seeds 48-65 (18 streams, never used): selected variant total committed pre-swap remaps
   <= 4 with no stream > 2, mean ans_lat <= z5 - 8 ticks, pre_acc >= z5 - 0.3 points; and on dev seed 26 committed
   pre-swap remaps <= 4. Kill: any bar missed. (Null = the false copies win their pairs because the confusable 4/9
   border really does carry the other label often enough, so retrieval cannot resolve what storage could not.)
Usage: OMP_NUM_THREADS=1 python3 copy_prune.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
import guard_remap as G
import hyst_remap as HY
import presence_bg as PB
import remap_probe as RP
from data import load_mnist

OD = "results/copy_prune"
NV = ["z5", "m0.3n4", "m0.3n4hinf", "m0.3c300", "m0.3c600", "m0.3n4c300", "m0.3n4c600"]
CAND = [v for v in NV if "c" in v]


class CPStore(HY.HStore):
    def __init__(self, *a, V="z5", **kw):
        self.P = None
        if "c" in V:
            V, P = V.split("c"); self.P = int(P)
        self.pairs, self.commit, self.copies = [], [], []
        super().__init__(*a, V=V, **kw)

    def idx(self, c, dq):
        for i, h in enumerate(self.hist.get(c, [])):
            if h is dq:
                return i
        return None

    def pair_of(self, c, dq):
        for p in self.pairs:
            if (p["o"][0] == c and p["o"][1] is dq) or (p["c"][0] == c and p["c"][1] is dq):
                return p
        return None

    def winner(self, p):
        return "c" if p["w"]["c"] > p["w"]["o"] else "o"

    def delete(self, c, dq):
        i = self.idx(c, dq)
        if i is None or len(self.mu[c]) <= 1:
            return False
        for D in (self.mu, self.var, self.n):
            D[c] = np.delete(D[c], i, axis=0)
        self.rad[c].pop(i); self.hist[c].pop(i)
        return True

    def copy(self, s, j, y):
        from collections import deque
        mu, var, n, rad = self.mu[s][j].copy(), self.var[s][j].copy(), self.n[s][j], self.rad[s][j]
        if y not in self.mu:
            self.mu[y], self.var[y], self.n[y], self.hist[y], self.rad[y] = mu[None], var[None], np.array([n]), [], []
        else:
            self.mu[y], self.var[y], self.n[y] = np.vstack([self.mu[y], mu]), np.vstack([self.var[y], var]), np.append(self.n[y], n)
        dq = deque(maxlen=10); self.hist[y].append(dq); self.rad[y].append(rad)
        self.pairs.append(dict(o=(s, self.hist[s][j]), c=(y, dq), t0=self.t, w={"o": self.t, "c": self.t}))
        self.copies.append((self.t, s, y))
        if len(self.mu[y]) > 2 * self.k:
            paired = {id(q[k][1]) for q in self.pairs for k in "oc"}
            free = [i for i, h in enumerate(self.hist[y]) if id(h) not in paired]
            if free:
                k = min(free, key=lambda i: self.n[y][i])
                for D in (self.mu, self.var, self.n):
                    D[y] = np.delete(D[y], k, axis=0)
                self.rad[y].pop(k); self.hist[y].pop(k)
        self.remaps += 1

    def prune(self):
        keep = []
        for p in self.pairs:
            io, ic = self.idx(*p["o"]), self.idx(*p["c"])
            if io is None or ic is None:
                continue
            lo, lc = self.t - p["w"]["o"] > self.P, self.t - p["w"]["c"] > self.P
            if lo or lc:
                if lo and (not lc or p["w"]["c"] > p["w"]["o"]):
                    if self.delete(*p["o"]):
                        self.commit.append((self.t, p["o"][0], p["c"][0]))
                else:
                    self.delete(*p["c"])
                continue
            keep.append(p)
        self.pairs = keep

    def masked(self):
        """(class, index) of losing twins, hidden from retrieval."""
        out = set()
        for p in self.pairs:
            lose = p["o"] if self.winner(p) == "c" else p["c"]
            i = self.idx(*lose)
            if i is not None:
                out.add((lose[0], i))
        return out

    def answer(self, x):
        if not self.mu:
            return -1, 0.0
        m = self.masked()
        cs = sorted(self.mu)
        mu = np.concatenate([self.mu[c] for c in cs])
        ys = np.concatenate([np.full(len(self.mu[c]), c) for c in cs])
        ok = np.array([(c, i) not in m for c in cs for i in range(len(self.mu[c]))])
        d = ((mu - x) ** 2).sum(1); d[~ok] = np.inf
        j = d.argmin(); s = int(ys[j]); other = d[(ys != s) & ok]
        return s, (float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0)

    def observe(self, x, y):
        if self.P is None:
            return super().observe(x, y)
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        while self.cev and self.cev[0][0] < self.t - G.GW:
            self.cev.popleft()
        if not self.mu:
            return
        self.prune()
        sm = self.answer(x)[1] if self.mode == "m" else None
        s, j, d = self.nearest(x)
        p = self.pair_of(s, self.hist[s][j])
        if p is not None:
            lab = {p["o"][0]: "o", p["c"][0]: "c"}
            if y in lab:
                p["w"][lab[y]] = self.t
                s, dq = p[lab[y]]; j = self.idx(s, dq)
                d = float(((self.mu[s][j] - x[0]) ** 2).sum())
            else:
                s, dq = p[self.winner(p)]; j = self.idx(s, dq)
                d = float(((self.mu[s][j] - x[0]) ** 2).sum())
            self._s, self._j = s, j
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
            self.hist[s][j].append((1, y)); self.dbl += 1
        paired = {id(q[k][1]) for q in self.pairs for k in "oc"}
        hot = [i for i, h in enumerate(self.hist[s]) if id(h) not in paired and sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            if self.N and sum(1 for _, a, b in self.cev if a == s and b == y) < self.N:
                self.gblk += 1; return
            for i in hot:
                self.log.append((self.t, s, y))
                self.copy(s, i, y)
            for i in hot:
                self.hist[s][i].clear()


def ans(st, x):
    return st.answer(x[0]) if st.P is not None else RP.answer(st, x[0])


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **G.FRESH[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: CPStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in NV}
    rows = {v: [] for v in NV}
    sw_cls = set(w.get("swap", ()))
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for v, st in sts.items():
            rows[v].append((t, c, y, ans(st, x)[0]))
            st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    stats = {}
    for v, st in sts.items():
        R = np.array(rows[v])
        pre = R[R[:, 0] < sw]; post = R[(R[:, 0] >= sw) & np.isin(R[:, 1], list(sw_cls))]
        p300 = post[post[:, 0] < sw + 300]
        ok = post[:, 3] == post[:, 2]; al = 2000
        for i in range(len(ok) - 2):
            if ok[i] and ok[i + 1] and ok[i + 2]:
                al = int(post[i + 2, 0] - sw); break
        comm = [e for e in st.commit if e[0] < sw] if st.P is not None else [e for e in st.log if e[0] < sw]
        lat = [next((t - sw for t, s_, _ in st.log if t >= sw and s_ == c), None) for c in w.get("swap", ())]
        stats[v] = dict(commit=len(comm), first_commit=comm[:4],
                        copies=sum(e[0] < sw for e in st.copies) if st.P is not None else len(comm),
                        ans_lat=al, lat=lat, swap300=float((p300[:, 3] == p300[:, 2]).mean() * 100),
                        pre_acc=float((pre[:, 3] == pre[:, 2]).mean() * 100))
    print(f"{wk} seed={seed} {json.dumps({v: (s['commit'], s['copies'], s['ans_lat']) for v, s in stats.items()})}", flush=True)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    seeds = list(range(12, 30)) + list(range(48, 66))
    jobs = [(wk, s) for wk in G.FRESH for s in seeds]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 48)
    M = lambda cells, v, f: float(np.mean([fr[c][v][f] for c in cells]))
    S = lambda cells: [c for c in cells if c[0] == "s49f"]
    sel = min(CAND, key=lambda v: (M(S(dev), v, "commit"), M(dev, v, "ans_lat")))
    md = ["# Copy, then prune (`copy_prune.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29", dev), ("Held-out fresh seeds 48-65", test)):
        md += [f"\n## {name} (18 streams, 36 runs)\n",
               "| Store | Committed false remaps (sum over streams) | Streams with any | Max per stream | Tentative copies pre-swap / stream | ans_lat s49f | ans_lat bg002f | swap300 s49f | swap300 bg002f | pre_acc |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for v in NV:
            cm = [fr[c][v]["commit"] for c in S(cells)]
            sub = lambda wk: [c for c in cells if c[0] == wk]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {sum(cm)} | {sum(x > 0 for x in cm)} | {max(cm)} | {M(S(cells), v, 'copies'):.2f} | "
                      f"{M(sub('s49f'), v, 'ans_lat'):.1f} | {M(sub('bg002f'), v, 'ans_lat'):.1f} | {M(sub('s49f'), v, 'swap300'):.1f} | "
                      f"{M(sub('bg002f'), v, 'swap300'):.1f} | {M(cells, v, 'pre_acc'):.2f} |")
    md.append("\n## Streams with any committed pre-swap remap (s49f; variant: count, first events (tick, from, to))\n")
    for c in sorted(S(list(fr))):
        bad = {v: fr[c][v] for v in NV if fr[c][v]["commit"]}
        if bad:
            md.append(f"- seed {c[1]}: " + "; ".join(f"{v} {b['commit']} {b['first_commit']}" for v, b in bad.items()))
    tc = [fr[c][sel]["commit"] for c in S(test)]
    verdict = dict(commit_total=int(sum(tc)), commit_max=int(max(tc)),
                   ans_lat=M(test, sel, "ans_lat") - M(test, "z5", "ans_lat"),
                   pre_acc=M(test, sel, "pre_acc") - M(test, "z5", "pre_acc"),
                   seed26=int(fr[("s49f", 26)][sel]["commit"]))
    passed = (verdict["commit_total"] <= 4 and verdict["commit_max"] <= 2 and verdict["ans_lat"] <= -8
              and verdict["pre_acc"] >= -0.3 and verdict["seed26"] <= 4)
    md.append(f"\n## Verdict H10 ({sel} vs z5)\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
