"""Where the post-swap remap latency goes (loop 03:30 hyp 1, pre-registered 04:20 UTC+8 31 Oct).

Default store a25w250h2k2 (store-only run, no FF rows needed). For every arrival of a swapped digit c in {2, 3} from
swap_at until that old class's first remap (cap 300 ticks), classify what the store did with it:
  agree   nearest prototype already has the NEW label y (no conflict, it refreshes the wrong-basin prototype)
  other   nearest prototype is a third class
  out     nearest prototype is the old class c but the item is outside its radius (no evidence)
  sub     in-radius conflict, but fewer than K = 2 prototypes of c are hot toward y
  blocked in-radius conflict with >= K hot prototypes, remap vetoed by a25 (class c judged absent)
  remap   the remap fires
Prediction (03:30 hyp 1): >= 50% of the latency window is spent with >= K hot prototypes blocked by a25.
Kill: a25 blocks 0 post-swap remaps; then the floor sits in the evidence path (out/agree/sub).
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import presence_bg as PB
import fast_remap as FR

CATS = ["agree", "other", "out", "sub", "blocked", "remap"]


class DStore(FR.FStore):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.ev = []; self.sw = None; self.pair = None; self.first = {}

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        if not self.mu:
            return
        s, j, d = self.nearest(x)
        r = self.rad[s][j]
        c = self.cur_c
        track = self.sw is not None and self.sw <= self.t < self.sw + 300 and c in self.pair and c not in self.first
        rec = lambda cat, **k: self.ev.append(dict(t=self.t - self.sw, c=int(c), cat=cat, **k)) if track else None
        if s == y:
            rec("agree" if c != y else "own")
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
        inr = r is not None and d <= r
        self.hist[s][j].append((1, y) if inr else 2)
        if s != c:
            rec("other"); 
            if not inr: return
        elif not inr:
            rec("out", ratio=float(d / r) if r else None); return
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1
                if s == c: rec("blocked", nhot=len(hot))
                return
            if s == c: rec("remap", nhot=len(hot)); self.first.setdefault(c, self.t - self.sw if self.t >= self.sw else None)
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)
        elif s == c:
            rec("sub", nhot=len(hot), nproto=len(self.mu[s]))


def run(args):
    seed, days, bg = args
    w = dict(LF.WORLD, bg=bg)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    st = DStore(9, np.random.default_rng(seed + 1000), noise=0.0, K="a25w250h2k2")
    st.sw, st.pair = world.swap_at, tuple(w["swap"])
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        st.t = t; st.cur_c = c; st.observe(x, y); st.add(x, y)
    return bg, seed, st.ev, {str(k): v for k, v in st.first.items()}, sum(t >= world.swap_at for t, _, _ in st.log)


def main():
    out_dir = "results/lat_diag"; os.makedirs(out_dir, exist_ok=True)
    load_mnist("mnist")
    jobs = [(s, 16, 0.02) for s in range(6)] + [(s, 16, 0.2) for s in range(6)]
    with Pool(min(12, os.cpu_count())) as p:
        res = p.map(run, jobs, chunksize=1)
    md = ["# Post-swap remap latency diagnostic (`lat_diag.py`)\n", __doc__]
    allres = {}
    for bg in (0.02, 0.2):
        rows = [r for r in res if r[0] == bg]
        md += [f"\n## bg {bg}, seeds 0-5: swapped-digit arrivals before the old class's first remap\n",
               "| Seed | Digit | First remap (ticks) | Arrivals | " + " | ".join(CATS + ["own"]) + " |", "|" + "---|" * (4 + len(CATS) + 1)]
        tot = {k: 0 for k in CATS + ["own"]}
        for _, seed, ev, first, npost in rows:
            allres[f"{bg}_{seed}"] = dict(events=ev, first=first, post_remaps=npost)
            for c in (2, 3):
                e = [x for x in ev if x["c"] == c]
                cnt = {k: sum(x["cat"] == k for x in e) for k in CATS + ["own"]}
                for k in tot: tot[k] += cnt[k]
                md.append(f"| {seed} | {c} | {first.get(str(c), '>300')} | {len(e)} | " + " | ".join(str(cnt[k]) for k in CATS + ["own"]) + " |")
        n = sum(tot.values())
        md.append("\nTotals: " + ", ".join(f"{k} {tot[k]} ({tot[k]/max(n,1)*100:.0f}%)" for k in CATS + ["own"]))
        subs = [x for _, _, ev, _, _ in rows for x in ev if x["cat"] == "sub"]
        if subs:
            md.append(f"Sub-K conflicts: mean hot prototypes {np.mean([x['nhot'] for x in subs]):.2f}, mean prototypes of class {np.mean([x['nproto'] for x in subs]):.1f}")
        outs = [x["ratio"] for _, _, ev, _, _ in rows for x in ev if x["cat"] == "out" and x.get("ratio")]
        if outs:
            md.append(f"Out-of-radius distance / radius: median {np.median(outs):.2f}, q25 {np.quantile(outs, .25):.2f}, q75 {np.quantile(outs, .75):.2f}")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
