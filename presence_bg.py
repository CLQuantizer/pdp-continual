"""Is rate presence robust to background rate? (loop 23:05 hyp 2, pre-registered 00:20 HKT 31 Oct).

Fixed r3 gates a class-level remap of class s on ">= 5 arrivals of s in the last 500 ticks", a threshold that
assumes bg 0.02 (out-of-season ~1 per 500, in-season ~130). At bg 0.2 an out-of-season class gets ~8 per 500,
so the fixed gate should open for absent causes. Adaptive aR (scale-free, no knowledge of bg): s is present iff
its count in the last 500 ticks >= R/100 x the count of the most frequent label in that window. a25 is the test.
Worlds: bg 0.02 (gate_probe + gate_probe_s345 logs) and bg 0.2 (gate_probe_bg02 logs), seeds 0-5, all held out
for a25 (threshold chosen from arithmetic, not data).
Prediction: a25 keeps mean pre-swap remaps <= 10 in both worlds and matches r3 at bg 0.02 (returns within 1 pt
under mq10_s2); fixed r3 exceeds 20 pre-swap remaps at bg 0.2.
Kill (adaptive rule): a25 pre-swap remaps > 20 in either world, or a25 mq10_s2 swap below plain-store mq10_s2 swap.
"""
import json, os, sys
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP


class AdaptStore(RP.RemapStore):
    def __init__(self, *a, K=None, **kw):
        self.frac = None
        if isinstance(K, str) and K.startswith("a"):
            self.frac = int(K[1:]) / 100; K = "r3"
        super().__init__(*a, K=K, **kw)
        self.win = deque()

    def observe(self, x, y):
        if self.frac is not None:
            self.win.append((self.t, y))
            while self.win and self.win[0][0] < self.t - 500:
                self.win.popleft()
        super().observe(x, y)


def absent_rel(st, s):
    cnt = {}
    for _, c in st.win:
        cnt[c] = cnt.get(c, 0) + 1
    return cnt.get(s, 0) < st.frac * max(cnt.values())


# override the presence test inside RemapStore.observe by wrapping recent-count check
_RO = RP.RemapStore.observe
def observe_wrap(self, x, y):
    if getattr(self, "frac", None) is None:
        return _RO(self, x, y)
    # replicate pair branch with relative presence
    if not self.mu:
        return
    s, j, d = self.nearest(x)
    self.seen[y] = self.t
    r = self.rad[s][j]
    if s == y:
        self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
    inr = r is not None and d <= r
    self.hist[s][j].append((1, y) if inr else 2)
    if not inr:
        return
    hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= 2]
    if len(hot) >= self.K and len(self.mu[s]) > len(hot):
        if absent_rel(self, s):
            self.blocked += 1; return
        for i in sorted(hot, reverse=True):
            self.log.append((self.t, s, y)); self.move(s, i, y)
RP.RemapStore.observe = observe_wrap


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: AdaptStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    return w["bg"], seed, {str(K): np.array(v) for K, v in out.items()}, {str(K): dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log)) for K, s in stores.items()}


def main():
    Ks = [None, "r3", "a25", "a10", "a40"]
    out_dir = "results/presence_bg"; os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    with Pool(8) as p:
        res = p.map(run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"])
        assert np.array_equal(out["None"][:, 2], R[:, 6]), (bg, seed)
        w = dict(LF.WORLD, bg=bg)
        allres[f"{bg}_{seed}"] = {K: dict(RP.score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = ["# Rate presence vs background rate (`presence_bg.py`)\n", __doc__]
    for bg in (0.02, 0.2):
        keys = [k for k in allres if k.startswith(f"{bg}_")]
        md += [f"\n## bg {bg}, seeds 0-5, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) |", "|---|---|---|---|---|---|---|"]
        for K in map(str, Ks):
            for g in ("ff", "store", "mq10_s2") if K == "None" else ("store", "mq10_s2"):
                m = lambda k: f"{np.mean([allres[s][K][g][k] for s in keys])*100:.1f} ± {np.std([allres[s][K][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.0f}) |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
