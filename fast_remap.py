"""Faster remap under the a25 guard (loop 01:05 hyp 1, pre-registered 02:15 HKT 31 Oct).

The swap cost of the default store (a25w250 + mq30_s2) is remap latency: a prototype of class s is "hot" toward y
after H = 2 in-radius conflicting hits labelled y among its last 10, and remapping fires when >= K = 3 prototypes
of s are hot toward the same y (and s is present by the a25 ratio rule, W = 250). With a25 blocking absent-cause
remaps, lower the evidence: hHkK variants with H in {1, 2}, K in {2, 3}.
Joined offline with gate_probe FF logs as in a25_gate.py: bg 0.02 seeds 3-5 held out (0-2 shown), bg 0.2 seeds 0-5.
Test setting: h1k2.  Prediction: held-out bg 0.02 swap >= 74 under mq30_s2 with returns >= 67.5, and pre-swap
remaps <= 5 in both worlds.  Kill: pre-swap remaps > 10 in either world, or returns < 67.
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import presence_bg as PB
import a25_gate as AG
from loose_gate import score
import loose_gate as LG

LG.QS, LG.SS = (0.3,), (0.2,)


class FStore(AG.WStore):
    def __init__(self, *a, K=None, **kw):
        base, hk = K.split("h"); H, Kp = hk.split("k")
        super().__init__(*a, K=base, **kw)
        self.H, self.K = int(H), int(Kp)

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        if not self.mu:
            return
        s, j, d = self.nearest(x)
        r = self.rad[s][j]
        if s == y:
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
        inr = r is not None and d <= r
        self.hist[s][j].append((1, y) if inr else 2)
        if not inr:
            return
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: FStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log)) for K, s in stores.items()}


def main():
    Ks = ["a25w250h2k3", "a25w250h1k3", "a25w250h2k2", "a25w250h1k2"]
    out_dir = "results/fast_remap"; os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    with Pool(min(12, os.cpu_count())) as p:
        res = p.map(run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"])
        w = dict(LF.WORLD, bg=bg)
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = ["# Faster remap under the a25 guard (`fast_remap.py`)\n", __doc__]
    for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Novel first 300 | Store answers | Remaps (pre-swap) |", "|---|---|---|---|---|---|---|---|"]
        for K in Ks:
            for g in allres[keys[0]][K]["routes"]:
                m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('novel')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.0f}) |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
