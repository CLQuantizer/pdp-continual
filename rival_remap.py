"""Rival-aware remap evidence (loop 05:00 hyp 1, pre-registered 05:15 UTC+8 31 Oct).

onset_s678 showed the default store a25w250h2k2 is not safe on fresh seed 6: 34 pre-swap false remaps, all ping-pong
between co-present confusable classes (4<->9 26, 7<->9 8) in seasons where both are in season, so the a25 presence
guard (which only blocks remaps of absent classes) cannot help. Mechanism: a borderline 9 that lands in a 4
prototype's radius counts as evidence that the 4 prototype "now means 9", although the store already has a 9
prototype nearby. A real swap differs: a 2 relabelled 3 lands near 2 prototypes and far from every real 3.
Rule (pattern separation / mnemonic discrimination, Yassa & Stark 2011): an in-radius conflicting hit labelled y on a
prototype of s counts as evidence only if no prototype of y is a close rival: rad: the arrival is outside the radius
of the nearest y prototype; cC: squared distance to the nearest y prototype > C x squared distance to the s prototype.
Test: rad.  Prediction: seed-6 pre-swap remaps <= 5 (from 34); mean swap (mq30_s2) over bg 0.02 seeds 3-8 within 1.0 of
baseline; returns within 1.0; bg 0.2 pre-swap remaps <= baseline + 1.
Kill: seed-6 pre-swap remaps > 10, or swap loss > 1.5, or returns loss > 1.0.
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import presence_bg as PB
import fast_remap as FR
from loose_gate import score


class CStore(FR.FStore):
    def __init__(self, *a, K=None, **kw):
        base, rule = K.split("_"); super().__init__(*a, K=base, **kw); self.rule = rule

    def rival(self, x, y, d):
        if self.rule == "base" or y not in self.mu or not len(self.mu[y]):
            return False
        dy = ((self.mu[y] - x[0]) ** 2).sum(1); k = int(dy.argmin())
        if self.rule == "rad":
            ry = self.rad[y][k]
            return ry is not None and dy[k] <= ry
        return dy[k] <= float(self.rule[1:]) * d

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
        if inr and self.rival(x, y, d):
            self.rivals += 1; inr = False
        self.hist[s][j].append((1, y) if inr else 2)
        if not inr:
            return
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)
    rivals = 0


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: globals().get("STORE", CStore)(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), rivals=s.rivals, post_lat={str(c): next((t - sw for t, s_, _ in s.log if t >= sw and s_ == c), None) for c in w["swap"]}) for K, s in stores.items()}


def main():
    Ks = globals().get("KS_OVERRIDE", ["a25w250h2k2_base", "a25w250h2k2_rad", "a25w250h2k2_c1.5"])
    out_dir = globals().get("OUT_OVERRIDE", "results/rival_remap"); os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg, n in (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:n]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    with Pool(min(8, os.cpu_count())) as p:
        res = p.map(run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"]); w = dict(LF.WORLD, bg=bg)
        assert np.array_equal(out[Ks[0]][:, 1], R[:, 2])
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = [globals().get("TITLE", "# Rival-aware remap evidence (`rival_remap.py`)\n"), globals().get("DOC", __doc__)]
    md += ["\n## Pre-swap false remaps per run\n", "| Run | " + " | ".join(Ks) + " |", "|---" * (len(Ks) + 1) + "|"]
    for k in sorted(allres):
        md.append(f"| {k} | " + " | ".join(f"{allres[k][K]['pre_swap']}" for K in Ks) + " |")
    for bg, seeds, name in ((0.02, [3, 4, 5, 6, 7, 8], "held-out seeds 3-8"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) | Rival vetoes | First post-swap remap, mean ticks |", "|---|---|---|---|---|---|---|---|---|"]
        for K in Ks:
            lat = [v for s in keys for v in allres[s][K]["post_lat"].values() if v is not None]
            for g in ("store", "mq30_s2"):
                m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.1f}) | {np.mean([allres[s][K]['rivals'] for s in keys]):.0f} | {np.mean(lat) if lat else float('nan'):.0f} |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
