"""Distance-weighted remap evidence (loop 02:15 hyp, pre-registered 02:25 HKT 31 Oct).

fast_remap.py showed that remap latency is set by H, the conflicting hits a prototype needs: H = 1 buys ~4 swap
points but makes 16-67 false pre-swap remaps per run. Novelty-gated one-shot storage (Lisman & Grace 2005) says a
single event should suffice when the mismatch is strong. Here a conflicting in-radius hit counts 2 if it is deep
inside the prototype's radius (d <= c * r) and 1 otherwise; a prototype is hot when its weighted evidence toward y
over its last 10 hits is >= 2. Variants eCkK: c in {0.5, 0.7} (c = 0 is h2, c = 1 is h1), K in {2, 3}.
Test: e50k2.  Prediction: held-out bg 0.02 swap >= 74 under mq30_s2, returns >= 67.5, pre-swap remaps <= 5 in both
worlds.  Kill: pre-swap remaps > 10 in either world, or swap < 73.
"""
import os
import numpy as np
import presence_bg as PB
import fast_remap as FR


class EStore(FR.FStore):
    def __init__(self, *a, K=None, **kw):
        base, ck = K.split("e"); c, Kp = ck.split("k")
        super().__init__(*a, K=f"{base}h2k{Kp}", **kw)
        self.c = int(c) / 100

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
        self.hist[s][j].append((1, y, 2 if d <= self.c * r else 1) if inr else 2)
        if not inr:
            return
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e[2] for e in h if isinstance(e, tuple) and e[1] == y) >= 2]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


FR.FStore = EStore
if __name__ == "__main__":
    import sys
    FR.__doc__ = __doc__
    Ks = ["a25w250e0k2", "a25w250e50k2", "a25w250e70k2", "a25w250e50k3"]
    src = open(FR.__file__).read()
    FR.main.__globals__["FStore"] = EStore
    # reuse fast_remap.main with new store list and output dir
    import types
    code = FR.main.__code__
    FR.os.makedirs("results/close_remap", exist_ok=True)
    def main():
        import json
        from multiprocessing import Pool
        import life as LF
        from data import load_mnist
        from loose_gate import score
        logs = {}
        for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
            for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
                logs[(bg, r["seed"])] = r
        load_mnist("mnist")
        with Pool(min(12, os.cpu_count())) as p:
            res = p.map(FR.run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
        allres = {}
        for bg, seed, out, rm in res:
            r = logs[(bg, seed)]; R = np.array(r["rows"]); w = dict(LF.WORLD, bg=bg)
            allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
        md = ["# Distance-weighted remap evidence (`close_remap.py`)\n", __doc__]
        for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
            keys = [f"{bg}_{s}" for s in seeds]
            md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) |", "|---|---|---|---|---|---|---|"]
            for K in Ks:
                for g in ("store", "mq30_s2"):
                    m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                    md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.0f}) |")
        json.dump(allres, open("results/close_remap/results.json", "w"))
        open("results/close_remap/RESULTS.md", "w").write("\n".join(md) + "\n")
        print("\n".join(md))
    main()
