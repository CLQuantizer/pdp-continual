"""a25 store + loosened margin gate in both worlds (loop 00:10 hyp 1, pre-registered 01:10 HKT 31 Oct).

Stores: none, a25 (W = 500) and a25w250 (same ratio rule, window halved; loop 00:10 hyp 2, window half only).
Joined offline with gate_probe FF logs: bg 0.02 (gate_probe + gate_probe_s345, seeds 0-5) and bg 0.2
(gate_probe_bg02, seeds 0-5). Routes: ff, store, mq{10,30,50}_s2 (s2 only; s1 broke lifetime in loose_gate).
Settings mq30_s2 / mq50_s2 were selected in loose_gate on seeds 0-2 with r3; here seeds 3-5 are reported as
held-out for bg 0.02, and all six seeds of bg 0.2 are held out (never used for selection).
Prediction: a25 + mq30_s2 or mq50_s2 reaches held-out bg 0.02 returns >= 68 at lifetime >= 96.0, and bg 0.2
lifetime >= 94.5 (a25 mq10_s2 is 94.4, FF 94.1).
Kill: bg 0.2 lifetime falls below FF (94.1) at any setting that gains >= 2 returns over a25 mq10_s2 at bg 0.02.
Window: a25w250 keeps pre-swap remaps <= 10 in both worlds (rate theory: ratio rule is window-invariant).
"""
import json, os
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import presence_bg as PB
from loose_gate import score
import loose_gate as LG

LG.QS, LG.SS = (0.1, 0.3, 0.5), (0.2,)


class WStore(PB.AdaptStore):
    def __init__(self, *a, K=None, **kw):
        W = 500
        if isinstance(K, str) and "w" in K:
            K, W = K.split("w"); W = int(W)
        super().__init__(*a, K=K, **kw); self.W = W

    def observe(self, x, y):
        if self.frac is not None:
            self.win.append((self.t, y))
            while self.win and self.win[0][0] < self.t - self.W:
                self.win.popleft()
        PB.RP.RemapStore.observe(self, x, y)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: WStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
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
    Ks = [None, "a25", "a25w250"]
    out_dir = "results/a25_gate"; os.makedirs(out_dir, exist_ok=True)
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
        assert np.array_equal(out["None"][:, 2], R[:, 6]), (bg, seed)
        w = dict(LF.WORLD, bg=bg)
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = ["# a25 store + loosened gate in both worlds (`a25_gate.py`)\n", __doc__]
    for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "selection seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Novel first 300 | Store answers | Remaps (pre-swap) |", "|---|---|---|---|---|---|---|---|"]
        for K in map(str, Ks):
            for g in allres[keys[0]][K]["routes"]:
                if K != "None" and g == "ff":
                    continue
                m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('novel')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.0f}) |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
