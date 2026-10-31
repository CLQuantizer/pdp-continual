"""Diagnostic for reread_store.py (loop 17:00 UTC+8 31 Oct, exploratory): why does the re-read store never remap?
Seed 0 and 1, s49, up to swap + 3000 ticks. For every post-swap arrival of digit 4 or 9 (labels swapped), record the
store's nearest prototype class s, whether the hit is a confirmation (s == y), in-radius conflict, or out-of-radius
conflict, and d / r. Pixel z5 vs z5_rr1."""
import json, sys, numpy as np, life as LF, reread_store as R
from data import load_mnist
from multiprocessing import Pool


def run(seed):
    xtr, ytr, _, _ = load_mnist("mnist")
    w = dict(LF.WORLD, **R.WEX); world = LF.World(xtr, ytr, 16 * w["day"], seed, w)
    mach = LF.Machine(dict(LF.DEFAULTS, **LF.POLICIES["none"]), seed); coder = R.Coder(mach.m)
    Ks = R.KS[1:3]; sts = {}
    for K in Ks:
        st = R.RRStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K); st.coder = coder; sts[K] = st
    sw = world.swap_at; rec = {K: [] for K in Ks}
    for t in range(sw + 3000):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in sts.items():
            st.answer(x); st.t = t; st.observe(x, y)
            if t >= sw and c in (4, 9):
                s, j = st._s, st._j; e = st.hist[s][j][-1] if s in st.hist and j < len(st.hist[s]) and st.hist[s][j] else None
                kind = "conf" if e == 0 else ("inr" if isinstance(e, tuple) else ("out" if e == 2 else "moved"))
                rec[K].append((t - sw, c, y, int(s), kind))
            st.add(x, y)
        mach.learn(x, y)
    return seed, rec


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(run, [0, 1])
    lines = ["# reread_diag.py: post-swap hits of 4/9 arrivals, first 3000 ticks after swap\n", __doc__, "",
             "| Seed | Store | Window (ticks after swap) | n | nearest = old label (true digit) | nearest = new label | confirm | in-radius conflict | out-of-radius conflict |", "|---|---|---|---|---|---|---|---|---|"]
    for seed, rec in res:
        for K, v in rec.items():
            for lo, hi in ((0, 300), (300, 1000), (1000, 3000)):
                vv = [r for r in v if lo <= r[0] < hi]
                n = len(vv)
                if not n:
                    continue
                f = lambda p: f"{sum(map(p, vv)) / n:.2f}"
                lines.append(f"| {seed} | {K.split('_', 1)[1]} | {lo}-{hi} | {n} | {f(lambda r: r[3] == r[1])} | {f(lambda r: r[3] == r[2])} | {f(lambda r: r[4] == 'conf')} | {f(lambda r: r[4] == 'inr')} | {f(lambda r: r[4] == 'out')} |")
    open("results/reread_store/DIAG.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
