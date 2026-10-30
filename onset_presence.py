"""Onset-aware presence for the a25 guard (loop 04:10 hyp 1, pre-registered 04:30 UTC+8 31 Oct).

lat_diag showed that in the latency window before the first post-swap remap, 15-19% of swapped-digit arrivals already
have >= K hot prototypes and are vetoed by a25: the swap always opens the pair's season, so the 250-tick window
still holds the previous season (label s count 0-4 vs top label ~90) and calls the old class absent. Fix: a class is
present if it passes a25 on the W = 250 window OR on a short window of the last S ticks (count of s >= 25% of the
top label count in those S ticks). Store a25w250h2k2sS; s0 = baseline. FF rows joined from gate_probe logs.
Test: s50.  Prediction: held-out bg 0.02 swap >= 73.5 under mq30_s2 (baseline 72.2), blocked post-swap vetoes ~0,
mean first post-swap remap latency drops >= 10 ticks.  Kill: pre-swap remaps > 5 in either world, or returns < 67.5,
or swap gain < 1.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import presence_bg as PB
import fast_remap as FR
from loose_gate import score

_AR = PB.absent_rel


def absent_rel(st, s):
    if not _AR(st, s):
        return False
    S = getattr(st, "S", 0)
    if not S:
        return True
    cnt = {}
    for t, c in st.win:
        if t >= st.t - S:
            cnt[c] = cnt.get(c, 0) + 1
    return not cnt or cnt.get(s, 0) < st.frac * max(cnt.values())


PB.absent_rel = absent_rel


class SStore(FR.FStore):
    def __init__(self, *a, K=None, **kw):
        base, S = K.split("s"); super().__init__(*a, K=base, **kw); self.S = int(S)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: SStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), blocked=s.blocked, post_lat={str(c): next((t - sw for t, s_, _ in s.log if t >= sw and s_ == c), None) for c in w["swap"]}) for K, s in stores.items()}


def main():
    Ks = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a25w250h2k2s0", "a25w250h2k2s50", "a25w250h2k2s100"]
    out_dir = "results/onset_presence"; os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    with Pool(min(12, os.cpu_count())) as p:
        res = p.map(run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"]); w = dict(LF.WORLD, bg=bg)
        assert np.array_equal(out[Ks[0]][:, 1], R[:, 2])
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = ["# Onset-aware presence (`onset_presence.py`)\n", __doc__]
    for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) | First post-swap remap, mean ticks |", "|---|---|---|---|---|---|---|---|"]
        for K in Ks:
            lat = [v for s in keys for v in allres[s][K]["post_lat"].values() if v is not None]
            for g in ("store", "mq30_s2"):
                m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.1f}) | {np.mean(lat):.0f} |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
