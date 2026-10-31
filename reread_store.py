"""Re-read index: store keeps pixel prototypes, compares them through the CURRENT cortex (loop 17:00 UTC+8 31 Oct,
pre-registered before running; 16:00 hypotheses 1 and 2).

code_probe.py showed FF hidden codes separate 4 from 9 ~3x better than pixels but go stale after a swap if the store
keeps the codes; re-encoding the stored pixels through the post-swap layer 1 keeps the separation (impostor 0.045 vs
pixels 0.077). Teyler & DiScenna index: the store holds pointers (pixels), similarity is read through today's cortex.
This is an in-loop life run: a never-sleeping life.py machine (policy none, wake LR x1, the gate_probe s49 stream,
seeds 0-5) learns as in gate_probe; at every arrival, BEFORE the cortex learns it, each store variant answers and
observes with distances on code(x) vs code(proto), code = L2-normalised ReLU of FF layer 1 (rr1) or layers 1+2
concatenated (rr12) under a neutral label (0.1 on the label pixels). Prototype means stay in pixel space (add()
unchanged); radii (EMA of confirmation distance) and the c1.5 rival ratio live in code units. FF answers/margins come
from the logged gate_probe_s49 rows (asserted equal to the in-run cortex on every 50th arrival).
Stores: base, z5 (pixel controls, must reproduce rival_s49), z5_rr1, z5_rr12.
  H1 (architecture): z5_rr1 first post-swap remap <= 65 ticks AND swap (mq30_s2) >= z5 + 0.8, 0 pre-swap false
     remaps. Kill: latency > 70 or store-only lifetime < pixel z5 - 1.0.
  H2 (neuron): rr1 latency lower than rr12. Kill if rr12 is better by > 5 ticks.
Usage: OMP_NUM_THREADS=1 python3 reread_store.py
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist
from loose_gate import score

EPS = 1e-6  # id column; ids restart at 1 on every call so (id*EPS)^2 <= 4e-8 (loop 17:35 fix: a global
# id counter grew to ~4.5M, adding a time-growing offset to every distance and pushing all conflicts out of radius)
SEEDS = [0, 1, 2, 3, 4, 5]
WEX = dict(bg=0.02, swap=(4, 9))
KS = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_rr1", "a25w250h2k2_c1.5_uinf_z5_rr12"]


class Coder:
    def __init__(self, ff):
        self.ff, self.tick, self.cache = ff, None, {}

    def __call__(self, X, layers):
        Xn = np.array(X, np.float32, copy=True); Xn[:, :10] = 0.1
        hs = [h for _, h in self.ff._layers(Xn)]
        nrm = lambda h: h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-6)
        return nrm(hs[0]) if layers == 1 else np.concatenate([nrm(hs[0]), nrm(hs[1])], 1)


class RRStore(RS.ZStore):
    coder = None

    def __init__(self, *a, K=None, **kw):
        self.L = None
        if K.endswith("_rr1") or K.endswith("_rr12"):
            self.L = 1 if K.endswith("_rr1") else 2; K = K.rsplit("_", 1)[0]
        super().__init__(*a, K=K, **kw)
        self.pix, self.nid = {}, 0

    def stacked(self, x):
        """mu -> code rows + an id column (scaled by EPS); x -> code + id 0. Distances are code distances (+<1e-6)."""
        cls = sorted(self.mu)
        allp = np.concatenate([self.mu[c] for c in cls])
        codes = self.coder(np.vstack([allp, x]), self.L)
        cx, cp = codes[-1:], codes[:-1]
        ids, smu, i = {}, {}, 0
        self.nid = 0
        for c in cls:
            n = len(self.mu[c]); col = np.empty((n, 1), np.float32)
            for r in range(n):
                self.nid += 1; self.pix[self.nid] = self.mu[c][r]; col[r, 0] = self.nid * EPS
            smu[c] = np.hstack([cp[i:i + n], col]); i += n
        return smu, np.hstack([cx, np.zeros((1, 1), np.float32)])

    def observe(self, x, y):
        if self.L is None or not self.mu:
            return super().observe(x, y)
        self.pix = {}
        smu, sx = self.stacked(x)
        self.mu = smu
        try:
            super().observe(sx, y)
        finally:
            self.mu = {c: np.stack([self.pix[int(round(v / EPS))] for v in m[:, -1]]) for c, m in self.mu.items()}
            self.pix = {}

    def answer(self, x):
        if self.L is None or not self.mu:
            return RP.answer(self, x)
        cls = sorted(self.mu)
        mu = np.concatenate([self.mu[c] for c in cls]); ys = np.concatenate([np.full(len(self.mu[c]), c) for c in cls])
        codes = self.coder(np.vstack([mu, x]), self.L)
        d = ((codes[:-1] - codes[-1:]) ** 2).sum(1); j = d.argmin(); s = int(ys[j]); other = d[ys != s]
        return s, (float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0)


def run(args):
    seed, logrows = args
    w = dict(LF.WORLD, **WEX)
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    mach = LF.Machine(dict(LF.DEFAULTS, **LF.POLICIES["none"]), seed)
    coder = Coder(mach.m)
    stores = {}
    for K in KS:
        st = RRStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K); st.coder = coder; stores[K] = st
    out = {K: [] for K in KS}
    t0, i, bad = time.time(), 0, 0
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        if i % 50 == 0:
            bad += int(int(mach.m.scores(x)[0].argmax()) != int(logrows[i][5]))
        for K, st in stores.items():
            out[K].append((t, y) + st.answer(x)); st.t = t; st.observe(x, y); st.add(x, y)
        mach.learn(x, y)
        i += 1
        if i % 5000 == 0:
            print(f"seed={seed} arrivals={i} {time.time() - t0:.0f}s mismatch={bad}", flush=True)
    sw = world.swap_at
    rm = {K: dict(n=s.remaps, pre_swap=sum(tt < sw for tt, _, _ in s.log), rivals=s.rivals,
                  post_lat={str(c): next((tt - sw for tt, s_, _ in s.log if tt >= sw and s_ == c), None) for c in w["swap"]})
          for K, s in stores.items()}
    print(f"seed={seed} done {time.time() - t0:.0f}s mismatch={bad}/{(i + 49) // 50}", flush=True)
    return seed, {K: np.array(v) for K, v in out.items()}, rm, bad


def main():
    J = json.load(open("results/gate_probe_s49/results.json"))
    logs = {r["seed"]: r for r in J["raw"] if r["seed"] in SEEDS}
    load_mnist("mnist")
    with Pool(len(SEEDS)) as p:
        res = p.map(run, [(s, logs[s]["rows"]) for s in SEEDS], chunksize=1)
    w = dict(LF.WORLD, **WEX); allres = {}
    for seed, out, rm, bad in res:
        r = logs[seed]; R = np.array(r["rows"])
        assert np.array_equal(out[KS[0]][:, 1], R[:, 2])
        allres[seed] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
        allres[seed]["_mismatch"] = bad
    md = ["# Re-read index: pixel store compared through the current cortex (`reread_store.py`)\n", __doc__,
          "\n## Pre-swap false remaps per seed\n", "| Seed | " + " | ".join(KS) + " | FF mismatches |", "|---" * (len(KS) + 2) + "|"]
    for s in SEEDS:
        md.append(f"| {s} | " + " | ".join(str(allres[s][K]["pre_swap"]) for K in KS) + f" | {allres[s]['_mismatch']} |")
    md += ["\n## s49, seeds 0-5, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) | Rival vetoes | First post-swap remap, mean ticks |", "|---|---|---|---|---|---|---|---|---|"]
    for K in KS:
        lat = [v for s in SEEDS for v in allres[s][K]["post_lat"].values() if v is not None]
        for g in ("store", "mq30_s2"):
            m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in SEEDS])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in SEEDS])*100:.1f}"
            md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in SEEDS]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in SEEDS]):.1f}) | {np.mean([allres[s][K]['rivals'] for s in SEEDS]):.0f} | {np.mean(lat) if lat else float('nan'):.0f} |")
    out_dir = "results/reread_store"; os.makedirs(out_dir, exist_ok=True)
    json.dump({str(k): v for k, v in allres.items()}, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
