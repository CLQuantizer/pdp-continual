"""H23: curiosity as compression progress (Schmidhuber 2009, arXiv 0812.4360) on an *active* H20 world.

Pre-registration: docs/h23_curiosity.md. Guideline: docs/curiosity.md.
The agent chooses which channel to look at; each glance shows the next B symbols of that channel. Channels = the 7 H20
processes (they resume where they left off) + N (uniform noise over the 41 symbols: the "noisy TV") + Z (dark: a
constant symbol). The compressor is fixed across conditions, so only the controller differs:
  compressor  mixture of KT orders 0..KMAX, contexts keyed by (channel, last k symbols of that channel); the channel is
              the agent's own action, so conditioning on it is legal (Schmidhuber's history includes actions).
  controllers rand  uniform channel choice
              err   reward = bits of the glance before learning it (1990 curiosity: prediction error)
              prog  reward = compression progress on the channel's stored history: C(p_old, h) - C(p_new, h), both
                    compressors scored on the same h = the last H symbols seen on that channel *before* this glance
                    (2009 eq. 5 with f(a, b) = a - b); the glance itself is excluded so memorising it buys nothing
              cnt   count bonus 1/sqrt(n_c) (novelty baseline)
              rand7 uniform over the 7 learnable channels only (oracle: knows N and Z are worthless; after the first pass)
  bandit      the same for every reward: optimistic first pass over all channels, then eps-greedy on an EMA of reward.
Score W (headline): mean bits/symbol of the *final* frozen compressor on a fresh continuation (EVAL symbols) of each of
the 7 learnable channels, drawn from a copy of the world at the end of life (the agent's own world is not advanced).
Usage: python3 curio.py [--seeds 0 1 2] [--glances 1024] [--conds rand err prog cnt] [--out results/h23/x.json]
"""
import argparse, copy, json, math, os
from multiprocessing import Pool
import numpy as np
from h20_world import World, ALPHA, PROCS, A

CH = PROCS + ["N", "Z"]
LEARN = PROCS
KMAX, FS, B, H, EVAL, EPS, BETA = 6, 1e-3, 64, 256, 2048, 0.1, 0.3

class Compressor:
    """Fixed-share mixture of KT order-0..KMAX models, one context table per channel. Weights are per channel too."""
    def __init__(self):
        self.tab = [dict() for _ in range(KMAX + 1)]
        self.w = {c: np.full(KMAX + 1, 1.0 / (KMAX + 1)) for c in CH}
    def probs(self, c, ctx, x):
        p = np.empty(KMAX + 1)
        for k in range(KMAX + 1):
            cnt = self.tab[k].get((c, ctx[len(ctx) - k:] if k else ()))
            p[k] = 1.0 / A if cnt is None else (cnt[0].get(x, 0) + 0.5) / (cnt[1] + A / 2)
        return p
    def bits(self, c, hist, seq, learn):
        """Code length of seq following hist on channel c; learn=True updates counts and weights as it goes."""
        ctx, w, tot = list(hist[-KMAX:]), self.w[c].copy(), 0.0
        for x in seq:
            t = tuple(ctx); p = self.probs(c, t, x); pm = float(w @ p); tot -= math.log2(pm)
            w = w * p / pm; w = (1 - FS) * w + FS / (KMAX + 1)
            if learn:
                for k in range(KMAX + 1):
                    key = (c, t[len(t) - k:] if k else ())
                    cnt = self.tab[k].get(key)
                    if cnt is None: cnt = self.tab[k][key] = [dict(), 0]
                    cnt[0][x] = cnt[0].get(x, 0) + 1; cnt[1] += 1
            ctx = (ctx + [x])[-KMAX:]
        if learn: self.w[c] = w
        return tot

    def score(self, c, hist, seq):
        return self.bits(c, hist, seq, False)

def glance(world, c, rng):
    if c == "N": return [int(v) for v in rng.integers(0, A, B)]
    if c == "Z": return [ALPHA.index(" ")] * B
    return [ALPHA.index(s) for s in world.take(c, B)]

def live(args):
    cond, seed, n_gl = args
    world, rng, comp = World(seed), np.random.default_rng(10_000 + seed), Compressor()
    hist = {c: [] for c in CH}
    v, n = {c: 0.0 for c in CH}, {c: 0 for c in CH}
    log = []
    for g in range(n_gl):
        if g < len(CH): c = CH[g]                                   # optimistic first pass: try every channel once
        elif cond == "rand7": c = LEARN[int(rng.integers(len(LEARN)))]   # oracle control: knows which channels are learnable
        elif cond == "rand" or rng.random() < EPS: c = CH[int(rng.integers(len(CH)))]
        elif cond == "cnt": c = min(CH, key=lambda k: (n[k], rng.random()))
        else: c = max(CH, key=lambda k: (v[k], rng.random()))
        x = glance(world, c, rng)
        old = hist[c][-H:]
        hc, hs = old[:KMAX], old[KMAX:]                             # score h with its first KMAX symbols as context
        before = comp.score(c, hc, hs) if cond == "prog" and hs else 0.0
        b = comp.bits(c, hist[c], x, learn=True)                    # prequential bits of this glance, then learn it
        after = comp.score(c, hc, hs) if cond == "prog" and hs else 0.0
        r = {"err": b / B, "prog": (before - after) / max(len(hs), 1), "cnt": 1 / math.sqrt(n[c] + 1)}.get(cond, 0.0)
        v[c] = r if n[c] == 0 else (1 - BETA) * v[c] + BETA * r
        n[c] += 1; hist[c] += x
        log.append((c, b / B, r))
    # headline: frozen final compressor on a fresh continuation of each learnable channel, from a copy of the world
    wc = copy.deepcopy(world)
    held = {c: comp.score(c, hist[c], [ALPHA.index(s) for s in wc.take(c, EVAL)]) / EVAL for c in LEARN}
    share = {c: n[c] / n_gl for c in CH}
    q = n_gl // 4
    share_q = [{c: sum(1 for e in log[i * q:(i + 1) * q] if e[0] == c) / q for c in CH} for i in range(4)]
    return dict(cond=cond, seed=seed, glances=n_gl, W=float(np.mean(list(held.values()))), held=held, share=share,
                share_by_quarter=share_q, life_bits=float(np.mean([e[1] for e in log])),
                life_bits_learnable=float(np.mean([e[1] for e in log if e[0] in LEARN] or [float("nan")])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--glances", nargs="+", type=int, default=[1024])
    ap.add_argument("--conds", nargs="+", default=["rand", "rand7", "err", "prog", "cnt"]); ap.add_argument("--out", default="results/h23/pilot.json")
    a = ap.parse_args()
    jobs = [(c, s, g) for g in a.glances for c in a.conds for s in a.seeds]
    with Pool(2) as pool: res = pool.map(live, jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(f'{r["glances"]:5d} {r["cond"]:4s} s{r["seed"]} W {r["W"]:.3f} | ' + " ".join(f'{c} {v:.2f}' for c, v in r["held"].items())
              + ' | N {:.2f} Z {:.2f}'.format(r["share"]["N"], r["share"]["Z"]))
