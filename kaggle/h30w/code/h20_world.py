"""H20 world: a label-free symbol stream from hidden processes at several timescales, plus its code-length floors.

Pre-registration: docs/h20_world.md. Deterministic: (world, seed) -> identical stream. Nothing here touches cls_sleep.py.
Processes (hidden parameters drawn once per seed, kept across absences so returns reward memory):
  G     stochastic grammar (PCFG with recursion via relative clauses), seeded random 2-5 letter words
  C1/C7 4-digit decimal counter, step 1 / step 7, records "dddd;" (carry = an algorithm, not a context)
  E30/E110  elementary cellular automaton rows (rule 30 / 110), width 31 + "\n", cell flip noise 0.005
  M7/M37    a seeded motif of length 7 / 37 repeated, 2% substitution noise
Seasons: the stream switches process every `season` symbols (slow 16384, fast 2048) to a uniformly drawn other process.
Floors (prequential, scored before each symbol is learned): uniform, adaptive order-0 KT, KT order-k, fixed-share
mixture of KT orders 0..8, and gzip/bz2/lzma bytes as a sanity reference.
Usage: python3 h20_world.py [--worlds slow fast] [--seeds 6 7 8] [--n 524288]
"""
import argparse, bz2, gzip, json, lzma, math, os, string
from multiprocessing import Pool
import numpy as np

ALPHA = string.ascii_lowercase + string.digits + ".#; \n"   # 41 symbols, declared up front
A = len(ALPHA)
PROCS = ["G", "C1", "C7", "E30", "E110", "M7", "M37"]
WORLDS = {"slow": 16384, "fast": 2048}
KMAX, FS_ALPHA = 8, 1e-3

def word(r): return "".join(r.choice(list(string.ascii_lowercase), r.integers(2, 6)))

class World:
    def __init__(self, seed):
        r = self.r = np.random.default_rng(seed)
        self.lex = {k: [word(r) for _ in range(n)] for k, n in [("D", 3), ("N", 6), ("A", 3), ("V", 5)]}
        self.motif = {L: "".join(r.choice(list(string.ascii_lowercase), L)) for L in (7, 37)}
        self.count = {1: int(r.integers(10000)), 7: int(r.integers(10000))}
        self.buf, self.eca = {p: [] for p in PROCS}, {}
    def np_(self, d):
        r, w = self.r, self.lex
        u = r.random(); out = [r.choice(w["D"])] + ([r.choice(w["A"])] if 0.6 <= u < 0.85 else []) + [r.choice(w["N"])]
        return out + (["that"] + self.vp(d + 1) if u >= 0.85 and d < 4 else [])
    def vp(self, d):
        return [self.r.choice(self.lex["V"])] + (self.np_(d) if self.r.random() < 0.6 else [])
    def chunk(self, p):
        r = self.r
        if p == "G": return " ".join(self.np_(0) + self.vp(0)) + ". "
        if p[0] == "C":
            s = int(p[1:]); self.count[s] = (self.count[s] + s) % 10000; return f"{self.count[s]:04d};"
        if p[0] == "M":
            m = list(self.motif[int(p[1:])])
            for i in np.nonzero(r.random(len(m)) < 0.02)[0]: m[i] = r.choice(list(string.ascii_lowercase))
            return "".join(m)
        rule, row = int(p[1:]), self.eca.get(p)            # ECA: a fresh random row per visit, then the rule runs
        if row is None: row = r.integers(0, 2, 31)
        out = []
        for _ in range(64):
            out.append("".join(".#"[c] for c in row) + "\n")
            idx = 4 * np.roll(row, 1) + 2 * row + np.roll(row, -1)
            row = (rule >> idx) & 1
            row ^= (r.random(31) < 0.005).astype(row.dtype)
        self.eca[p] = row
        return "".join(out)
    def take(self, p, n):
        while len(self.buf[p]) < n: self.buf[p].extend(self.chunk(p))
        out, self.buf[p] = self.buf[p][:n], self.buf[p][n:]
        return out

def stream(world, seed, n):
    w, season = World(seed), WORLDS[world]
    syms, procs, p = [], [], PROCS[int(w.r.integers(len(PROCS)))]
    while len(syms) < n:
        if p[0] == "E": w.buf[p], w.eca[p] = [], None       # ECA restarts on each visit; others resume
        syms += w.take(p, season); procs += [p] * season
        p = PROCS[(PROCS.index(p) + 1 + int(w.r.integers(len(PROCS) - 1))) % len(PROCS)]
    return "".join(syms[:n]), procs[:n]

def kt_floors(s):
    """Prequential bits per symbol for KT order-k (k=0..KMAX) and their fixed-share mixture."""
    x = [ALPHA.index(c) for c in s]
    tabs = [dict() for _ in range(KMAX + 1)]
    bits = np.zeros((KMAX + 1, len(x))); mix = np.zeros(len(x))
    w = np.full(KMAX + 1, 1.0 / (KMAX + 1))
    for t, c in enumerate(x):
        p = np.empty(KMAX + 1)
        for k in range(KMAX + 1):
            ctx = tuple(x[t - k:t]) if t >= k else None
            cnt = tabs[k].get(ctx)
            if cnt is None: cnt = tabs[k][ctx] = [np.zeros(A), 0]
            p[k] = (cnt[0][c] + 0.5) / (cnt[1] + A / 2)
            cnt[0][c] += 1; cnt[1] += 1
        bits[:, t] = -np.log2(p)
        pm = float(w @ p); mix[t] = -math.log2(pm)
        w = w * p / pm; w = (1 - FS_ALPHA) * w + FS_ALPHA / (KMAX + 1)
    ents = [sum(int((v[0] > 0).sum()) for v in tab.values()) for tab in tabs]
    return bits, mix, ents

def run(args):
    world, seed, n = args
    s, procs = stream(world, seed, n)
    bits, mix, ents = kt_floors(s)
    procs = np.array(procs); season = WORLDS[world]
    sw = np.zeros(n, bool)
    for i in range(0, n, season): sw[i:i + 1024] = True     # first 1024 symbols after every switch
    b = s.encode()
    out = dict(world=world, seed=seed, n=n, uniform=math.log2(A),
               kt={k: float(bits[k].mean()) for k in range(KMAX + 1)}, mix=float(mix.mean()),
               mix_switch=float(mix[sw].mean()), mix_by_proc={p: float(mix[procs == p].mean()) for p in PROCS},
               table_entries=ents,
               gzip=8 * len(gzip.compress(b, 9)) / n, bz2=8 * len(bz2.compress(b, 9)) / n,
               lzma=8 * len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)) / n)
    # two-part reference: the mixture's count tables priced like store entries at their final size (each nonzero
    # (context, symbol) count = (k + 1) * log2 A bits of key + 16 bits of value), amortised over n symbols
    out["mix_twopart"] = out["mix"] + sum(e * (16 + (k + 1) * math.log2(A)) for k, e in enumerate(ents)) / n
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS)); ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
    ap.add_argument("--n", type=int, default=1 << 19); ap.add_argument("--out", default="results/h20_floors/results.json")
    a = ap.parse_args()
    jobs = [(w, s, a.n) for w in a.worlds for s in a.seeds]
    with Pool(len(jobs)) as pool: res = pool.map(run, jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        best = min(r["kt"], key=r["kt"].get)
        print(f'{r["world"]:4s} s{r["seed"]} uniform {r["uniform"]:.3f} kt0 {r["kt"][0]:.3f} best kt{best} {r["kt"][best]:.3f} '
              f'mix {r["mix"]:.3f} (switch {r["mix_switch"]:.3f}, 2-part {r["mix_twopart"]:.2f}) gzip {r["gzip"]:.3f} '
              f'bz2 {r["bz2"]:.3f} lzma {r["lzma"]:.3f} | ' + " ".join(f'{p} {v:.3f}' for p, v in r["mix_by_proc"].items()))
