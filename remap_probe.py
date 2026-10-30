"""Label-conflict remap of store prototypes (loop 21:10 hyp 1).

FF learning is independent of the store when the machine never sleeps (policy none, no mix), so we replay the
same world stream through the store alone and join with the FF answers/margins logged by gate_probe.py
(results/gate_probe seeds 0-2, results/gate_probe_s345 held-out seeds 3-5, bg 0.02). Sanity: the plain store
must reproduce the logged store answers exactly.

Remap rule (pre-registered): on every labelled arrival, find the global nearest prototype j (label s).
A hit on j is a *conflict* if s != y and d_j <= j's radius (EMA, rate 0.1, of its distance on agreeing hits;
initialised to the first hit distance). After >= K conflicts among j's last 10 hits (K = 3), j is moved to
class y (relabelled) and its history cleared. A class keeps at least one prototype; a class over 2k=18
prototypes drops its least-used one. Routing = mq10_s2 margin gate, no veto.
Prediction: swap first 300 >= 76, returns >= 64, lifetime >= 96.0 on held-out seeds 3-5.
Kill: returns < 62 or swap < 74.  Sweep K in {2, 3, 5} reported but K = 3 is the pre-registered test.
Follow-up (added after the K sweep, chosen on seeds 0-2, confirmed on 3-5): fK = "flip" rule, remap only when
>= K in-radius conflicts AND <= 1 agreeing hit among the last 10 (a prototype whose label changed, not a
prototype that sits on a permanently confusable border). Remap counts split into before the swap (all false)
and swap-pair remaps after it.
Follow-up 2 (22:15, pre-registered before running): pM = coordinated class-level rule (latent-cause style,
Gershman, Blei & Niv 2010): a prototype is "hot" toward y when >= 2 of its last 10 hits are in-radius conflicts
labelled y; when >= M prototypes of class s are hot toward the same y, all of them move to y at once. M = 3 is
the test. Prediction: pre-swap remaps <= 10 and held-out mq10_s2 swap >= 76 with returns >= 64. Kill as above.
Follow-up 3 (23:00, pre-registered): gM = pM plus a presence gate (Gershman et al. 2017: only modify memories whose
latent cause is active): class s's prototypes may move only if label s arrived within the last W = 200 ticks.
g3 is the test. Prediction on held-out seeds: pre-swap remaps <= 10, swap >= 76, returns >= 64, lifetime >= 96.0.
Kill: pre-swap remaps > 20 or returns < 63.
Follow-up 4 (23:00, pre-registered after g3 failed and a diagnostic showed 90 of 108 held-out pre-swap p3 remaps
move an out-of-season class s): rM = pM with a rate-based presence gate, s may move only if label s arrived
>= 5 times in the last 500 ticks (in season ~150, background only ~1). r3 is the test; same prediction and kill.
"""
import argparse, json, os, time
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
import sleep as SL
from data import load_mnist
from gate_probe import rolling_quantile


class RemapStore(SL.ProtoStore):
    def __init__(self, *a, K=None, **kw):
        super().__init__(*a, **kw)
        self.flip = isinstance(K, str) and K.startswith("f")
        self.gate = isinstance(K, str) and K[0] in "gr"
        self.rate = isinstance(K, str) and K.startswith("r")
        self.recent = {}
        self.pair = isinstance(K, str) and K[0] in "pgr"
        self.K = int(K[1:]) if (self.flip or self.pair) else K
        self.hist, self.rad, self.remaps, self.log = {}, {}, 0, []
        self.seen, self.W, self.blocked = {}, getattr(RemapStore, "W_default", 200), 0

    def add(self, x, y, surprise=False):
        before = len(self.mu.get(y, []))
        super().add(x, y, surprise)
        if len(self.mu[y]) > before:
            self.hist.setdefault(y, []).append(deque(maxlen=10))
            self.rad.setdefault(y, []).append(None)

    def nearest(self, x):
        best = (None, None, np.inf)
        for c in self.mu:
            d = ((self.mu[c] - x[0]) ** 2).sum(1)
            j = int(d.argmin())
            if d[j] < best[2]:
                best = (c, j, float(d[j]))
        return best

    def observe(self, x, y):
        if self.K is None or not self.mu:
            return
        s, j, d = self.nearest(x)
        self.seen[y] = self.t
        self.recent.setdefault(y, deque()).append(self.t)
        r = self.rad[s][j]
        if s == y:
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r)
            self.hist[s][j].append(0)
            return
        inr = r is not None and d <= r
        self.hist[s][j].append((1, y) if inr else 2)
        if self.pair:
            if not inr:
                return
            hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= 2]
            if len(hot) >= self.K and len(self.mu[s]) > len(hot):
                q = self.recent.get(s, deque())
                while q and q[0] < self.t - 500:
                    q.popleft()
                absent = len(q) < 5 if self.rate else self.t - self.seen.get(s, -10**9) > self.W
                if self.gate and absent:
                    self.blocked += 1
                    return
                for i in sorted(hot, reverse=True):
                    self.log.append((self.t, s, y)); self.move(s, i, y)
            return
        h = [1 if isinstance(e, tuple) else e for e in self.hist[s][j]]
        if h.count(1) >= self.K and (not self.flip or h.count(0) <= 1) and len(self.mu[s]) > 1:
            self.log.append((self.t, s, y))
            self.move(s, j, y)

    def move(self, s, j, y):
        mu, var, n = self.mu[s][j].copy(), self.var[s][j].copy(), self.n[s][j]
        for D in (self.mu, self.var, self.n):
            D[s] = np.delete(D[s], j, axis=0)
        rad = self.rad[s].pop(j); self.hist[s].pop(j)
        if y not in self.mu:
            self.mu[y], self.var[y], self.n[y], self.hist[y], self.rad[y] = mu[None], var[None], np.array([n]), [], []
        else:
            self.mu[y], self.var[y], self.n[y] = np.vstack([self.mu[y], mu]), np.vstack([self.var[y], var]), np.append(self.n[y], n)
        self.hist[y].append(deque(maxlen=10)); self.rad[y].append(rad)
        if len(self.mu[y]) > 2 * self.k:
            k = int(self.n[y].argmin())
            for D in (self.mu, self.var, self.n):
                D[y] = np.delete(D[y], k, axis=0)
            self.rad[y].pop(k); self.hist[y].pop(k)
        self.remaps += 1


def answer(st, x):
    if not st.mu:
        return -1, 0.0
    mu = np.concatenate([st.mu[c] for c in sorted(st.mu)])
    ys = np.concatenate([np.full(len(st.mu[c]), c) for c in sorted(st.mu)])
    d = ((mu - x) ** 2).sum(1); j = d.argmin(); s = int(ys[j]); other = d[ys != s]
    return s, (float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: RemapStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    t0 = time.time()
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + answer(st, x))
            st.t = t; st.observe(x, y); st.add(x, y)
    print(f"seed={seed} {time.time() - t0:.0f}s remaps={ {K: s.remaps for K, s in stores.items()} }", flush=True)
    sw = world.swap_at
    return seed, {str(K): np.array(v) for K, v in out.items()}, {str(K): dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log),
        swap_pair=sum(t >= sw and {a, b} == set(w["swap"]) for t, a, b in s.log)) for K, s in stores.items()}


def score(R, S, SM, w, swap_at, novel_at):
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F); prot = CF >= 1000; low10 = FFM < rolling_quantile(FFM, 0.1)
    res = {}
    for g, use in dict(ff=np.zeros_like(dis), store=dis, mq10_s2=dis & prot & low10 & (SM >= 0.2)).items():
        P = np.where(use, S, F); ok = P == Y
        first = lambda dg, at: float(ok[np.flatnonzero(np.isin(C, dg) & (T >= at))[:300]].mean())
        res[g] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), off=float(ok[INS == 0].mean()),
                      swap=first(list(w["swap"]), swap_at), novel=first(list(w["novel"]), novel_at), fired=float(use.mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", nargs="+", default=["none", "3", "f3", "p2", "p3", "p4"])
    ap.add_argument("--out", default="results/remap_probe")
    ap.add_argument("--W", type=int, default=200)
    a = ap.parse_args()
    Ks = [None if k == "none" else (k if k[0] in "fpgr" else int(k)) for k in a.Ks]
    os.makedirs(a.out, exist_ok=True); RemapStore.W_default = a.W
    w = dict(LF.WORLD, bg=0.02)
    logs = {}
    for path in ("results/gate_probe", "results/gate_probe_s345"):
        J = json.load(open(os.path.join(path, "results.json")))
        for r in J["raw"][:3]:  # first world is bg002
            logs[r["seed"]] = r
    load_mnist("mnist")
    with Pool(len(logs)) as p:
        res = p.map(run, [(s, 16, dict(bg=0.02), Ks) for s in sorted(logs)], chunksize=1)
    allres, md = {}, ["# Label-conflict remap of store prototypes (`remap_probe.py`)\n", __doc__, ""]
    for seed, out, remaps in res:
        r = logs[seed]; R = np.array(r["rows"])
        base = out["None"]
        assert np.array_equal(base[:, 0], R[:, 0]) and np.array_equal(base[:, 1], R[:, 2]), "stream mismatch"
        assert np.array_equal(base[:, 2], R[:, 6]), f"plain store does not reproduce logged answers (seed {seed})"
        allres[seed] = {K: dict(score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), remaps=remaps[K]["n"], pre_swap=remaps[K]["pre_swap"], swap_pair=remaps[K]["swap_pair"]) for K, v in out.items()}
    for name, seeds in (("seeds 0-2 (seen)", [0, 1, 2]), ("held-out seeds 3-5", [3, 4, 5])):
        md += [f"\n## {name}, bg 0.02, mean ± std %\n", "| Store | Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers | Remaps (pre-swap / swap pair) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for K in out:
            for g in ("ff", "store", "mq10_s2") if K == "None" else ("store", "mq10_s2"):
                m = lambda key: f"{np.mean([allres[s][K][g][key] for s in seeds]) * 100:.1f} ± {np.std([allres[s][K][g][key] for s in seeds]) * 100:.1f}"
                md.append(f"| K={K} | {g} | {m('lifetime')} | {m('returns')} | {m('off')} | {m('swap')} | {m('novel')} | {m('fired')} | {np.mean([allres[s][K]['remaps'] for s in seeds]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in seeds]):.0f} / {np.mean([allres[s][K]['swap_pair'] for s in seeds]):.0f}) |")
    json.dump({str(s): v for s, v in allres.items()}, open(os.path.join(a.out, "results.json"), "w"))
    open(os.path.join(a.out, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    os.makedirs("results/remap_probe", exist_ok=True)
    main()
