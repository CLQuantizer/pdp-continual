"""CLS complementary routing: let the fast store *answer*, not only replay.

One FF + clock-sleep system (sleep.CONDITIONS proto9_nonoise recipe: ProtoStore k/class, no pixel noise,
sleep every 2000 wake examples, 50x32 replay = 0.8x) is run on Split-MNIST and then on the 2<->3 label
swap. At every step all answer routes are scored on the SAME network and store, so they're paired:

  ff        the FF network (argmax goodness)                    = "FF + sleep" baseline
  store     nearest prototype in the store (raw pixels)          = "store alone" baseline
  m{tau}    store answers if its relative margin (d2-d1)/d1 >= tau, or its predicted class is new
            (< NEW labelled examples seen); FF otherwise           (hypothesis 1, fixed rule)
  rel       per-class running reliability arbiter: for each route an EMA (alpha) of whether its
            answer was right, indexed by the class it answered; the more reliable route answers
            (ties -> store). Updated prequentially from the labelled stream only.
  relp      rel, but FF's reliability for class c is discounted by fp[c], the share of prototypes of
            classes absent from the last 2000 labels that FF calls c (awake memory probe every 250 steps)
  gate{phi} FF answers only about classes in the recent stream (last 2000 labels): if both routes name
            recent classes, stream precision (rel) decides; if FF names a recent class and the store an
            old one, FF answers only if fp[ff] < phi; otherwise the store answers
  oracle    either route right (upper bound, not a system)

Metrics: prequential stream accuracy (answer before learning), anytime class-IL on 2000 fixed test images
restricted to seen classes every 1000 wake examples (mid-interval and just before each sleep), final
class-IL after the last sleep, and during the swap (8000 examples of 2/3 relabelled) the new-mapping
accuracy on test 2/3 + retention on the other 8 every 500 examples, plus swap-phase prequential.

Usage: python cls_route.py --seeds 0 1 2 --k 9 --jobs 2 --out results/cls_route
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import argparse, json, time
from multiprocessing import Pool
import numpy as np

TAUS = [0.05, 0.1, 0.2, 0.4]
NEW = 1000
ALPHA = 0.02
PHIS = [0.02, 0.1]
W_RECENT = 2000


def routes_names():
    return ["ff", "store"] + [f"m{t}" for t in TAUS] + ["rel", "relp"] + [f"gate{p}" for p in PHIS] + ["oracle"]


def store_nn(store, X):
    """Nearest prototype: predicted class, relative margin to the best other class."""
    cs = sorted(store.mu)
    M = np.vstack([store.mu[c] for c in cs]); lab = np.concatenate([[c] * len(store.mu[c]) for c in cs])
    d = (X ** 2).sum(1)[:, None] - 2 * X @ M.T + (M ** 2).sum(1)[None]
    d = np.maximum(d, 1e-6)
    per = np.full((len(X), 10), np.inf)
    for c in cs:
        per[:, c] = d[:, lab == c].min(1)
    pred = per.argmin(1)
    srt = np.sort(per, 1)
    margin = np.where(np.isfinite(srt[:, 1]), (srt[:, 1] - srt[:, 0]) / srt[:, 0], np.inf)
    return pred, margin


class Router:
    def __init__(self):
        self.rel = {"ff": np.full(10, 0.5), "store": np.full(10, 0.5)}
        self.count = np.zeros(10, int)
        self.fp = np.zeros(10)       # awake memory probe: share of old-class prototypes FF calls c
        self.recent = []

    def probe(self, m, store):
        """Awake replay check: FF on the prototypes of classes absent from the last W labels; fp[c] = share
        it labels c. The stream can't show FF's false alarms on absent classes; the store can."""
        rec = set(self.recent[-W_RECENT:]); old = [c for c in sorted(store.mu) if c not in rec]
        self.recent = self.recent[-W_RECENT:]
        if not old:
            self.fp[:] = 0; return
        P = np.vstack([store.mu[c] for c in old]); lab = np.concatenate([[c] * len(store.mu[c]) for c in old])
        pr = m.scores(P).argmax(1)
        self.fp = np.bincount(pr[pr != lab], minlength=10)[:10] / len(P)

    def answer(self, ff, sp, sm, truth=None):
        """ff, sp: (n,) predictions; sm: (n,) store margins. Returns dict route -> (n,) predictions."""
        new = self.count[sp] < NEW
        out = {"ff": ff, "store": sp}
        for t in TAUS:
            out[f"m{t}"] = np.where((sm >= t) | new, sp, ff)
        use_store = self.rel["store"][sp] >= self.rel["ff"][ff]
        out["rel"] = np.where(use_store, sp, ff)
        use_p = self.rel["store"][sp] >= self.rel["ff"][ff] * (1 - self.fp[ff])
        out["relp"] = np.where(use_p, sp, ff)
        # gate: FF (plastic cortex) is trusted only about classes in the recent stream; the store about
        # everything else. Both recent -> stream precision decides (unbiased when both are present).
        # FF recent vs store old -> FF only if its awake-probe false-alarm rate fp[ff] < phi.
        rec = np.zeros(10, bool); rec[list(set(self.recent[-W_RECENT:]))] = True
        both = rec[ff] & rec[sp]
        prec_ff = self.rel["ff"][ff] > self.rel["store"][sp]
        for p in PHIS:
            use_ff = (ff != sp) & ((both & prec_ff) | (rec[ff] & ~rec[sp] & (self.fp[ff] < p)))
            out[f"gate{p}"] = np.where(use_ff, ff, sp)
        if truth is not None:
            out["oracle"] = np.where(sp == truth, sp, ff)
        return out

    def update(self, ff, sp, y):
        # a route can't be blamed for a class nobody has seen yet (first example of a class)
        for r, p in (("ff", ff), ("store", sp)) if self.count[y] > 0 else ():
            self.rel[r][p] += ALPHA * (float(p == y) - self.rel[r][p])
        self.count[y] += 1
        self.recent.append(y)


def run(args):
    seed, k, n_swap, dataset, every, limit = args
    import sleep as S
    import learners as L
    from bench import TASKS
    from data import load_mnist
    xtr, ytr, xte, yte = load_mnist(dataset)
    xtr = xtr.astype(np.float32); xte = xte.astype(np.float32)
    cfg = dict(store="proto", size=k, downscale=0.0, pnoise=0.0)
    rng = np.random.default_rng(seed)
    m = L.ForwardForward(lr=1e-3, loss="symba", homeo=0.001, seed=seed)
    store = S.ProtoStore(k, np.random.default_rng(seed + 1000), noise=0.0)
    sleep_rng = np.random.default_rng(seed + 2000)
    batches = max(1, round(50 * every / 2000))
    R = Router(); names = routes_names()
    any_idx = np.random.default_rng(seed + 4000).choice(len(xte), 2000, replace=False)
    preq = {r: [] for r in names}; anytime = []; seen = set(); wake = 0; t0 = time.time()

    def test(X, Y, chunk=2000):
        out = {r: [] for r in names}
        for i in range(0, len(X), chunk):
            x, y = X[i:i + chunk], Y[i:i + chunk]
            ff = m.scores(x).argmax(1); sp, sm = store_nn(store, x)
            for r, p in R.answer(ff, sp, sm, y).items():
                out[r].append(p)
        return {r: np.concatenate(v) for r, v in out.items()}

    def step(x, y, rec):
        nonlocal wake
        if store.mu:
            ff = m.scores(x).argmax(1); sp, sm = store_nn(store, x)
            for r, p in R.answer(ff, sp, sm, np.array([y])).items():
                rec[r].append(int(p[0] == y))
            R.update(int(ff[0]), int(sp[0]), y)
        else:
            R.count[y] += 1; R.recent.append(y)
        m.fit_batch(x, np.array([y]))
        store.add(x, y); seen.add(y); wake += 1
        if wake % every == 0:
            S.sleep_once(m, cfg, store, batches, 32, sorted(seen), 20, 16, sleep_rng)
        if wake % 250 == 0:
            R.probe(m, store)

    for t in TASKS:
        for i in rng.permutation(np.flatnonzero(np.isin(ytr, t)))[:limit or None]:
            step(xtr[i:i + 1], int(ytr[i]), preq)
            if wake % 1000 == 0:
                j = any_idx[np.isin(yte[any_idx], sorted(seen))]
                p = test(xte[j], yte[j])
                anytime.append(dict(t=wake, **{r: float((p[r] == yte[j]).mean()) for r in names}))
    pf = test(xte, yte)
    final = {r: float((pf[r] == yte).mean()) for r in names}
    pre_swap_s = time.time() - t0
    # swap phase: digits 2,3 swap labels; only they are seen
    a, b = 2, 3; flip = {a: b, b: a}
    idx = np.resize(rng.permutation(np.flatnonzero(np.isin(ytr, [a, b]))), n_swap)
    sel, rest = np.isin(yte, [a, b]), ~np.isin(yte, [a, b])
    ynew = yte.copy(); ynew[yte == a] = b; ynew[yte == b] = a
    sw_preq = {r: [] for r in names}; traj = []

    def probe(t):
        p = test(xte, ynew)  # oracle uses the new mapping as truth
        traj.append(dict(t=t, **{f"{r}_new": float((p[r][sel] == ynew[sel]).mean()) for r in names},
                         **{f"{r}_oth": float((p[r][rest] == yte[rest]).mean()) for r in names}))
    probe(0)
    for kk, i in enumerate(idx, 1):
        step(xtr[i:i + 1], flip[int(ytr[i])], sw_preq)
        if kk % 500 == 0:
            probe(kk)
    return dict(seed=seed, k=k, dataset=dataset, final=final,
                preq={r: float(np.mean(v)) for r, v in preq.items()},
                anytime_mean={r: float(np.mean([a_[r] for a_ in anytime])) for r in names},
                anytime_min={r: float(np.min([a_[r] for a_ in anytime])) for r in names},
                anytime=anytime, swap=traj,
                swap_preq={r: float(np.mean(v)) for r, v in sw_preq.items()},
                swap_preq_first500={r: float(np.mean(v[:500])) for r, v in sw_preq.items()},
                rel_final={kk: v.tolist() for kk, v in R.rel.items()},
                seconds_pre_swap=pre_swap_s, seconds=time.time() - t0)


def summarise(rows):
    names = routes_names(); out = []
    for k in sorted({r["k"] for r in rows}):
        rs = [r for r in rows if r["k"] == k]
        seeds = sorted(r["seed"] for r in rs)
        out += [f"\n## Store {k} prototypes/class (seeds {seeds}, mean %)\n",
                "| Route | Split final class-IL | Anytime mean | Anytime min | Prequential | Swap new @500 | @2000 | @8000 | Others @8000 | Swap preq |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: f"{100 * np.mean(v):.1f}"
        def at(r, t, key):
            return [p[key] for x in rs for p in x["swap"] if p["t"] == t]
        tmax = max(p["t"] for p in rs[0]["swap"])
        for n in names:
            out.append(f"| {n} | {f([r['final'][n] for r in rs])} | {f([r['anytime_mean'][n] for r in rs])} | "
                       f"{f([r['anytime_min'][n] for r in rs])} | {f([r['preq'][n] for r in rs])} | "
                       f"{f(at(None, 500, n + '_new'))} | {f(at(None, 2000, n + '_new'))} | {f(at(None, tmax, n + '_new'))} | "
                       f"{f(at(None, tmax, n + '_oth'))} | {f([r['swap_preq'][n] for r in rs])} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", nargs="+", type=int, default=[9])
    ap.add_argument("--swap", type=int, default=8000)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: examples per task")
    ap.add_argument("--out", default="results/cls_route")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from data import load_mnist; load_mnist(a.data)
    path = os.path.join(a.out, "results.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["k"], r["seed"]) for r in rows}
    todo = [(s, k, a.swap, a.data, a.every, a.limit) for k in a.k for s in a.seeds if (k, s) not in done]
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(run, todo):
            rows.append(r)
            json.dump(rows, open(path, "w"))
            open(os.path.join(a.out, "RESULTS.md"), "w").write(
                "# CLS complementary routing (`cls_route.py`)\n\nPartial until all seeds are in.\n" + summarise(rows) + "\n")
            print(f"done k={r['k']} seed={r['seed']} {r['seconds']:.0f}s final={ {n: round(v*100,1) for n, v in r['final'].items()} }", flush=True)


if __name__ == "__main__":
    main()
