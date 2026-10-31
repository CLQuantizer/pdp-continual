"""Which store code separates a 4 from a 9? (loop 16:00 UTC+8 31 Oct, pre-registered before running.)

Decides the 15:00 fork offline before any life run. Store-like nearest-prototype geometry (9 k-means prototypes per
class, as ProtoStore(9)) in each code space; test 4s and 9s only.
  imp    impostor rate: share of test 4s/9s whose nearest 4-or-9 prototype has the other label (what a swapped 9
         labelled 4 does to a 4 prototype: confirms it)
  ratio  mean d(own nearest proto) / d(rival nearest proto) on 4s and 9s (Babadi & Sompolinsky within/between)
  acc10  10-class nearest-prototype accuracy on the full test set
Codes: pix; rand = fixed random expansion 2000 + k-WTA 100 (15:00, killed in life); hebb = same expansion tuned
online by competitive Hebbian learning on the stream (Babadi & Sompolinsky's remedy; 14:00/15:00 hyp 2); ff1/ff2/ff12 =
hidden layers of the online FF cortex (life.py settings: symba, homeo 0.001, lr 1e-3, one image per update) under a
neutral label (0.1 on every label pixel), the CLS index of a cortical code (15:00 hyp 1).
Drift: prototypes built at the end of the normal stream (20k arrivals), codes re-read after 2000 further arrivals with
4<->9 labels swapped (the s49 world): stale = old prototypes on new codes; fresh = prototypes rebuilt from the same
stored images re-encoded by the post-swap cortex (store keeps pixels, indexes them through the current cortex).
Added after the first pass showed drift (fresh is exploratory, not pre-registered).
Pre-registered:
  H2 (neuron, hebb): ratio at least 10% below pix. Kill otherwise.
  H1 (architecture, ff code): imp at most 0.5x pix AND stale imp at most 0.75x pix. Kill otherwise (no life rerun).
"""
import json, os, sys, time
import numpy as np
import learners as L
from data import load_mnist

N1, N2, SW = 20000, 2000, (4, 9)


def kmeans(X, k, rng, it=20):
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(it):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C


def protos(Z, y, rng, n=2000):
    P, Y = [], []
    for c in range(10):
        Zc = Z[y == c][:n]
        P.append(kmeans(Zc, 9, rng)); Y.append(np.full(9, c))
    return np.concatenate(P), np.concatenate(Y)


def metrics(P, Y, Z, y):
    d = (Z ** 2).sum(1)[:, None] - 2 * Z @ P.T + (P ** 2).sum(1)[None]
    d = np.maximum(d, 0)
    acc10 = float((Y[d.argmin(1)] == y).mean())
    m = np.isin(y, SW)
    d, yy = d[m], y[m]
    d4, d9 = d[:, Y == 4].min(1), d[:, Y == 9].min(1)
    own = np.where(yy == 4, d4, d9); riv = np.where(yy == 4, d9, d4)
    return dict(imp=float((riv < own).mean()), ratio=float(np.mean(np.sqrt(own) / np.sqrt(riv + 1e-12))), acc10=acc10)


def kwta(H, k):
    out = np.zeros_like(H)
    idx = np.argpartition(-H, k, axis=1)[:, :k]
    np.put_along_axis(out, idx, 1.0, axis=1)
    return out


def run(seed):
    rng = np.random.default_rng(seed)
    xtr, ytr, xte, yte = load_mnist("mnist")
    order = rng.permutation(len(xtr))[:N1 + N2]
    xbar = xtr.mean(0)
    Wr = np.random.default_rng(12345).standard_normal((784, 2000)).astype(np.float32) / np.sqrt(784)
    Wh = Wr.copy(); Wh /= np.linalg.norm(Wh, axis=0, keepdims=True)
    ff = L.ForwardForward(lr=1e-3, loss="symba", homeo=0.001, seed=seed)

    def ffcode(X):
        Xn = X.copy(); Xn[:, :10] = 0.1
        hs = [h for _, h in ff._layers(Xn)]
        return hs

    def codes(X):
        hs = ffcode(X)
        nrm = lambda h: h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-6)
        return dict(pix=X, rand=kwta((X - xbar) @ Wr, 100), hebb=kwta((X - xbar) @ Wh, 100),
                    ff1=nrm(hs[0]), ff2=nrm(hs[1]), ff12=np.concatenate([nrm(hs[0]), nrm(hs[1])], 1))

    t0 = time.time()
    eta = 0.01
    for i, j in enumerate(order):
        x = xtr[j:j + 1]; y = int(ytr[j])
        if i >= N1 and y in SW:
            y = SW[1 - SW.index(y)]
        ff.fit_batch(x, np.array([y]))
        # competitive Hebbian on the expansion (hebb code): winners move toward the centred input
        xc = (x[0] - xbar)
        win = np.argpartition(-(xc @ Wh), 100)[:100]
        Wh[:, win] += eta * (xc[:, None] / (np.linalg.norm(xc) + 1e-6) - Wh[:, win])
        Wh[:, win] /= np.linalg.norm(Wh[:, win], axis=0, keepdims=True)
        if i == N1 - 1:
            xs = xtr[order[:N1]]; ys = ytr[order[:N1]]
            Cs = codes(xs[:12000]); Ct = codes(xte)
            P = {k: protos(Cs[k], ys[:12000], np.random.default_rng(seed)) for k in Cs}
            pre = {k: metrics(*P[k], Ct[k], yte) for k in Cs}
    Ct2 = codes(xte)
    stale = {k: metrics(*P[k], Ct2[k], yte) for k in Ct2}
    Cs2 = codes(xs[:12000])  # re-encode the same stored images with the post-swap cortex (index re-read at query)
    fresh = {k: metrics(*protos(Cs2[k], ys[:12000], np.random.default_rng(seed)), Ct2[k], yte) for k in Ct2}
    print(f"seed {seed} {time.time() - t0:.0f}s", flush=True)
    return dict(pre=pre, stale=stale, fresh=fresh)


if __name__ == "__main__":
    from multiprocessing import Pool
    seeds = [0, 1, 2]
    with Pool(3) as p:
        R = p.map(run, seeds)
    out = "results/code_probe"; os.makedirs(out, exist_ok=True)
    json.dump(dict(seeds=seeds, runs=R), open(f"{out}/results.json", "w"))
    md = ["# Which store code separates 4 from 9 (`code_probe.py`)\n", __doc__,
          f"\nSeeds {seeds}, mean ± std.\n", "| Code | imp | ratio | acc10 | stale imp | stale ratio | stale acc10 | fresh imp | fresh ratio | fresh acc10 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in R[0]["pre"]:
        f = lambda ph, m: f"{np.mean([r[ph][k][m] for r in R]):.3f} ± {np.std([r[ph][k][m] for r in R]):.3f}"
        md.append(f"| {k} | {f('pre','imp')} | {f('pre','ratio')} | {f('pre','acc10')} | {f('stale','imp')} | {f('stale','ratio')} | {f('stale','acc10')} | {f('fresh','imp')} | {f('fresh','ratio')} | {f('fresh','acc10')} |")
    open(f"{out}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
