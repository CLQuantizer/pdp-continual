"""Why do decoded sandals fail? A shape check of what each method replays.

Trains a condition on Split Fashion-MNIST (every 2000, 0.8x), keeps the
final network and store, and compares the store's sandal replays with real
test sandals (label pixels excluded, distances = mean squared pixel error):

  fidelity   each replay item -> its nearest real sandal (is it a real-looking sandal?)
  coverage   each real sandal -> its nearest replay item (are all styles represented?),
             with 10 items per method so stores of different sizes compare fairly
  openness   empty fraction of the ink's bounding box (sandals are straps around
             holes; boots and sneakers are filled)
  nearest    class of each item's nearest real image among sandal / sneaker / boot
  styles     k-means (8) on real training sandals; how many styles the items hit
  detail     correlation of each item's high-pass detail (image minus its 3x3 blur)
             with that of its nearest real sandal: is the fine structure real or noise?

For the top-down decoders it also decodes the final network's codes of real
test sandals ("recon"). Bad dreams with good recon = stale stored codes; bad
recon too = the decoder itself averages styles.

Usage
  python sandal.py            # 5 conditions x 3 seeds, parallel -> results/sandal/
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np

from data import load_mnist

CONDS = ["proto10_nonoise", "proto10_nods", "proto3_nonoise", "ridge_p8res40", "p40_drop08_avg_3k"]
SANDAL, SNEAKER, BOOT = 5, 7, 9
FOOT = (SANDAL, SNEAKER, BOOT)


def d2(a, b):
    """Mean squared pixel distance between every row of a and every row of b (pixels 10: only)."""
    a, b = a[:, 10:].astype(np.float64), b[:, 10:].astype(np.float64)
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T) / a.shape[1]


def openness(x):
    """Fraction of the ink's bounding box that is empty (< 0.1)."""
    out = []
    for img in x.reshape(-1, 28, 28):
        img = img.copy()
        img[0, :10] = 0
        r, c = np.nonzero(img > 0.3)
        if len(r) == 0:
            out.append(np.nan)
            continue
        box = img[r.min():r.max() + 1, c.min():c.max() + 1]
        out.append(float((box < 0.1).mean()))
    return np.array(out)


def highpass(x):
    img = x.reshape(-1, 28, 28).copy()
    img[:, 0, :10] = 0
    p = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return (img - sum(p[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)) / 9).reshape(len(img), -1)


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = d2(x, c).argmin(1)
        c = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return c


def measure(items, real, styles, foot_real, foot_y, rng):
    """All metrics for one set of sandal images."""
    D = d2(items, real)
    sub = [d2(real, items[rng.choice(len(items), min(10, len(items)), replace=False)]).min(1).mean()
           for _ in range(20)]
    near = foot_y[d2(items, foot_real).argmin(1)]
    hit = d2(items, styles).argmin(1)
    hi, hr = highpass(items), highpass(real[D.argmin(1)])
    detail = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(hi, hr) if a.std() > 0])
    return dict(detail=float(detail), n=len(items), fidelity=float(D.min(1).mean()), coverage=float(D.min(0).mean()),
                coverage10=float(np.mean(sub)), openness=float(np.nanmean(openness(items))),
                near={str(c): float((near == c).mean()) for c in FOOT},
                styles=int(len(np.unique(hit))), style_hist=np.bincount(hit, minlength=len(styles)).tolist())


def job(args):
    cond, seed = args
    import sleep as S
    xtr, ytr, xte, yte = data = load_mnist("fashion")
    r, m, store = S.run(cond, seed, data, sleep_every=2000, sleep_batches=50, sleep_bs=32, return_store=True)
    rng = np.random.default_rng(seed)
    real = xte[yte == SANDAL]
    styles = kmeans(xtr[ytr == SANDAL], 8)
    fm = np.isin(yte, FOOT)
    foot_real, foot_y = xte[fm], yte[fm]
    out = dict(condition=cond, seed=seed, class_il_final=r["class_il_final"], confusion=r["confusion"])
    if isinstance(store, S.ProtoStore):
        items = store.mu[SANDAL]                 # what is stored
        replay = store.sample_class(SANDAL, 400)  # what is replayed (noise added if the store uses it)
    else:
        store.consolidate()
        items = store.decode(np.stack(store.prior.buf[SANDAL]).astype(np.float32))
        replay = items
        z = [h for _, h in m._layers(m.embed(real, np.full(len(real), SANDAL)))][0]
        out["recon"] = measure(store.decode(z), real, styles, foot_real, foot_y, rng)
        out["recon"]["mse"] = float(((store.decode(z) - real)[:, 10:] ** 2).mean())
    out["items"] = measure(items, real, styles, foot_real, foot_y, rng)
    out["replay_openness"] = float(np.nanmean(openness(replay)))
    if seed == 0:
        out["images"] = np.round(items[:40], 3).tolist()
    print(f"{cond:18s} seed={seed} classIL={r['class_il_final']:.4f} fid={out['items']['fidelity']:.4f} "
          f"cov10={out['items']['coverage10']:.4f} open={out['items']['openness']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs("results/sandal", exist_ok=True)
    xtr, ytr, xte, yte = load_mnist("fashion")
    rng = np.random.default_rng(0)
    real = xte[yte == SANDAL]
    styles = kmeans(xtr[ytr == SANDAL], 8)
    fm = np.isin(yte, FOOT)
    ref = {"real test sandals": real, "real test sneakers": xte[yte == SNEAKER], "real test boots": xte[yte == BOOT],
           "sandal style centroids (k-means 8)": styles}
    # baselines: real sandals measured against themselves use a held-out half
    half = rng.permutation(len(real))
    base = {k: dict(openness=float(np.nanmean(openness(v)))) for k, v in ref.items()}
    base["real test sandals"].update(measure(real[half[:500]], real[half[500:]], styles, xte[fm], yte[fm], rng))
    with Pool(args.jobs) as pool:
        rows = pool.map(job, [(c, s) for c in CONDS for s in args.seeds], chunksize=1)
    json.dump(dict(seeds=args.seeds, rows=rows, base=base), open("results/sandal/results.json", "w"))
    import sleep as S
    from dreams import write_png
    # one image: row per condition (seed 0), up to 20 sandal items each; last row = style centroids
    grid = np.ones((len(CONDS) + 1, 20, 28, 28))
    for i, c in enumerate(CONDS):
        im = np.array(next(r["images"] for r in rows if r["condition"] == c and r["seed"] == args.seeds[0]))[:20]
        grid[i, :len(im)] = im.reshape(-1, 28, 28)
    grid[-1, :8] = styles.reshape(-1, 28, 28)
    grid[:, :, 0, :10] = 0
    big = np.ones(((len(CONDS) + 1) * 30, 20 * 30))
    for i in range(len(CONDS) + 1):
        for j in range(20):
            big[i * 30 + 1:i * 30 + 29, j * 30 + 1:j * 30 + 29] = grid[i, j]
    write_png("results/sandal/sandals.png", np.kron(big, np.ones((3, 3))))

    f = lambda rs, key, sub="items": f"{np.mean([r[sub][key] for r in rs]):.4f}"
    md = ["# Sandal shape check\n", f"Split Fashion-MNIST, every 2000, 0.8x, seeds {args.seeds}. Items = what the store "
          "holds for sandal (prototype means, or the decoder's dreams of the 40 stored codes). Recon = the final "
          "decoder applied to the final network's codes of real test sandals.\n",
          "| Condition | Class-IL | Items | Fidelity ↓ | Coverage (10 items) ↓ | Openness | Nearest is sandal / sneaker / boot | Styles hit (of 8) | Fine detail | Replay openness |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    b = base["real test sandals"]
    md.append(f"| Real sandals (500 vs other 500) | – | 500 | {b['fidelity']:.4f} | {b['coverage10']:.4f} | {b['openness']:.3f} | "
              f"{b['near'][str(SANDAL)]:.2f} / {b['near'][str(SNEAKER)]:.2f} / {b['near'][str(BOOT)]:.2f} | {b['styles']} | {b['detail']:.2f} | – |")
    for c in CONDS:
        rs = [r for r in rows if r["condition"] == c]
        nr = lambda k, sub="items": np.mean([r[sub]["near"][str(k)] for r in rs])
        md.append(f"| {S.LABELS[c]} | {np.mean([r['class_il_final'] for r in rs]) * 100:.2f} | {rs[0]['items']['n']} | "
                  f"{f(rs, 'fidelity')} | {f(rs, 'coverage10')} | {np.mean([r['items']['openness'] for r in rs]):.3f} | "
                  f"{nr(SANDAL):.2f} / {nr(SNEAKER):.2f} / {nr(BOOT):.2f} | "
                  f"{np.mean([r['items']['styles'] for r in rs]):.1f} | {f(rs, 'detail')} | {np.mean([r['replay_openness'] for r in rs]):.3f} |")
        if "recon" in rs[0]:
            md.append(f"| ↳ recon of real sandals | | 1000 | {f(rs, 'fidelity', 'recon')} | {f(rs, 'coverage10', 'recon')} | "
                      f"{np.mean([r['recon']['openness'] for r in rs]):.3f} | {nr(SANDAL, 'recon'):.2f} / {nr(SNEAKER, 'recon'):.2f} / "
                      f"{nr(BOOT, 'recon'):.2f} | {np.mean([r['recon']['styles'] for r in rs]):.1f} | {f(rs, 'detail', 'recon')} | – |")
    md += ["", "Openness of real classes: " + ", ".join(f"{k} {v['openness']:.3f}" for k, v in base.items()), ""]
    open("results/sandal/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
