"""Dreams and Hinton's wake/sleep split, on Split-MNIST.

Every condition sleeps on a clock at ~0.8x replay (every 2000 wake examples,
50 batches of 32), the cheapest setting that kept prototype replay at 90%
(sleep_cost.py). Hinton's split is also run sleeping every 100 (3 batches,
0.96x), because it fails at 2000.

  proto10_nods      reference: prototype + noise replay as positives
  dream             the network dreams its own positives (no store)
  split_proto       wake: positives only; sleep: negatives only (replay, wrong label)
  split_dream       same, negatives are dreams with a wrong label
  split_dream_own   same, negatives are dreams with their own label (Hinton)

Usage
  python dreams.py                 # all conditions, 3 seeds, parallel
  python dreams.py --set topdown   # top-down generator vs prototype replay
  python dreams.py --set local     # local accumulate-only generator rules vs ridge
  python dreams.py --set prior     # contrastive rule with priors matching real codes
  python dreams.py --set fold      # contrastive rule with C folded into G
  python dreams.py --set cheapc    # contrastive rule with a low-bit / top-k C
  python dreams.py --set budget    # 8-bit C with 10 or 5 real codes/class
  python dreams.py --set fashion --data fashion   # best setups on Split Fashion-MNIST
  python dreams.py --set deep --data fashion      # two-level generator / more sleep steps, Fashion
  python dreams.py --set long                     # 3000 sleep steps on MNIST
  python dreams.py --set expand --data fashion    # nonlinear decoder (random expansion), Fashion
  python dreams.py --set perclass --data fashion  # per-class accuracy / confusions, Fashion
  python dreams.py --set nonoise                  # noise-free prototype replay on MNIST
  python dreams.py --set sharpen --data fashion   # dreams sharpened by lateral inhibition, Fashion
  python dreams.py --set tenseed --seeds 3 4 5 6 7 8 9   # seeds 3-9 (0-2 are in long / nonoise)
  python dreams.py --set frontier --data fashion  # 3 and 5 noise-free prototypes/class, Fashion
  python dreams.py --set sgmnist --seeds 3 4 5 6 7 8 9   # surprise-gated prototypes, MNIST
  python dreams.py --set sgfashion --data fashion # surprise-gated prototypes, Fashion
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import zlib
import struct
from multiprocessing import Pool

import numpy as np

from data import load_mnist

BS = 32
# (condition, sleep_every, sleep_batches)
GRID = [
    ("proto10_nods", 2000, 50),
    ("dream", 2000, 50),
    ("split_proto", 2000, 50),
    ("split_dream", 2000, 50),
    ("split_dream_own", 2000, 50),
    ("split_proto", 100, 3),
]
# --set topdown: Helmholtz-style top-down generator, all at every 2000 / 0.8x
TOPDOWN = [
    ("proto10_nods", 2000, 50),
    ("proto1_nods", 2000, 50),
    ("gen1_k10", 2000, 50),
    ("gen1_k10_count", 2000, 50),
    ("gen1_k10_ridge", 2000, 50),
    ("gen2_k10_ridge", 2000, 50),
    ("gen1_k1_ridge", 2000, 50),
]
# --set local: local, accumulate-only generator rules vs ridge
LOCAL = [
    ("proto10_nods", 2000, 50),
    ("gen1_k10_ridge", 2000, 50),
    ("gen1_k10_outstar1", 2000, 50),
    ("gen1_k10_outstar2", 2000, 50),
    ("gen1_k1_outstar1", 2000, 50),
    ("gen1_k10_contrast", 2000, 50),
    ("gen1_k10_contrast300", 2000, 50),
]
# --set prior: contrastive rule with a code prior that matches real codes
PRIOR = [(c, 2000, 50) for c in ["reservoir20_nods", "ridge_res20", "c_res20", "c_res20_300", "c_res10",
                                  "c_ppca20", "c_ppca20_300", "c_proto_neg10"]]
# --set fold: contrastive with its accumulator folded into G (one number per synapse)
FOLD = [(c, 2000, 50) for c in ["c_res20", "fold_res20_k025", "fold_res20_k1", "fold_res20_k4", "fold_res20_k16"]]
# --set cheapc: contrastive with a low-precision and/or sparse accumulator C
CHEAPC = [(c, 2000, 50) for c in ["c_res20", "cq8", "cq4", "cq2", "ck200", "ck100", "ck50", "ck100_q8", "ck50_q4"]]
# --set budget: 8-bit C with smaller priors, to get under the prototype store's memory
BUDGET = [(c, 2000, 50) for c in ["cq8_res10", "cq8_res5"]]
# --set gap: why the contrastive decoder trails ridge on the same prior
GAP = [(c, 2000, 50) for c in ["ridge_res10", "cq8_res10_exact", "cq8_res10_1000"]]
# --set codes8: real codes stored at 8 bits, so more fit per class
CODES8 = [(c, 2000, 50) for c in ["proto9_nods", "cq8_p8res20", "cq8_p8res40", "cq8_p8res80"]]
# --set decay: a regularised, converged contrastive decoder on 40 8-bit codes/class
DECAY = [(c, 2000, 50) for c in ["ridge_p8res40", "cq8_p8res40_d1", "cq8_p8res40_d3", "cq8_p8res40_d03", "cq8_p8res40_n1"]]
# --set noise: weaker, noise-based regularisation with 3000 sleep steps
NOISE = [(c, 2000, 50) for c in ["cq8_p8res40_n01_3k", "cq8_p8res40_n03_3k", "cq8_p8res40_d03_3k"]]
# --set jacobi: per-unit (Jacobi) sleep steps, and dropout negatives with Polyak averaging
JACOBI = [(c, 2000, 50) for c in ["p40_d03_300", "p40_cen_d03_300", "p40_jac_d03_300",
                                   "p40_drop08_1k", "p40_drop08_avg_1k", "p40_drop05_avg_1k"]]
# --set fashion (use with --data fashion): the best setups, Split Fashion-MNIST
FASHION = [(c, 2000, 50) for c in ["off", "proto9_nods", "proto10_nods", "ridge_p8res40",
                                    "p40_d03_300", "p40_drop08_avg_1k"]]
# --set deep (use with --data fashion): a sharper local decoder, two levels and/or more sleep steps
DEEP = [(c, 2000, 50) for c in ["p40_drop08_avg_3k", "g2_drop08_avg_1k", "g2_drop08_avg_3k", "ridge2_p8res40"]]
# --set long (MNIST): the best local decoder with 3000 sleep steps
LONG = [(c, 2000, 50) for c in ["p40_drop08_avg_3k"]]
# --set expand (use with --data fashion): a nonlinear image decoder, code + sparse random features
EXPAND = [(c, 2000, 50) for c in ["nl1000_3k", "nl2000_3k", "ridge_nl1000"]]
# --set perclass (use with --data fashion): per-class accuracy and confusions, prototypes vs decoded codes
PERCLASS = [(c, 2000, 50) for c in ["proto10_nods", "proto10_nonoise", "ridge_p8res40", "nl2000_3k"]]
# --set nonoise (MNIST): prototypes replayed without pixel noise, with the noisy references rerun
NONOISE = [(c, 2000, 50) for c in ["proto9_nods", "proto9_nonoise", "proto10_nods", "proto10_nonoise"]]
# --set sharpen (use with --data fashion): dreams sharpened by centre-surround lateral inhibition
SHARPEN = [(c, 2000, 50) for c in ["ridge_sharp1", "p40_3k_sharp1", "p40_3k_sharp2"]]
# --set tenseed (MNIST, run with --seeds 3 ... 9): more seeds for the local decoder vs prototypes
TENSEED = [(c, 2000, 50) for c in ["p40_drop08_avg_3k", "proto9_nods", "proto9_nonoise"]]
# --set frontier (use with --data fashion): memory curve of noise-free prototypes
FRONTIER = [(c, 2000, 50) for c in ["proto3_nonoise", "proto5_nonoise"]]
# --set sgmnist (seeds 3-9; 0-2 come from swap.py) and --set sgfashion (--data fashion): surprise-gated prototypes
SGMNIST = [(c, 2000, 50) for c in ["proto9_nn_sg"]]
SGFASHION = [(c, 2000, 50) for c in ["proto10_nn_sg"]]
SETS = {"sgmnist": SGMNIST, "sgfashion": SGFASHION, "frontier": FRONTIER, "tenseed": TENSEED, "sharpen": SHARPEN, "nonoise": NONOISE, "perclass": PERCLASS, "expand": EXPAND, "long": LONG, "deep": DEEP, "fashion": FASHION, "jacobi": JACOBI, "noise": NOISE, "decay": DECAY, "codes8": CODES8, "gap": GAP, "dreams": GRID, "topdown": TOPDOWN, "local": LOCAL, "prior": PRIOR, "fold": FOLD, "cheapc": CHEAPC,
        "budget": BUDGET}


def job(args):
    cond, every, batches, seed, dataset = args
    import sleep as S
    r = S.run(cond, seed, load_mnist(dataset), sleep_every=every, sleep_batches=batches, sleep_bs=BS)
    r.update(condition=cond, every=every, batches=batches, seed=seed, ratio=batches * BS / every, dataset=dataset)
    R = np.array(r["R"])
    r["task_end_avg"] = float(np.mean([R[i, :i + 1].mean() for i in range(1, len(R))]))
    print(f"{cond:16s} every={every:5d} seed={seed} classIL={r['class_il_final']:.4f} "
          f"taskend={r['task_end_avg']:.4f} taskIL={r['task_il_final']:.4f} {r['seconds']:.0f}s", flush=True)
    return r


def write_png(path, img):
    """Greyscale PNG from a 2-D array in [0, 1], no imaging library needed."""
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    raw = b"".join(b"\x00" + row.tobytes() for row in a)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    hdr = struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, 0, 0, 0, 0)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", hdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def dream_grid(images, scale=3):
    """10 dreams per class (rows = classes 0-9) -> one image."""
    x = np.array(images).reshape(10, 10, 28, 28).copy()
    x[:, :, 0, :10] = 0  # hide the label pixels
    g = np.ones((10 * 30, 10 * 30))
    for c in range(10):
        for k in range(10):
            g[c * 30 + 1:c * 30 + 29, k * 30 + 1:k * 30 + 29] = x[c, k]
    return np.kron(g, np.ones((scale, scale)))


def summarise(rows, seeds, grid=GRID):
    import sleep as S
    f = lambda rs, k: f"{np.mean([r[k] for r in rs]) * 100:.2f} ± {np.std([r[k] for r in rs]) * 100:.2f}"
    out = [f"Seeds: {seeds}. Split class-IL. Task-end avg = class-IL over classes seen so far at the "
           "end of tasks 2-5 (points anywhere in the sleep cycle).\n",
           "| Condition | Sleep every | Replay ratio | Store (floats) | Class-IL final | Task-end avg | Task-IL final | Dream p(label) | Dead units |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cond, every, batches in grid:
        rs = [r for r in rows if r["condition"] == cond and r["every"] == every]
        if not rs:
            continue
        conf = f"{np.mean([r['dream_conf'] for r in rs]):.2f}" if rs[0]["dream_conf"] is not None else "–"
        out.append(f"| {S.LABELS[cond]} | {every} | {batches * BS / every:.2f}× | {int(rs[0]['store_floats']):,} | "
                   f"{f(rs, 'class_il_final')} | {f(rs, 'task_end_avg')} | {f(rs, 'task_il_final')} | {conf} | "
                   f"{f(rs, 'dead_units')} |")
    out += ["\n### Do the final network's dreams look like digits?\n",
            "Correlation between each class's average dream and the average real test digit of the same "
            "class (diagonal) vs the best other class. Mean over seeds.\n",
            "| Condition | Same class | Best other class |", "|---|---|---|"]
    for cond, every, _ in grid:
        rs = [r for r in rows if r["condition"] == cond and r["every"] == every and r.get("dreams")]
        if not rs:
            continue
        C = np.mean([np.array(r["dreams"]["corr"]) for r in rs], 0)
        off = (C - np.eye(10) * 9).max(1)
        out.append(f"| {S.LABELS[cond]} (every {every}) | {np.diag(C).mean():.2f} | {off.mean():.2f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--set", default="dreams", choices=list(SETS))
    ap.add_argument("--out", default=None, help="default results/<set>")
    ap.add_argument("--data", default="mnist", choices=["mnist", "fashion"])
    args = ap.parse_args()
    args.out = args.out or f"results/{args.set}"
    grid = SETS[args.set]
    os.makedirs(args.out, exist_ok=True)
    load_mnist(args.data)
    tasks = [(c, e, b, s, args.data) for c, e, b in grid for s in args.seeds]
    with Pool(args.jobs) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(dict(seeds=args.seeds, rows=rows), open(os.path.join(args.out, "results.json"), "w"))
    for r in rows:
        if r.get("dreams") and r["seed"] == args.seeds[0]:
            write_png(os.path.join(args.out, f"dreams_{r['condition']}_every{r['every']}.png"),
                      dream_grid(r["dreams"]["images"]))
    md = summarise(rows, args.seeds, grid)
    title = {"dreams": "Dreams and Hinton's split", "topdown": "Top-down generative dreams",
             "local": "Local accumulate-only generator rules",
             "prior": "Contrastive generator with matched code priors",
             "fold": "Contrastive generator, one number per synapse",
             "cheapc": "Contrastive generator with a cheap accumulator",
             "budget": "8-bit accumulator with a smaller prior",
             "gap": "Contrastive vs ridge: where the gap comes from",
             "codes8": "More codes per class: the prior stored at 8 bits",
             "decay": "Regularised, converged contrastive decoder",
             "noise": "Noisy negatives, weaker, with 3000 sleep steps",
             "jacobi": "Per-unit sleep steps and dropout negatives",
             "fashion": "Best setups on Split Fashion-MNIST",
             "deep": "A sharper local decoder: two levels, more sleep steps",
             "long": "Longer sleeps on Split MNIST",
             "expand": "A nonlinear local decoder: code + sparse random features",
             "perclass": "Per-class accuracy: what prototypes do on coat/sandal",
             "nonoise": "Prototypes without pixel noise on Split MNIST",
             "sharpen": "Sharpened dreams: centre-surround lateral inhibition",
             "tenseed": "Local decoder vs prototypes on Split MNIST, more seeds",
             "frontier": "Noise-free prototype memory curve on Split Fashion-MNIST",
             "sgmnist": "Surprise-gated prototypes on Split MNIST, more seeds",
             "sgfashion": "Surprise-gated prototypes on Split Fashion-MNIST"}[args.set]
    open(os.path.join(args.out, "RESULTS.md"), "w").write(f"# {title}\n\nData: {args.data}.\n\n" + md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
