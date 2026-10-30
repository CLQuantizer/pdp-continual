"""Benchmark: backprop vs. five non-backprop learners on MNIST.

Protocols
  standard  Several epochs of minibatches, shuffled. Test accuracy.
  iid       ONE pass, batch size 1, shuffled. The online control.
  split     ONE pass, batch size 1, digits arrive as tasks
            0/1 -> 2/3 -> 4/5 -> 6/7 -> 8/9. Evaluated after every task on
            all tasks. Class-incremental: one 10-way output, no task labels
            at test time. Task-incremental accuracy (choose between the two
            digits of the task) is also reported as the easier variant.

Usage
  python bench.py                       # everything, 1 seed
  python bench.py --seeds 0 1 2         # mean over seeds
  python bench.py --protocols split --methods bp hebb
"""
import argparse
import json
import os
import time

import numpy as np

import learners as L
from data import load_mnist

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# name -> (label, constructor(seed), standard-protocol settings)
# Standard settings: epochs, batch size, base lr (decayed linearly to 0).
METHODS = {
    "bp": ("Backprop MLP", lambda s: L.MLP("bp", lr=1e-3, seed=s), dict(epochs=10, bs=64, lr=1e-3)),
    "fa": ("Feedback alignment", lambda s: L.MLP("fa", lr=1e-3, seed=s), dict(epochs=10, bs=64, lr=1e-3)),
    "dfa": ("Direct feedback alignment", lambda s: L.MLP("dfa", lr=1e-3, seed=s), dict(epochs=10, bs=64, lr=1e-3)),
    "ff_orig": ("Forward-forward (threshold loss)", lambda s: L.ForwardForward(lr=3e-3, seed=s), dict(epochs=10, bs=64, lr=3e-3)),
    "ff": ("Forward-forward (SymBa + homeostasis)",
           lambda s: L.ForwardForward(lr=3e-3, loss="symba", homeo=0.001, seed=s), dict(epochs=10, bs=64, lr=3e-3)),
    "hebb": ("Competitive Hebbian + ridge", lambda s: L.SoftHebb(tau=0.005, lr=0.02, homeo=0.1, seed=s), dict(epochs=2, bs=32, lr=0.02)),
    "rf": ("Random features + ridge", lambda s: L.RandomFeatures(seed=s), dict(epochs=1, bs=1000, lr=None)),
}

# Online (batch size 1) learning rates. Chosen on the iid stream using a
# held-out slice of the training set (see tune_online), never the test set.
ONLINE_LR = {"bp": 3e-4, "fa": 3e-4, "dfa": 3e-4, "ff_orig": 1e-3, "ff": 1e-3, "hebb": 0.02, "rf": None}


def set_lr(m, lr):
    if lr is None:
        return
    if hasattr(m, "opt"):
        m.opt.lr = lr
    else:
        m.lr = lr


def accuracy(m, x, y, chunk=2000):
    pred = np.concatenate([m.scores(x[i:i + chunk]).argmax(1) for i in range(0, len(x), chunk)])
    return float((pred == y).mean())


def task_il_accuracy(m, x, y, classes):
    s = m.scores(x)[:, list(classes)]
    return float((np.array(classes)[s.argmax(1)] == y).mean())


# --------------------------------------------------------------------------
def run_standard(key, seed, data):
    xtr, ytr, xte, yte = data
    _, make, cfg = METHODS[key]
    m, rng, t0 = make(seed), np.random.default_rng(seed), time.time()
    for ep in range(cfg["epochs"]):
        if cfg["lr"] is not None:
            set_lr(m, cfg["lr"] * (1 - ep / cfg["epochs"]))
        p = rng.permutation(len(xtr))
        for i in range(0, len(p), cfg["bs"]):
            j = p[i:i + cfg["bs"]]
            m.fit_batch(xtr[j], ytr[j])
    if hasattr(m, "end_training"):
        m.end_training(xtr, ytr)
    return dict(test_acc=accuracy(m, xte, yte), seconds=time.time() - t0)


def run_stream(m, x, y, order, lr):
    set_lr(m, lr)
    for i in order:
        m.fit_batch(x[i:i + 1], y[i:i + 1])


def run_iid(key, seed, data):
    xtr, ytr, xte, yte = data
    m, t0 = METHODS[key][1](seed), time.time()
    run_stream(m, xtr, ytr, np.random.default_rng(seed).permutation(len(xtr)), ONLINE_LR[key])
    return dict(test_acc=accuracy(m, xte, yte), seconds=time.time() - t0)


def run_split(key, seed, data):
    xtr, ytr, xte, yte = data
    m, rng, t0 = METHODS[key][1](seed), np.random.default_rng(seed), time.time()
    test_sets = [np.isin(yte, t) for t in TASKS]
    R, R_task = [], []  # R[i][j]: class-IL acc on task j after learning task i
    for t in TASKS:
        idx = np.flatnonzero(np.isin(ytr, t))
        run_stream(m, xtr, ytr, rng.permutation(idx), ONLINE_LR[key])
        R.append([accuracy(m, xte[s], yte[s]) for s in test_sets])
        R_task.append([task_il_accuracy(m, xte[s], yte[s], c) for s, c in zip(test_sets, TASKS)])
    R = np.array(R)
    forgetting = float(np.mean([R[:, j].max() - R[-1, j] for j in range(len(TASKS) - 1)]))
    return dict(
        class_il_final=accuracy(m, xte, yte),  # all 10 digits, 10-way
        task_il_final=float(np.mean(R_task[-1])),
        forgetting=forgetting,
        R=R.tolist(),
        R_task=np.array(R_task).tolist(),
        seconds=time.time() - t0,
    )


RUNNERS = {"standard": run_standard, "iid": run_iid, "split": run_split}


# --------------------------------------------------------------------------
def tune_online(keys, grid=(1e-4, 3e-4, 1e-3, 3e-3)):
    """Pick each method's batch-1 learning rate on an iid single pass over
    50k training images, scored on the other 10k. Prints a table only."""
    xtr, ytr, _, _ = load_mnist()
    xa, ya, xv, yv = xtr[:50000], ytr[:50000], xtr[50000:], ytr[50000:]
    for k in keys:
        g = [0.005, 0.02, 0.05] if k == "hebb" else grid
        if k == "rf":
            continue
        for lr in g:
            m = METHODS[k][1](0)
            run_stream(m, xa, ya, np.random.default_rng(0).permutation(len(xa)), lr)
            print(f"  tune {k:5s} lr={lr:<7g} val={accuracy(m, xv, yv):.4f}", flush=True)


def summarise(results, seeds):
    def ms(vals):
        vals = np.array(vals) * 100
        return f"{vals.mean():.2f}" + (f" ± {vals.std():.2f}" if len(vals) > 1 else "")

    lines = [f"Seeds: {seeds}\n"]
    if "standard" in results or "iid" in results:
        lines += ["## Standard MNIST (test accuracy %)\n",
                  "| Method | Multi-epoch | Single pass, batch 1 | Time (multi-epoch) |",
                  "|---|---|---|---|"]
        for k in METHODS:
            st, ii = results.get("standard", {}).get(k), results.get("iid", {}).get(k)
            if not st and not ii:
                continue
            a = ms([r["test_acc"] for r in st]) if st else "-"
            b = ms([r["test_acc"] for r in ii]) if ii else "-"
            t = f"{np.mean([r['seconds'] for r in st]):.0f}s" if st else "-"
            lines.append(f"| {METHODS[k][0]} | {a} | {b} | {t} |")
    if "split" in results:
        lines += ["\n## Split-MNIST, online, single pass, batch 1\n",
                  "Class-IL = 10-way over all digits at the end (chance 10%; "
                  "remembering only the last task gives ≈20%). Task-IL = told which pair, pick one of two. "
                  "Forgetting = mean drop on tasks 1-4 from their best to their final class-IL accuracy.\n",
                  "| Method | Class-IL final | Task-IL final | Forgetting | iid single pass (no task order) |",
                  "|---|---|---|---|---|"]
        for k in METHODS:
            sp = results["split"].get(k)
            if not sp:
                continue
            ii = results.get("iid", {}).get(k)
            lines.append(
                f"| {METHODS[k][0]} | {ms([r['class_il_final'] for r in sp])} | "
                f"{ms([r['task_il_final'] for r in sp])} | {ms([r['forgetting'] for r in sp])} | "
                f"{ms([r['test_acc'] for r in ii]) if ii else '-'} |")
        lines += ["\n### Class-IL accuracy per task after each task (seed %d)\n" % seeds[0]]
        for k in METHODS:
            sp = results["split"].get(k)
            if not sp:
                continue
            lines += [f"**{METHODS[k][0]}**\n", "| after \\ on | " + " | ".join(f"{a}/{b}" for a, b in TASKS) + " |",
                      "|---" * (len(TASKS) + 1) + "|"]
            for i, row in enumerate(sp[0]["R"]):
                lines.append(f"| {TASKS[i][0]}/{TASKS[i][1]} | " + " | ".join(
                    f"{v * 100:.0f}" if j <= i else "·" for j, v in enumerate(row)) + " |")
            lines.append("")
    return "\n".join(lines)


def merge(dirs, out):
    seeds, results = [], {}
    for d in dirs:
        with open(os.path.join(d, "results.json")) as f:
            run = json.load(f)
        seeds = sorted(set(seeds) | set(run["seeds"]))
        for proto, by_method in run["results"].items():
            for k, rs in by_method.items():
                results.setdefault(proto, {}).setdefault(k, []).extend(rs)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(dict(seeds=seeds, results=results), f, indent=1)
    md = summarise(results, seeds)
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write("# Results\n\n" + md + "\n")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--protocols", nargs="+", default=["standard", "iid", "split"], choices=list(RUNNERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tune", action="store_true", help="only run the online learning-rate sweep")
    ap.add_argument("--out", default="results")
    ap.add_argument("--merge", nargs="+", metavar="DIR",
                    help="combine results.json from separate runs (e.g. one per seed) into --out")
    args = ap.parse_args()

    if args.tune:
        tune_online(args.methods)
        return
    if args.merge:
        merge(args.merge, args.out)
        return

    data = load_mnist()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    results = {}
    for proto in args.protocols:
        for k in args.methods:
            for s in args.seeds:
                r = RUNNERS[proto](k, s, data)
                results.setdefault(proto, {}).setdefault(k, []).append(r)
                brief = {kk: (round(v, 4) if isinstance(v, float) else None) for kk, v in r.items()}
                print(f"[{proto:8s}] {k:5s} seed={s} " + " ".join(
                    f"{kk}={v}" for kk, v in brief.items() if v is not None), flush=True)
                with open(path, "w") as f:
                    json.dump(dict(seeds=args.seeds, results=results), f, indent=1)

    md = summarise(results, args.seeds)
    with open(os.path.join(args.out, "RESULTS.md"), "w") as f:
        f.write("# Results\n\n" + md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
