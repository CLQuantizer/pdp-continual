"""Cost/retention curve for wake/sleep FF (10 prototypes/class).

Sweeps how often it sleeps (every 500 / 2000 / 6000 wake examples) and how
much it replays per sleep, at four replay ratios (replayed examples per
wake example: ~3.2, 0.8, 0.2, 0.05). All intervals divide 60,000, so every
run ends with a sleep and the final numbers are measured at the same point
in the cycle (right after a sleep). The per-task matrices are measured at
task ends, which fall at different points in the cycle.

Pruning (downscaling) runs once per sleep, so with it on, frequent sleeps
also prune more. Use --condition proto10_nods to isolate the schedule.

Usage
  python sleep_cost.py                # full grid, 3 seeds, parallel
  python sleep_cost.py --condition proto10_nods --out results/sleep_cost_nods
  python sleep_cost.py --seeds 0 --jobs 4
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np

from data import load_mnist

BS = 32
# (sleep_every, sleep_batches); replay ratio = batches * BS / every
GRID = [
    (500, 50), (500, 12), (500, 3), (500, 1),
    (2000, 200), (2000, 50), (2000, 12), (2000, 3),
    (6000, 600), (6000, 150), (6000, 38), (6000, 9),
]


def job(args):
    every, batches, seed, cond = args
    import sleep as S
    r = S.run(cond, seed, load_mnist(), sleep_every=every, sleep_batches=batches, sleep_bs=BS)
    r.update(every=every, batches=batches, seed=seed, ratio=batches * BS / every, condition=cond)
    print(f"every={every:5d} batches={batches:4d} ratio={r['ratio']:.3f} seed={seed} "
          f"classIL={r['class_il_final']:.4f} forget={r['forgetting']:.3f} {r['seconds']:.0f}s", flush=True)
    return r


def summarise(rows, seeds):
    out = [f"Seeds: {seeds}. FF + sleep, 10 prototypes/class, Split-MNIST class-IL.\n",
           "Replay ratio = examples replayed during sleep per example seen awake.\n",
           "Class-IL final is measured right after the last sleep, which favours long "
           "intervals (a big sleep just before the test). Task-end avg is class-IL over "
           "all digits seen so far, at the end of tasks 2-5, averaged; those points fall "
           "anywhere in the sleep cycle.\n",
           "| Sleep every | Replay per sleep | Replay ratio | Class-IL final | Task-end avg | Forgetting | Task-IL final | Weight sparsity | Dead units |",
           "|---|---|---|---|---|---|---|---|---|"]
    for every, batches in GRID:
        rs = [r for r in rows if r["every"] == every and r["batches"] == batches]
        if not rs:
            continue
        for r in rs:
            R = np.array(r["R"])
            r["task_end_avg"] = float(np.mean([R[i, :i + 1].mean() for i in range(1, len(R))]))
        f = lambda k: f"{np.mean([r[k] for r in rs]) * 100:.2f} ± {np.std([r[k] for r in rs]) * 100:.2f}"
        out.append(f"| {every} | {batches * BS} | {batches * BS / every:.3f}× | {f('class_il_final')} | {f('task_end_avg')} | "
                   f"{f('forgetting')} | {f('task_il_final')} | {f('weight_sparsity')} | {f('dead_units')} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--condition", default="proto10", help="a sleep.py condition, e.g. proto10_nods (no downscaling)")
    ap.add_argument("--out", default="results/sleep_cost")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    load_mnist()  # download/cache once before forking
    tasks = [(e, b, s, args.condition) for e, b in GRID for s in args.seeds]
    with Pool(args.jobs) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(dict(seeds=args.seeds, rows=rows), open(os.path.join(args.out, "results.json"), "w"), indent=1)
    md = f"Condition: `{args.condition}`. " + summarise(rows, args.seeds)
    open(os.path.join(args.out, "RESULTS.md"), "w").write("# Sleep cost vs retention\n\n" + md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
