"""Adaptation test: does a method forget when it should?

Split-MNIST as usual (every 2000, 0.8x), then the world changes: digits a and b
swap labels, and the stream shows only those two digits (relabelled) for n
more wake examples, on the same sleep clock. Every `check` examples:

  new        test a/b images predicted with the new (swapped) label: adaptation
  old        still predicted with the old label: perseveration
  others     class-IL accuracy on the other 8 digits: retention
  store_new  fraction of the store's replays labelled a/b that look like the other
             digit: how far the store itself has taken in the new mapping

Usage
  python swap.py                       # 4 conditions x 3 seeds -> results/swap/
  python swap.py --conds proto9_nonoise --out results/swap_x
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np

from data import load_mnist

CONDS = ["off", "proto9_nonoise", "ridge_p8res40", "p40_drop08_avg_3k"]
SHOW = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000]


def job(args):
    cond, seed, sw, dataset, every, anytime, batches = args
    import sleep as S
    # the replay ratio stays 0.8x: batches scale with the sleep interval
    r = S.run(cond, seed, load_mnist(dataset), sleep_every=every, sleep_batches=batches or max(1, round(50 * every / 2000)),
              sleep_bs=32, swap=sw, anytime=anytime)
    tr = r["swap"]
    print(f"{cond:18s} seed={seed} classIL={r['class_il_final']:.4f} " +
          " ".join(f"t{p['t']}:new={p['new']:.2f}/oth={p['others']:.2f}" for p in tr if p["t"] in (0, 2000, 8000)),
          flush=True)
    keep = ("class_il_final", "store_floats", "seconds", "swap", "confusion", "sleeps", "replayed", "surprises",
            "swap_sleeps", "swap_replayed", "task_il_final", "probes", "swap_probes", "trace", "pre_swap_trace_len", "anytime", "anytime_late", "anytime_min", "anytime_trace", "replay")
    return dict({k: r[k] for k in keep if k in r}, condition=cond, seed=seed)


def summarise(rows, conds, seeds, sw):
    import sleep as S
    out = [f"Split-MNIST, 0.8x replay, seeds {seeds}. After the five tasks digits {sw['a']} and {sw['b']} swap "
           f"labels and only they are seen ({sw['n']} examples). Sleeps continue on the clock (one per 2000 wake "
           "examples). Values are means over seeds, in %.\n"]
    for key, title in [("new", "New mapping on the swapped digits (adaptation)"),
                       ("old", "Old mapping on the swapped digits (perseveration)"),
                       ("others", "Other 8 digits, class-IL (retention)"),
                       ("store_new", "Store replays labelled a/b that look like the other digit")]:
        ts = [t for t in SHOW if t <= sw["n"]]
        out += [f"\n### {title}\n", "| Condition | " + " | ".join(f"t={t}" for t in ts) + " |",
                "|---|" + "---|" * len(ts)]
        for c in conds:
            rs = [r for r in rows if r["condition"] == c]
            vals = []
            for t in ts:
                v = [p[key] for r in rs for p in r["swap"] if p["t"] == t and p[key] is not None]
                vals.append(f"{np.mean(v) * 100:.1f}" if v else "–")
            out.append(f"| {S.LABELS[c]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--a", type=int, default=2)
    ap.add_argument("--b", type=int, default=3)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--check", type=int, default=500)
    ap.add_argument("--data", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--every", type=int, default=2000, help="sleep clock (replay stays 0.8x)")
    ap.add_argument("--anytime", type=int, default=0, help="class-IL every this many wake examples (pre-swap)")
    ap.add_argument("--batches", type=int, default=0, help="batches per sleep (default: keep 0.8x)")
    ap.add_argument("--out", default="results/swap")
    args = ap.parse_args()
    sw = dict(a=args.a, b=args.b, n=args.n, check=args.check)
    os.makedirs(args.out, exist_ok=True)
    load_mnist(args.data)
    with Pool(args.jobs) as pool:
        rows = pool.map(job, [(c, s, sw, args.data, args.every, args.anytime, args.batches) for c in args.conds for s in args.seeds], chunksize=1)
    json.dump(dict(seeds=args.seeds, swap=sw, rows=rows), open(os.path.join(args.out, "results.json"), "w"))
    md = summarise(rows, args.conds, args.seeds, sw)
    open(os.path.join(args.out, "RESULTS.md"), "w").write(f"# Label-swap adaptation test\n\nData: {args.data}.\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
