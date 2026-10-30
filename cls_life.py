"""CLS complementary routing in a life (life.py world, default bg 0.2, 16 days).

The machine is life.py's (FF + 9-proto store, a sleep policy); learning is untouched. Every answered
arrival, all cls_route.py answer routes are computed on the same network and store and scored (paired),
so lifetime per route = mean over all arrivals (misses count wrong, as in life.py). The 'ff' route is
exactly life.py's machine. Target from results/life: never-sleep lifetime 94.1.

Usage: python cls_life.py --conds none --seeds 0 1 2 --days 16 --jobs 3 --out results/cls_life
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from multiprocessing import Pool
import numpy as np
import life
import cls_route as C


class RoutedMachine(life.Machine):
    def __init__(self, cfg, seed, lr=1e-3):
        super().__init__(cfg, seed, lr)
        self.R = C.Router(); self.names = C.routes_names()
        self.ok = {r: [] for r in self.names}; self.last = None; self.ticks_since_probe = 0

    def act(self, x):
        ff = self.m.scores(x).argmax(1)
        if self.store.mu:
            sp, sm = C.store_nn(self.store, x.astype(np.float32))
        else:
            sp, sm = ff.copy(), np.zeros(1)
        self.last = (ff, sp, sm)
        return int(ff[0])

    def learn(self, x, y):
        ff, sp, sm = self.last
        for r, p in self.R.answer(ff, sp, sm, np.array([y])).items():
            self.ok[r].append(int(p[0] == y))
        if self.store.mu:
            self.R.update(int(ff[0]), int(sp[0]), y)
        else:
            self.R.count[y] += 1; self.R.recent.append(y)
        super().learn(x, y)
        self.ticks_since_probe += 1
        if self.ticks_since_probe % 250 == 0:
            self.R.probe(self.m, self.store)


def job(args):
    cond, seed, days, bg = args
    holder = {}
    orig = life.Machine
    class M(RoutedMachine):
        def __init__(s, *a, **k):
            super().__init__(*a, **k); holder["m"] = s
    life.Machine = M
    try:
        r = life.live(cond, seed, days, dict(bg=bg))
    finally:
        life.Machine = orig
    m = holder["m"]; n_arr = r["arrivals"]
    out = dict(condition=cond, seed=seed, days=days, bg=bg, arrivals=n_arr, life_lifetime=r["lifetime"],
               missed=r["missed"], seconds=r["seconds"],
               lifetime={k: float(np.sum(v)) / n_arr for k, v in m.ok.items()},
               answered={k: float(np.mean(v)) for k, v in m.ok.items()})
    print(f"done {cond} seed={seed} life={r['lifetime']:.4f} " +
          " ".join(f"{k}={v*100:.1f}" for k, v in out["lifetime"].items()), flush=True)
    return out


def summarise(rows):
    names = C.routes_names(); out = []
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        out += [f"\n## Policy `{cond}`, bg {rs[0]['bg']}, {rs[0]['days']} days, seeds {sorted(r['seed'] for r in rs)} (mean %)\n",
                "| Route | Lifetime | Answered |", "|---|---|---|"]
        for n in names:
            out.append(f"| {n} | {100*np.mean([r['lifetime'][n] for r in rs]):.2f} | {100*np.mean([r['answered'][n] for r in rs]):.2f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["none"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--bg", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="results/cls_life")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["condition"], r["seed"]) for r in rows}
    todo = [(c, s, a.days, a.bg) for c in a.conds for s in a.seeds if (c, s) not in done]
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(job, todo):
            rows.append(r); json.dump(rows, open(path, "w"))
            open(os.path.join(a.out, "RESULTS.md"), "w").write("# CLS routing in a life (`cls_life.py`)\n" + summarise(rows) + "\n")


if __name__ == "__main__":
    main()
