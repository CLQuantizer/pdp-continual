"""Need-prioritised night replay (loop 19:40 hyp 1; Mattar & Daw 2018 need term).

The machine remembers, per label, the tick it last learned that label while awake. At night (circadian policy,
same 8 chunks x 10 batches x 32 as `night`) each replay item's class is drawn with probability proportional to
w(c) = max(ticks since c was last seen, floor)^p, then a prototype within the class as usual.
  night_need1*  p = 1 (linear in time away); _f100 floor 100 ticks, else 4000 (one day)
  night_needH*  p = 0.5 (softer)
Pre-registered (bg 0.02 world, seeds 0-2, control `night` 95.0 lifetime / 64.5 returns / 52.1 off season):
  prediction returns >= 68 and off season >= 56 at lifetime >= 94.9; kill if returns < 65.5 for both.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist

LF.POLICIES.update({
    "night_need1": dict(sleep="night", chunks=8, need=1.0, floor=4000),
    "night_needH": dict(sleep="night", chunks=8, need=0.5, floor=4000),
    "night_need1_f100": dict(sleep="night", chunks=8, need=1.0, floor=100),
    "night_needH_f100": dict(sleep="night", chunks=8, need=0.5, floor=100),
})
LF.LABELS.update({k: k for k in LF.POLICIES if k.startswith("night_need")})


class NeedMachine(LF.Machine):
    def __init__(self, cfg, seed, lr=1e-3):
        super().__init__(cfg, seed, lr)
        self.now, self.last, self.wlog = 0, {}, []

    def routes(self, x, y, t=0):
        self.now = t
        return super().routes(x, y, t)

    def woke(self):
        super().woke()

    def learn(self, x, y):
        self.last[int(y)] = self.now
        super().learn(x, y)

    def sleep_chunk(self):
        p = self.cfg.get("need")
        if not p:
            return super().sleep_chunk()
        cl = np.array(sorted(self.store.mu))
        w = np.array([max(self.now - self.last.get(int(c), self.now), self.cfg["floor"]) for c in cl], float) ** p
        w /= w.sum()
        self.wlog.append(float(w.max() / w.min()))
        for _ in range(LF.CHUNK):
            ys = self.rng.choice(cl, LF.BS, p=w)
            xs = np.empty((LF.BS, self.store.dim), np.float32)
            for c in np.unique(ys):
                idx = np.flatnonzero(ys == c)
                xs[idx] = self.store.sample_class(c, len(idx))
            self.m.fit_batch(xs, ys)
        self.replayed += LF.CHUNK * LF.BS


LF.Machine = NeedMachine


def job(args):
    import life
    box = {}
    orig = NeedMachine.__init__
    def init(self, *a, **k):
        orig(self, *a, **k); box["m"] = self
    NeedMachine.__init__ = init
    r = LF.job(args)
    wl = box["m"].wlog
    r["need_ratio"] = dict(median=float(np.median(wl)) if wl else 1.0, max=float(max(wl)) if wl else 1.0)
    print(args[0], args[1], "need max/min weight ratio", r["need_ratio"], flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conds", nargs="+", default=["night_need1_f100", "night_needH_f100"])
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--bg", type=float, default=0.02)
    ap.add_argument("--out", default="results/need_replay")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load_mnist("mnist")
    w = dict(LF.WORLD, bg=a.bg)
    jobs = [(c, s, a.days, w) for c in a.conds for s in a.seeds]
    with Pool(len(jobs)) as p:
        rows = p.map(job, jobs, chunksize=1)
    json.dump(dict(seeds=a.seeds, days=a.days, world=w, rows=rows), open(os.path.join(a.out, "results.json"), "w"))
    md = LF.summarise(rows, a.conds, a.seeds, a.days, w)
    open(os.path.join(a.out, "RESULTS.md"), "w").write(f"# Need-prioritised night replay (`need_replay.py`)\n\n{__doc__}\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
