"""Gain-prioritised night replay (loop 18:00 hyp 1; Mattar & Daw 2018 gain term).

By day the machine logs "gain" events: the store and FF disagree, FF's goodness margin is in its causal
bottom 10% (last 500 answered arrivals), and the store's relative margin is >= 0.2 (the mq10_s2 condition
from gate_probe.py). Each event records the store slot (class, prototype index) nearest to the input.
At night (circadian policy, same 8 chunks x 10 batches x 32 as `night`) a fraction `gfrac` of every
replay batch is drawn from the logged slots (their current prototype, labelled with the slot's class);
the rest is uniform store replay as before. The buffer is cleared after each sleep.
  gainU*  unlabelled events (store's claim may be wrong)
  gainL*  label-verified events: the arriving label agrees with the store and FF was wrong
          (the label is used only to choose what to replay, never to answer)
Pre-registered (bg 0.02 world, seeds 0-2, compare to night 95.0 lifetime / 64.5 returns):
  prediction returns >= 70 and lifetime >= 95.5; kill if returns < 66 for every variant.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist

LF.POLICIES.update({
    "night_gainU50": dict(sleep="night", chunks=8, gain="U", gfrac=0.5),
    "night_gainL50": dict(sleep="night", chunks=8, gain="L", gfrac=0.5),
    "night_gainL100": dict(sleep="night", chunks=8, gain="L", gfrac=1.0),
})
LF.LABELS.update({k: k for k in LF.POLICIES if k.startswith("night_gain")})


class GainMachine(LF.Machine):
    def __init__(self, cfg, seed, lr=1e-3):
        super().__init__(cfg, seed, lr)
        self.margins, self.events, self.n_events = deque(maxlen=500), [], 0

    def learn(self, x, y):
        cfg = self.cfg
        if cfg.get("gain") and self.store.mu:
            sc = self.m.scores(x)[0]; o = np.sort(sc)
            f, ffm = int(sc.argmax()), float(o[-1] - o[-2])
            mu, ys = self._protos()
            d = ((mu - x) ** 2).sum(1); j = int(d.argmin()); s = int(ys[j])
            other = d[ys != s]
            sm = float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0
            low = len(self.margins) == 500 and ffm < np.quantile(self.margins, 0.1)
            if s != f and low and sm >= 0.2 and (cfg["gain"] == "U" or (s == y and f != y)):
                slot = j - int(np.flatnonzero(ys == s)[0])
                self.events.append((s, slot)); self.n_events += 1
            self.margins.append(ffm)
        super().learn(x, y)

    def sleep_chunk(self):
        g, k = self.cfg.get("gain"), int(round(self.cfg.get("gfrac", 0) * LF.BS))
        for _ in range(LF.CHUNK):
            if g and self.events and k:
                pick = self.rng.integers(len(self.events), size=k)
                xg = np.stack([self.store.mu[c][min(sl, len(self.store.mu[c]) - 1)] for c, sl in (self.events[i] for i in pick)])
                yg = np.array([self.events[i][0] for i in pick])
                if k < LF.BS:
                    xu, yu = self.store.sample(LF.BS - k)
                    xg, yg = np.concatenate([xg, xu]), np.concatenate([yg, yu])
                self.m.fit_batch(xg.astype(np.float32), yg)
            else:
                x, y = self.store.sample(LF.BS)
                self.m.fit_batch(x, y)
        self.replayed += LF.CHUNK * LF.BS

    def woke(self):
        super().woke()
        self.events = []


LF.Machine = GainMachine


def job(args):
    r = LF.job(args)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conds", nargs="+", default=["night_gainU50", "night_gainL50", "night_gainL100"])
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--bg", type=float, default=0.02)
    ap.add_argument("--out", default="results/gain_replay")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load_mnist("mnist")
    w = dict(LF.WORLD, bg=a.bg)
    jobs = [(c, s, a.days, w) for c in a.conds for s in a.seeds]
    with Pool(len(jobs)) as p:
        rows = p.map(job, jobs, chunksize=1)
    json.dump(dict(seeds=a.seeds, days=a.days, world=w, rows=rows), open(os.path.join(a.out, "results.json"), "w"))
    md = LF.summarise(rows, a.conds, a.seeds, a.days, w)
    open(os.path.join(a.out, "RESULTS.md"), "w").write(f"# Gain-prioritised night replay (`gain_replay.py`)\n\n{__doc__}\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
