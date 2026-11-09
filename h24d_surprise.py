"""H24d, curiosity principles 1 and 2: retain surprises and replay them.

Only the two new cortices run. Consume H24c's world-RNG replay draws even
though its baseline cortex is not rerun, preserving the exact arrival stream.
Margin = pre-update FF goodness(true) - max goodness(other), without softmax.
The s2 threshold is numpy's linear 20th percentile of the preceding up-to-500
arrival margins (all arrivals, never replay). No confidence write at tick 1.
Predict both cortex and route before observing the label, writing, or training.
Store payload accounting follows H24c: 784 pixels + one label = 785 floats.
"""
import argparse
from collections import deque
import hashlib
import json
import os
import time
from multiprocessing import Pool

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from h24c_combined import DELTA, K, Q, REPLAY, DeletingStore, exposure_masks
from data import load_mnist
from learners import ForwardForward

BASE = ["ff", "exem", "gate", "ffr", "gate_r"]
NEW = ["ffr_s", "gate_s", "ffr_s2", "gate_s2"]
MARGIN = dict(formula="scores[y] - max(scores[c] for c != y)",
              percentile=20, window=500, interpolation="linear",
              history="preceding arrival margins, excluding current; shorter at startup",
              comparison="<=", empty_history_low_confidence=False)


class SurpriseStore:
    nearest = DeletingStore.nearest

    def __init__(self, capacity):
        self.x = np.empty((capacity, 784), np.float32)
        self.y = np.empty(capacity, int)
        self.n = 0

    def add(self, x, y):
        self.x[self.n], self.y[self.n] = x[0], y
        self.n += 1


def live(args):
    seed, ticks = args
    started = time.monotonic()
    xtr, ytr, xte, _ = load_mnist("mnist")
    rng = np.random.default_rng(seed)
    items = rng.choice(len(xte), K, replace=False)
    ex_x, ex_y = xte[items], rng.integers(0, 10, K)
    order = rng.permutation(len(xtr))
    models = [ForwardForward(loss="symba", homeo=0.001, seed=seed) for _ in range(2)]
    stores = [SurpriseStore(ticks) for _ in range(2)]
    replay_rngs = [np.random.default_rng(seed + offset) for offset in (2000, 3000)]
    history = deque(maxlen=500)
    seen = np.zeros(K, int)
    sizes, writes, firing, correct = [], [], [], []
    episodes, exposures, margins, thresholds = [], [], [], []
    stream_hash = hashlib.sha256()
    j = 0
    for t in range(ticks):
        if rng.random() < Q:
            i = int(rng.integers(K))
            x, y, epi = ex_x[i:i + 1], int(ex_y[i]), True
            seen[i] += 1
            expo = int(seen[i])
        else:
            index = order[j % len(order)]
            x, y, epi, expo = xtr[index][None], int(ytr[index]), False, 0
            j += 1
        stream_hash.update(x.tobytes())
        stream_hash.update(np.array([y, epi, expo], dtype="<i8").tobytes())
        # All prediction and margin measurements precede either model's update.
        scores = [model.scores(x)[0] for model in models]
        predictions = [int(s.argmax()) for s in scores]
        neighbours = [store.nearest(x) for store in stores]
        fired = [distance < DELTA for _, distance in neighbours]
        answers = [neighbours[k][0] if fired[k] else predictions[k] for k in range(2)]
        margin = float(scores[1][y] - np.max(np.delete(scores[1], y)))
        threshold = float(np.percentile(history, 20, method="linear")) if history else None
        low = threshold is not None and margin <= threshold
        write = [answers[0] != y, answers[1] != y or low]
        correct.append([predictions[0] == y, answers[0] == y,
                        predictions[1] == y, answers[1] == y])
        episodes.append(epi)
        exposures.append(expo)
        writes.append(write)
        firing.append(fired)
        margins.append(margin)
        thresholds.append(threshold)
        history.append(margin)
        for k, (model, store) in enumerate(zip(models, stores)):
            if write[k]:
                store.add(x, y)
            model.fit_batch(x, np.array([y]))
            if t >= REPLAY and store.n:
                indices = replay_rngs[k].integers(0, store.n, REPLAY)
                model.fit_batch(store.x[indices], store.y[indices])
        sizes.append([store.n for store in stores])
        if t >= REPLAY:
            rng.integers(0, t + 1, REPLAY)  # H24c baseline replay's world-RNG draws
        if (t + 1) % 400 == 0 or t + 1 == ticks:
            print(f"seed {seed}: {t+1}/{ticks}, stores {sizes[-1]}, "
                  f"elapsed {time.monotonic()-started:.1f}s", flush=True)
    ok, epi, expo = np.array(correct), np.array(episodes), np.array(exposures)
    write, fire, size = np.array(writes), np.array(firing), np.array(sizes)
    def acc(mask):
        return {route: float(ok[mask, k].mean() * 100) if mask.any() else None
                for k, route in enumerate(NEW)}
    masks = exposure_masks(epi, expo)
    result = dict(seed=seed, ticks=ticks, lifetime=acc(np.ones(ticks, bool)),
                  normal=acc(~epi), episodic=acc(epi),
                  by_exposure={e: acc(mask) for e, mask in masks.items()},
                  episodes=int(epi.sum()),
                  counts=dict(normal=int((~epi).sum()), exposure={e: int(m.sum()) for e, m in masks.items()}),
                  stream_sha256=stream_hash.hexdigest(), store={}, writes={}, gate_firing={},
                  margin_config=MARGIN, margins=margins, margin_thresholds=thresholds,
                  runtime_seconds=time.monotonic() - started)
    for k, route in enumerate(("gate_s", "gate_s2")):
        n = int(write[:, k].sum())
        n_epi = int(write[epi, k].sum())
        result["store"][route] = dict(peak=int(size[:, k].max()), final=int(size[-1, k]),
            peak_floats=int(size[:, k].max())*785, final_floats=int(size[-1, k])*785,
            count_over_time=size[:, k].tolist())
        result["writes"][route] = dict(total=n, episodes=n_epi, normal=n-n_epi,
            fraction_arrivals=n/ticks, fraction_episodes=n_epi/n if n else None,
            fraction_normal=(n-n_epi)/n if n else None)
        result["gate_firing"][route] = {
            name: float(fire[mask, k].mean()*100) if mask.any() else None
            for name, mask in [("lifetime", np.ones(ticks, bool)), ("normal", ~epi),
                               ("episodic_repeat", epi & (expo >= 2))]}
    return result


def reuse_baselines(results, reference):
    with open(reference) as inp:
        archived = {r["seed"]: r for r in json.load(inp)}
    for r in results:
        old = archived[r["seed"]]
        assert r["ticks"] == old["ticks"] and r["counts"] == old["counts"]
        assert r["episodes"] == old["episodes"]
        def finite(value):
            return value if value is None or np.isfinite(value) else None
        for metric in ("lifetime", "normal", "episodic"):
            r[metric].update({route: finite(old[metric][route]) for route in BASE})
        for e in r["by_exposure"]:
            r["by_exposure"][e].update({route: finite(old["by_exposure"][e][route]) for route in BASE})
        for route in ("gate", "gate_r"):
            r["gate_firing"][route] = old["gate_firing"][route]
        for route in ("exem", "gate_r"):
            r["store"][route] = old["store"][route]
        r["baseline_source"] = reference


def write_report(results, path, wall):
    def f(v):
        return "NA" if v is None or not np.isfinite(v) else f"{v:.3f}"
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"]*len(headers)) + " |"] +
            ["| " + " | ".join(map(str, row)) + " |" for row in rows])
    def metrics(r, route):
        return [r["lifetime"][route], r["normal"][route]] + [
            r["by_exposure"][e][route] for e in ("1", "2", "3-5", "6+")]
    sections = ["# H24d", table(
        ["Seeds", "Ticks / seed", "Q", "K", "Replay / arrival", "Pool", "Wall seconds"],
        [[", ".join(str(r["seed"]) for r in results), results[0]["ticks"], Q, K, REPLAY, 2, f(wall)]]),
        "Baseline source: `" + results[0]["baseline_source"] + "` (reused).",
        "Margin: `FF_score(true) - max(FF_score(other))`, before learning, without softmax. "
        "Low confidence: margin <= linear 20th percentile of the preceding up-to-500 arrival margins; "
        "current margin excluded; no low-confidence write with empty history. Replay begins at tick 9, "
        "with replacement from the route's store after any current write, as in H24c. "
        "Store payload: 785 floats/exemplar; margin history: at most 500 extra scalars for gate_s2."]
    headers = ["Route", "Lifetime %", "Normal %", "Episode 1 %", "Episode 2 %", "Episode 3–5 %", "Episode 6+ %"]
    sections += ["## Mean accuracy", table(headers, [[route] + [f(v) for v in np.mean(
        np.array([metrics(r, route) for r in results], dtype=float), axis=0)] for route in BASE + NEW])]
    for r in results:
        sections += [f"## Seed {r['seed']}", table(headers,
            [[route] + [f(v) for v in metrics(r, route)] for route in BASE + NEW])]
    sections += ["## Gate firing", table(["Seed", "Route", "Lifetime %", "Normal %", "Repeat episodes %"],
        [[r["seed"], route] + [f(r["gate_firing"][route][m]) for m in ("lifetime", "normal", "episodic_repeat")]
         for r in results for route in ("gate", "gate_r", "gate_s", "gate_s2")])]
    sections += ["## Store payload", table(
        ["Seed", "Route", "Peak exemplars", "Final exemplars", "Peak floats", "Final floats"],
        [[r["seed"], route] + [r["store"][route][m] for m in ("peak", "final", "peak_floats", "final_floats")]
         for r in results for route in ("exem", "gate_r", "gate_s", "gate_s2")])]
    sections += ["## Mean store payload", table(
        ["Route", "Peak exemplars", "Final exemplars", "Peak floats", "Final floats"],
        [[route] + [f(np.mean([r["store"][route][m] for r in results]))
                    for m in ("peak", "final", "peak_floats", "final_floats")]
         for route in ("gate_r", "gate_s", "gate_s2")])]
    sections += ["## Writes", table(
        ["Seed", "Route", "Total", "Episodes", "Normal", "Arrivals written %", "Episode writes %", "Normal writes %"],
        [[r["seed"], route] + [r["writes"][route][m] for m in ("total", "episodes", "normal")] +
         [f(r["writes"][route][m]*100) for m in ("fraction_arrivals", "fraction_episodes", "fraction_normal")]
         for r in results for route in ("gate_s", "gate_s2")])]
    sections += ["## Store samples", table(["Seed", "Tick", "exem / gate_r", "gate_s", "gate_s2"],
        [[r["seed"], t, t] + [r["store"][route]["count_over_time"][t-1] for route in ("gate_s", "gate_s2")]
         for r in results for t in sorted({min(400, r["ticks"]), r["ticks"]} |
                                           set(range(1000, r["ticks"]+1, 1000)))])]
    sections += ["## Cortex normal accuracy difference from ffr (percentage points)", table(
        ["Seed", "ffr_s − ffr", "ffr_s2 − ffr"], [[r["seed"]] +
        [f(r["normal"][route]-r["normal"]["ffr"]) for route in ("ffr_s", "ffr_s2")] for r in results]),
        table(["Seed", "Runtime seconds", "Episodes"],
              [[r["seed"], f(r["runtime_seconds"]), r["episodes"]] for r in results])]
    with open(path, "w") as out:
        out.write("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--out", default="results/h24d/results.json")
    ap.add_argument("--reference", default="results/h24c/results.json")
    a = ap.parse_args()
    started = time.monotonic()
    with Pool(2) as pool:
        results = pool.map(live, [(seed, a.ticks) for seed in a.seeds])
    wall = time.monotonic()-started
    reuse_baselines(results, a.reference)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for r in results:
        r["run_wall_seconds"] = wall
    with open(a.out, "w") as out:
        json.dump(results, out, indent=1, allow_nan=False)
    write_report(results, os.path.join(os.path.dirname(a.out), "RESULTS.md"), wall)
    print(f"Total wall seconds: {wall:.1f}", flush=True)
