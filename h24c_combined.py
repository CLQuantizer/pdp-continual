"""H24c: combined exact-match routing, replay, and right-twice deletion.

Serves curiosity principles 1 and 2: retain experience and consolidate it.
Preserves H24a's world RNG and all five comparison routes exactly.
The deletion route has an independent cortex and replay RNG. Predictions and
correctness streaks use real arrivals before learning; replay is not an exposure.
Each arrival is stored, including duplicates, unless its cortex has just answered
that exact image correctly twice consecutively. Then all copies are removed.
A subsequent mistake resets the streak and stores that arrival again.
Store floats count only exemplar payloads: 784 pixels plus one label per copy.
"""
import argparse, hashlib, json, os, time
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import learners as L
import sleep as S
from data import load_mnist

Q, K, DELTA, REPLAY = 0.2, 200, 1.0, 8

def live(args):
    started = time.monotonic()
    seed, ticks = args
    xtr, ytr, xte, _ = load_mnist()
    rng = np.random.default_rng(seed)
    items = rng.choice(len(xte), K, replace=False)
    ex_x, ex_y = xte[items], rng.integers(0, 10, K)               # episodes: specific images, arbitrary labels
    order = rng.permutation(len(xtr))
    ff = L.ForwardForward(loss="symba", homeo=0.001, seed=seed)
    ffr = L.ForwardForward(loss="symba", homeo=0.001, seed=seed)
    ffrd = L.ForwardForward(loss="symba", homeo=0.001, seed=seed)
    rd_rng = np.random.default_rng(seed + 2000)
    rd = DeletingStore(ticks)
    sizes, deletion_log = [], []
    proto = S.ProtoStore(9, np.random.default_rng(seed + 1000), noise=0.0)
    mem_x, mem_y = np.empty((ticks, 784), np.float32), np.empty(ticks, int)
    seen = np.zeros(K, int)
    log = []                                                       # (episodic, exposure, ok per route..., gate_fired, nn_d2)
    j = 0
    for t in range(ticks):
        if rng.random() < Q:
            i = int(rng.integers(K)); x, y, epi = ex_x[i:i + 1], int(ex_y[i]), True; seen[i] += 1; expo = int(seen[i])
        else:
            x, y, epi, expo = xtr[order[j % len(order)]][None], int(ytr[order[j % len(order)]]), False, 0; j += 1
        a_ff = int(ff.scores(x).argmax()); a_ffr = int(ffr.scores(x).argmax())
        a_ffrd = int(ffrd.scores(x).argmax())
        if proto.mu:
            cls = sorted(proto.mu); d = [((proto.mu[c] - x) ** 2).sum(1).min() for c in cls]; a_pr = cls[int(np.argmin(d))]
        else: a_pr = -1
        if t:
            d2 = ((mem_x[:t] - x) ** 2).sum(1); k = int(d2.argmin()); a_ex, nn = int(mem_y[k]), float(d2[k])
        else: a_ex, nn = -1, float("inf")
        fired = nn < DELTA
        a_gate = a_ex if fired else a_ff
        a_gate_r = a_ex if fired else a_ffr
        a_rd, nn_rd = rd.nearest(x)
        fired_rd = nn_rd < DELTA
        a_gate_rd = a_rd if fired_rd else a_ffrd
        log.append((epi, expo, a_ff == y, a_pr == y, a_ex == y, a_gate == y,
                    a_ffr == y, a_gate_r == y, a_gate_rd == y, fired, fired_rd, nn))
        removed, restored = rd.observe(x, y, a_ffrd == y)
        deletion_log.append((removed, restored))
        ff.fit_batch(x, np.array([y])); ffr.fit_batch(x, np.array([y]))
        ffrd.fit_batch(x, np.array([y]))
        proto.add(x, y); mem_x[t], mem_y[t] = x[0], y
        sizes.append(rd.n)
        if t >= REPLAY:
            r = rng.integers(0, t + 1, REPLAY); ffr.fit_batch(mem_x[r], mem_y[r])
        if t >= REPLAY and rd.n:
            r = rd_rng.integers(0, rd.n, REPLAY)
            ffrd.fit_batch(rd.x[r], rd.y[r])
        if (t + 1) % 1000 == 0:
            print(f"seed {seed}: {t + 1}/{ticks}, retained {rd.n}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    A = np.array([row[:11] for row in log], dtype=float); nn = np.array([row[11] for row in log])
    epi, expo = A[:, 0] == 1, A[:, 1]
    routes = ["ff", "proto", "exem", "gate", "ffr", "gate_r", "gate_rd"]
    def acc(mask): return {r: float(A[mask, 2 + i].mean() * 100) if mask.any() else float("nan") for i, r in enumerate(routes)}
    return dict(seed=seed, ticks=ticks, lifetime=acc(np.ones(len(A), bool)), normal=acc(~epi), episodic=acc(epi),
                by_exposure={"1": acc(epi & (expo == 1)), "2": acc(epi & (expo == 2)), "3-5": acc(epi & (expo >= 3) & (expo <= 5)),
                             "6+": acc(epi & (expo >= 6))},
                gate_fire_normal=float(A[~epi, 9].mean() * 100), gate_fire_episodic_rep=float(A[epi & (expo >= 2), 9].mean() * 100),
                nn_d2_normal_median=float(np.median(nn[~epi][1:])), episodes=int(epi.sum()),
                counts=dict(normal=int((~epi).sum()),
                            exposure={e: int(m.sum()) for e, m in exposure_masks(epi, expo).items()}),
                gate_firing={r: dict(lifetime=float(A[:, col].mean() * 100),
                                    normal=float(A[~epi, col].mean() * 100),
                                    episodic_repeat=float(A[epi & (expo >= 2), col].mean() * 100)
                                    if (epi & (expo >= 2)).any() else None)
                             for r, col in [("gate", 9), ("gate_r", 9), ("gate_rd", 10)]},
                store={r: dict(peak=max(trace), final=trace[-1],
                               peak_floats=785 * max(trace), final_floats=785 * trace[-1],
                               count_over_time=trace)
                       for r, trace in [("exem", list(range(1, ticks + 1))),
                                        ("gate_r", list(range(1, ticks + 1))), ("gate_rd", sizes)]},
                deletion=dict(removed_exemplars=sum(d[0] for d in deletion_log),
                              restored_arrivals=sum(d[1] for d in deletion_log),
                              tracked_identities=len(rd.streak)),
                runtime_seconds=time.monotonic() - started)

def exposure_masks(epi, expo):
    return {"1": epi & (expo == 1), "2": epi & (expo == 2),
            "3-5": epi & (expo >= 3) & (expo <= 5), "6+": epi & (expo >= 6)}


class DeletingStore:
    """Packed exemplar copies; digest metadata never supplies answers or replay."""
    def __init__(self, capacity):
        self.x = np.empty((capacity, 784), np.float32)
        self.y = np.empty(capacity, int)
        self.keys = []
        self.streak = {}
        self.deleted = set()
        self.n = 0

    def nearest(self, x):
        if not self.n:
            return -1, float("inf")
        d2 = ((self.x[:self.n] - x) ** 2).sum(1)
        k = int(d2.argmin())
        return int(self.y[k]), float(d2[k])

    def observe(self, x, y, correct):
        key = hashlib.sha256(x.tobytes()).digest()
        self.streak[key] = min(2, self.streak.get(key, 0) + 1) if correct else 0
        removed, restored = 0, 0
        if self.streak[key] >= 2:
            # Remove every duplicate, keeping surviving arrival order for NN ties.
            keep = np.array([k != key for k in self.keys], dtype=bool)
            removed = self.n - int(keep.sum())
            if removed:
                self.x[:self.n - removed] = self.x[:self.n][keep]
                self.y[:self.n - removed] = self.y[:self.n][keep]
                self.keys = [k for k in self.keys if k != key]
                self.n -= removed
            self.deleted.add(key)
        else:
            restored = int(key in self.deleted and not correct)
            self.deleted.discard(key)
            self.x[self.n], self.y[self.n] = x[0], y
            self.keys.append(key)
            self.n += 1
        return removed, restored


def write_report(res, path, wall):
    routes = ["ff", "proto", "exem", "gate", "ffr", "gate_r", "gate_rd"]
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "| " + " | ".join(["---"] * len(headers)) + " |"] +
                         ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])
    def metrics(r, route):
        return [r["lifetime"][route], r["normal"][route]] + [
            r["by_exposure"][e][route] for e in ["1", "2", "3-5", "6+"]]
    def f(v):
        return f"{v:.3f}"
    sections = ["# H24c", table(["Seeds", "Ticks / seed", "Q", "K", "Replay / arrival", "Pool", "Wall seconds"],
                 [[", ".join(str(r["seed"]) for r in res), res[0]["ticks"], Q, K, REPLAY, 2, f(wall)]])]
    headers = ["Route", "Lifetime %", "Normal %", "Episode 1 %", "Episode 2 %", "Episode 3–5 %", "Episode 6+ %"]
    sections += ["## Mean accuracy", table(headers, [[route] + [f(v) for v in np.mean(
        [metrics(r, route) for r in res], axis=0)] for route in routes])]
    for r in res:
        sections += [f"## Seed {r['seed']}", table(headers, [[route] + [f(v) for v in metrics(r, route)] for route in routes])]
    sections += ["## Gate firing", table(["Seed", "Route", "Lifetime %", "Normal %", "Repeat episodes %"],
        [[r["seed"], route] + [f(r["gate_firing"][route][m]) for m in ["lifetime", "normal", "episodic_repeat"]]
         for r in res for route in ["gate", "gate_r", "gate_rd"]])]
    sections += ["## Store payload (785 floats / exemplar)", table(
        ["Seed", "Route", "Peak exemplars", "Final exemplars", "Peak floats", "Final floats"],
        [[r["seed"], route] + [r["store"][route][m] for m in ["peak", "final", "peak_floats", "final_floats"]]
         for r in res for route in ["exem", "gate_r", "gate_rd"]])]
    sections += ["## Deletion and runtime", table(
        ["Seed", "Removed copies", "Restored arrivals", "Tracked identities", "Runtime seconds", "Episodes"],
        [[r["seed"], r["deletion"]["removed_exemplars"], r["deletion"]["restored_arrivals"],
          r["deletion"]["tracked_identities"], f(r["runtime_seconds"]), r["episodes"]] for r in res])]
    sections += ["## Store samples", table(["Seed", "Tick", "exem / gate_r", "gate_rd"],
        [[r["seed"], t, t, r["store"]["gate_rd"]["count_over_time"][t-1]]
         for r in res for t in sorted(set([min(400, r["ticks"])] + list(range(1000, r["ticks"] + 1, 1000)) + [r["ticks"]]))])]
    reference = "results/h24a/results.json"
    if os.path.exists(reference):
        with open(reference) as inp:
            archived = {r["seed"]: r for r in json.load(inp)}
        matched = [r for r in res if r["seed"] in archived
                   and r["ticks"] == archived[r["seed"]]["ticks"]]
        if matched:
            baseline = ["ff", "proto", "exem", "gate", "ffr"]
            sections += ["## Lifetime minus archived H24a (percentage points)", table(
                ["Seed"] + baseline,
                [[r["seed"]] + [f(r["lifetime"][route] - archived[r["seed"]]["lifetime"][route])
                                 for route in baseline] for r in matched])]
    with open(path, "w") as out:
        out.write("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8]); ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--out", default="results/h24c/results.json")
    a = ap.parse_args()
    started = time.monotonic()
    with Pool(2) as pool: res = pool.map(live, [(s, a.ticks) for s in a.seeds])
    wall = time.monotonic() - started
    for r in res:
        r["run_wall_seconds"] = wall
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as out:
        json.dump(res, out, indent=1)
    if a.ticks >= 10000:
        write_report(res, os.path.join(os.path.dirname(a.out), "RESULTS.md"), wall)
    print(f"Total wall seconds: {time.monotonic() - started:.1f}", flush=True)
    for r in res:
        print(f's{r["seed"]} lifetime ' + " ".join(f'{k} {v:.1f}' for k, v in r["lifetime"].items()))
        print('   normal ' + " ".join(f'{k} {v:.1f}' for k, v in r["normal"].items()))
        for e, d in r["by_exposure"].items(): print(f'   episodic exp {e:3s} ' + " ".join(f'{k} {v:.1f}' for k, v in d.items()))
        print(f'   gate fires: normal {r["gate_fire_normal"]:.2f}%, repeat episodes {r["gate_fire_episodic_rep"]:.1f}%; nn d2 median {r["nn_d2_normal_median"]:.1f}')
