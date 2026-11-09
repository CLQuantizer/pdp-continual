"""H24a: is the hippocampus useless, or were the world and the store wrong? (diagnostic; pre-registration in
docs/hippocampus_diagnosis.md)

World: the MNIST stream with true labels, plus, with probability Q per tick, an *episode*: one of K specific images
(from the MNIST test split, disjoint from the stream) shown with an arbitrary fixed label ("this face is called 7").
Episodes recur, about every K/Q ticks. Their labels are a fact about that item, not about its digit class, so no class
statistics predict them: only memory of the item does.
Routes, all answering before learning (prequential), all on the same arrivals:
  ff      the FF cortex (symba + homeostasis, as in life.py), wake update on every arrival
  proto   the core's ProtoStore (9 running-mean prototypes per class, noise 0), nearest prototype in pixel space
  exem    an exemplar store: every arrival kept verbatim (pattern-separated), nearest neighbour in pixel space
  gate    exem answers when its nearest neighbour is (near-)identical (squared distance < DELTA), else ff
  ffr     a second FF that also replays 8 exemplars (uniform from exem) after every arrival: consolidation
Usage: python3 h24_episodic.py [--seeds 6 7 8] [--ticks 10000]
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
import learners as L
import sleep as S
from data import load_mnist

Q, K, DELTA, REPLAY = 0.2, 200, 1.0, 8

def live(args):
    seed, ticks = args
    xtr, ytr, xte, _ = load_mnist()
    rng = np.random.default_rng(seed)
    items = rng.choice(len(xte), K, replace=False)
    ex_x, ex_y = xte[items], rng.integers(0, 10, K)               # episodes: specific images, arbitrary labels
    order = rng.permutation(len(xtr))
    ff = L.ForwardForward(loss="symba", homeo=0.001, seed=seed)
    ffr = L.ForwardForward(loss="symba", homeo=0.001, seed=seed)
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
        if proto.mu:
            cls = sorted(proto.mu); d = [((proto.mu[c] - x) ** 2).sum(1).min() for c in cls]; a_pr = cls[int(np.argmin(d))]
        else: a_pr = -1
        if t:
            d2 = ((mem_x[:t] - x) ** 2).sum(1); k = int(d2.argmin()); a_ex, nn = int(mem_y[k]), float(d2[k])
        else: a_ex, nn = -1, float("inf")
        fired = nn < DELTA
        a_gate = a_ex if fired else a_ff
        log.append((epi, expo, a_ff == y, a_pr == y, a_ex == y, a_gate == y, a_ffr == y, fired, nn))
        ff.fit_batch(x, np.array([y])); ffr.fit_batch(x, np.array([y]))
        proto.add(x, y); mem_x[t], mem_y[t] = x[0], y
        if t >= REPLAY:
            r = rng.integers(0, t + 1, REPLAY); ffr.fit_batch(mem_x[r], mem_y[r])
    A = np.array([row[:8] for row in log], dtype=float); nn = np.array([row[8] for row in log])
    epi, expo = A[:, 0] == 1, A[:, 1]
    routes = ["ff", "proto", "exem", "gate", "ffr"]
    def acc(mask): return {r: float(A[mask, 2 + i].mean() * 100) if mask.any() else float("nan") for i, r in enumerate(routes)}
    return dict(seed=seed, ticks=ticks, lifetime=acc(np.ones(len(A), bool)), normal=acc(~epi), episodic=acc(epi),
                by_exposure={"1": acc(epi & (expo == 1)), "2": acc(epi & (expo == 2)), "3-5": acc(epi & (expo >= 3) & (expo <= 5)),
                             "6+": acc(epi & (expo >= 6))},
                gate_fire_normal=float(A[~epi, 7].mean() * 100), gate_fire_episodic_rep=float(A[epi & (expo >= 2), 7].mean() * 100),
                nn_d2_normal_median=float(np.median(nn[~epi][1:])), episodes=int(epi.sum()))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8]); ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--out", default="results/h24a/results.json")
    a = ap.parse_args()
    with Pool(2) as pool: res = pool.map(live, [(s, a.ticks) for s in a.seeds])
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(f's{r["seed"]} lifetime ' + " ".join(f'{k} {v:.1f}' for k, v in r["lifetime"].items()))
        print('   normal ' + " ".join(f'{k} {v:.1f}' for k, v in r["normal"].items()))
        for e, d in r["by_exposure"].items(): print(f'   episodic exp {e:3s} ' + " ".join(f'{k} {v:.1f}' for k, v in d.items()))
        print(f'   gate fires: normal {r["gate_fire_normal"]:.2f}%, repeat episodes {r["gate_fire_episodic_rep"]:.1f}%; nn d2 median {r["nn_d2_normal_median"]:.1f}')
