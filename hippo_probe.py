"""Can the fast store answer? Nearest-prototype readout of sleep.ProtoStore on the Split-MNIST stream.
Tests whether a CLS-style 'hippocampal route' (the store answers, not only replays) is strong enough to
cover what slow waking (wake lr x0.3) loses: early accuracy on novel / swapped digits.
Usage: python hippo_probe.py --seeds 0 1 2 --k 9 40"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, time, numpy as np
from data import load_mnist
from sleep import ProtoStore

def nn(store, X):
    cs = sorted(store.mu); M = np.vstack([store.mu[c] for c in cs]); lab = np.concatenate([[c]*len(store.mu[c]) for c in cs])
    d = (X**2).sum(1)[:, None] - 2*X@M.T + (M**2).sum(1)[None]
    return lab[d.argmin(1)]

def run(seed, k, data):
    Xtr, ytr, Xte, yte = data
    rng = np.random.default_rng(seed); st = ProtoStore(k, rng, noise=0.0)
    preq, first, cnt = [], {}, {}
    for t in range(5):
        idx = np.flatnonzero((ytr == 2*t) | (ytr == 2*t+1)); idx = rng.permutation(idx)
        for i in idx:
            x, y = Xtr[i:i+1].astype(np.float32), int(ytr[i])
            if st.mu:
                ok = int(nn(st, x)[0] == y); preq.append(ok)
                cnt[y] = cnt.get(y, 0) + 1
                if cnt[y] <= 300: first.setdefault(y, []).append(ok)
            st.add(x, y)
    final = float((nn(st, Xte.astype(np.float32)) == yte).mean())
    # swap test: 2<->3 for 8000 examples, prequential on the swapped digits' first 300 and on others
    idx = rng.permutation(len(ytr))[:8000]; sw, oth = [], []
    for i in idx:
        x, y = Xtr[i:i+1].astype(np.float32), int(ytr[i]); y = {2: 3, 3: 2}.get(y, y)
        ok = int(nn(st, x)[0] == y)
        (sw if y in (2, 3) else oth).append(ok); st.add(x, y)
    return dict(seed=seed, k=k, preq=float(np.mean(preq)), novel_first300=float(np.mean([np.mean(v) for c, v in first.items() if c >= 2])),
                final_classIL=final, swap_first300=float(np.mean(sw[:300])), swap_last1000=float(np.mean(sw[-1000:])), others_during_swap=float(np.mean(oth)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--k", nargs="+", type=int, default=[9, 40])
    ap.add_argument("--out", default="results/hippo_route"); a = ap.parse_args()
    data = load_mnist("mnist"); rows = []
    for k in a.k:
        for s in a.seeds:
            t0 = time.time(); r = run(s, k, data); r["seconds"] = time.time() - t0; rows.append(r); print(r, flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)
