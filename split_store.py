"""Split store: remap in pixels, answer in cortex code (loop 18:00 UTC+8 31 Oct, pre-registered before running;
17:00 hypothesis 1).

reread_store.py showed the cortex-read store is a better answerer (store-only lifetime 93.1-93.5 vs 91.4) but a worse
remapper (2.3-2.7 false remaps per run). Split the roles: ONE pixel z5 store (a25w250h2k2_c1.5_uinf_z5) observes and
remaps exactly as before (0 false remaps by construction), and answers three ways off the same prototypes: pixel
distance (control, must reproduce rival_s49 / reread z5), code layer 1 (sp1) and code layers 1+2 (sp12), code =
L2-normalised ReLU of the CURRENT FF layers under a neutral label. The mq gate's store-margin threshold s is re-set in
code units: sweep q in (0.2, 0.3) x s in (0.1 .. 0.8); choose on seeds 0-2 (best mq lifetime), confirm on seeds 3-5.
  H1 (architecture/routing): held-out mq lifetime >= 96.0 AND returns >= 75 (pixel z5 mq30_s2: 96.0 / 72.0).
     Kill: lifetime < 96.0 or returns < 74.
  H1b: swap first 300 not below pixel z5 by more than 1.0.
Usage: OMP_NUM_THREADS=1 python3 split_store.py
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import life as LF
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist
from gate_probe import rolling_quantile
from reread_store import Coder

SEEDS = [0, 1, 2, 3, 4, 5]
WEX = dict(bg=0.02, swap=(4, 9))
K = "a25w250h2k2_c1.5_uinf_z5"
VARS = ["pix", "sp1", "sp12"]
QS, SS = (0.2, 0.3), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)


def code_answer(st, coder, x, L):
    cls = sorted(st.mu)
    mu = np.concatenate([st.mu[c] for c in cls]); ys = np.concatenate([np.full(len(st.mu[c]), c) for c in cls])
    codes = coder(np.vstack([mu, x]), L)
    d = ((codes[:-1] - codes[-1:]) ** 2).sum(1); j = d.argmin(); s = int(ys[j]); other = d[ys != s]
    return s, (float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0)


def run(args):
    seed, logrows = args
    w = dict(LF.WORLD, **WEX)
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    mach = LF.Machine(dict(LF.DEFAULTS, **LF.POLICIES["none"]), seed)
    coder = Coder(mach.m)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K)
    out = {v: [] for v in VARS}
    t0, i, bad = time.time(), 0, 0
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        if i % 50 == 0:
            bad += int(int(mach.m.scores(x)[0].argmax()) != int(logrows[i][5]))
        if st.mu:
            out["pix"].append((t, y) + RP.answer(st, x))
            out["sp1"].append((t, y) + code_answer(st, coder, x, 1))
            out["sp12"].append((t, y) + code_answer(st, coder, x, 2))
        else:
            for v in VARS:
                out[v].append((t, y) + RP.answer(st, x))
        st.t = t; st.observe(x, y); st.add(x, y)
        mach.learn(x, y)
        i += 1
        if i % 5000 == 0:
            print(f"seed={seed} arrivals={i} {time.time() - t0:.0f}s mismatch={bad}", flush=True)
    sw = world.swap_at
    rm = dict(n=st.remaps, pre_swap=sum(tt < sw for tt, _, _ in st.log), rivals=st.rivals,
              post_lat={str(c): next((tt - sw for tt, s_, _ in st.log if tt >= sw and s_ == c), None) for c in w["swap"]})
    print(f"seed={seed} done {time.time() - t0:.0f}s mismatch={bad}/{(i + 49) // 50} remaps={rm}", flush=True)
    return seed, {v: np.array(o) for v, o in out.items()}, rm, bad


def score(R, S, SM, w, swap_at, novel_at):
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    routes = dict(store=dis)
    for q in QS:
        lo = FFM < rolling_quantile(FFM, q)
        for s in SS:
            routes[f"mq{int(q*100)}_s{int(round(s*10))}"] = dis & prot & lo & (SM >= s)
    res = {}
    for g, use in routes.items():
        P = np.where(use, S, F); ok = P == Y
        first = lambda dg, at: float(ok[np.flatnonzero(np.isin(C, dg) & (T >= at))[:300]].mean())
        res[g] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()),
                      swap=first(list(w["swap"]), swap_at), fired=float(use.mean()))
    return res


def main():
    J = json.load(open("results/gate_probe_s49/results.json"))
    logs = {r["seed"]: r for r in J["raw"] if r["seed"] in SEEDS}
    load_mnist("mnist")
    with Pool(len(SEEDS)) as p:
        res = p.map(run, [(s, logs[s]["rows"]) for s in SEEDS], chunksize=1)
    w = dict(LF.WORLD, **WEX); allres = {}
    for seed, out, rm, bad in res:
        r = logs[seed]; R = np.array(r["rows"])
        assert np.array_equal(out["pix"][:, 1], R[:, 2])
        allres[seed] = {v: score(R, o[:, 2], o[:, 3], w, r["swap_at"], r["novel_at"]) for v, o in out.items()}
        allres[seed]["_remap"] = rm; allres[seed]["_mismatch"] = bad
    mean = lambda v, g, k, ss: float(np.mean([allres[s][v][g][k] for s in ss]))
    md = ["# Split store: remap in pixels, answer in cortex code (`split_store.py`)\n", __doc__,
          "\n## Remapping (shared pixel z5 store)\n",
          f"Pre-swap false remaps per seed: {[allres[s]['_remap']['pre_swap'] for s in SEEDS]}; FF mismatches "
          f"{[allres[s]['_mismatch'] for s in SEEDS]}; first post-swap remap mean "
          f"{np.mean([v for s in SEEDS for v in allres[s]['_remap']['post_lat'].values() if v is not None]):.0f} ticks.\n"]
    sel = {}
    for v in VARS:
        gs = [g for g in allres[0][v] if g != "store"]
        sel[v] = max(gs, key=lambda g: mean(v, g, "lifetime", [0, 1, 2]))
    md.append("Selected on seeds 0-2 (best mq lifetime): " + ", ".join(f"{v} {g}" for v, g in sel.items()) + "\n")
    for name, ss in (("selection seeds 0-2", [0, 1, 2]), ("held-out seeds 3-5", [3, 4, 5]), ("all seeds 0-5", SEEDS)):
        md += [f"\n## {name}, s49, mean %\n", "| Answer | Route | Lifetime | Returns | Swap first 300 | Store answers |", "|---|---|---|---|---|---|"]
        for v in VARS:
            for g in ["store", "mq30_s2", "mq20_s2", sel[v]]:
                md.append(f"| {v} | {g} | {mean(v,g,'lifetime',ss)*100:.2f} | {mean(v,g,'returns',ss)*100:.1f} | {mean(v,g,'swap',ss)*100:.1f} | {mean(v,g,'fired',ss)*100:.2f} |")
    md += ["\n## Full sweep, all seeds 0-5 (lifetime / returns / fired %)\n", "| Answer | " + " | ".join(g for g in allres[0]["pix"] if g != "store") + " |", "|---" * (len(allres[0]["pix"])) + "|"]
    for v in VARS:
        md.append(f"| {v} | " + " | ".join(f"{mean(v,g,'lifetime',SEEDS)*100:.1f}/{mean(v,g,'returns',SEEDS)*100:.1f}/{mean(v,g,'fired',SEEDS)*100:.1f}" for g in allres[0][v] if g != "store") + " |")
    out_dir = "results/split_store"; os.makedirs(out_dir, exist_ok=True)
    json.dump({"sel": sel, "raw": {str(k): v for k, v in allres.items()}}, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
