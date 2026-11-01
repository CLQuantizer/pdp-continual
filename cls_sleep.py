"""Minimal CLS wake/sleep dual (loop 09:40 UTC+8 1 Nov; Gall's law: one working sleep path on the working default).

Hippocampus = the default store (a25w250h2k2_c1.5_uinf_z5): one-shot episodic writes and its own remaps, by day only.
Neocortex  = the FF network. By day it answers and learns from the world at wake LR x`wl`.
Answer     = route c3 (store answers when it disagrees with FF and its relative margin >= 0.3, or the mq30_s2 gate):
             the combined system, scored in-loop. FF-only and store-only routes are scored on the same arrivals.
Sleep      = circadian: at the start of each night the machine stops answering for `chunks` chunks (10 batches of 32,
             40 ticks each; arrivals then are misses and nothing is learned). The cortex learns ONLY from replay:
  uni      uniform replay of store prototypes (life.py's `night`, control)
  swr      sharp-wave-ripple selective replay: by day the hippocampus tags episodes the cortex got wrong (prediction
           error; FIFO cap `cap`, cleared after each sleep). At night each tagged episode is replayed as the store's
           CURRENT nearest prototype with the store's label (so committed remaps are what the cortex is taught),
           interleaved 50/50 with uniform store prototypes (CLS interleaving protects old schemas).
Conditions (s49 world: bg 0.02, 4<->9 swap; seeds 3-5, held out for the c3 router; nothing tuned here):
  none        never sleeps, wake LR x1 (baseline; must reproduce the offline c3 numbers 96.12 / 74.8)
  uni8        uniform night replay, 8 chunks, wake LR x1
  swr2        selective, 2 chunks, wake LR x1
  swr8        selective, 8 chunks, wake LR x1
  swr2_wl3    selective, 2 chunks, wake LR x0.3 (slow cortex + sleep: the textbook CLS split)
Pre-registered (combined c3 route, vs `none`, mean of seeds 3-5):
  PASS if some sleep condition has lifetime >= none - 0.10 AND (returns >= none + 1.0 OR swap300 >= none + 2.0 with
  returns >= none - 0.5).  KILL if every sleep condition misses that.  Secondary (CLS mechanism check, not a gate): swr
  should lift FF-only returns over `none` by >= 3 (the cortex actually consolidates what the store holds).
Replication (reported, same bars): bg002 world (bg 0.02, 2<->3 swap), seeds 6-8.
Result 09:40 run (both worlds): KILL. No sleep condition met the bar (see results/cls_sleep/RESULTS.md).
Follow-up H2, pre-registered 10:25 UTC+8 1 Nov before running ("temporary hippocampus"; s49 seeds 3-5):
  Why sleep can't win above: the store never forgets, so returns are already answered by the store and sleep has
  nothing to hand over. CLS assumes the hippocampus is temporary. `*_f2k` drops every class unseen for >= 2000 ticks
  (= RET_GAP), so on a return only the cortex can know the class.
  PASS if swr2_f2k or uni8_f2k has combined returns >= none_f2k + 3.0 AND lifetime >= none_f2k - 0.10.
  KILL otherwise (sleep does not even rescue a forgetful hippocampus -> the FF cortex can't consolidate usefully).
  Reported, not a gate: gap of the best sleep_f2k to `none` (permanent store, never sleeps).
Usage: OMP_NUM_THREADS=1 python3 cls_sleep.py [--world s49|bg002] [--seeds 3 4 5] [--conds ...] [--days 16]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, time
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
import learners as L
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist

BASEK = "a25w250h2k2_c1.5_uinf_z5"
WORLDS = {"s49": dict(bg=0.02, swap=(4, 9)), "bg002": dict(bg=0.02)}
WORLD = WORLDS["s49"]
CONDS = {
    "none": dict(sleep=None, wl=1.0),
    "uni8": dict(sleep="uni", chunks=8, wl=1.0),
    "swr2": dict(sleep="swr", chunks=2, wl=1.0),
    "swr8": dict(sleep="swr", chunks=8, wl=1.0),
    "swr2_wl3": dict(sleep="swr", chunks=2, wl=0.3),
    # 10:00 loop: temporary hippocampus. Every 100 ticks (and after each night's replay) the store drops every class not observed
    # for >= F ticks (F = RET_GAP = 2000, so every return finds its class gone from the store).
    "none_f2k": dict(sleep=None, wl=1.0, forget=2000),
    "swr2_f2k": dict(sleep="swr", chunks=2, wl=1.0, forget=2000),
    "uni8_f2k": dict(sleep="uni", chunks=8, wl=1.0, forget=2000),
}
CAP, CHUNK_TICKS = 128, 40


def protos(st):
    mu = np.concatenate([st.mu[c] for c in sorted(st.mu)])
    ys = np.concatenate([np.full(len(st.mu[c]), c) for c in sorted(st.mu)])
    return mu, ys


def live(args):
    cond, seed, days, wk = args
    cfg = CONDS[cond]
    w = dict(LF.WORLD, **WORLDS[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = days * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    lr = 1e-3
    ff = L.ForwardForward(lr=lr, loss="symba", homeo=0.001, seed=seed)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=BASEK)
    rng = np.random.default_rng(seed + 3000)
    tags = deque(maxlen=CAP)
    cnt = np.zeros(10); ffms = deque(maxlen=500)
    last_seen, returning, rows = {}, {}, []
    n_sleep = replayed = tagged_replayed = 0
    t0, t = time.time(), 0
    forgot = 0

    def forget():
        nonlocal forgot
        F = cfg.get("forget")
        if not F:
            return
        for c in [c for c in list(st.mu) if t - last_seen.get(c, -10 ** 9) >= F]:
            if len(st.mu) <= 1:
                break
            for D in (st.mu, st.var, st.n, st.hist, st.rad):
                D.pop(c, None)
            forgot += 1

    def sleep_chunk():
        nonlocal replayed, tagged_replayed
        mu, ys = protos(st)
        for _ in range(LF.CHUNK):
            if cfg["sleep"] == "swr" and tags:
                k = LF.BS // 2
                xs = np.concatenate([tags[i] for i in rng.choice(len(tags), k)])
                j = ((mu[None] - xs[:, None]) ** 2).sum(2).argmin(1)   # the store's current memory of each episode
                u = rng.choice(len(mu), LF.BS - k)
                bx, by = np.concatenate([mu[j], mu[u]]), np.concatenate([ys[j], ys[u]])
                tagged_replayed += k
            else:
                u = rng.choice(len(mu), LF.BS)
                bx, by = mu[u], ys[u]
            ff.fit_batch(bx.astype(np.float32), by.astype(int))
            replayed += LF.BS

    while t < ticks:
        if cfg["sleep"] and st.mu and t % w["day"] == w["day"] - w["night"]:
            n_sleep += 1
            ff.opt.lr = lr
            for _ in range(cfg["chunks"]):
                sleep_chunk()
                for _ in range(CHUNK_TICKS):
                    if t >= ticks:
                        break
                    a = world.arrive(t)
                    if a is not None:
                        x, c, y, ins = a
                        age = t - last_seen.get(c, -10 ** 9)
                        if c in last_seen and age >= LF.RET_GAP:
                            returning[c] = LF.RET_N
                        ret = int(returning.get(c, 0) > 0); returning[c] = returning.get(c, 0) - 1; last_seen[c] = t
                        rows.append((t, c, y, ret, 0, -1, -1, -1))
                    t += 1
            tags.clear()
            forget()
            continue
        if t % 100 == 0:
            forget()
        a = world.arrive(t)
        if a is not None:
            x, c, y, ins = a
            age = t - last_seen.get(c, -10 ** 9)
            if c in last_seen and age >= LF.RET_GAP:
                returning[c] = LF.RET_N
            ret = int(returning.get(c, 0) > 0); returning[c] = returning.get(c, 0) - 1; last_seen[c] = t
            sc = ff.scores(x)[0]; o = np.sort(sc); f = int(sc.argmax()); ffm = float(o[-1] - o[-2])
            s, sm = RP.answer(st, x)
            q30 = np.quantile(ffms, 0.3) if len(ffms) == 500 else -np.inf
            dis = s >= 0 and s != f
            use = dis and ((cnt[f] >= 1000 and ffm < q30 and sm >= 0.2) or sm >= 0.3)
            d = s if use else f
            rows.append((t, c, y, ret, 1, f, s, d))
            ffms.append(ffm); cnt[y] += 1
            if f != y:
                tags.append(x.copy())
            ff.opt.lr = lr * cfg["wl"]
            ff.fit_batch(x, np.array([y]))
            ff.opt.lr = lr
            st.t = t; st.observe(x, y); st.add(x, y)
        t += 1

    R = np.array(rows)
    T, C, Y, RET, ANS, F, S, D = R.T
    sw = world.swap_at
    idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300]
    nidx = np.flatnonzero(np.isin(C, list(w["novel"])) & (T >= world.novel_at))[:300]
    out = dict(condition=cond, world=wk, seed=seed, days=days, arrivals=len(R), missed=float(1 - ANS.mean()), sleeps=n_sleep,
               replay_per_wake=replayed / max(ANS.sum(), 1), tagged_share=tagged_replayed / max(replayed, 1),
               pre_swap_remaps=sum(tt < sw for tt, _, _ in st.log), classes_forgotten=forgot, seconds=time.time() - t0, route={})
    for name, P in (("combined", D), ("ff", F), ("store", S)):
        ok = (P == Y) & (ANS == 1)
        out["route"][name] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()),
                                  swap=float(ok[idx].mean()), novel=float(ok[nidx].mean()))
    print(f"{wk} {cond:9s} seed={seed} " + " ".join(f"{k}={v['lifetime']*100:.2f}/{v['returns']*100:.1f}/{v['swap']*100:.1f}"
                                            for k, v in out["route"].items()) + f" missed={out['missed']*100:.2f} ({out['seconds']:.0f}s)", flush=True)
    return out


def summarise(rows, conds):
    M = lambda rs, r, k: 100 * np.mean([x["route"][r][k] for x in rs])
    base = [x for x in rows if x["condition"] == "none"]
    out = ["| Machine | Lifetime | Returns | Swap first 300 | Novel first 300 | FF-only lifetime / returns / swap | Store-only lifetime | Missed % | Replay ÷ wake | Δ lifetime | Δ returns | Δ swap |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for c in conds:
        rs = [x for x in rows if x["condition"] == c]
        if not rs:
            continue
        dl = M(rs, "combined", "lifetime") - M(base, "combined", "lifetime") if base else float("nan")
        dr = M(rs, "combined", "returns") - M(base, "combined", "returns") if base else float("nan")
        ds = M(rs, "combined", "swap") - M(base, "combined", "swap") if base else float("nan")
        out.append(f"| {c} | {M(rs,'combined','lifetime'):.2f} | {M(rs,'combined','returns'):.1f} | {M(rs,'combined','swap'):.1f} | "
                   f"{M(rs,'combined','novel'):.1f} | {M(rs,'ff','lifetime'):.2f} / {M(rs,'ff','returns'):.1f} / {M(rs,'ff','swap'):.1f} | "
                   f"{M(rs,'store','lifetime'):.2f} | {100*np.mean([x['missed'] for x in rs]):.2f} | {np.mean([x['replay_per_wake'] for x in rs]):.2f} | "
                   f"{dl:+.2f} | {dr:+.1f} | {ds:+.1f} |")
        if c != "none":
            verdict[c] = bool(dl >= -0.10 and (dr >= 1.0 or (ds >= 2.0 and dr >= -0.5)))
    return "\n".join(out), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/cls_sleep")
    ap.add_argument("--world", default="s49")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load_mnist("mnist")
    jobs = [(c, s, a.days, a.world) for c in a.conds for s in a.seeds]
    rows = []
    with Pool(a.jobs) as p:
        for r in p.imap_unordered(live, jobs):
            rows.append(r)
            json.dump(dict(world=WORLDS[a.world], conds=CONDS, rows=rows), open(os.path.join(a.out, "results.json"), "w"))
    md, verdict = summarise(rows, a.conds)
    json.dump(dict(world=WORLDS[a.world], conds=CONDS, rows=rows, verdict=verdict), open(os.path.join(a.out, "results.json"), "w"))
    print(md); print(verdict)


if __name__ == "__main__":
    main()
