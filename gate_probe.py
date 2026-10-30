"""Uncertainty-gated store (loop 17:00 hyp 1) and a returns-heavy world (hyp 3).

A never-sleeping life.py machine lives one life; per answered arrival we log both routes' answers and
their uncertainty (FF goodness margin, store relative distance margin), class ages and label counts.
Learning is identical for every route, so all gates are scored offline on the same arrivals.
Gates (pre-registered thresholds, no tuning on these numbers):
  age      store answers on disagreement if its class has been away longer and >= 500 ticks (= life.py dual_age)
  age_p    age AND FF's class has >= 1000 labelled arrivals (protect novel / swapped classes)
  ug_m     age_p AND FF margin < causal median of FF's last 500 margins
  ug_s     age_p AND store relative margin >= 0.2
  ug_or    age_p AND (ug_m cond OR ug_s cond)     <- hyp 1 as written
  ug_and   age_p AND ug_m cond AND ug_s cond
  mq10_s1/s2  NO age: disagreement AND FF margin in its causal bottom 10% AND store rel. margin >= 0.1/0.2
              AND protection (post hoc on seeds 0-2; held-out confirmation on seeds 3-5)
Kill for hyp 1: lifetime < 95.5 at returns >= 70 (bg 0.02 world).
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist


def run(args):
    seed, days, w = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    cfg = dict(LF.DEFAULTS, **LF.POLICIES["none"])
    ticks = days * w["day"]
    world, mach = LF.World(xtr, ytr, ticks, seed, w), LF.Machine(cfg, seed)
    seen_t = np.full(10, -10 ** 9)
    cnt = np.zeros(10)
    last_seen, returning, rows = {}, {}, []
    t0 = time.time()
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        age = t - last_seen.get(c, -10 ** 9)
        if c in last_seen and age >= LF.RET_GAP:
            returning[c] = LF.RET_N
        ret = int(returning.get(c, 0) > 0)
        returning[c] = returning.get(c, 0) - 1
        last_seen[c] = t
        sc = mach.m.scores(x)[0]
        o = np.sort(sc)
        f = int(sc.argmax()); ffm = float(o[-1] - o[-2])
        s, sm = -1, 0.0
        if mach.store.mu:
            mu, ys = mach._protos()
            d = ((mu - x) ** 2).sum(1)
            j = d.argmin(); s = int(ys[j])
            other = d[ys != s]
            sm = float((other.min() - d[j]) / max(other.min(), 1e-9)) if len(other) else 1.0
        rows.append((t, c, y, ins, ret, f, s, ffm, sm, t - seen_t[s] if s >= 0 else 0, t - seen_t[f], cnt[f]))
        seen_t[y] = t; cnt[y] += 1
        mach.learn(x, y)
    print(f"seed={seed} ctx={w['ctx']} bg={w['bg']} {time.time() - t0:.0f}s", flush=True)
    return dict(seed=seed, swap_at=world.swap_at, novel_at=world.novel_at, rows=np.array(rows, float).tolist())


def rolling_median(v, n=500):
    out = np.empty(len(v))
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - n):i]) if i else np.inf
    return out


def rolling_quantile(v, q, n=500):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    out = np.full(len(v), -np.inf)
    if len(v) > n:
        out[n:] = np.quantile(sw(v, n), q, axis=1)[:len(v) - n]
    return out


def score(r, w):
    R = np.array(r["rows"])
    T, C, Y, INS, RET, F, S, FFM, SM, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F)
    age = dis & (AS > AF) & (AS >= 500)
    prot = CF >= 1000
    lowm = FFM < rolling_median(FFM)
    hism = SM >= 0.2
    gates = dict(ff=np.zeros_like(dis), store=dis, age=age, age_p=age & prot, ug_m=age & prot & lowm,
                 ug_s=age & prot & hism, ug_or=age & prot & (lowm | hism), ug_and=age & prot & lowm & hism)
    # margin gate, no age (found post hoc at 18:30 on seeds 0-2; confirmed on held-out seeds 3-5):
    # store answers on disagreement when FF's margin is in its bottom 10% of the last 500 and the store's
    # relative margin >= 0.1 / 0.2; FF's class protected as above
    low10 = FFM < rolling_quantile(FFM, 0.1)
    gates.update(mq10_s1=dis & prot & low10 & (SM >= 0.1), mq10_s2=dis & prot & low10 & (SM >= 0.2))
    out = {}
    for g, use in gates.items():
        P = np.where(use, S, F); ok = P == Y
        first = lambda digits, at: float(ok[np.flatnonzero(np.isin(C, digits) & (T >= at))[:300]].mean())
        out[g] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), off=float(ok[INS == 0].mean()),
                      swap=first(list(w["swap"]), r["swap_at"]), novel=first(list(w["novel"]), r["novel_at"]),
                      fired=float(use.mean()))
    out["oracle"] = dict(lifetime=float(((F == Y) | (S == Y)).mean()), returns=float(((F == Y) | (S == Y))[RET == 1].mean()))
    out["_n"] = dict(arrivals=len(R), returns_share=float(RET.mean()), off_share=float(1 - INS.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--worlds", nargs="+", default=["bg002", "ctx2bg0"])
    ap.add_argument("--out", default="results/gate_probe")
    a = ap.parse_args()
    W = dict(bg002=dict(bg=0.02), ctx2bg0=dict(bg=0.0, ctx=2))
    os.makedirs(a.out, exist_ok=True)
    load_mnist("mnist")
    jobs = [(s, a.days, W[k]) for k in a.worlds for s in a.seeds]
    with Pool(len(jobs)) as p:
        res = p.map(run, jobs, chunksize=1)
    allres, md = {}, ["# Uncertainty-gated store (`gate_probe.py`)\n", __doc__, ""]
    for i, k in enumerate(a.worlds):
        w = dict(LF.WORLD, **W[k])
        sc = [score(res[i * len(a.seeds) + j], w) for j in range(len(a.seeds))]
        allres[k] = sc
        n = sc[0]["_n"]
        md += [f"\n## World {k} ({W[k]}), seeds {a.seeds}, mean ± std %. Returns share {np.mean([s['_n']['returns_share'] for s in sc]):.1%}, off-season {np.mean([s['_n']['off_share'] for s in sc]):.1%}\n",
               "| Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers (% of arrivals) |", "|---|---|---|---|---|---|---|"]
        for g in [x for x in sc[0] if not x.startswith("_") and x != "oracle"]:
            m = lambda key: f"{np.mean([s[g][key] for s in sc]) * 100:.1f} ± {np.std([s[g][key] for s in sc]) * 100:.1f}"
            md.append(f"| {g} | {m('lifetime')} | {m('returns')} | {m('off')} | {m('swap')} | {m('novel')} | {m('fired')} |")
        md.append(f"| oracle | {np.mean([s['oracle']['lifetime'] for s in sc]) * 100:.1f} | {np.mean([s['oracle']['returns'] for s in sc]) * 100:.1f} | | | | |")
    json.dump(dict(scores=allres, raw=res), open(os.path.join(a.out, "results.json"), "w"))
    open(os.path.join(a.out, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
