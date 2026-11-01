"""A life, not a benchmark: a machine that acts and learns all the time, and sleeps when it chooses.

THE WORLD  ticks of time, grouped into days (`day` ticks; the last `night` of them are night).
           An input arrives on a tick with probability 1 by day and `night_rate` at night.
           Each input is a digit from the "season": `ctx` digits that dominate for a while
           (seasons last `season` ticks on average); with probability `bg` it is any digit
           instead. Seasons recur, so digits leave and come back. Digits 8 and 9 don't exist
           until `novel_at` of the life (novelty); at `swap_at` digits 2 and 3 swap labels
           for good (the world's rule changes). Images come from the MNIST training set,
           each class cycled in a fresh random order (a long life sees an image ~1.3x).
THE LOOP   every arrival: if asleep, it goes unanswered (a miss, and nothing is learned).
           Otherwise the machine answers (scored) and then gets the label and learns from it:
           FF wake update (optionally slow, optionally with awake replay) and the fast store
           (9 noise-free prototypes/class) records it.
SLEEP      the machine stops answering. It replays its store in chunks of 10 batches of 32,
           and each chunk takes `chunk_ticks` of world time. Inputs keep arriving, unanswered.
           What decides when and how long is the policy:
             none      never sleeps
             clock     every `every` answered inputs, `chunks` chunks (sleep.py's default)
             probe     sleep pressure only: every `check` ticks the network re-classifies 30
                       stored prototypes drawn afresh; sleep `chunks` chunks when a fraction
                       `theta` of them fail
             night     circadian only: sleep `chunks` chunks at the start of every night
             twoproc   Borbely's two processes: pressure S (the probe drop) against a
                       threshold set by the time of day C (`theta` by day, `theta_night` at
                       night: sleep is cheap when the world is quiet). S = probe failures. Sleep lasts until the
                       store is recalled again (all stored prototypes >= `recall` correct,
                       checked after each chunk, at least `min_chunks`), capped at
                       `max_day` / `max_night` chunks.

SCORES     lifetime   prequential accuracy over the arrivals that come while it is awake: the north star.
                      Arrivals while asleep are not evaluated (rule from 1 Nov 2026 14:00 UTC+8; before that they
                      counted as wrong). The old score is kept as `lifetime_all`; `missed` still reports the downtime.
           answered   same as lifetime (kept for old readers)
           frozen     class-IL on a held-out test subset (digits that exist, current labels),
                      taken at noon and midnight of every day: the legacy train-then-test view
           age        prequential accuracy by ticks since that digit was last seen
           adaptation prequential accuracy on 2/3 in their first arrivals after the swap
           novelty    prequential accuracy on 8/9 in their first arrivals

Usage
  python life.py                            # all contenders x 3 seeds -> results/life/
  python life.py --conds clock twoproc --seeds 0 --days 4     # smoke test
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from data import load_mnist

WORLD = dict(day=4000, night=1000, night_rate=0.1, ctx=3, bg=0.2, season=6000,
             novel=(8, 9), novel_at=0.3, swap=(2, 3), swap_at=0.6)
CHUNK = 10  # batches per sleep chunk
BS = 32
AGE_BINS = [0, 100, 1000, 10000, 10 ** 9]
RET_GAP, RET_N = 2000, 20  # a "return": a digit back after >= 2000 ticks away; its first 20 arrivals

POLICIES = {
    "none": dict(sleep="none"),
    "clock": dict(sleep="clock", every=2000, chunks=5),
    "probe": dict(sleep="probe", theta=0.3, chunks=5),
    "night": dict(sleep="night", chunks=8),
    "twoproc": dict(sleep="twoproc", theta=0.3, theta_night=0.1),
    "clock_wl3": dict(sleep="clock", every=2000, chunks=5, wake_lr=0.3),
    "twoproc_wl3": dict(sleep="twoproc", theta=0.3, theta_night=0.1, wake_lr=0.3),
    "mix8": dict(sleep="none", mix=0.8),
    "twoproc_mix8": dict(sleep="twoproc", theta=0.3, theta_night=0.1, mix=0.8),
    "twoproc_mix8_wl3": dict(sleep="twoproc", theta=0.3, theta_night=0.1, mix=0.8, wake_lr=0.3),
}
LABELS = {
    "none": "Never sleeps",
    "clock": "Clock: 50 batches every 2000 answered inputs",
    "probe": "Pressure only: 50 batches when 30% of probed memories fail",
    "night": "Circadian only: 80 batches at the start of each night",
    "twoproc": "Two-process: pressure vs time of day, sleeps until recalled",
    "clock_wl3": "Clock, wake lr x0.3",
    "twoproc_wl3": "Two-process, wake lr x0.3",
    "mix8": "Awake replay 0.8, never sleeps",
    "twoproc_mix8": "Two-process + awake replay 0.8",
    "twoproc_mix8_wl3": "Two-process + awake replay 0.8, wake lr x0.3",
}
DEFAULTS = dict(check=100, probe_n=30, min_gap=200, recall=0.97, min_chunks=1, max_day=5, max_night=25,
                chunk_ticks=40, wake_lr=1.0, mix=0.0)


class World:
    """The stream. Everything about it is drawn from its own seed, so every policy lives the same life
    (same arrivals, same images, same seasons) and differences are the machine's alone."""

    def __init__(self, xtr, ytr, ticks, seed, w):
        self.w, self.ticks = w, ticks
        rng = np.random.default_rng(seed + 7000)
        self.xtr = xtr
        self.pools = {c: rng.permutation(np.flatnonzero(ytr == c)) for c in range(10)}
        self.ptr = {c: 0 for c in range(10)}
        self.rng = rng
        self.novel_at, self.swap_at = int(w["novel_at"] * ticks), int(w["swap_at"] * ticks)
        # seasons: (start tick, context digits). Novel digits and the swapped pair get a season when
        # they arrive, so every policy meets them at the same moment.
        self.seasons, t, forced = [], 0, set()
        while t < ticks:
            # the season that would start after a forced moment is cut short and the forced one starts on time
            for name, at, pair in (("novel", self.novel_at, w["novel"]), ("swap", self.swap_at, w["swap"])):
                if t >= at and name not in forced:
                    forced.add(name)
                    t = at
                    others = [c for c in self.exists(t) if c not in pair]
                    ctx = list(pair) + list(rng.choice(others, w["ctx"] - 2, replace=False))
                    break
            else:
                ctx = list(rng.choice(self.exists(t), w["ctx"], replace=False))
            self.seasons.append((t, ctx))
            t += int(rng.uniform(0.5, 1.5) * w["season"])
        self.season_start = np.array([s[0] for s in self.seasons])

    def night(self, t):
        return t % self.w["day"] >= self.w["day"] - self.w["night"]

    def exists(self, t):
        return [c for c in range(10) if t >= self.novel_at or c not in self.w["novel"]]

    def label(self, c, t):
        """The world's current rule: after swap_at, the swapped pair trade labels."""
        a, b = self.w["swap"]
        if t >= self.swap_at and c in (a, b):
            return b if c == a else a
        return c

    def arrive(self, t):
        """None, or (image, digit, label) for tick t. Draws in tick order."""
        if self.rng.random() >= (self.w["night_rate"] if self.night(t) else 1.0):
            return None
        ctx = self.seasons[np.searchsorted(self.season_start, t, side="right") - 1][1]
        c = int(self.rng.choice(self.exists(t))) if self.rng.random() < self.w["bg"] else int(self.rng.choice(ctx))
        p = self.pools[c]
        i = p[self.ptr[c] % len(p)]
        self.ptr[c] += 1
        return self.xtr[i:i + 1], c, self.label(c, t), int(c in ctx)


class Machine:
    """FF network + fast store + a sleep policy. It only ever sees what the world shows it."""

    def __init__(self, cfg, seed, lr=1e-3):
        import learners as L
        import sleep as S
        self.cfg, self.lr = cfg, lr
        self.m = L.ForwardForward(lr=lr, loss="symba", homeo=0.001, seed=seed)
        self.store = S.ProtoStore(9, np.random.default_rng(seed + 1000), noise=0.0)
        self.rng = np.random.default_rng(seed + 2000)
        self.answered = 0          # answered inputs since the last sleep (for the clock)
        self.awake_ticks = 0       # ticks since the last sleep ended (min_gap)
        self.trace = []            # every pressure reading
        self.replayed = self.wake_updates = self.probe_items = 0

    def act(self, x):
        return int(self.m.scores(x).argmax(1)[0])

    # --- dual route (CLS: the store also answers). Learning is unchanged; we only score extra answerers.
    def store_act(self, x):
        if not self.store.mu:
            return -1
        mu, ys = self._protos()
        return int(ys[((mu - x) ** 2).sum(1).argmin()])

    def routes(self, x, y, t=0):
        """(ff, store, dual) predictions, then update the arbiter with the label. Arbiter: a running
        reliability per route per *predicted* class (EMA of being right when that route said c); on
        disagreement the route whose claim is more reliable answers."""
        if not hasattr(self, "rel"):
            self.rel = np.full((2, 10), 0.5)
        f, s = self.act(x), self.store_act(x)
        if s < 0 or f == s:
            d = f
        else:
            d = f if self.rel[0, f] >= self.rel[1, s] else s
        # age arbiter: FF errs off season by naming a recent class; on disagreement, trust the store when the
        # class it names has been away longer than the class FF names (and away >= age_gate ticks)
        if not hasattr(self, "seen_t"):
            self.seen_t = np.full(10, -10 ** 9)
        if s < 0 or f == s:
            g = f
        else:
            g = s if (t - self.seen_t[s] > t - self.seen_t[f] and t - self.seen_t[s] >= self.cfg.get("age_gate", 500)) else f
        self.seen_t[y] = t
        self.last_age = g
        a = self.cfg.get("rel_rate", 0.05)
        self.rel[0, f] += a * ((f == y) - self.rel[0, f])
        if s >= 0:
            self.rel[1, s] += a * ((s == y) - self.rel[1, s])
        return f, s, d

    def learn(self, x, y):
        cfg = self.cfg
        self.m.opt.lr = self.lr * cfg["wake_lr"]
        k = 0
        if cfg["mix"] and self.store.mu:
            k = int(cfg["mix"]) + int(self.rng.random() < cfg["mix"] % 1)
        if k:
            xr, yr = self.store.sample(k)
            self.m.fit_batch(np.concatenate([x, xr]), np.concatenate([[y], yr]))
            self.replayed += k
        else:
            self.m.fit_batch(x, np.array([y]))
        self.m.opt.lr = self.lr
        self.store.add(x, y)
        self.wake_updates += 1
        self.answered += 1

    def _protos(self):
        mu = np.concatenate([self.store.mu[c] for c in sorted(self.store.mu)])
        ys = np.concatenate([np.full(len(self.store.mu[c]), c) for c in sorted(self.store.mu)])
        return mu, ys

    def recall(self):
        """Fraction of all stored prototypes the network classifies correctly."""
        mu, ys = self._protos()
        self.probe_items += len(mu)
        return float((self.m.scores(mu).argmax(1) == ys).mean())

    def pressure(self):
        """Process S: the fraction of a fresh random draw of `probe_n` stored memories (any class) that the
        network now gets wrong. (life.py's first version measured the drop on a probe drawn once and redrawn
        only after a sleep; a machine that never slept kept its first-season probe for life.)"""
        mu, ys = self._protos()
        j = self.rng.choice(len(mu), min(self.cfg["probe_n"], len(mu)), replace=False)
        self.probe_items += len(j)
        s = 1.0 - float((self.m.scores(mu[j]).argmax(1) == ys[j]).mean())
        self.trace.append(s)
        return s

    def wants_sleep(self, t, world, night_started):
        """None, or (night?) if the policy says sleep now. Called every tick while awake."""
        cfg, pol = self.cfg, self.cfg["sleep"]
        if pol == "none" or not self.store.mu:
            return None
        if pol == "clock":
            return world.night(t) if self.answered >= cfg["every"] else None
        if pol == "night":
            return True if night_started else None
        if self.awake_ticks < cfg["min_gap"] or t % cfg["check"]:
            return None
        s = self.pressure()
        if pol == "probe":
            return world.night(t) if s >= cfg["theta"] else None
        theta = cfg["theta_night"] if world.night(t) else cfg["theta"]  # process C: the price of sleep now
        return world.night(t) if s >= theta else None

    def sleep_chunk(self):
        for _ in range(CHUNK):
            x, y = self.store.sample(BS)
            self.m.fit_batch(x, y)
        self.replayed += CHUNK * BS

    def keep_sleeping(self, done, night):
        cfg = self.cfg
        if cfg["sleep"] != "twoproc":
            return done < cfg["chunks"]
        if done < cfg["min_chunks"]:
            return True
        if done >= (cfg["max_night"] if night else cfg["max_day"]):
            return False
        return self.recall() < cfg["recall"]

    def woke(self):
        self.answered = self.awake_ticks = 0


def live(cond, seed, days, w=WORLD, data=None):
    w = dict(WORLD, **(w or {}))
    xtr, ytr, xte, yte = data or load_mnist("mnist")
    cfg = dict(DEFAULTS, **POLICIES[cond])
    ticks = days * w["day"]
    world, mach = World(xtr, ytr, ticks, seed, w), Machine(cfg, seed)
    test_idx = np.random.default_rng(seed + 4000).choice(len(xte), 2000, replace=False)
    log = []          # per arrival: tick, digit, answered, correct, age, in season
    last_seen, returning = {}, {}   # digit -> arrivals left in its current "return" window
    frozen, sleeps, t0 = [], [], time.time()
    t = 0

    def frozen_test(t):
        j = test_idx[np.isin(yte[test_idx], world.exists(t))]
        lab = np.array([world.label(c, t) for c in yte[j]])
        frozen.append((t, float((mach.m.scores(xte[j]).argmax(1) == lab).mean())))

    def tick(t, asleep):
        if t % (w["day"] // 2) == w["day"] // 2 - 1:
            frozen_test(t)
        a = world.arrive(t)
        if a is None:
            return
        x, c, y, ins = a
        age = t - last_seen.get(c, -10 ** 9)
        if c in last_seen and age >= RET_GAP:
            returning[c] = RET_N
        ret = int(returning.get(c, 0) > 0)
        returning[c] = returning.get(c, 0) - 1
        last_seen[c] = t
        if asleep:
            log.append((t, c, 0, 0, age, ins, ret, 0, 0, 0, 0))
            return
        f, s_, d = mach.routes(x, y, t)
        log.append((t, c, 1, int(f == y), age, ins, ret, int(s_ == y), int(d == y), int(f == y or s_ == y), int(mach.last_age == y)))
        mach.learn(x, y)

    while t < ticks:
        night_started = t % w["day"] == w["day"] - w["night"]
        night = mach.wants_sleep(t, world, night_started)
        if night is not None:
            start, done = t, 0
            while t < ticks and mach.keep_sleeping(done, night):
                mach.sleep_chunk()
                done += 1
                for _ in range(cfg["chunk_ticks"]):
                    if t >= ticks:
                        break
                    tick(t, asleep=True)
                    t += 1
            sleeps.append((start, t - start, done))
            mach.woke()
            continue
        tick(t, asleep=False)
        mach.awake_ticks += 1
        t += 1

    L = np.array(log)
    T, C, ANS, OK, AGE, INS, RET, OKS, OKD, OKO, OKA = L.T
    nightly = np.array([world.night(s) for s, _, _ in sleeps]) if sleeps else np.zeros(0)
    slept = sum(d for _, d, _ in sleeps)
    night_ticks = sum(sum(world.night(s + k) for k in range(d)) for s, d, _ in sleeps)

    W = ANS == 1      # evaluated arrivals: awake ones only (asleep arrivals are not scored)

    def firstk(ok, digits, after, n, awake=True):
        j = np.flatnonzero(np.isin(C, digits) & (T >= after) & (W if awake else True))[:n]
        return float(ok[j].mean()) if len(j) else None

    def first(digits, after, n, awake=True):
        return firstk(OK, digits, after, n, awake)

    route = {}
    for name, ok in (("ff", OK), ("store", OKS), ("dual", OKD), ("dual_age", OKA), ("oracle", OKO)):
        route[name] = dict(lifetime=float(ok[W].mean()), returns=float(ok[W & (RET == 1)].mean()) if (W & (RET == 1)).any() else None,
                           lifetime_all=float(ok.mean()), returns_all=float(ok[RET == 1].mean()) if (RET == 1).any() else None,
                           off_season=float(ok[(ANS == 1) & (INS == 0)].mean()),
                           adapt=firstk(ok, list(w["swap"]), world.swap_at, 300), novel=firstk(ok, list(w["novel"]), world.novel_at, 300))
    days_acc = [float(OK[W & (T >= d * w["day"]) & (T < (d + 1) * w["day"])].mean()) for d in range(days)]
    age = []
    for lo, hi in zip(AGE_BINS[:-1], AGE_BINS[1:]):
        j = (AGE >= lo) & (AGE < hi) & (ANS == 1)
        age.append((float(OK[j].mean()) if j.any() else None, int(j.sum())))
    return dict(
        condition=cond, seed=seed, days=days, route=route,
        arrivals=len(L), lifetime=float(OK[W].mean()), answered=float(OK[W].mean()), lifetime_all=float(OK.mean()),
        # answered accuracy on the season's digits vs the rest (retention of what is out of season)
        in_season=float(OK[(ANS == 1) & (INS == 1)].mean()), off_season=float(OK[(ANS == 1) & (INS == 0)].mean()),
        off_share=float(1 - INS.mean()),
        # a digit's first RET_N arrivals after >= RET_GAP ticks away, scored on the awake ones: memory that has to last
        returns=float(OK[W & (RET == 1)].mean()) if (W & (RET == 1)).any() else None, n_returns=int((W & (RET == 1)).sum()),
        returns_all=float(OK[RET == 1].mean()) if (RET == 1).any() else None,
        pressure_mean=float(np.mean(mach.trace)) if mach.trace else None,
        pressure_p90=float(np.percentile(mach.trace, 90)) if mach.trace else None,
        missed=float(1 - ANS.mean()),
        asleep=slept / ticks, sleep_night_share=night_ticks / slept if slept else None,
        sleeps=len(sleeps), sleeps_at_night=int(nightly.sum()),
        mean_sleep_chunks=float(np.mean([n for _, _, n in sleeps])) if sleeps else 0.0,
        replay_per_wake=mach.replayed / max(mach.wake_updates, 1),
        probe_per_wake=mach.probe_items / max(mach.wake_updates, 1),
        frozen=float(np.mean([a for _, a in frozen])), frozen_final=frozen[-1][1], frozen_trace=frozen,
        days_acc=days_acc, age=age,
        # the first 300 awake arrivals of the swapped digits after the swap and of the novel digits after they appear
        adapt=first(list(w["swap"]), world.swap_at, 300), adapt_late=first(list(w["swap"]), world.swap_at + 3 * w["day"], 300),
        novel=first(list(w["novel"]), world.novel_at, 300),
        adapt_all=first(list(w["swap"]), world.swap_at, 300, awake=False), novel_all=first(list(w["novel"]), world.novel_at, 300, awake=False),
        seasons=[(s, [int(c) for c in ctx]) for s, ctx in world.seasons],
        sleep_log=sleeps, seconds=time.time() - t0,
    )


def job(args):
    cond, seed, days, w = args
    r = live(cond, seed, days, w)
    print(f"{cond:18s} seed={seed} lifetime={r['lifetime']:.4f} answered={r['answered']:.4f} "
          f"missed={r['missed']:.3f} frozen={r['frozen']:.4f} sleeps={r['sleeps']} night={r['sleep_night_share']} "
          f"replay/wake={r['replay_per_wake']:.2f} returns={r['returns']} S={r['pressure_mean']} ({r['seconds']:.0f}s) "
          + " ".join(f"{k}={v['lifetime']:.4f}" for k, v in r["route"].items()), flush=True)
    return r


def summarise(rows, conds, seeds, days, w=WORLD):
    def ms(v, scale=100, fmt=".1f"):
        v = [x for x in v if x is not None]
        if not v:
            return "–"
        return f"{np.mean(v) * scale:{fmt}}" + (f" ± {np.std(v) * scale:{fmt}}" if len(v) > 1 else "")

    out = [f"{days} days of {w['day']} ticks (night: last {w['night']}, input rate {w['night_rate']}), "
           f"seasons of ~{w['season']} ticks with {w['ctx']} digits (+{w['bg']:.0%} background), digits "
           f"{w['novel']} appear at {w['novel_at']:.0%}, {w['swap']} swap labels at {w['swap_at']:.0%}. "
           f"A sleep chunk (10 batches of 32) takes {DEFAULTS['chunk_ticks']} ticks. Returns = a digit's first "
           f"{RET_N} arrivals after >= {RET_GAP} ticks away (misses count; {np.mean([r['n_returns'] for r in rows]):.0f} per life). "
           f"Sleep pressure S (probe failures): mean {np.mean([r['pressure_mean'] or 0 for r in rows]):.2f}. Off-season inputs: "
           f"{np.mean([r['off_share'] for r in rows]):.0%}. Seeds {seeds}; mean ± std, in %.\n",
           "| Machine | Lifetime | Answered | In season / off season | Returns | Missed | Frozen test (avg / final) | Asleep | Sleep at night | Sleeps | Chunks/sleep | Replay ÷ wake |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        g = lambda k: [r[k] for r in rs]
        out.append(f"| {LABELS[c]} | **{ms(g('lifetime'))}** | {ms(g('answered'))} | {ms(g('in_season'))} / {ms(g('off_season'))} | {ms(g('returns'))} | {ms(g('missed'))} | "
                   f"{ms(g('frozen'))} / {ms(g('frozen_final'))} | {ms(g('asleep'))} | {ms(g('sleep_night_share'), fmt='.0f')} | "
                   f"{np.mean(g('sleeps')):.0f} | {np.mean(g('mean_sleep_chunks')):.1f} | {np.mean(g('replay_per_wake')):.2f} |")
    bins = [f"{lo}–{hi}" if hi < 10 ** 9 else f"≥{lo}" for lo, hi in zip(AGE_BINS[:-1], AGE_BINS[1:])]
    out += ["\n### Answered accuracy by ticks since the digit was last seen\n",
            "| Machine | " + " | ".join(bins) + " | Swap: first 300 awake arrivals of 2/3 | 3 days later | Novel 8/9: first 300 |",
            "|---|" + "---|" * (len(bins) + 3)]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        cells = [ms([r["age"][i][0] for r in rs]) for i in range(len(bins))]
        out.append(f"| {LABELS[c]} | " + " | ".join(cells) +
                   f" | {ms([r['adapt'] for r in rs])} | {ms([r['adapt_late'] for r in rs])} | {ms([r['novel'] for r in rs])} |")
    n = [int(np.mean([r["age"][i][1] for r in rows])) for i in range(len(bins))]
    out.append(f"\nAnswered inputs per age bin (mean over runs): {', '.join(f'{b}: {k}' for b, k in zip(bins, n))}.")
    out += ["\n### Lifetime accuracy by day\n", "| Machine | " + " | ".join(f"d{d}" for d in range(days)) + " |",
            "|---|" + "---|" * days]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        out.append(f"| {LABELS[c]} | " + " | ".join(f"{np.mean([r['days_acc'][d] for r in rs]) * 100:.0f}" for d in range(days)) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conds", nargs="+", default=list(POLICIES))
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="results/life")
    ap.add_argument("--bg", type=float, default=WORLD["bg"], help="chance an input is any digit, not the season's")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    load_mnist("mnist")
    with Pool(args.jobs) as pool:
        w = dict(WORLD, bg=args.bg)
        rows = pool.map(job, [(c, s, args.days, w) for c in args.conds for s in args.seeds], chunksize=1)
    json.dump(dict(seeds=args.seeds, days=args.days, world=w, defaults=DEFAULTS, rows=rows),
              open(os.path.join(args.out, "results.json"), "w"))
    md = summarise(rows, args.conds, args.seeds, args.days, w)
    open(os.path.join(args.out, "RESULTS.md"), "w").write(f"# A life (`life.py`)\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
