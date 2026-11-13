"""M6b-W3: sleep replays one store and is scored on another.

Imports m6_w3 and the modules it uses. Does not edit them. No verdict.

Each room-glance pair is scored, learned, then assigned by a fair coin to an exam
reservoir or a replay reservoir. Both reservoirs hold R = 2,048 pairs per channel.
Progress rewards use the exam reservoirs only. Sleep draws from the replay reservoirs
only.
"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("CURIO_LR") != "0.004":
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["CURIO_LR"] = "0.004"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import time
from multiprocessing import get_context

import numpy as np

import curio
import m5b_w3
import m6_w3
from w3_headroom import LR, W3Compressor

SEEDS = (24, 25, 26, 27, 28, 29)
CONTROLLERS = m6_w3.CONTROLLERS
LENGTHS = m6_w3.LENGTHS
SCORED_L = 2048
REPORT_L = 512
R = 2048
KERNEL_WALL_LIMIT_S = 90 * 60
SMOKE_SEED = 24
KERNEL_ID = "ezzzzzx/m6b-w3"
NFEAT = m6_w3.NFEAT


def _pair_term(comp, feats, x):
    """One pair's codelength. Same arithmetic as one step of SGDCompressor.score_pairs."""
    z = comp.W[feats].sum(0)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return -math.log2(max(float(p[int(x)]), 1e-12))


def _touched(comp, channel, hist, symbols):
    """Feature rows a learning step writes. Matches W3Compressor.bits, including corridor."""
    if channel == "corridor":
        ci = len(curio.CH)
    else:
        ci = curio.CH.index(channel)
    ctx = list(hist[-max(curio.ORDERS):])
    dirty = set()
    max_k = max(curio.ORDERS)
    for x in symbols:
        for f in comp.feats(ci, ctx):
            dirty.add(int(f))
        ctx.append(int(x))
        if len(ctx) > max_k:
            del ctx[:-max_k]
    return dirty


def _touched_pair(comp, channel, pair):
    ctx, _x = pair
    ci = curio.CH.index(channel)
    return [int(f) for f in comp.feats(ci, list(ctx))]


class Reservoir:
    """One reservoir per room channel, cap R, its own offer counter.

    stream index is the symbol's position in that channel's observed stream.
    The exam store caches each pair's codelength and updates a pair only when a
    learning step writes one of its feature rows. The sum is score_pairs.
    """

    def __init__(self, comp, scored):
        self.comp = comp
        self.scored = bool(scored)
        self.pairs = {c: [] for c in curio.CH}
        self.sids = {c: [] for c in curio.CH}
        self.ids = {c: set() for c in curio.CH}
        self.seen = {c: 0 for c in curio.CH}
        self.ci = {c: curio.CH.index(c) for c in curio.CH}
        self._pool = None
        self.score_s = 0.0
        self.score_pair_evals = 0
        if self.scored:
            self.idx = {c: np.zeros((R, NFEAT), np.int64) for c in curio.CH}
            self.mask = {c: np.zeros((R, NFEAT), np.bool_) for c in curio.CH}
            self.xs = {c: np.zeros(R, np.int64) for c in curio.CH}
            self.feat_rows = {c: [] for c in curio.CH}
            self.contrib = {c: [] for c in curio.CH}
            self.total = {c: 0.0 for c in curio.CH}
            self.stale = {c: False for c in curio.CH}

    def size(self, channel):
        return len(self.pairs[channel])

    def sizes(self):
        return {c: self.size(c) for c in curio.CH}

    def offer(self, channel, pair, stream_index, rng):
        """Reservoir-sample one pair. rng is drawn only once the cap is full."""
        self.seen[channel] += 1
        self._pool = None
        n = len(self.pairs[channel])
        if n < R:
            self._put(channel, n, pair, stream_index, append=True)
            return
        k = int(rng.integers(self.seen[channel]))
        if k < R:
            self._put(channel, k, pair, stream_index, append=False)

    def _put(self, channel, row, pair, stream_index, append):
        if append:
            self.pairs[channel].append(pair)
            self.sids[channel].append(int(stream_index))
        else:
            old = self.sids[channel][row]
            self.ids[channel].discard(old)
            self.pairs[channel][row] = pair
            self.sids[channel][row] = int(stream_index)
        self.ids[channel].add(int(stream_index))
        if not self.scored:
            return
        ctx, x = pair
        feats = [int(f) for f in self.comp.feats(self.ci[channel], list(ctx))]
        k = len(feats)
        self.idx[channel][row, :] = 0
        self.mask[channel][row, :] = False
        self.idx[channel][row, :k] = feats
        self.mask[channel][row, :k] = True
        self.xs[channel][row] = int(x)
        term = _pair_term(self.comp, feats, x)
        if append:
            self.feat_rows[channel].append(feats)
            self.contrib[channel].append(term)
        else:
            self.feat_rows[channel][row] = feats
            self.contrib[channel][row] = term
        self.stale[channel] = True

    def pool(self):
        """Union of stored (channel, pair, stream_index), cached until the next offer."""
        if self._pool is None:
            self._pool = [
                (c, pair, sid)
                for c in curio.CH
                for pair, sid in zip(self.pairs[c], self.sids[c])
            ]
        return self._pool

    def _resum(self, channel):
        tot = 0.0
        for v in self.contrib[channel]:
            tot += v
        self.total[channel] = tot
        self.stale[channel] = False

    def rescore(self, dirty):
        """Recompute pairs whose features intersect the rows this decision wrote."""
        if not self.scored:
            raise RuntimeError("rescore called on the replay store")
        if not dirty:
            return
        t0 = time.perf_counter()
        table = np.zeros(curio.HSZ, np.bool_)
        table[list(dirty)] = True
        hits = 0
        for c in curio.CH:
            n = len(self.pairs[c])
            if n == 0:
                continue
            hit = np.where(self.mask[c][:n], table[self.idx[c][:n]], False).any(axis=1)
            chosen = np.flatnonzero(hit)
            if len(chosen) == 0:
                continue
            rows = self.feat_rows[c]
            xs = self.xs[c]
            contrib = self.contrib[c]
            for i in chosen:
                i = int(i)
                contrib[i] = _pair_term(self.comp, rows[i], int(xs[i]))
                hits += 1
            self._resum(c)
        self.score_s += time.perf_counter() - t0
        self.score_pair_evals += hits

    def score_channel(self, channel):
        n = len(self.pairs[channel])
        if n == 0:
            return 0.0
        if not self.scored:
            raise RuntimeError("score called on the replay store")
        if self.stale[channel]:
            t0 = time.perf_counter()
            self._resum(channel)
            self.score_s += time.perf_counter() - t0
        return self.total[channel]

    def score_all(self):
        return {c: self.score_channel(c) for c in curio.CH}

    def score_gap(self):
        """Absolute gap versus SGDCompressor.score_pairs on the stored pairs."""
        worst = 0.0
        for c in curio.CH:
            pairs = self.pairs[c]
            ref = self.comp.score_pairs(c, pairs) if pairs else 0.0
            got = self.score_channel(c) if self.scored else ref
            worst = max(worst, abs(got - ref))
        return float(worst)


def _overlap(exam, replay):
    n = 0
    for c in curio.CH:
        n += len(exam.ids[c] & replay.ids[c])
    return int(n)


def _replay_draw(comp, replay, exam, rng, counter):
    """64 uniform draws with replacement from the replay union. Exam is not read for the draw.

    counter[0] counts drawn stream indices that are currently stored in the exam.
    Returns the feature rows the draws write.
    """
    dirty = set()
    pool = replay.pool()
    if not pool:
        return dirty
    n = len(pool)
    for _ in range(m6_w3.REPLAY_N):
        channel, pair, sid = pool[int(rng.integers(n))]
        if sid in exam.ids[channel]:
            counter[0] += 1
        for f in _touched_pair(comp, channel, pair):
            dirty.add(f)
        m6_w3._learn_pair(comp, channel, pair)
    return dirty


def _room_decision(comp, hist, exam, replay, exam_rng, replay_rng, coin_rng, channel, symbols):
    """Learn one room glance. Score the exam before this glance's pairs are inserted."""
    before = exam.score_all()
    comp.bits(channel, hist[channel], symbols, learn=True)
    exam.rescore(_touched(comp, channel, hist[channel], symbols))
    after = exam.score_all()
    full = hist[channel] + symbols
    max_k = max(curio.ORDERS)
    start = len(hist[channel])
    for i in range(start, len(full)):
        pair = (tuple(full[max(0, i - max_k):i]), full[i])
        if int(coin_rng.integers(2)) == 0:
            exam.offer(channel, pair, i, exam_rng)
        else:
            replay.offer(channel, pair, i, replay_rng)
    hist[channel] += symbols
    return m6_w3._bits_saved(before, after)


def _corridor_decision(comp, hist, exam, symbols):
    before = exam.score_all()
    comp.bits("corridor", hist["corridor"], symbols, learn=True)
    exam.rescore(_touched(comp, "corridor", hist["corridor"], symbols))
    after = exam.score_all()
    hist["corridor"] += symbols
    return m6_w3._bits_saved(before, after)


def _sleep_decision(comp, exam, replay, rng, counter):
    before = exam.score_all()
    exam.rescore(_replay_draw(comp, replay, exam, rng, counter))
    after = exam.score_all()
    return m6_w3._bits_saved(before, after)


def _store_payload(exam, replay):
    return {
        "exam": {c: int(exam.size(c)) for c in curio.CH},
        "replay": {c: int(replay.size(c)) for c in curio.CH},
    }


def _seen_payload(exam, replay):
    return {
        "exam": {c: int(exam.seen[c]) for c in curio.CH},
        "replay": {c: int(replay.seen[c]) for c in curio.CH},
    }


def _check_stores(exam, replay, channels, tag):
    room_symbols = sum(curio.B for ch in channels if ch not in ("corridor", "sleep"))
    offered = sum(exam.seen.values()) + sum(replay.seen.values())
    if offered != room_symbols:
        raise RuntimeError(f"{tag}: offered {offered} pairs, room symbols {room_symbols}")
    overlap = _overlap(exam, replay)
    if overlap:
        raise RuntimeError(f"{tag}: {overlap} stream indices sit in both stores")
    for c in curio.CH:
        for store in (exam, replay):
            if len(store.sids[c]) != len(set(store.sids[c])):
                raise RuntimeError(f"{tag}: duplicate stream index in {c}")
            if set(store.sids[c]) != store.ids[c]:
                raise RuntimeError(f"{tag}: id set drifted on {c}")
            if store.size(c) > R:
                raise RuntimeError(f"{tag}: {c} holds {store.size(c)} > {R}")
    return overlap


def run_prog(seed, L, score=True):
    """M5's prog controller. No split store and no sleep."""
    row = m6_w3.run_prog(seed, L, score=score)
    row["store_size"] = None
    row["store_seen"] = None
    row["exam_pairs_drawn"] = 0
    row["score_s"] = 0.0
    row["score_gap"] = None
    return row


def run_whole(seed, kind, L, score=True, n_sleep_target=None):
    """prog_total, prog_sleep, or fixed_sleep. Exam-store reward. Life length L."""
    if kind not in ("prog_total", "prog_sleep", "fixed_sleep"):
        raise ValueError(kind)
    if L % 4 != 0:
        raise ValueError(f"L={L} is not divisible by 4")
    if kind == "fixed_sleep":
        if n_sleep_target is None:
            raise ValueError("fixed_sleep needs prog_sleep's sleep-decision count")
        plan = m6_w3.sleep_plan(L, int(n_sleep_target))
    else:
        plan = []
        if n_sleep_target not in (None, 0):
            raise ValueError("n_sleep_target is only for fixed_sleep")
    t0 = time.perf_counter()
    world = m6_w3.World3(seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    world_rng = np.random.default_rng(1_000 + seed)
    exam_rng = np.random.default_rng(20_000 + seed)
    replay_rng = np.random.default_rng(50_000 + seed)
    coin_rng = np.random.default_rng(40_000 + seed)
    sleep_rng = np.random.default_rng(30_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = m6_w3.m5_w3._empty_hist()
    exam = Reservoir(comp, scored=True)
    replay = Reservoir(comp, scored=False)
    n_arms = 10 if kind == "prog_sleep" else 9
    v = [0.0] * n_arms
    nvis = [0] * n_arms
    unvisited = set(range(9))
    first_sleep = kind == "prog_sleep"
    channels = []
    reward_q = m6_w3._empty_reward_quarters()
    drawn = [0]
    si = 0
    t = 0

    def one_sleep_visit(length):
        nonlocal t
        start = t
        sum_r = 0.0
        n = 0
        while n < length and t < L:
            sum_r += _sleep_decision(comp, exam, replay, sleep_rng, drawn)
            channels.append("sleep")
            n += 1
            t += 1
        reward = sum_r / n if n else 0.0
        return start, n, reward

    while t < L:
        if t and t % 256 == 0:
            gap_now = exam.score_gap()
            if gap_now > 0.0:
                raise RuntimeError(f"{kind} seed {seed} L={L} t={t} score gap {gap_now:.3e}")
        if kind == "fixed_sleep" and m6_w3._sleep_due(t, L, plan, si):
            _start, n, reward = one_sleep_visit(plan[si][1])
            m6_w3._note_visit(reward_q, "sleep", _start, L, reward)
            si += 1
            continue
        arm, first_sleep = m6_w3._choose_arm(
            world, unvisited, first_sleep, v, n_arms, choice_rng,
        )
        if arm == m6_w3.SLEEP_ARM:
            _start, n, reward = one_sleep_visit(min(m6_w3.DWELL, L - t))
            m6_w3.m5_w3._ema(v, nvis, m6_w3.SLEEP_ARM, reward)
            m6_w3._note_visit(reward_q, "sleep", _start, L, reward)
            continue
        target = int(arm)
        start = t
        path = m6_w3.shortest(world.room, target)
        sum_r = 0.0
        travel = 0
        while path and t < L:
            action = path.pop(0)
            _room, channel, symbols = world.step(action, world_rng)
            sum_r += _corridor_decision(comp, hist, exam, symbols)
            channels.append(channel)
            travel += 1
            t += 1
        dwell = 0
        while dwell < m6_w3.DWELL and t < L:
            _room, channel, symbols = world.step("stay", world_rng)
            sum_r += _room_decision(
                comp, hist, exam, replay, exam_rng, replay_rng, coin_rng, channel, symbols,
            )
            channels.append(channel)
            dwell += 1
            t += 1
        n_dec = travel + dwell
        reward = sum_r / n_dec if n_dec else 0.0
        m6_w3.m5_w3._ema(v, nvis, target, reward)
        m6_w3._note_visit(reward_q, "room", start, L, reward)
    if len(channels) != L:
        raise RuntimeError(f"{kind} seed {seed} L={L} ran {len(channels)} decisions")
    if kind == "fixed_sleep" and si != len(plan):
        raise RuntimeError(
            f"fixed_sleep seed {seed} L={L} took {si} of {len(plan)} planned sleep visits"
        )
    tag = f"{kind} seed {seed} L={L}"
    overlap = _check_stores(exam, replay, channels, tag)
    # The gap check rescores. Keep score_s as the life, which the wall-time model uses.
    scored_s = exam.score_s
    gap = exam.score_gap()
    exam.score_s = scored_s
    if gap > 0.0:
        raise RuntimeError(f"{tag}: packed score gap {gap:.3e}")
    if drawn[0] != 0:
        raise RuntimeError(f"{tag}: sleep drew {drawn[0]} exam pairs")
    row = m6_w3._finish_row(
        seed, kind, L, comp, world, hist, channels, t0, score, reward_q,
        (int(n_sleep_target) if kind == "fixed_sleep" else None),
    )
    if kind == "fixed_sleep" and row["sleep_decisions"] != int(n_sleep_target):
        raise RuntimeError(
            f"fixed_sleep seed {seed} L={L} sleep decisions {row['sleep_decisions']} "
            f"!= target {int(n_sleep_target)}"
        )
    row["store_size"] = _store_payload(exam, replay)
    row["store_seen"] = _seen_payload(exam, replay)
    row["exam_pairs_drawn"] = int(drawn[0])
    row["score_s"] = float(exam.score_s)
    row["score_gap"] = float(gap)
    row["store_overlap"] = int(overlap)
    row["visits"] = None
    return row


def run_pair(seed, L, score=True):
    """prog_sleep, then fixed_sleep at that life's sleep dose."""
    slept = run_whole(seed, "prog_sleep", L, score=score)
    fixed = run_whole(
        seed, "fixed_sleep", L, score=score, n_sleep_target=slept["sleep_decisions"],
    )
    return slept, fixed


def _check_row(row, tag):
    m6_w3._check_row(row, tag)
    if row["controller"] == "prog":
        if row["store_size"] is not None:
            raise SystemExit(f"{tag}: prog has a split store")
        if row["exam_pairs_drawn"] != 0:
            raise SystemExit(f"{tag}: prog drew exam pairs")
        return
    if row["exam_pairs_drawn"] != 0:
        raise SystemExit(f"{tag}: sleep drew {row['exam_pairs_drawn']} exam pairs")
    if row.get("store_overlap", 0) != 0:
        raise SystemExit(f"{tag}: stores overlap")
    if set(row["store_size"]["exam"]) != set(curio.CH):
        raise SystemExit(f"{tag}: exam sizes missing a channel")
    if set(row["store_size"]["replay"]) != set(curio.CH):
        raise SystemExit(f"{tag}: replay sizes missing a channel")
    for side in ("exam", "replay"):
        for c, n in row["store_size"][side].items():
            if n > R or n < 0:
                raise SystemExit(f"{tag}: {side} {c} size {n}")


def _ref_prog_w():
    return m6_w3._ref_prog_w()


def _print_sizes(exam_sizes, replay_sizes, exam_seen, replay_seen):
    lines = []
    for c in curio.CH:
        lines.append(
            f"  {c} exam {exam_sizes[c]} (offered {exam_seen[c]}) "
            f"replay {replay_sizes[c]} (offered {replay_seen[c]})"
        )
    return lines


def _bench_full_score(repeats=8):
    """Seconds for one pessimistic decision.

    One channel holds R copies of a single pair, the way a constant channel does, so a
    glance there rescores all R of its pairs. The other channels are full of distinct
    pairs and are rescored only where a feature row collides. A decision writes at most
    one glance of features, so a dirty set of every stored feature is not a real case.
    """
    rng = np.random.default_rng(0)
    comp = W3Compressor(lr=LR)
    comp.W += rng.normal(0.0, 0.05, comp.W.shape).astype(np.float32)
    exam = Reservoir(comp, scored=True)
    ctx = tuple(int(x) for x in rng.integers(0, curio.A, 6))
    symbol = int(rng.integers(0, curio.A))
    head = curio.CH[0]
    for i in range(R):
        exam.offer(head, (ctx, symbol), i, rng)
    for c in curio.CH[1:]:
        for i in range(R):
            ctx_i = tuple(int(v) for v in rng.integers(0, curio.A, 6))
            exam.offer(c, (ctx_i, int(rng.integers(0, curio.A))), i, rng)
    dirty = set(exam.feat_rows[head][0])
    exam.rescore(dirty)
    gap = exam.score_gap()
    if gap > 0.0:
        raise SystemExit(f"bench cache gap {gap:.3e}")
    exam.score_s = 0.0
    t0 = time.perf_counter()
    for _ in range(repeats):
        exam.rescore(dirty)
        exam.score_all()
    return (time.perf_counter() - t0) / repeats


def _self_check_score():
    """Packed rows match a full pack, and both match score_pairs."""
    rng = np.random.default_rng(0)
    comp = W3Compressor(lr=LR)
    comp.W += rng.normal(0.0, 0.05, comp.W.shape).astype(np.float32)
    worst = 0.0
    for trial in range(8):
        exam = Reservoir(comp, scored=True)
        channel = curio.CH[int(rng.integers(len(curio.CH)))]
        n_pairs = int(rng.integers(1, 64))
        dummy = np.random.default_rng(1)
        for i in range(n_pairs):
            nctx = int(rng.integers(0, 7))
            ctx = tuple(int(x) for x in rng.integers(0, curio.A, nctx))
            pair = (ctx, int(rng.integers(0, curio.A)))
            exam.offer(channel, pair, i, dummy)
        packed = m6_w3._pack(comp, channel, exam.pairs[channel])
        n = len(exam.pairs[channel])
        got_idx = exam.idx[channel][:n]
        got_mask = exam.mask[channel][:n]
        got_xs = exam.xs[channel][:n]
        if not np.array_equal(got_idx, packed[0]) or not np.array_equal(got_mask, packed[1]):
            raise SystemExit(f"pack rows diverge at trial {trial}")
        if not np.array_equal(got_xs, packed[2]):
            raise SystemExit(f"pack symbols diverge at trial {trial}")
        ref = comp.score_pairs(channel, exam.pairs[channel])
        got = exam.score_channel(channel)
        worst = max(worst, abs(got - ref))
        if worst > 0.0:
            raise SystemExit(f"score pack diverges from score_pairs by {worst:.3e} at trial {trial}")
        hist = [int(x) for x in rng.integers(0, curio.A, 6)]
        symbols = [int(x) for x in rng.integers(0, curio.A, 32)]
        before_w = comp.W.copy()
        comp.bits(channel, hist, symbols, learn=True)
        dirty = _touched(comp, channel, hist, symbols)
        changed = set(np.flatnonzero(np.any(comp.W != before_w, axis=1)).tolist())
        if not changed <= dirty:
            raise SystemExit(f"touched features missed a written row at trial {trial}")
        exam.rescore(dirty)
        before_w = comp.W.copy()
        comp.bits("corridor", [], symbols, learn=True)
        dirty = _touched(comp, "corridor", [], symbols)
        changed = set(np.flatnonzero(np.any(comp.W != before_w, axis=1)).tolist())
        if not changed <= dirty:
            raise SystemExit(f"corridor touched features missed a written row at trial {trial}")
        exam.rescore(dirty)
        # A stored pair's own row, written directly, must be picked up by rescore.
        feat = int(exam.feat_rows[channel][0][0])
        comp.W[feat] += np.float32(0.01)
        exam.rescore({feat})
        gap = exam.score_gap()
        worst = max(worst, gap)
        if gap > 0.0:
            raise SystemExit(f"cache diverges from score_pairs by {gap:.3e} at trial {trial}")
    return worst


def _project_life(row, L, full_decision_s):
    """Upper bound. Learn time scales with L. Scoring is charged at the repetitive-channel cost.

    Evaluation length is fixed, so eval time is not scaled.
    """
    scale = L / float(row["L"])
    learn = max(0.0, row["wall_s"] - row["eval_s"] - row["score_s"])
    return learn * scale + full_decision_s * L + row["eval_s"]


def _pool_wall(durations, workers):
    loads = [0.0] * workers
    for dur in sorted(durations, reverse=True):
        loads[0] += dur
        loads.sort()
    return max(loads)


def _estimate(timed, full_decision_s):
    """One seed, pool of 4, then six seeds run one after another.

    fixed_sleep waits on prog_sleep, so those two are one chain job.
    prog keeps M5's reservoir, so its measured wall is scaled linearly.
    """
    def at(controller, L):
        if controller == "prog":
            return timed["prog"]["wall_s"] * (L / float(timed["prog"]["L"]))
        return _project_life(timed[controller], L, full_decision_s)

    jobs = []
    for L in LENGTHS:
        jobs.append(at("prog", L))
        jobs.append(at("prog_total", L))
        jobs.append(at("prog_sleep", L) + at("fixed_sleep", L))
    one = _pool_wall(jobs, 4)
    six = one * len(SEEDS)
    return {
        "L128_wall_s": {c: timed[c]["wall_s"] for c in CONTROLLERS},
        "full_exam_score_decision_s": full_decision_s,
        "one_seed_wall_s": one,
        "one_seed_wall_min": one / 60.0,
        "six_seed_wall_s": six,
        "six_seed_wall_min": six / 60.0,
        "limit_s": KERNEL_WALL_LIMIT_S,
        "one_seed_over": one > KERNEL_WALL_LIMIT_S,
        "six_seed_over": six > KERNEL_WALL_LIMIT_S,
        "jobs_s": jobs,
    }


def smoke():
    """Four checks from the spec. Exit before any kernel if one fails."""
    m5b_w3.install()
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    say("smoke check 0: packed score matches score_pairs")
    worst = _self_check_score()
    say(f"  packed score abs {worst:.3e} on 8 random reservoirs")

    say("smoke check 1: prog seed 12 L=512 vs results/m5bw3/results.json")
    ref = _ref_prog_w()
    repro = run_prog(12, 512, score=True)
    dW = abs(repro["W"] - ref)
    say(
        f'  prog W {repro["W"]:.12f} ref {ref:.12f} abs {dW:.3e} '
        f'W9 {repro["W9"]:.6f} finite {repro["finite"]} wall {repro["wall_s"]:.2f}s'
    )
    _check_row(repro, "smoke check 1")
    if dW > 1e-9:
        raise SystemExit(f"smoke check 1 failed: abs W {dW:.3e} > 1e-9")
    if repro["sleep_decisions"] != 0 or repro["sleep_share"] != 0.0:
        raise SystemExit("smoke check 1 failed: prog slept")

    say("smoke check 2: prog_sleep seed 24 L=256, disjoint stores")
    say("  stream index is the symbol position in that channel's observed stream")
    life = run_whole(SMOKE_SEED, "prog_sleep", 256, score=True)
    _check_row(life, "smoke check 2")
    for line in _print_sizes(
        life["store_size"]["exam"], life["store_size"]["replay"],
        life["store_seen"]["exam"], life["store_seen"]["replay"],
    ):
        say(line)
    say(
        f'  overlap {life["store_overlap"]} score_gap {life["score_gap"]:.3e} '
        f'W {life["W"]:.6f} finite {life["finite"]} wall {life["wall_s"]:.2f}s'
    )
    if life["store_overlap"] != 0:
        raise SystemExit("smoke check 2 failed: a stream index is in both stores")

    say("smoke check 3: exam pairs drawn by sleep during that life")
    say(f'  exam_pairs_drawn {life["exam_pairs_drawn"]}')
    if life["exam_pairs_drawn"] != 0:
        raise SystemExit(
            f'smoke check 3 failed: sleep drew {life["exam_pairs_drawn"]} exam pairs'
        )

    say("smoke check 4: seed 24 L=128, four controllers, W and W9 finite")
    timed = {}
    rows = []
    order = ("prog", "prog_total", "prog_sleep", "fixed_sleep")
    slept_n = None
    for controller in order:
        if controller == "fixed_sleep":
            row = run_whole(SMOKE_SEED, "fixed_sleep", 128, score=True, n_sleep_target=slept_n)
        elif controller == "prog":
            row = run_prog(SMOKE_SEED, 128, score=True)
        else:
            row = run_whole(SMOKE_SEED, controller, 128, score=True)
            if controller == "prog_sleep":
                slept_n = row["sleep_decisions"]
        say(
            f'  {controller} W {row["W"]:.6f} W9 {row["W9"]:.6f} finite {row["finite"]} '
            f'sleep {row["sleep_share"]:.4f} corridor {row["corridor_share"]:.4f} '
            f'wall {row["wall_s"]:.2f}s learn {row["learn_s"]:.2f}s eval {row["eval_s"]:.2f}s '
            f'score {row["score_s"]:.2f}s'
        )
        _check_row(row, f"smoke check 4 {controller}")
        if not row["finite"]:
            raise SystemExit(f"smoke check 4 failed: {controller} W is not finite")
        timed[controller] = row
        rows.append(row)
    fixed = rows[-1]
    if fixed["sleep_decisions"] != slept_n:
        raise SystemExit(
            f"smoke check 4 failed: fixed_sleep decisions {fixed['sleep_decisions']} != prog_sleep {slept_n}"
        )
    if abs(fixed["sleep_share"] - rows[-2]["sleep_share"]) > 1e-12:
        raise SystemExit("smoke check 4 failed: sleep shares differ")

    say("  pessimistic exam rescore, one repetitive channel of R pairs")
    full_decision_s = _bench_full_score()
    say(f"  pessimistic exam rescore per decision {full_decision_s:.4f}s")
    est = _estimate(timed, full_decision_s)
    say(
        f'estimate one-seed kernel wall {est["one_seed_wall_s"]:.0f}s '
        f'({est["one_seed_wall_min"]:.1f} min), six seeds sequential '
        f'{est["six_seed_wall_s"]:.0f}s ({est["six_seed_wall_min"]:.1f} min), '
        f'pool 4 limit {KERNEL_WALL_LIMIT_S / 60:.0f} min'
    )
    say("estimate charges every decision a full exam store and is an upper bound from this VPS")
    if est["one_seed_over"]:
        say("estimate exceeds 90 min for one seed; do not push kernels")
    elif est["six_seed_over"]:
        say("estimate exceeds 90 min for six sequential seeds; split the kernel by seed")
    else:
        say("estimate is within 90 min for one kernel of six seeds")
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "score_abs": worst,
        "reproduce": {
            "seed": 12,
            "L": 512,
            "controller": "prog",
            "W": repro["W"],
            "W_ref": ref,
            "abs_W": dW,
            "W9": repro["W9"],
            "wall_s": repro["wall_s"],
            "finite": repro["finite"],
        },
        "disjoint": {
            "seed": SMOKE_SEED,
            "L": 256,
            "controller": "prog_sleep",
            "overlap": life["store_overlap"],
            "exam_pairs_drawn": life["exam_pairs_drawn"],
            "store_size": life["store_size"],
            "store_seen": life["store_seen"],
            "score_gap": life["score_gap"],
            "W": life["W"],
            "W9": life["W9"],
            "finite": life["finite"],
            "wall_s": life["wall_s"],
            "sleep_decisions": life["sleep_decisions"],
        },
        "L128": [{
            "controller": r["controller"],
            "seed": r["seed"],
            "L": r["L"],
            "W": r["W"],
            "W9": r["W9"],
            "finite": r["finite"],
            "sleep_share": r["sleep_share"],
            "sleep_decisions": r["sleep_decisions"],
            "corridor_share": r["corridor_share"],
            "wall_s": r["wall_s"],
            "learn_s": r["learn_s"],
            "eval_s": r["eval_s"],
            "score_s": r["score_s"],
            "store_size": r["store_size"],
        } for r in rows],
        "estimate": est,
        "over_limit": est["one_seed_over"],
        "split_by_seed": (not est["one_seed_over"]) and est["six_seed_over"],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
    }


def _job(args):
    seed, kind, L = args
    m5b_w3.install()
    if kind == "prog":
        rows = [run_prog(seed, L, score=True)]
    elif kind == "prog_total":
        rows = [run_whole(seed, "prog_total", L, score=True)]
    elif kind == "chain":
        rows = list(run_pair(seed, L, score=True))
    else:
        raise ValueError(kind)
    for row in rows:
        row.pop("visits", None)
        print(
            f's{seed} {row["controller"]} L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
            f'sleep {row["sleep_share"]:.3f} corridor {row["corridor_share"]:.3f} '
            f'wall {row["wall_s"]:.1f}s score {row["score_s"]:.1f}s',
            flush=True,
        )
        _check_row(row, f's{seed} {row["controller"]} L={L}')
    return rows


def kernel_jobs(seed):
    """Long lives first. fixed_sleep is inside the chain, after prog_sleep."""
    jobs = []
    for L in (SCORED_L, REPORT_L):
        jobs.append((seed, "chain", L))
        jobs.append((seed, "prog_total", L))
    for L in (SCORED_L, REPORT_L):
        jobs.append((seed, "prog", L))
    return jobs


def kernel_main(seed, out, workers=4, kernel_id=None):
    if int(workers) != 4:
        raise SystemExit("kernel pool is 4")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 24-29")
    m5b_w3.install()
    t0 = time.perf_counter()
    jobs = kernel_jobs(seed)
    print(f"kernel {kernel_id} seed {seed} jobs {len(jobs)} workers {workers}", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        nested = list(pool.imap_unordered(_job, jobs, chunksize=1))
    runs = [row for batch in nested for row in batch]
    wall = time.perf_counter() - t0
    if len(runs) != len(CONTROLLERS) * len(LENGTHS):
        raise SystemExit(f"expected {len(CONTROLLERS) * len(LENGTHS)} runs, got {len(runs)}")
    for row in runs:
        if not row["finite"]:
            raise SystemExit(f's{seed} {row["controller"]} L={row["L"]} is not finite')
        if row["controller"] == "fixed_sleep":
            match = [
                r for r in runs
                if r["controller"] == "prog_sleep" and r["L"] == row["L"] and r["seed"] == seed
            ]
            if len(match) != 1 or row["sleep_decisions"] != match[0]["sleep_decisions"]:
                raise SystemExit(f"fixed_sleep dose mismatch at L={row['L']}")
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "workers": workers,
        "wall_s": wall,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "runs": runs,
    }
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"m6b_s{seed}.json")
    json.dump(payload, open(path, "w"), indent=1)
    open(path, "a").write("\n")
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _pick(runs, controller, L, seed):
    hit = [r for r in runs if r["controller"] == controller and r["L"] == L and r["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} L={L} seed {seed}, got {len(hit)}")
    return hit[0]


def _mean_std(xs):
    return m6_w3._mean_std(xs)


def summarize(runs):
    by = {}
    for L in LENGTHS:
        for controller in CONTROLLERS:
            rows = sorted(
                (r for r in runs if r["controller"] == controller and r["L"] == L),
                key=lambda r: r["seed"],
            )
            if len(rows) != len(SEEDS):
                raise RuntimeError(f"{controller} L={L}: {len(rows)} rows, expected {len(SEEDS)}")
            w_mu, w_sd = _mean_std(r["W"] for r in rows)
            w9_mu, w9_sd = _mean_std(r["W9"] for r in rows)
            held_mu = {c: _mean_std(r["held"][c] for r in rows)[0] for c in curio.LEARN}
            held9_mu = {c: _mean_std(r["held9"][c] for r in rows)[0] for c in curio.CH}
            share_mu = {c: _mean_std(r["share"][c] for r in rows)[0] for c in curio.CH}
            corr_mu = _mean_std(r["corridor_share"] for r in rows)[0]
            sleep_mu, sleep_sd = _mean_std(r["sleep_share"] for r in rows)
            quarter = []
            for i in range(4):
                q_share = {c: _mean_std(r["quarter"][i]["share"][c] for r in rows)[0] for c in curio.CH}
                q_corr = _mean_std(r["quarter"][i]["corridor"] for r in rows)[0]
                q_sleep = _mean_std(r["quarter"][i]["sleep"] for r in rows)[0]
                quarter.append({"share": q_share, "corridor": q_corr, "sleep": q_sleep})
            by[f"{controller}/L{L}"] = {
                "controller": controller,
                "L": L,
                "n": len(rows),
                "W_mean": w_mu,
                "W_std_ddof0": w_sd,
                "W9_mean": w9_mu,
                "W9_std_ddof0": w9_sd,
                "held_mean": held_mu,
                "held9_mean": held9_mu,
                "share_mean": share_mu,
                "corridor_mean": corr_mu,
                "sleep_mean": sleep_mu,
                "sleep_std_ddof0": sleep_sd,
                "quarter_mean": quarter,
                "per_seed": [{
                    "seed": r["seed"],
                    "W": r["W"],
                    "W9": r["W9"],
                    "held": r["held"],
                    "held9": r["held9"],
                    "share": r["share"],
                    "corridor_share": r["corridor_share"],
                    "sleep_share": r["sleep_share"],
                    "sleep_decisions": r["sleep_decisions"],
                    "quarter": r["quarter"],
                    "reward_quarter": r["reward_quarter"],
                    "room_channel": r["room_channel"],
                    "wall_s": r["wall_s"],
                    "store_size": r["store_size"],
                    "store_seen": r["store_seen"],
                    "exam_pairs_drawn": r["exam_pairs_drawn"],
                } for r in rows],
            }
    gaps = []
    for L in LENGTHS:
        for s in SEEDS:
            prog = _pick(runs, "prog", L, s)
            total = _pick(runs, "prog_total", L, s)
            slept = _pick(runs, "prog_sleep", L, s)
            fixed = _pick(runs, "fixed_sleep", L, s)
            gaps.append({
                "L": L,
                "seed": s,
                "prog_sleep": slept["W"],
                "prog_total": total["W"],
                "fixed_sleep": fixed["W"],
                "prog": prog["W"],
                "prog_sleep_minus_prog_total": slept["W"] - total["W"],
                "prog_sleep_minus_fixed_sleep": slept["W"] - fixed["W"],
                "prog_total_minus_prog": total["W"] - prog["W"],
            })
    return {"by_controller": by, "gaps": gaps}


def _fmt(x):
    return f"{x:.6f}"


def _pm(mu, sd):
    return f"{_fmt(mu)} ± {_fmt(sd)}"


def _seed_header():
    return " | ".join(f"seed {s}" for s in SEEDS)


def _cell_mean(value):
    if value is None:
        return "—"
    return _fmt(value)


def _store_total(sizes):
    if sizes is None:
        return None
    return int(sum(sizes.values()))


def render_markdown(payload):
    by = payload["summary"]["by_controller"]
    lines = []
    lines.append("# M6b-W3 sleep")
    lines.append("")
    lines.append(
        "Sleep is an action on the W3 ring. Replay and exam are disjoint reservoirs of 2,048 pairs "
        "per channel. Progress is scored on the exam stores. "
        "W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. "
        "W9 is that mean over all 9 room channels. Lower is better. Numbers only."
    )
    lines.append("")
    for title, key_mu, key_sd, field in (
        ("W", "W_mean", "W_std_ddof0", "W"),
        ("W9", "W9_mean", "W9_std_ddof0", "W9"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for L in LENGTHS:
            lines.append(f"### L = {L}")
            lines.append("")
            lines.append(f"| controller | {_seed_header()} | mean ± std ddof=0 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for controller in CONTROLLERS:
                block = by[f"{controller}/L{L}"]
                cells = []
                for s in SEEDS:
                    hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                    cells.append(_fmt(hit[field]))
                lines.append(
                    f"| {controller} | " + " | ".join(cells)
                    + f' | {_pm(block[key_mu], block[key_sd])} |'
                )
            lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("On W. prog_sleep minus prog_total, prog_sleep minus fixed_sleep, prog_total minus prog.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |")
        lines.append("|---:|---:|---:|---:|")
        gaps = [g for g in payload["summary"]["gaps"] if g["L"] == L]
        for g in gaps:
            lines.append(
                f'| {g["seed"]} | {_fmt(g["prog_sleep_minus_prog_total"])} | '
                f'{_fmt(g["prog_sleep_minus_fixed_sleep"])} | {_fmt(g["prog_total_minus_prog"])} |'
            )
        mu1, sd1 = _mean_std(g["prog_sleep_minus_prog_total"] for g in gaps)
        mu2, sd2 = _mean_std(g["prog_sleep_minus_fixed_sleep"] for g in gaps)
        mu3, sd3 = _mean_std(g["prog_total_minus_prog"] for g in gaps)
        n1 = sum(g["prog_sleep_minus_prog_total"] < 0 for g in gaps)
        n2 = sum(g["prog_sleep_minus_fixed_sleep"] < 0 for g in gaps)
        n3 = sum(g["prog_total_minus_prog"] < 0 for g in gaps)
        lines.append(f"| mean | {_fmt(mu1)} | {_fmt(mu2)} | {_fmt(mu3)} |")
        lines.append(f"| std ddof=0 | {_fmt(sd1)} | {_fmt(sd2)} | {_fmt(sd3)} |")
        lines.append(f"| left lower (count) | {n1} of 6 | {n2} of 6 | {n3} of 6 |")
        lines.append("")
    lines.append("## Sleep share")
    lines.append("")
    lines.append("Sleep decisions divided by L. fixed_sleep uses prog_sleep's count on the same seed and L.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append(f"| controller | {_seed_header()} | mean ± std ddof=0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = []
            for s in SEEDS:
                hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                cells.append(_fmt(hit["sleep_share"]))
            lines.append(
                f"| {controller} | " + " | ".join(cells)
                + f' | {_pm(block["sleep_mean"], block["sleep_std_ddof0"])} |'
            )
        lines.append("")
    lines.append("## prog_sleep sleep share by quarter")
    lines.append("")
    lines.append("Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |")
        lines.append("|---:|---:|---:|---:|---:|")
        block = by[f"prog_sleep/L{L}"]
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            cells = [_fmt(hit["quarter"][i]["sleep"]) for i in range(4)]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        means = [_fmt(block["quarter_mean"][i]["sleep"]) for i in range(4)]
        lines.append("| mean | " + " | ".join(means) + " |")
        lines.append("")
    lines.append("## Decision shares")
    lines.append("")
    lines.append(
        "Mean over seeds. A stay counts for the room channel. A move counts for the corridor. "
        "A sleep decision counts for sleep. Shares sum to 1."
    )
    lines.append("")
    header = "| controller | L | " + " | ".join(list(curio.CH) + ["corridor", "sleep"]) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * 11)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [f'{block["share_mean"][c]:.4f}' for c in curio.CH]
            cells.append(f'{block["corridor_mean"]:.4f}')
            cells.append(f'{block["sleep_mean"]:.4f}')
            lines.append(f"| {controller} | {L} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## prog_sleep visit rewards by quarter")
    lines.append("")
    lines.append(
        "Exam-store bits saved per decision, mean over the visits whose first decision falls in the quarter. "
        "A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, "
        "over seeds that have at least one such visit."
    )
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        block = by[f"prog_sleep/L{L}"]
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            for i, slot in enumerate(hit["reward_quarter"], 1):
                lines.append(
                    f"| {s} | {i} | {slot['sleep_n']} | {_cell_mean(slot['sleep_mean'])} | "
                    f"{slot['room_n']} | {_cell_mean(slot['room_mean'])} |"
                )
        for i in range(4):
            sleep_vals = []
            room_vals = []
            sleep_n = 0
            room_n = 0
            for s in SEEDS:
                hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                slot = hit["reward_quarter"][i]
                sleep_n += slot["sleep_n"]
                room_n += slot["room_n"]
                if slot["sleep_mean"] is not None:
                    sleep_vals.append(slot["sleep_mean"])
                if slot["room_mean"] is not None:
                    room_vals.append(slot["room_mean"])
            sleep_mu = _fmt(_mean_std(sleep_vals)[0]) if sleep_vals else "—"
            room_mu = _fmt(_mean_std(room_vals)[0]) if room_vals else "—"
            lines.append(
                f"| mean | {i + 1} | {sleep_n} | {sleep_mu} | {room_n} | {room_mu} |"
            )
        lines.append("")
    lines.append("## Mean held-out bits")
    lines.append("")
    lines.append("Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.")
    lines.append("")
    lines.append("| controller | L | " + " | ".join(list(curio.CH)) + " | W | W9 |")
    lines.append("|---|---:|" + "---:|" * 11)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [_fmt(block["held9_mean"][c]) for c in curio.CH]
            lines.append(
                f"| {controller} | {L} | " + " | ".join(cells)
                + f' | {_fmt(block["W_mean"])} | {_fmt(block["W9_mean"])} |'
            )
    lines.append("")
    lines.append("## Store sizes")
    lines.append("")
    lines.append(
        "Pairs stored at the end of life. R = 2,048 per channel. "
        "prog keeps M5's own reservoir and has no split store. "
        "A total is the sum over the 9 room channels."
    )
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| controller | seed | store | " + " | ".join(curio.CH) + " | total |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for controller in CONTROLLERS:
            if controller == "prog":
                continue
            block = by[f"{controller}/L{L}"]
            for s in SEEDS:
                hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                for side in ("exam", "replay"):
                    sizes = hit["store_size"][side]
                    cells = [str(int(sizes[c])) for c in curio.CH]
                    lines.append(
                        f"| {controller} | {s} | {side} | " + " | ".join(cells)
                        + f" | {sum(int(sizes[c]) for c in curio.CH)} |"
                    )
        lines.append("")
    lines.append("## Ring order")
    lines.append("")
    lines.append("Channel in rooms 0..8, seeds 24–29. One order per seed; controllers share it.")
    lines.append("")
    lines.append("| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        rooms = _pick(payload["runs"], "prog", SCORED_L, s)["room_channel"]
        lines.append(f"| {s} | " + " | ".join(rooms) + " |")
    lines.append("")
    lines.append("## Walls")
    lines.append("")
    kernel_wall = payload.get("kernel_wall_s")
    if kernel_wall is None:
        lines.append(f'Smoke wall {payload["smoke_wall_s"]:.3f} s.')
    else:
        lines.append(
            f'Smoke wall {payload["smoke_wall_s"]:.3f} s. '
            f'Kernel wall {kernel_wall:.3f} s.'
        )
    lines.append("")
    lines.append("| kernel | seed | wall seconds |")
    lines.append("|---|---:|---:|")
    for item in payload["kernels"]:
        lines.append(f'| {item["kernel_id"]} | {item["seed"]} | {item["wall_s"]:.3f} |')
    lines.append("")
    lines.append("| controller | L | seed | wall seconds |")
    lines.append("|---|---:|---:|---:|")
    order = [(c, L) for L in LENGTHS for c in CONTROLLERS]
    runs_sorted = sorted(payload["runs"], key=lambda r: (order.index((r["controller"], r["L"])), r["seed"]))
    for r in runs_sorted:
        lines.append(f'| {r["controller"]} | {r["L"]} | {r["seed"]} | {r["wall_s"]:.3f} |')
    lines.append("")
    lines.append("## Smoke test")
    lines.append("")
    lines.append("```")
    lines.append(payload["smoke_text"].rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- `prog` is `m5_w3.run_controller` with the M5 per-channel reward and no sleep. "
        "W9 is `m5b_w3.score_nz` on that frozen compressor. `m6_w3.py`, `m5_w3.py`, `m5b_w3.py`, "
        "`curio.py` and `w3_world.py` are imported and not edited."
    )
    lines.append(
        "- `prog_total` is M5's 9-room bandit. `prog_sleep` is the same bandit over 10 arms. "
        "Sleep is the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3. "
        "Choice rng is `default_rng(10_000 + seed)`. N-channel noise and corridor steps use "
        "`default_rng(1_000 + seed)`."
    )
    lines.append(
        "- Every (context, symbol) pair from a room glance is scored and learned, then assigned by a fair "
        "coin, `default_rng(40_000 + seed).integers(2) == 0`, to the exam store, and otherwise to the replay "
        "store. Each store is a reservoir of R = 2,048 pairs per channel with its own offer counter. "
        "Exam replacement uses `default_rng(20_000 + seed)`. Replay replacement uses "
        "`default_rng(50_000 + seed)`. The replacement rng is drawn only when that channel's reservoir is full. "
        "Corridor glances are learned and enter neither store."
    )
    lines.append(
        "- A decision's progress is the sum over the 9 room channels of `score_pairs` on the exam store "
        "before the learning step minus `score_pairs` after it, on the pairs stored before the decision's "
        "own pairs enter. The visit reward is that sum over the visit's decisions, divided by the decisions "
        "the visit took. Travel counts."
    )
    lines.append(
        "- The exam score is the sum of per-pair codelengths. A pair is recomputed when a learning step "
        "writes one of its feature rows. The sum is checked against `SGDCompressor.score_pairs`. "
        "The smoke check reports the absolute gap. A stream index is the symbol's position in that "
        "channel's observed stream. Smoke check 2 prints the store sizes after a 256-decision life."
    )
    lines.append(
        "- A sleep decision draws 64 pairs uniformly with replacement from the union of the replay stores, "
        "rng `default_rng(30_000 + seed)`, and applies the compressor's learning step to each pair on its "
        "own channel. It does not draw from an exam store. Drawn pairs enter no reservoir. An empty union "
        "learns nothing. A sleep visit is 8 such decisions where the agent stands, with no travel. Its "
        "reward is the summed exam-store progress divided by the decisions it took. "
        f"N is EVAL uniform symbols from `default_rng({m5b_w3.N_EVAL_BASE} + seed)`, a fresh generator, "
        "not the sleep rng."
    )
    lines.append(
        "- `fixed_sleep` is the `prog_total` controller plus planned sleep visits. The visit lengths are 8, "
        "with a shorter last visit when the count is not a multiple of 8, and the lengths sum to "
        "`prog_sleep`'s sleep decisions on that seed and L. Awake gaps before, between and after the visits "
        "differ by at most one decision. A due sleep that falls inside a room visit starts on the next "
        "decision after that visit. A room visit is not started when the decisions left would not cover "
        "the sleeps still owed plus one maximum room visit (4 moves + 8 dwells)."
    )
    lines.append(
        "- W is `w3_headroom.evaluate`. W9 adds N and Z on the same frozen compressor. "
        "Z is EVAL space symbols. Corridor is not in W or W9."
    )
    lines.append(
        "- Quarter i is decisions [(i−1)L/4, iL/4). A visit reward is counted in the quarter of its first "
        "decision. Seeds 24–29. L = 2048 is the scored life. L = 512 is the report life. Nothing retuned."
    )
    lines.append(
        "- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Private Kaggle kernels, pool of 4, no GPU, no internet."
    )
    lines.append("")
    return "\n".join(lines)


def assemble(parts, smoke_payload, smoke_wall_s, out_dir, kernel_wall_s=None):
    runs = []
    kernels = []
    for part in parts:
        kernels.append({
            "kernel_id": part["kernel_id"],
            "seed": part["seed"],
            "wall_s": part["wall_s"],
            "workers": part["workers"],
        })
        runs.extend(part["runs"])
    if len(runs) != len(SEEDS) * len(CONTROLLERS) * len(LENGTHS):
        raise SystemExit(f"expected {len(SEEDS) * len(CONTROLLERS) * len(LENGTHS)} runs, got {len(runs)}")
    for row in runs:
        _check_row(row, f's{row["seed"]} {row["controller"]} L={row["L"]}')
    summary = summarize(runs)
    payload = {
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "controllers": list(CONTROLLERS),
        "R": R,
        "kernels": kernels,
        "kernel_wall_s": kernel_wall_s,
        "smoke_wall_s": smoke_wall_s,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "runs": runs,
        "summary": summary,
        "smoke": {
            k: smoke_payload[k]
            for k in (
                "reproduce", "disjoint", "L128", "estimate", "over_limit", "split_by_seed",
                "score_abs", "pythonhashseed", "curio_lr",
            )
            if k in smoke_payload
        },
        "smoke_text": smoke_payload["text"],
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    md_path = os.path.join(out_dir, "RESULTS.md")
    json.dump(payload, open(json_path, "w"), indent=1)
    open(json_path, "a").write("\n")
    open(md_path, "w").write(render_markdown(payload))
    print(f"wrote {md_path} and {json_path}", flush=True)
    return payload


def _load_smoke(path):
    payload = json.load(open(path))
    if "text" not in payload or "reproduce" not in payload:
        raise SystemExit(f"smoke file {path} is missing fields")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kernel-id", default=None)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--kernel-summary", default=None)
    ap.add_argument("--smoke-json", default="results/m6bw3/smoke.json")
    ap.add_argument("--out-dir", default="results/m6bw3")
    a = ap.parse_args()
    if a.kernel:
        kernel_main(a.seed, a.out_dir, workers=a.workers, kernel_id=a.kernel_id)
        return
    if a.assemble:
        smoke_payload = _load_smoke(a.smoke_json)
        parts = [json.load(open(path)) for path in a.parts]
        kernel_wall_s = None
        if a.kernel_summary:
            kernel_wall_s = float(json.load(open(a.kernel_summary))["wall_s"])
        assemble(
            parts, smoke_payload, smoke_payload.get("smoke_wall_s", float("nan")),
            a.out_dir, kernel_wall_s=kernel_wall_s,
        )
        return
    t_smoke = time.perf_counter()
    result = smoke()
    smoke_wall = time.perf_counter() - t_smoke
    print(f"smoke wall {smoke_wall:.3f}s", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {**result, "smoke_wall_s": smoke_wall}
    json.dump(out, open(os.path.join(a.out_dir, "smoke.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "smoke.json"), "a").write("\n")
    if result["over_limit"]:
        raise SystemExit("smoke estimate exceeds 90 min for one seed; kernels were not pushed")


if __name__ == "__main__":
    main()
