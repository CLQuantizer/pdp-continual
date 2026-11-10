"""M4a: one life with cortex, utility store, and a curious controller.

Protocol: handoffs/2026-11-10_grok_m4a_one_life.md. Numbers only; no verdict.
Creates only this file and results/m4a/*. Does not edit earlier scripts.

Readings of the spec, fixed before the scored runs:
- Channels are F_PROCS + N + Z (10). Alphabet is F_ALPHA (42). N is uniform on
  those 42 symbols. Z is the space symbol, as in curio.
- Optimistic first pass over all 10 channels, then the controller, as in H25.
  rand8 is uniform on the 8 learnable channels after that pass (H25's rand7).
- One UtilityStore per channel so order-5 keys cannot cross channels. Mixture
  weights are per channel (initial 0.5/0.5, fixed share 0.001).
- Budget is 0.15 * (observed symbols * log2(42)), not H24g's full5-peak formula.
  M is still stored symbols * log2(42) + 32 bits/window, summed over channels.
  Decay and the 32-glance protection use the global glance index.
- Replay samples stored windows only, one channel per column, zero hidden state,
  zero embedding at each window boundary. Skipped when the store is empty.
  A deferred surprise tail whose head was already evicted is not rewritten.
- GRU Adam steps minimise cross-entropy. The mixture is the coder for writes,
  match credit, prequential bits and W. gmc_prop / slope_prop use the imported
  H25b / H25d estimators on the GRU, unchanged.
- rand_nostore has no store, no match and no replay. Its W is the GRU alone.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bisect import bisect_right

from h24b_world import F_ALPHA, F_PROCS, FactWorld
from h24e_priced_store import A
from h24g_utility_store import (DECAY, PROTECT, THRESHOLD, UtilityStore,
                               surprise_intervals)
from h25a_curio_neural import RES
from h25b_progress_fix import NeuralCompressor, gmc_reward
from h25d_visit_slope import slope_reward
from h21a_substrate import mean_std, now

CH = F_PROCS + ['N', 'Z']
LEARN = list(F_PROCS)
B = 64
EVAL = 2048
EPS = 0.1
BETA = 0.3
BUDGET_FRACTION = 0.15
LOG2A = math.log2(A)
SPACE = F_ALPHA.index(' ')
CONDS = ['rand', 'rand8', 'gmc_prop', 'slope_prop', 'rand_nostore']
WALL_BUDGET = 7200.0
assert A == len(F_ALPHA) == 42 and len(CH) == 10 and len(LEARN) == 8
assert SPACE == F_ALPHA.index(' ') == 39 and F_ALPHA[-1] == '='


class Cortex(NeuralCompressor):
    """H25 GRU with the 42-symbol alphabet and 10 channel ids."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(A, 32)
        self.channel = nn.Embedding(len(CH), 8)
        self.readout = nn.Linear(256, A)


def configure_threads():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def observe(world, channel, rng):
    if channel == 'N':
        return rng.integers(0, A, size=B).astype(np.int64)
    if channel == 'Z':
        return np.full(B, SPACE, dtype=np.int64)
    text = world.take(channel, B)
    assert len(text) == B
    return np.fromiter((F_ALPHA.index(ch) for ch in text), dtype=np.int64, count=B)


def choose(cond, g, values, rng):
    if g < len(CH):
        return g
    if cond in ('rand', 'rand_nostore'):
        return int(rng.integers(len(CH)))
    if cond == 'rand8':
        return int(rng.integers(len(LEARN)))
    weights = np.maximum(values, 0.0) + 0.01
    return int(rng.choice(len(CH), p=weights / weights.sum()))


def frozen_likelihood(store, context, symbol):
    """Match5 probability without updating counts or the window pointer."""
    if len(context) < 5 or not store.starts:
        return 1.0 / A
    pos = store.last.get(bytes(context[-5:]))
    if pos is None:
        return 1.0 / A
    j = bisect_right(store.starts, pos) - 1
    if j < 0:
        return 1.0 / A
    lo = store.starts[j]
    payload = store.windows[j]
    if pos >= lo + len(payload):
        return 1.0 / A
    offset = pos - lo
    if offset < 5:
        return 1.0 / A
    length = 5
    while (length < 32 and length < offset and length < len(context)
           and context[-length - 1] == payload[offset - length - 1]):
        length += 1
    bucket = 0 if length < 8 else 1 if length < 12 else 2 if length < 16 else 3 if length < 24 else 4
    confidence = (store.hits[bucket] + 1) / (store.trials[bucket] + 2)
    correct = int(payload[offset]) == int(symbol)
    return float(confidence if correct else (1.0 - confidence) / (A - 1))


def window_ends_at(store, pos):
    j = bisect_right(store.starts, pos) - 1
    if j < 0:
        return False
    return store.starts[j] + len(store.windows[j]) == pos


def retained_memory(stores):
    total = 0.0
    for store in stores:
        store.stored = sum(len(w) for w in store.windows)
        total += store.memory()
    return total


def evict_global(stores, glance, budget):
    dropped = 0
    while True:
        mem = retained_memory(stores)
        if mem <= budget:
            break
        best, victim = None, None
        for ci, store in enumerate(stores):
            for i, born in enumerate(store.born):
                if glance < born + PROTECT:
                    continue
                key = (store.utils[i], store.starts[i], ci, i)
                if best is None or key < best:
                    best, victim = key, (ci, i)
        if victim is None:
            break
        stores[victim[0]]._drop(victim[1])
        dropped += 1
    for store in stores:
        if store.index_dirty:
            store.reindex()
    return dropped


def sample_replay(stores, rng):
    owners = [ci for ci, store in enumerate(stores) if store.windows]
    if not owners:
        return None
    counts = np.array([len(stores[ci].windows) for ci in owners], dtype=np.float64)
    probs = counts / counts.sum()
    targets = np.empty((B, 8), dtype=np.int64)
    previous = np.full((B, 8), -1, dtype=np.int64)
    cis = np.empty(8, dtype=np.int64)
    for col in range(8):
        ci = int(rng.choice(np.array(owners), p=probs))
        cis[col] = ci
        store = stores[ci]
        filled, nwin = 0, len(store.windows)
        while filled < B:
            window = store.windows[int(rng.integers(nwin))]
            offset = int(rng.integers(len(window) - B + 1)) if len(window) >= B else 0
            take = min(B - filled, len(window) - offset)
            targets[filled:filled + take, col] = window[offset:offset + take]
            if offset:
                previous[filled, col] = int(window[offset - 1])
            if take > 1:
                previous[filled + 1:filled + take, col] = window[offset:offset + take - 1]
            filled += take
    return torch.from_numpy(previous), torch.from_numpy(targets), torch.from_numpy(cis)


def run(cond, seed, glances, phase):
    assert cond in CONDS
    started = time.monotonic()
    torch.manual_seed(seed)
    model = Cortex()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world = FactWorld(seed)
    rng = np.random.default_rng(10_000 + seed)
    rrng = np.random.default_rng(20_000 + seed)
    replay_rng = np.random.default_rng(30_000 + seed)
    nostore = cond == 'rand_nostore'
    hist = torch.zeros((len(CH), glances * B), dtype=torch.long)
    lengths = np.zeros(len(CH), dtype=np.int64)
    states = [None] * len(CH)
    values = np.zeros(len(CH))
    counts = np.zeros(len(CH), dtype=int)
    reservoirs = [[] for _ in CH]
    visit_scores = [[] for _ in CH]
    contexts = [bytearray() for _ in CH]
    weights = [np.full(2, 0.5) for _ in CH]
    bufs = [np.zeros(glances * B, dtype=np.uint8) for _ in CH]
    pendings = [0 for _ in CH]
    stores = None if nostore else [
        UtilityStore(math.inf, np.zeros(glances * B + 8, dtype=np.int16)) for _ in CH]
    log, loss_log, rewards = [], [], []
    peak = 0.0
    max_over = 0.0
    glances_over = 0
    replay_steps = 0
    orphaned_tails = 0
    evicted = 0

    def step(loss, coupling=False):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        floors = [1e-3 * p.detach().abs() for p in model.parameters()] if coupling else None
        opt.step()
        return gmc_reward(model, opt, floors) if coupling else 0.0

    for g in range(glances):
        ci = choose(cond, g, values, rng)
        channel = CH[ci]
        if stores is not None:
            for store in stores:
                store.chunk = g
                store.decay()
            if stores[ci].index_dirty:
                stores[ci].reindex()
        oldlen = int(lengths[ci])
        if cond == 'slope_prop':
            score = model.reservoir_bits(ci, reservoirs[ci], hist[ci]) / RES
            visit_scores[ci].append(score)
            visit_scores[ci] = visit_scores[ci][-4:]
            reward = slope_reward(visit_scores[ci], int(counts[ci]) + 1)
        else:
            reward = 0.0
        x = torch.tensor(observe(world, channel, rng), dtype=torch.long)
        bufs[ci][oldlen:oldlen + B] = x.numpy().astype(np.uint8)
        hist[ci, oldlen:oldlen + B] = x
        lengths[ci] = oldlen + B
        previous = torch.cat((hist[ci, oldlen - 1:oldlen] if oldlen else torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        probs = logits[:, 0].detach().double().softmax(-1).numpy()
        gru_bits = losses.detach().double().numpy() / math.log(2)
        mix_bits = np.empty(B, dtype=np.float64)
        ctx, w = contexts[ci], weights[ci]
        store = None if nostore else stores[ci]
        for offset in range(B):
            symbol = int(x[offset])
            p_gru = float(probs[offset, symbol])
            if nostore:
                mix_bits[offset] = -math.log2(p_gru)
            else:
                p_match = store.likelihood(ctx, symbol, oldlen + offset)
                liks = np.array([p_gru, p_match], dtype=np.float64)
                mix = float(w @ liks)
                mix_bits[offset] = -math.log2(mix)
                store.credit_match(float(gru_bits[offset]), float(mix_bits[offset]))
                w *= liks / mix
                w[:] = 0.999 * w + 0.001 / 2
            ctx.append(symbol)
            if len(ctx) > 32:
                del ctx[0]
        states[ci] = state.detach()
        if cond == 'gmc_prop':
            reward = step(losses.mean(), coupling=True)
        else:
            step(losses.mean())
        if store is not None:
            if pendings[ci] > oldlen and window_ends_at(store, oldlen):
                hi = min(oldlen + B, pendings[ci])
                store.write(oldlen, bufs[ci][oldlen:hi], join_touching=True, fresh=False)
            elif pendings[ci] > oldlen:
                orphaned_tails += 1
            indices = np.flatnonzero(mix_bits > THRESHOLD) + oldlen
            intervals, pendings[ci] = surprise_intervals(indices, oldlen + B + 4, oldlen + B, pendings[ci])
            for lo, hi in intervals:
                if hi > lo:
                    store.write(lo, bufs[ci][lo:hi], fresh=True)
            n_obs = (g + 1) * B
            budget = BUDGET_FRACTION * n_obs * LOG2A
            evicted += evict_global(stores, g, budget)
            mem = retained_memory(stores)
            if mem > peak:
                peak = mem
            over = mem - budget
            if over > max_over:
                max_over = over
            if over > 1e-6:
                glances_over += 1
                assert all(g < born + PROTECT for s in stores for born in s.born)
            else:
                assert mem <= budget + 1e-6
        if stores is not None and any(s.windows for s in stores):
            for _ in range(2):
                batch = sample_replay(stores, replay_rng)
                previous_r, targets, cis = batch
                replay_logits, _ = model(previous_r, cis)
                step(F.cross_entropy(replay_logits.reshape(-1, A), targets.reshape(-1)))
                replay_steps += 1
        if cond == 'slope_prop':
            for position in range(oldlen, oldlen + B):
                if len(reservoirs[ci]) < RES:
                    reservoirs[ci].append(position)
                else:
                    k = int(rrng.integers(position + 1))
                    if k < RES:
                        reservoirs[ci][k] = position
        if cond == 'gmc_prop':
            values[ci] = reward if counts[ci] == 0 else (1 - BETA) * values[ci] + BETA * reward
        elif cond == 'slope_prop':
            values[ci] = reward
        counts[ci] += 1
        log.append(ci)
        loss_log.append(float(mix_bits.mean()))
        rewards.append(float(reward))
        if (g + 1) % 64 == 0 or g + 1 == glances:
            extra = '' if nostore else f' windows={sum(len(s.starts) for s in stores)} M/N={peak / ((g + 1) * B):.4f}'
            print(f'{now()} {phase} {cond} s{seed} T={g + 1}/{glances} '
                  f'bits={np.mean(loss_log):.4f}{extra} wall={time.monotonic() - started:.1f}s', flush=True)

    held = {}
    wc = copy.deepcopy(world)
    model.eval()
    with torch.no_grad():
        for lci, lch in enumerate(LEARN):
            seq = torch.tensor([F_ALPHA.index(s) for s in wc.take(lch, EVAL)], dtype=torch.long)
            last = hist[lci, lengths[lci] - 1:lengths[lci]] if lengths[lci] else torch.tensor([-1])
            previous = torch.cat((last, seq[:-1]))
            state = states[lci]
            w = weights[lci]
            ctx = bytearray(contexts[lci])
            store = None if nostore else stores[lci]
            total = 0.0
            for start in range(0, EVAL, B):
                logits, state = model(previous[start:start + B, None], torch.tensor([lci]), state)
                gp = logits[:, 0].double().softmax(-1).numpy()
                for offset, symbol in enumerate(seq[start:start + B].tolist()):
                    p_gru = float(gp[offset, symbol])
                    if nostore:
                        total += -math.log2(p_gru)
                    else:
                        p_match = frozen_likelihood(store, ctx, symbol)
                        total += -math.log2(float(w[0] * p_gru + w[1] * p_match))
                    ctx.append(symbol)
                    if len(ctx) > 32:
                        del ctx[0]
            held[lch] = total / EVAL
    choices = np.array(log)
    n_obs = glances * B
    assert counts.sum() == glances and np.all(lengths == counts * B)
    assert abs(sum(counts[i] / glances for i in range(len(CH))) - 1) < 1e-9
    assert np.isfinite(loss_log).all() and np.isfinite(list(held.values())).all()
    if nostore:
        assert peak == 0 and replay_steps == 0
    result = dict(
        phase=phase, cond=cond, seed=seed, glances=glances,
        W=float(np.mean([held[c] for c in LEARN])), W_F=float(held['F']), held=held,
        share={c: float(counts[i] / glances) for i, c in enumerate(CH)},
        share_by_quarter=[{c: float(np.mean(q == i)) for i, c in enumerate(CH)}
                          for q in np.array_split(choices, 4)],
        life_bits=float(np.mean(loss_log)), wall_seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        adam_steps=int(glances + replay_steps), replay_steps=int(replay_steps),
        observed_symbols=int(n_obs),
        peak_M_bits=float(peak), peak_M_over_N=float(peak / n_obs),
        final_M_bits=0.0 if nostore else float(retained_memory(stores)),
        final_windows=0 if nostore else int(sum(len(s.starts) for s in stores)),
        evicted_windows=int(evicted), orphaned_tails=int(orphaned_tails),
        glances_over_budget=int(glances_over), max_over_budget_bits=float(max(0.0, max_over)),
        mixture_weights={c: [float(x) for x in weights[i]] for i, c in enumerate(CH)})
    result['final_M_over_N'] = result['final_M_bits'] / n_obs
    print(f'{now()} done {phase} {cond} s{seed} T={glances} W={result["W"]:.6f} '
          f'W_F={result["W_F"]:.6f} peak_M/N={result["peak_M_over_N"]:.6f} '
          f'wall={result["wall_seconds"]:.1f}s', flush=True)
    return result, choices, np.array(loss_log), np.array(rewards)


def save(data, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    scoring = [r for r in data['runs'] if r['phase'] == 'score']
    lines = ['# M4a measurements', '',
             f'Started: {data["started"]}. Updated: {now()}.', '',
             'CPU threads: 2. Standard deviation: population (ddof=0). No verdict.', '',
             f'Chosen T: {data.get("chosen_T")}. Scoring seeds: {data.get("scoring_seeds", [])}.',
             f'Dropped: {data.get("dropped_conditions", [])}.', '']
    lines += ['| Condition | W mean ± std | W_F mean ± std | N share | Z share | Peak M/N mean | Wall seconds (sum) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for cond in CONDS:
        rs = [r for r in scoring if r['cond'] == cond]
        if not rs:
            continue
        lines.append('| ' + cond
                     + f' | {mean_std([r["W"] for r in rs])} | {mean_std([r["W_F"] for r in rs])} '
                     + f'| {np.mean([r["share"]["N"] for r in rs]):.6f} '
                     + f'| {np.mean([r["share"]["Z"] for r in rs]):.6f} '
                     + f'| {np.mean([r["peak_M_over_N"] for r in rs]):.6f} '
                     + f'| {sum(r["wall_seconds"] for r in rs):.3f} |')
    for phase in ('smoke', 'score'):
        rs = [r for r in data['runs'] if r['phase'] == phase]
        if not rs:
            continue
        lines += ['', f'{phase}: per run.', '',
                  '| T | Seed | Condition | W | W_F | ' + ' | '.join(LEARN)
                  + ' | Peak M/N | Final M/N | Windows | Evicted | Over-budget glances | Wall seconds |',
                  '|---:|---:|---|---:|---:|' + '---:|' * len(LEARN) + '---:|---:|---:|---:|---:|---:|']
        for r in rs:
            lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | {r["W"]:.6f} | {r["W_F"]:.6f} | '
                         + ' | '.join(f'{r["held"][c]:.6f}' for c in LEARN)
                         + f' | {r["peak_M_over_N"]:.6f} | {r["final_M_over_N"]:.6f} | {r["final_windows"]} '
                         + f'| {r["evicted_windows"]} | {r["glances_over_budget"]} | {r["wall_seconds"]:.3f} |')
    if scoring:
        lines += ['', 'Mean overall glance shares.', '',
                  '| Condition | ' + ' | '.join(CH) + ' |',
                  '|---|' + '---:|' * len(CH)]
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            if rs:
                lines.append('| ' + cond + ' | ' + ' | '.join(
                    f'{np.mean([r["share"][c] for r in rs]):.6f}' for c in CH) + ' |')
        lines += ['', 'Mean glance shares by quarter.', '',
                  '| Condition | Quarter | ' + ' | '.join(CH) + ' |',
                  '|---|---:|' + '---:|' * len(CH)]
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            for q in range(4):
                if rs:
                    lines.append(f'| {cond} | {q + 1} | ' + ' | '.join(
                        f'{np.mean([r["share_by_quarter"][q][c] for r in rs]):.6f}' for c in CH) + ' |')
        lines += ['', 'Glance shares per seed, overall and by quarter.', '',
                  '| Seed | Condition | Period | ' + ' | '.join(CH) + ' |',
                  '|---:|---|---|' + '---:|' * len(CH)]
        for r in scoring:
            for label, shares in [('all', r['share'])] + [(f'Q{i + 1}', q) for i, q in enumerate(r['share_by_quarter'])]:
                lines.append(f'| {r["seed"]} | {r["cond"]} | {label} | '
                             + ' | '.join(f'{shares[c]:.6f}' for c in CH) + ' |')
        lines += ['', 'Paired differences (measurements; bars are not scored here).', '',
                  '| Seed | Condition | W − rand | W − rand8 | W_F − W_F(rand) | N share | Z share |',
                  '|---:|---|---:|---:|---:|---:|---:|']
        for seed in data.get('scoring_seeds', []):
            rows = {r['cond']: r for r in scoring if r['seed'] == seed}
            if 'rand' not in rows:
                continue
            for cond in CONDS:
                if cond not in rows:
                    continue
                lines.append(f'| {seed} | {cond} | {rows[cond]["W"] - rows["rand"]["W"]:.6f} | '
                             + (f'{rows[cond]["W"] - rows["rand8"]["W"]:.6f}' if 'rand8' in rows else 'NA')
                             + f' | {rows[cond]["W_F"] - rows["rand"]["W_F"]:.6f} | '
                             + f'{rows[cond]["share"]["N"]:.6f} | {rows[cond]["share"]["Z"]:.6f} |')
        if all(any(r['cond'] == c and r['seed'] == s for r in scoring)
               for c in ('rand', 'rand_nostore') for s in data.get('scoring_seeds', [])):
            lines += ['', 'Store against no-store, per seed. Negative means the store is lower.', '',
                      '| Seed | W(rand) − W(rand_nostore) | W_F(rand) − W_F(rand_nostore) |',
                      '|---:|---:|---:|']
            for seed in data.get('scoring_seeds', []):
                a = next(r for r in scoring if r['cond'] == 'rand' and r['seed'] == seed)
                b = next(r for r in scoring if r['cond'] == 'rand_nostore' and r['seed'] == seed)
                lines.append(f'| {seed} | {a["W"] - b["W"]:.6f} | {a["W_F"] - b["W_F"]:.6f} |')
    lines += ['', 'Protocol notes:',
              '- World: FactWorld, channels G C1 C7 E30 E110 M7 M37 F N Z. Glance = 64 symbols. Processes resume.',
              '- N: uniform over 42 symbols. Z: the space symbol repeated. T glances, seeds as listed.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance (mean cross-entropy), then up to two replay steps.',
              '- Replay: 8 columns of 64, sampled from stored windows only, one channel per column, zero initial state.',
              '  Zero embedding at a window boundary. No replay when every store is empty, or in rand_nostore.',
              '- Store: per channel, H24g util_m. Write where mixture bits > 4, window [i−16, i+4], overlap merges,',
              '  touching windows stay separate, deferred tail joins its head. Match credit only. Decay 0.999 per glance.',
              '  Protected for 32 glances. Evict the lowest utility when M > 0.15 * observed_symbols * log2(42).',
              '  A tail whose head was evicted is counted and not rewritten.',
              '- Match: order 5 inside that channel\'s windows. Fixed-share mixture with the GRU, α = 0.001, per channel.',
              '- Prequential: mixture (or GRU, if rand_nostore) bits are recorded before the Adam step and before writes.',
              '- gmc_prop: imported gmc_reward after the live step and before replay; EMA β = 0.3; proportional',
              '  max(value, 0) + 0.01; no epsilon after the first pass.',
              '- slope_prop: imported slope_reward on H25b reservoir bits / 128; no EMA; same proportional allocation.',
              '  Reservoir positions are added after learning and replay. Algorithm R, capacity 128, context 32.',
              '- W: frozen GRU, frozen match, frozen mixture weights, fresh 2048 symbols on each learnable channel',
              '  from a deep copy of the world, carried GRU state. Mean over the 8 channels. W_F is channel F.',
              '  rand_nostore uses the GRU probability alone. History is not added to W.',
              '- First pass: glance g visits channel g for g < 10, in every condition, then the controller.',
              '']
    lines += [f'Note: {n}' for n in data.get('notes', [])]
    lines += ['', f'Elapsed wall seconds: {data.get("total_wall_seconds", 0):.3f}.',
              f'Status: {data["status"]}.']
    if 'smoke_wall_seconds' in data:
        lines += [f'Smoke wall seconds: {data["smoke_wall_seconds"]:.3f}.',
                  f'Wall seconds including smoke: {data.get("wall_seconds_including_smoke", 0):.3f}.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def checks():
    assert abs(DECAY - 0.999) < 1e-15 and PROTECT == 32 and THRESHOLD == 4.0
    intervals, pending = surprise_intervals([60], 68, 64, 0)
    assert intervals == [(44, 64)] and pending == 65, (intervals, pending)
    intervals, pending = surprise_intervals([20, 24], 100, 64, 0)
    assert intervals == [(4, 29)] and pending == 29
    world = FactWorld(6)
    text = world.take('F', B)
    assert len(text) == B and '=' in text and set(text) <= set(F_ALPHA)
    other = world.take('C1', B)
    assert set(other) <= set(F_ALPHA) and '=' not in other
    cloned = copy.deepcopy(world)
    assert world.take('G', 40) == cloned.take('G', 40)
    payload = np.arange(30, dtype=np.uint8) % A
    store = UtilityStore(math.inf, np.zeros(80, dtype=np.int16))
    store.chunk = 0
    store.write(0, payload, fresh=True)
    ctx = bytearray(int(x) for x in payload[:10])
    before = store.hits.copy(), store.trials.copy()
    frozen = frozen_likelihood(store, ctx, int(payload[10]))
    assert store.hits.tolist() == before[0].tolist() and store.trials.tolist() == before[1].tolist()
    live = store.likelihood(ctx, int(payload[10]), 11)
    assert abs(frozen - live) < 1e-12
    assert store.trials.sum() == before[1].sum() + 1
    frozen_likelihood(store, ctx, int(payload[10]))
    assert store.trials.sum() == before[1].sum() + 1
    # Two unprotected windows, global budget drops the lower utility.
    a = UtilityStore(math.inf, np.zeros(80, dtype=np.int16))
    b = UtilityStore(math.inf, np.zeros(80, dtype=np.int16))
    a.chunk = b.chunk = 0
    a.write(0, payload[:20], fresh=True)
    b.write(0, payload[:20], fresh=True)
    a.utils[0], b.utils[0] = 0.1, 5.0
    a.born[0] = b.born[0] = 0
    budget = retained_memory([a])  # one window's worth: b must survive, a must go
    dropped = evict_global([a, b], glance=PROTECT, budget=budget + 1e-9)
    assert dropped == 1 and len(a.windows) == 0 and len(b.windows) == 1
    assert not a.index_dirty and not b.index_dirty
    # Head missing → caller must not join. window_ends_at is the guard.
    c = UtilityStore(math.inf, np.zeros(80, dtype=np.int16))
    c.chunk = 0
    c.write(0, payload[:10], fresh=True)
    assert window_ends_at(c, 10) and not window_ends_at(c, 9)
    print(f'{now()} checks passed', flush=True)


def preflight():
    """Every controller path, thrown away. Not part of the reported runs."""
    for cond, t in (('rand', 4), ('rand_nostore', 4), ('rand8', 4), ('gmc_prop', 4), ('slope_prop', 8)):
        result, choices, _, _ = run(cond, 6, t, 'preflight')
        assert choices.tolist() == list(range(min(t, len(CH))))[:t] or t > len(CH)
        assert choices[:len(CH)].tolist() == list(range(min(t, len(CH))))
        assert np.isfinite(result['W'])
        if cond == 'rand_nostore':
            assert result['peak_M_bits'] == 0 and result['replay_steps'] == 0
        else:
            assert result['replay_steps'] > 0
            assert result['final_windows'] > 0
    print(f'{now()} preflight passed', flush=True)


def execute(data, out, cond, seed, glances, phase):
    result, choices, bits, rewards = run(cond, seed, glances, phase)
    data['runs'].append(result)
    data['total_wall_seconds'] = time.monotonic() - data['clock']
    np.savez_compressed(out / f'{phase}_{cond}_{seed}_{glances}.npz',
                        channels=choices, prequential_bits=bits, rewards=rewards)
    save(data, out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    configure_threads()
    checks()
    if args.checks_only:
        preflight()
        return
    clock = time.monotonic()
    smoke_dir = Path('results/m4a/smoke')
    smoke_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke or not (smoke_dir / 'results.json').exists():
        smoke = dict(started=now(), status='running', phase='smoke', runs=[], notes=[],
                     chosen_T=32, scoring_seeds=[6], dropped_conditions=[], clock=clock,
                     torch_version=torch.__version__, principles=[1, 2, 3, 4])
        for cond in CONDS:
            execute(smoke, smoke_dir, cond, 6, 32, 'smoke')
        smoke['status'] = 'complete'
        smoke['finished'] = now()
        smoke['total_wall_seconds'] = time.monotonic() - clock
        save(smoke, smoke_dir)
        print(f'{now()} smoke complete wall={smoke["total_wall_seconds"]:.1f}s', flush=True)
    else:
        smoke = json.loads((smoke_dir / 'results.json').read_text())
        assert smoke['status'] == 'complete' and len(smoke['runs']) == len(CONDS)
    if args.smoke:
        return
    out = Path('results/m4a')
    rates = {r['cond']: r['wall_seconds'] / r['glances'] for r in smoke['runs']}
    projected = sum(rates[c] * 512 * 3 for c in CONDS)
    active = list(CONDS)
    notes = [f'Smoke-based projected score wall seconds: {projected:.1f}.']
    dropped = []
    smoke_wall = float(smoke['total_wall_seconds'])
    if smoke_wall + projected > WALL_BUDGET:
        active = [c for c in CONDS if c != 'slope_prop']
        dropped = ['slope_prop']
        notes.append('Dropped slope_prop before scoring: smoke projection plus smoke exceeds 120 minutes.')
        projected = sum(rates[c] * 512 * 3 for c in active)
        notes.append(f'Projected score wall seconds without slope_prop: {projected:.1f}.')
    data = dict(started=now(), status='running', runs=[], notes=notes, chosen_T=512,
                scoring_seeds=[6, 7, 8], dropped_conditions=dropped, clock=time.monotonic(),
                torch_version=torch.__version__, principles=[1, 2, 3, 4],
                smoke_wall_seconds=smoke_wall, projected_score_seconds=projected,
                protocol='handoffs/2026-11-10_grok_m4a_one_life.md')
    save(data, out)
    # slope_prop last, so a late budget cut cannot drop the store or oracle runs.
    slowdown = 1.0
    for cond in ('rand', 'rand_nostore', 'rand8', 'gmc_prop', 'slope_prop'):
        if cond not in active:
            continue
        for seed in (6, 7, 8):
            spent = smoke_wall + (time.monotonic() - data['clock'])
            eta = rates[cond] * slowdown * 512
            if cond == 'slope_prop' and spent + eta > WALL_BUDGET:
                name = f'slope_prop/seed{seed}'
                data['dropped_conditions'].append(name)
                data['notes'].append(
                    f'Dropped {name}: 120-minute wall budget (spent {spent:.0f}s, eta {eta:.0f}s).')
                save(data, out)
                continue
            result = execute(data, out, cond, seed, 512, 'score')
            slowdown = max(slowdown, (result['wall_seconds'] / result['glances']) / rates[cond])
    data['status'] = 'complete'
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic() - data['clock']
    data['wall_seconds_including_smoke'] = data['total_wall_seconds'] + smoke_wall
    save(data, out)
    print(f'{now()} ALL_DONE score_wall={data["total_wall_seconds"]:.1f}s '
          f'with_smoke={data["wall_seconds_including_smoke"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
