"""M4b: separate replay source, match mixture, and the memory budget.

Protocol: handoffs/2026-11-10_grok_m4b_ablation.md. Numbers only; no verdict.
Creates only this file and results/m4b/*. Imports earlier scripts; does not edit them.

Readings of the spec, fixed before the scored runs:
- World, cortex, store, match, mixture, writes, utility and decay are M4a's.
  Channels F_PROCS + N + Z, glance 64, alphabet 42. First pass visits channel g
  for g < 10, then the controller. Seeds 0 1 2. Smoke is seed 0, T = 32.
- Controller is rand except gmc_hist, which is M4a's gmc_prop (same estimator,
  EMA β = 0.3, proportional max(value, 0) + 0.01).
- Rand conditions share the choice/observation RNG (10_000 + seed) and do not
  draw replay from it, so one seed has one glance sequence and one observation
  log. That match is checked, not assumed. Replay uses 30_000 + seed.
- Whole-history replay draws 64-symbol segments uniformly from every valid
  start in the raw per-channel log (H21c / H25a). Each segment starts from a
  zero hidden state and uses that segment's channel embedding. The input is
  the preceding symbol, and the embedding is zero only at position 0.
  Two Adam steps, batch 8, after the live step. The log is not charged to M;
  its size is reported as symbols and as symbols × log2(42) bits.
- store_rep replays from the utility store only, with M4a's sampler.
  store_norep and nostore do not replay. hist_rep has no store and no match.
- Hard budget, every store condition: while M > 0.15 × observed symbols ×
  log2(42), evict the lowest-utility unprotected window; once none remain,
  evict the oldest protected window (smallest birth glance, then smallest
  start). Stop at M ≤ budget. After every glance, M ≤ budget. Peak M/N is at
  most the budget ratio plus one canonical window (21 symbols + 32 bits) / N.
- W_mix is M4a's frozen mixture on a fresh 2048-symbol continuation of each
  learnable channel. W_gru scores the same continuation with the frozen GRU
  probability alone. nostore and hist_rep have no match, so W_mix = W_gru.
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

from h21a_substrate import mean_std, now
from h24b_world import F_ALPHA, FactWorld
from h24g_utility_store import DECAY, PROTECT, THRESHOLD, UtilityStore, surprise_intervals
from h25b_progress_fix import gmc_reward
from m4a_one_life import (A, B, BETA, BUDGET_FRACTION, CH, EVAL, LEARN, LOG2A,
                         Cortex, configure_threads, frozen_likelihood, observe,
                         retained_memory, sample_replay, window_ends_at)

CONDS = ['nostore', 'hist_rep', 'store_norep', 'store_rep', 'store_hist', 'gmc_hist']
RAND_CONDS = ['nostore', 'hist_rep', 'store_norep', 'store_rep', 'store_hist']
STORE_CONDS = ['store_norep', 'store_rep', 'store_hist', 'gmc_hist']
HIST_REPLAY = ['hist_rep', 'store_hist', 'gmc_hist']
SEEDS = [0, 1, 2]
WALL_BUDGET = 7200.0
CANON_WINDOW_SYMBOLS = 21  # surprise window [i−16, i+4]
CANON_WINDOW_BITS = CANON_WINDOW_SYMBOLS * LOG2A + 32.0
BUDGET_RATIO = BUDGET_FRACTION * LOG2A
assert abs(DECAY - 0.999) < 1e-15 and PROTECT == 32 and THRESHOLD == 4.0
assert abs(BUDGET_FRACTION - 0.15) < 1e-15 and A == 42 and B == 64
assert len(CH) == 10 and len(LEARN) == 8 and EVAL == 2048


def has_store(cond):
    return cond in STORE_CONDS


def replay_source(cond):
    if cond in HIST_REPLAY:
        return 'history'
    if cond == 'store_rep':
        return 'store'
    return 'none'


def choose(cond, g, values, rng):
    if g < len(CH):
        return g
    if cond == 'gmc_hist':
        weights = np.maximum(values, 0.0) + 0.01
        return int(rng.choice(len(CH), p=weights / weights.sum()))
    return int(rng.integers(len(CH)))


def evict_hard(stores, glance, budget):
    """M4a eviction, then oldest protected windows until M ≤ budget."""
    dropped = 0
    protected_dropped = 0
    guard = sum(len(store.starts) for store in stores) + 1
    while True:
        mem = retained_memory(stores)
        if mem <= budget:
            break
        best, victim = None, None
        oldest, oldest_victim = None, None
        for ci, store in enumerate(stores):
            for i, born in enumerate(store.born):
                if glance >= born + PROTECT:
                    key = (store.utils[i], store.starts[i], ci, i)
                    if best is None or key < best:
                        best, victim = key, (ci, i)
                okey = (born, store.starts[i], ci, i)
                if oldest is None or okey < oldest:
                    oldest, oldest_victim = okey, (ci, i)
        if victim is not None:
            choice, protected = victim, False
        elif oldest_victim is not None:
            choice, protected = oldest_victim, True
        else:
            break
        stores[choice[0]]._drop(choice[1])
        dropped += 1
        protected_dropped += int(protected)
        guard -= 1
        if guard < 0:
            raise RuntimeError('hard eviction failed to shrink the store')
    for store in stores:
        if store.index_dirty:
            store.reindex()
    mem = retained_memory(stores)
    if mem > budget + 1e-6:
        raise AssertionError(f'hard budget failed: M={mem} budget={budget}')
    return dropped, protected_dropped


def segment_index(lengths, draws):
    """Uniform over valid 64-symbol starts. A channel with L symbols has L−63."""
    valid = np.maximum(lengths - B + 1, 0)
    total = int(valid.sum())
    cumulative = np.cumsum(valid)
    channels = np.searchsorted(cumulative, draws, side='right')
    starts = draws - np.concatenate(([0], cumulative[:-1]))[channels]
    return channels, starts, total


def sample_history(hist, lengths, rng):
    total = int(np.maximum(np.asarray(lengths) - B + 1, 0).sum())
    if total <= 0:
        return None
    draws = rng.integers(0, total, size=8)
    channels, starts, _ = segment_index(lengths, draws)
    starts_t = torch.as_tensor(starts, dtype=torch.long)
    cis = torch.as_tensor(np.ascontiguousarray(channels), dtype=torch.long)
    indices = starts_t[None, :] + torch.arange(B, dtype=torch.long)[:, None]
    targets = hist[cis[None, :], indices]
    previous = hist[cis[None, :], (indices - 1).clamp_min(0)].clone()
    previous[indices == 0] = -1
    return previous, targets, cis


def checksum(values):
    acc = 0
    for value in values:
        acc = (acc * 1_000_003 + int(value) + 1) & ((1 << 62) - 1)
    return acc


def observation_checksum(hist, lengths):
    parts = []
    for ci, n in enumerate(lengths.tolist()):
        parts.append((ci + 1) * 1_000_000 + int(n))
        if n:
            parts.append(int(hist[ci, :n].sum().item()))
    return checksum(parts)


def replay_batch(model, opt, previous, targets, cis):
    opt.zero_grad(set_to_none=True)
    logits, _ = model(previous, cis)
    loss = F.cross_entropy(logits.reshape(-1, A), targets.reshape(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def run(cond, seed, glances, phase):
    assert cond in CONDS
    started = time.monotonic()
    torch.manual_seed(seed)
    model = Cortex()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world = FactWorld(seed)
    rng = np.random.default_rng(10_000 + seed)
    replay_rng = np.random.default_rng(30_000 + seed)
    use_store = has_store(cond)
    source = replay_source(cond)
    hist = torch.zeros((len(CH), glances * B), dtype=torch.long)
    lengths = np.zeros(len(CH), dtype=np.int64)
    states = [None] * len(CH)
    values = np.zeros(len(CH))
    counts = np.zeros(len(CH), dtype=int)
    contexts = [bytearray() for _ in CH]
    weights = [np.full(2, 0.5) for _ in CH]
    bufs = [np.zeros(glances * B, dtype=np.uint8) for _ in CH]
    pendings = [0 for _ in CH]
    stores = None if not use_store else [
        UtilityStore(math.inf, np.zeros(glances * B + 8, dtype=np.int16)) for _ in CH]
    log, loss_log = [], []
    life_gru = []
    peak = 0.0
    peak_ratio = 0.0
    replay_steps = 0
    orphaned_tails = 0
    evicted = 0
    protected_evicted = 0
    max_window_symbols = 0

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
        store = None if stores is None else stores[ci]
        for offset in range(B):
            symbol = int(x[offset])
            p_gru = float(probs[offset, symbol])
            if store is None:
                mix_bits[offset] = -math.log2(p_gru)
            else:
                p_match = store.likelihood(ctx, symbol, oldlen + offset)
                liks = np.array([p_gru, p_match], dtype=np.float64)
                mix = float(w @ liks)
                mix_bits[offset] = -math.log2(mix)
                store.credit_match(float(gru_bits[offset]), float(mix_bits[offset]))
                w *= liks / mix
                w[:] = 0.999 * w + 0.001 / 2.0
            ctx.append(symbol)
            if len(ctx) > 32:
                del ctx[0]
        states[ci] = state.detach()
        if cond == 'gmc_hist':
            reward = step(losses.mean(), coupling=True)
        else:
            step(losses.mean())
            reward = 0.0
        if store is not None:
            if pendings[ci] > oldlen and window_ends_at(store, oldlen):
                hi = min(oldlen + B, pendings[ci])
                store.write(oldlen, bufs[ci][oldlen:hi], join_touching=True, fresh=False)
            elif pendings[ci] > oldlen:
                orphaned_tails += 1
            indices = np.flatnonzero(mix_bits > THRESHOLD) + oldlen
            intervals, pendings[ci] = surprise_intervals(
                indices, oldlen + B + 4, oldlen + B, pendings[ci])
            for lo, hi in intervals:
                if hi > lo:
                    store.write(lo, bufs[ci][lo:hi], fresh=True)
            n_now = (g + 1) * B
            budget = BUDGET_FRACTION * n_now * LOG2A
            dropped, protected_dropped = evict_hard(stores, g, budget)
            evicted += dropped
            protected_evicted += protected_dropped
            mem = retained_memory(stores)
            if mem > peak:
                peak = mem
            ratio = mem / n_now
            if ratio > peak_ratio:
                peak_ratio = ratio
            for kept in stores:
                for window in kept.windows:
                    if len(window) > max_window_symbols:
                        max_window_symbols = len(window)
            limit = BUDGET_RATIO + CANON_WINDOW_BITS / n_now
            if ratio > limit + 1e-9:
                raise AssertionError(
                    f'peak ratio {ratio} exceeds budget {BUDGET_RATIO} + one window / N')
        if source == 'store' and any(s.windows for s in stores):
            for _ in range(2):
                previous_r, targets, cis = sample_replay(stores, replay_rng)
                replay_batch(model, opt, previous_r, targets, cis)
                replay_steps += 1
        elif source == 'history':
            for _ in range(2):
                batch = sample_history(hist, lengths, replay_rng)
                previous_r, targets, cis = batch
                replay_batch(model, opt, previous_r, targets, cis)
                replay_steps += 1
        if cond == 'gmc_hist':
            values[ci] = reward if counts[ci] == 0 else (1 - BETA) * values[ci] + BETA * reward
        counts[ci] += 1
        log.append(ci)
        loss_log.append(float(mix_bits.mean()))
        life_gru.append(float(gru_bits.mean()))
        if (g + 1) % 64 == 0 or g + 1 == glances:
            extra = '' if stores is None else (
                f' windows={sum(len(s.starts) for s in stores)} M/N={peak / ((g + 1) * B):.4f}')
            print(f'{now()} {phase} {cond} s{seed} T={g + 1}/{glances} '
                  f'bits={np.mean(loss_log):.4f}{extra} wall={time.monotonic() - started:.1f}s',
                  flush=True)

    held_mix, held_gru = {}, {}
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
            store = None if stores is None else stores[lci]
            total_gru = 0.0
            total_mix = 0.0
            for start in range(0, EVAL, B):
                logits, state = model(previous[start:start + B, None], torch.tensor([lci]), state)
                gp = logits[:, 0].double().softmax(-1).numpy()
                for offset, symbol in enumerate(seq[start:start + B].tolist()):
                    p_gru = float(gp[offset, symbol])
                    gru_b = -math.log2(p_gru)
                    total_gru += gru_b
                    if store is None:
                        total_mix += gru_b
                    else:
                        p_match = frozen_likelihood(store, ctx, symbol)
                        total_mix += -math.log2(float(w[0] * p_gru + w[1] * p_match))
                    ctx.append(symbol)
                    if len(ctx) > 32:
                        del ctx[0]
            held_gru[lch] = total_gru / EVAL
            held_mix[lch] = held_gru[lch] if store is None else total_mix / EVAL
    choices = np.array(log)
    n_obs = glances * B
    assert counts.sum() == glances and np.all(lengths == counts * B)
    assert abs(sum(counts[i] / glances for i in range(len(CH))) - 1) < 1e-9
    assert np.isfinite(loss_log).all() and np.isfinite(life_gru).all()
    assert np.isfinite(list(held_mix.values())).all() and np.isfinite(list(held_gru.values())).all()
    if not use_store:
        assert peak == 0.0 and replay_steps == (0 if source == 'none' else 2 * glances)
        for name in LEARN:
            assert held_mix[name] == held_gru[name]
    else:
        allowance = BUDGET_RATIO + CANON_WINDOW_BITS / n_obs
        if peak / n_obs > allowance + 1e-9:
            raise AssertionError(
                f'peak M/N {peak / n_obs} > budget {BUDGET_RATIO} + one window/N {CANON_WINDOW_BITS / n_obs}')
        if peak_ratio > BUDGET_RATIO + 1e-9:
            raise AssertionError(f'max M/N {peak_ratio} exceeds the hard budget ratio {BUDGET_RATIO}')
    if source == 'history':
        assert replay_steps == 2 * glances
    if source == 'none':
        assert replay_steps == 0
    obs_sum = observation_checksum(hist, lengths)
    glance_sum = checksum(choices.tolist())
    result = dict(
        phase=phase, cond=cond, seed=seed, glances=glances,
        W_mix=float(np.mean([held_mix[c] for c in LEARN])),
        W_gru=float(np.mean([held_gru[c] for c in LEARN])),
        W_F_mix=float(held_mix['F']), W_F_gru=float(held_gru['F']),
        held_mix=held_mix, held_gru=held_gru,
        share={c: float(counts[i] / glances) for i, c in enumerate(CH)},
        share_by_quarter=[{c: float(np.mean(q == i)) for i, c in enumerate(CH)}
                          for q in np.array_split(choices, 4)],
        life_mix_bits=float(np.mean(loss_log)), life_gru_bits=float(np.mean(life_gru)),
        wall_seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        adam_steps=int(glances + replay_steps), replay_steps=int(replay_steps),
        replay_source=source, controller=('gmc_prop' if cond == 'gmc_hist' else 'rand'),
        observed_symbols=int(n_obs),
        history_symbols=int(n_obs), history_bits_unpriced=float(n_obs * LOG2A),
        history_charged=False,
        peak_M_bits=float(peak), peak_M_over_N=float(peak / n_obs),
        peak_ratio_M_over_N=float(peak_ratio),
        final_M_bits=0.0 if stores is None else float(retained_memory(stores)),
        final_windows=0 if stores is None else int(sum(len(s.starts) for s in stores)),
        evicted_windows=int(evicted), protected_evictions=int(protected_evicted),
        orphaned_tails=int(orphaned_tails),
        max_window_symbols=int(max_window_symbols),
        budget_ratio=float(BUDGET_RATIO),
        one_window_bits=float(CANON_WINDOW_BITS),
        glances_over_budget=0,
        mixture_weights={c: [float(x) for x in weights[i]] for i, c in enumerate(CH)},
        observation_checksum=int(obs_sum), glance_checksum=int(glance_sum))
    result['final_M_over_N'] = result['final_M_bits'] / n_obs
    result['hard_budget_ok'] = (not use_store) or (
        result['peak_ratio_M_over_N'] <= BUDGET_RATIO + 1e-9
        and result['peak_M_over_N'] <= BUDGET_RATIO + CANON_WINDOW_BITS / n_obs + 1e-9)
    print(f'{now()} done {phase} {cond} s{seed} T={glances} '
          f'W_mix={result["W_mix"]:.6f} W_gru={result["W_gru"]:.6f} '
          f'W_F_mix={result["W_F_mix"]:.6f} W_F_gru={result["W_F_gru"]:.6f} '
          f'M7={held_gru["M7"]:.6f} M37={held_gru["M37"]:.6f} C1={held_gru["C1"]:.6f} '
          f'peak_M/N={result["peak_M_over_N"]:.6f} replay={replay_steps} '
          f'wall={result["wall_seconds"]:.1f}s', flush=True)
    return result, choices, np.array(loss_log)


def cell(value):
    return f'{value:.6f}'


def save(data, out):
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in data.items() if k != 'clock'}
    (out / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    scoring = [r for r in data['runs'] if r['phase'] == 'score']
    summary = scoring or list(data['runs'])
    lines = ['# M4b measurements', '',
             f'Started: {data["started"]}. Updated: {now()}.', '',
             'CPU threads: 2. Standard deviation: population (ddof=0). No verdict.', '',
             f'Chosen T: {data.get("chosen_T")}. Scoring seeds: {data.get("scoring_seeds", [])}.',
             f'Dropped: {data.get("dropped_conditions", [])}.', '']
    lines += ['| Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru | '
              'N share | Z share | Peak M/N | Final M/N | Replay steps | History symbols | '
              'History bits unpriced | Wall seconds (sum) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    shown = [c for c in CONDS if any(r['cond'] == c for r in summary)]
    for cond in shown:
        rs = [r for r in summary if r['cond'] == cond]
        lines.append('| ' + cond
                     + f' | {mean_std([r["W_mix"] for r in rs])} | {mean_std([r["W_gru"] for r in rs])} '
                     + f'| {mean_std([r["W_F_mix"] for r in rs])} | {mean_std([r["W_F_gru"] for r in rs])} '
                     + f'| {mean_std([r["held_gru"]["M7"] for r in rs])} '
                     + f'| {mean_std([r["held_gru"]["M37"] for r in rs])} '
                     + f'| {mean_std([r["held_gru"]["C1"] for r in rs])} '
                     + f'| {np.mean([r["share"]["N"] for r in rs]):.6f} '
                     + f'| {np.mean([r["share"]["Z"] for r in rs]):.6f} '
                     + f'| {np.mean([r["peak_M_over_N"] for r in rs]):.6f} '
                     + f'| {np.mean([r["final_M_over_N"] for r in rs]):.6f} '
                     + f'| {np.mean([r["replay_steps"] for r in rs]):.1f} '
                     + f'| {np.mean([r["history_symbols"] for r in rs]):.0f} '
                     + f'| {np.mean([r["history_bits_unpriced"] for r in rs]):.3f} '
                     + f'| {sum(r["wall_seconds"] for r in rs):.3f} |')
    for phase in ('smoke', 'score'):
        rs = [r for r in data['runs'] if r['phase'] == phase]
        if not rs:
            continue
        lines += ['', f'{phase}: per run.', '',
                  '| T | Seed | Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru | '
                  'Peak M/N | Final M/N | Peak ratio M/N | Windows | Evicted | Protected evictions | '
                  'Replay steps | History symbols | History bits unpriced | Wall seconds |',
                  '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rs:
            lines.append(
                f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | {cell(r["W_mix"])} | {cell(r["W_gru"])} | '
                f'{cell(r["W_F_mix"])} | {cell(r["W_F_gru"])} | {cell(r["held_gru"]["M7"])} | '
                f'{cell(r["held_gru"]["M37"])} | {cell(r["held_gru"]["C1"])} | '
                f'{cell(r["peak_M_over_N"])} | {cell(r["final_M_over_N"])} | {cell(r["peak_ratio_M_over_N"])} | '
                f'{r["final_windows"]} | {r["evicted_windows"]} | {r["protected_evictions"]} | '
                f'{r["replay_steps"]} | {r["history_symbols"]} | {r["history_bits_unpriced"]:.3f} | '
                f'{r["wall_seconds"]:.3f} |')
        lines += ['', f'{phase}: held-out mixture bits per channel.', '',
                  '| T | Seed | Condition | ' + ' | '.join(LEARN) + ' |',
                  '|---:|---:|---|---:|' + '---:|' * (len(LEARN) - 1)]
        for r in rs:
            lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | '
                         + ' | '.join(cell(r['held_mix'][c]) for c in LEARN) + ' |')
        lines += ['', f'{phase}: held-out GRU bits per channel.', '',
                  '| T | Seed | Condition | ' + ' | '.join(LEARN) + ' |',
                  '|---:|---:|---|---:|' + '---:|' * (len(LEARN) - 1)]
        for r in rs:
            lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | '
                         + ' | '.join(cell(r['held_gru'][c]) for c in LEARN) + ' |')
    if scoring:
        lines += ['', 'Mean overall glance shares.', '',
                  '| Condition | ' + ' | '.join(CH) + ' |',
                  '|---|' + '---:|' * len(CH)]
        for cond in shown:
            rs = [r for r in scoring if r['cond'] == cond]
            lines.append('| ' + cond + ' | ' + ' | '.join(
                f'{np.mean([r["share"][c] for r in rs]):.6f}' for c in CH) + ' |')
        lines += ['', 'Mean glance shares by quarter.', '',
                  '| Condition | Quarter | ' + ' | '.join(CH) + ' |',
                  '|---|---:|' + '---:|' * len(CH)]
        for cond in shown:
            rs = [r for r in scoring if r['cond'] == cond]
            for q in range(4):
                lines.append(f'| {cond} | {q + 1} | ' + ' | '.join(
                    f'{np.mean([r["share_by_quarter"][q][c] for r in rs]):.6f}' for c in CH) + ' |')
        lines += ['', 'Glance shares per seed, overall and by quarter.', '',
                  '| Seed | Condition | Period | ' + ' | '.join(CH) + ' |',
                  '|---:|---|---|' + '---:|' * len(CH)]
        for r in scoring:
            for label, shares in [('all', r['share'])] + [
                    (f'Q{i + 1}', q) for i, q in enumerate(r['share_by_quarter'])]:
                lines.append(f'| {r["seed"]} | {r["cond"]} | {label} | '
                             + ' | '.join(f'{shares[c]:.6f}' for c in CH) + ' |')
        lines += ['', 'Observation alignment across rand conditions. Equal checksums are the same log.', '',
                  '| Seed | Condition | Glance checksum | Observation checksum |',
                  '|---:|---|---:|---:|']
        for r in scoring:
            lines.append(f'| {r["seed"]} | {r["cond"]} | {r["glance_checksum"]} | {r["observation_checksum"]} |')
        lines += ['', 'Hard budget. Allowance is 0.15 × log2(42) + (21 × log2(42) + 32) / N. '
                  'Peak ratio is the max of M/N after each glance.', '',
                  '| Seed | Condition | Peak M/N | Peak ratio M/N | Budget ratio | Allowance | '
                  'Over-budget glances | Protected evictions | Final windows |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in scoring:
            if r['cond'] not in STORE_CONDS:
                continue
            allowance = r['budget_ratio'] + r['one_window_bits'] / r['observed_symbols']
            lines.append(f'| {r["seed"]} | {r["cond"]} | {cell(r["peak_M_over_N"])} | '
                         f'{cell(r["peak_ratio_M_over_N"])} | {cell(r["budget_ratio"])} | '
                         f'{allowance:.6f} | {r["glances_over_budget"]} | {r["protected_evictions"]} | '
                         f'{r["final_windows"]} |')
        lines += ['', 'Differences (measurements; bars are not scored here).', '']
        by = {(r['seed'], r['cond']): r for r in scoring}

        def diff_line(seed, left, right, key):
            if (seed, left) not in by or (seed, right) not in by:
                return None
            return by[(seed, left)][key] - by[(seed, right)][key]

        lines += ['| Seed | W_gru(store_hist) − W_gru(store_rep) | M7 gru store_hist − nostore | '
                  'M37 gru store_hist − nostore | W(hist_rep) − W(nostore) | '
                  'W_mix(store_hist) − W_gru(store_hist) | W_F_mix(store_hist) − W_F_gru(store_hist) |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        acc = {name: [] for name in ('d_gru', 'd_m7', 'd_m37', 'd_hist', 'd_mix', 'd_f')}
        for seed in data.get('scoring_seeds', []):
            d_gru = diff_line(seed, 'store_hist', 'store_rep', 'W_gru')
            d_m7 = (by[(seed, 'store_hist')]['held_gru']['M7'] - by[(seed, 'nostore')]['held_gru']['M7']
                    if (seed, 'store_hist') in by and (seed, 'nostore') in by else None)
            d_m37 = (by[(seed, 'store_hist')]['held_gru']['M37'] - by[(seed, 'nostore')]['held_gru']['M37']
                     if (seed, 'store_hist') in by and (seed, 'nostore') in by else None)
            d_hist = diff_line(seed, 'hist_rep', 'nostore', 'W_mix')
            d_mix = None
            if (seed, 'store_hist') in by:
                d_mix = by[(seed, 'store_hist')]['W_mix'] - by[(seed, 'store_hist')]['W_gru']
            d_f = None
            if (seed, 'store_hist') in by:
                d_f = by[(seed, 'store_hist')]['W_F_mix'] - by[(seed, 'store_hist')]['W_F_gru']
            row = [d_gru, d_m7, d_m37, d_hist, d_mix, d_f]
            if all(v is None for v in row):
                continue
            for name, value in zip(acc, row):
                if value is not None:
                    acc[name].append(value)
            lines.append('| ' + str(seed) + ' | ' + ' | '.join('NA' if v is None else cell(v) for v in row) + ' |')
        if any(acc.values()):
            lines.append('| mean | ' + ' | '.join(
                'NA' if not acc[name] else cell(float(np.mean(acc[name])))
                for name in ('d_gru', 'd_m7', 'd_m37', 'd_hist', 'd_mix', 'd_f')) + ' |')
        lines += ['', 'gmc_hist against store_hist (measurements; not scored here).', '',
                  '| Seed | W_mix gmc | W_mix store_hist | W_F_mix gmc | W_F_mix store_hist | '
                  'W_F_gru gmc | W_F_gru store_hist | N gmc | N store_hist | Z gmc | Z store_hist |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for seed in data.get('scoring_seeds', []):
            if (seed, 'gmc_hist') not in by or (seed, 'store_hist') not in by:
                continue
            g, s = by[(seed, 'gmc_hist')], by[(seed, 'store_hist')]
            lines.append(f'| {seed} | {cell(g["W_mix"])} | {cell(s["W_mix"])} | {cell(g["W_F_mix"])} | '
                         f'{cell(s["W_F_mix"])} | {cell(g["W_F_gru"])} | {cell(s["W_F_gru"])} | '
                         f'{cell(g["share"]["N"])} | {cell(s["share"]["N"])} | '
                         f'{cell(g["share"]["Z"])} | {cell(s["share"]["Z"])} |')
    lines += ['', 'Protocol notes:',
              '- World: FactWorld, channels G C1 C7 E30 E110 M7 M37 F N Z. Glance = 64 symbols. Processes resume.',
              '- N: uniform over 42 symbols. Z: the space symbol repeated. Seeds 0 1 2. Smoke seed 0, T = 32.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance, then zero or two replay steps.',
              '- nostore: no store, no match, no replay. hist_rep: no store, no match, whole-history replay.',
              '- store_norep: M4a store and match, no replay. store_rep: replay from the utility store only.',
              '- store_hist: M4a store and match, whole-history replay. gmc_hist: same, controller gmc_prop.',
              '- Rand conditions use one choice/observation stream per seed. Replay has its own stream.',
              '- History replay: uniform over starts 0 through L−64 on every channel with L ≥ 64,',
              '  batch 8, two steps, zero hidden state, that channel\'s embedding, preceding symbol,',
              '  zero embedding at position 0. The log is every observed symbol. It is not added to M.',
              '  history_bits_unpriced = observed symbols × log2(42).',
              '- Store replay: M4a\'s sampler. Eight columns of 64 from stored windows, one channel per',
              '  column, zero hidden state, zero embedding at a window boundary. Skipped if the store is empty.',
              '- Store writes: mixture bits > 4, window [i−16, i+4], overlap merges, touching windows stay',
              '  separate, deferred tail joins its head. Match credit only. Decay 0.999 per glance.',
              '  Protected for 32 glances. A tail whose head was evicted is counted and not rewritten.',
              '- Hard budget: M ≤ 0.15 × observed symbols × log2(42). Unprotected windows leave by lowest',
              '  utility. If still over and every window is protected, the oldest protected window leaves,',
              '  repeating until M ≤ budget. Oldest is the smallest birth glance, then the smallest start.',
              '- Canonical one-window slack: 21 × log2(42) + 32 bits. Asserted: after each glance,',
              '  M ≤ budget, and peak M/N ≤ budget ratio + that slack / N.',
              '- Match: order 5 inside that channel\'s windows. Fixed-share mixture, α = 0.001, per channel.',
              '- Prequential: mixture bits, or GRU bits when there is no store, are recorded before the',
              '  Adam step and before writes.',
              '- gmc_hist: imported gmc_reward after the live step and before replay. EMA β = 0.3.',
              '  Allocation is proportional to max(value, 0) + 0.01. No epsilon after the first pass.',
              '- W_mix: frozen GRU, frozen match, frozen mixture weights, fresh 2048 symbols on each',
              '  learnable channel from a deep copy of the world, carried GRU state. Mean over 8 channels.',
              '- W_gru: the same continuations scored with the frozen GRU probability alone.',
              '  nostore and hist_rep set W_mix = W_gru. The history log is not part of W.',
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
    lengths = np.zeros(len(CH), dtype=np.int64)
    lengths[0], lengths[1], lengths[2] = 128, 0, 64
    # valid starts: channel 0 has 65, channel 1 has 0, channel 2 has 1. Total 66.
    draws = np.array([0, 64, 65])
    channels, starts, total = segment_index(lengths, draws)
    assert total == 66
    assert channels.tolist() == [0, 0, 2]
    assert starts.tolist() == [0, 64, 0]
    hist = torch.zeros((len(CH), 128), dtype=torch.long)
    hist[0, :64] = torch.arange(64)
    lengths = np.zeros(len(CH), dtype=np.int64)
    lengths[0] = 64
    previous, targets, cis = sample_history(hist, lengths, np.random.default_rng(0))
    assert previous.shape == (B, 8) and targets.shape == (B, 8) and cis.shape == (8,)
    assert torch.all(cis == 0) and torch.all(targets[:, 0] == torch.arange(B))
    assert int(previous[0, 0]) == -1 and int(previous[1, 0]) == 0
    payload = np.arange(30, dtype=np.uint8) % A
    ids = np.zeros(80, dtype=np.int16)
    # Unprotected lowest utility leaves before a protected window.
    low = UtilityStore(math.inf, ids.copy())
    high = UtilityStore(math.inf, ids.copy())
    low.chunk = high.chunk = 0
    low.write(0, payload[:20], fresh=True)
    high.write(0, payload[:20], fresh=True)
    low.utils[0], high.utils[0] = 0.1, 9.0
    low.born[0], high.born[0] = 0, PROTECT
    budget = retained_memory([high])
    dropped, protected = evict_hard([low, high], glance=PROTECT, budget=budget + 1e-9)
    assert dropped == 1 and protected == 0 and len(low.windows) == 0 and len(high.windows) == 1
    # Both protected: the older one leaves, even if its utility is higher.
    older = UtilityStore(math.inf, ids.copy())
    newer = UtilityStore(math.inf, ids.copy())
    older.chunk = newer.chunk = 0
    older.write(0, payload[:20], fresh=True)
    newer.write(0, payload[:20], fresh=True)
    older.utils[0], newer.utils[0] = 50.0, 0.01
    older.born[0], newer.born[0] = 1, 5
    budget = retained_memory([newer])
    dropped, protected = evict_hard([older, newer], glance=5, budget=budget + 1e-9)
    assert dropped == 1 and protected == 1 and len(older.windows) == 0 and len(newer.windows) == 1
    assert retained_memory([older, newer]) <= budget + 1e-6
    # A budget of zero clears protected windows.
    dropped, protected = evict_hard([newer], glance=5, budget=0.0)
    assert dropped == 1 and protected == 1 and len(newer.windows) == 0
    # Store replay still returns M4a's batch shape.
    kept = UtilityStore(math.inf, ids.copy())
    kept.chunk = 0
    kept.write(0, np.arange(B, dtype=np.uint8) % A, fresh=True)
    previous, targets, cis = sample_replay([kept], np.random.default_rng(0))
    assert previous.shape == (B, 8) and targets.shape == (B, 8)
    print(f'{now()} checks passed', flush=True)


def align(reference, result):
    if result['cond'] not in RAND_CONDS:
        return
    found = reference.get(result['seed'])
    if found is None:
        reference[result['seed']] = (result['glance_checksum'], result['observation_checksum'])
        return
    glance, observed = found
    if result['glance_checksum'] != glance or result['observation_checksum'] != observed:
        raise AssertionError(
            f'{result["cond"]} seed {result["seed"]} broke the shared rand glance sequence')


def execute(data, out, cond, seed, glances, phase, reference):
    result, choices, bits = run(cond, seed, glances, phase)
    align(reference, result)
    data['runs'].append(result)
    data['total_wall_seconds'] = time.monotonic() - data['clock']
    np.savez_compressed(out / f'{phase}_{cond}_{seed}_{glances}.npz',
                        channels=choices, prequential_bits=bits)
    save(data, out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    configure_threads()
    print(f'{now()} torch threads {torch.get_num_threads()}', flush=True)
    checks()
    if args.checks_only:
        return
    clock = time.monotonic()
    smoke_dir = Path('results/m4b/smoke')
    smoke_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke or not (smoke_dir / 'results.json').exists():
        smoke = dict(started=now(), status='running', phase='smoke', runs=[], notes=[],
                     chosen_T=32, scoring_seeds=[0], dropped_conditions=[], clock=clock,
                     torch_version=torch.__version__, protocol='handoffs/2026-11-10_grok_m4b_ablation.md')
        reference = {}
        for cond in CONDS:
            execute(smoke, smoke_dir, cond, 0, 32, 'smoke', reference)
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
    out = Path('results/m4b')
    rates = {r['cond']: r['wall_seconds'] / r['glances'] for r in smoke['runs']}
    projected = sum(rates[c] * 512 * len(SEEDS) for c in CONDS)
    active = list(CONDS)
    notes = [f'Smoke-based projected score wall seconds: {projected:.1f}.']
    dropped = []
    smoke_wall = float(smoke['total_wall_seconds'])
    if smoke_wall + projected > WALL_BUDGET:
        active = [c for c in CONDS if c != 'gmc_hist']
        dropped = ['gmc_hist']
        notes.append('Dropped gmc_hist before scoring: smoke projection plus smoke exceeds 120 minutes.')
        projected = sum(rates[c] * 512 * len(SEEDS) for c in active)
        notes.append(f'Projected score wall seconds without gmc_hist: {projected:.1f}.')
    data = dict(started=now(), status='running', runs=[], notes=notes, chosen_T=512,
                scoring_seeds=list(SEEDS), dropped_conditions=dropped, clock=time.monotonic(),
                torch_version=torch.__version__,
                protocol='handoffs/2026-11-10_grok_m4b_ablation.md',
                smoke_wall_seconds=smoke_wall, projected_score_seconds=projected)
    save(data, out)
    reference = {}
    slowdown = 1.0
    for seed in SEEDS:
        for cond in CONDS:
            if cond not in active:
                continue
            spent = smoke_wall + (time.monotonic() - data['clock'])
            eta = rates[cond] * slowdown * 512
            if cond == 'gmc_hist' and spent + eta > WALL_BUDGET:
                name = f'gmc_hist/seed{seed}'
                data['dropped_conditions'].append(name)
                data['notes'].append(
                    f'Dropped {name}: 120-minute wall budget (spent {spent:.0f}s, eta {eta:.0f}s).')
                save(data, out)
                continue
            result = execute(data, out, cond, seed, 512, 'score', reference)
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
