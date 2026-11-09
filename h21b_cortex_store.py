"""H21b: stored-history replay and exact recall; principles 1 and 2.

Fixed protocol from handoffs/2026-11-10_codex_h21b_cortex_store.md.
Run --smoke before the full run. No selection or verdicts.
"""
import argparse
import json
import lzma
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h21a_substrate import (GRU, CHUNK, N, detach, summarize, mean_std, now,
                            stream, ALPHA, PROCS, WORLDS, kt_floors)

NAMES = ['gru', 'gru_r2', 'match', 'gru+match', 'gru_r2+match', 'kt_mixture', 'lzma_9e']


def match_predict(x):
    """Table values are target positions j: x[j-8:j] is the indexed context.

    Query before inserting t, so j < t and the recalled target is observed.
    Overlapping occurrences are allowed; extension reads only symbols < t.
    """
    started = time.monotonic()
    n, a = len(x), len(ALPHA)
    probs = np.full((n, a), 1 / a, dtype=np.float64)
    last = {}
    hits = np.zeros(4, dtype=np.int64)
    trials = np.zeros(4, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)
    lengths = np.zeros(n, dtype=np.uint8)
    for t in range(8, n):
        key = bytes(x[t-8:t])
        j = last.get(key)
        if j is not None:
            assert j < t
            length = 8
            while length < 32 and j > length and x[t-length-1] == x[j-length-1]:
                length += 1
            bucket = 0 if length < 12 else 1 if length < 16 else 2 if length < 24 else 3
            confidence = (hits[bucket] + 1) / (trials[bucket] + 2)
            predicted = x[j]
            probs[t].fill((1 - confidence) / (a - 1))
            probs[t, predicted] = confidence
            matched[t] = True
            lengths[t] = length
            correct[t] = predicted == x[t]
            # Both adaptive confidence and context index update only after scoring.
            trials[bucket] += 1
            hits[bucket] += int(correct[t])
        last[key] = t
    return probs, matched, correct, lengths, dict(
        wall_seconds=time.monotonic()-started, table_entries=len(last),
        bucket_hits=hits.tolist(), bucket_trials=trials.tolist())


def cortex(text, seed, replay, progress_every=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRU(256)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor([ALPHA.index(c) for c in text], dtype=torch.long)
    probs = np.empty((len(x), len(ALPHA)), dtype=np.float64)
    bits = np.empty(len(x), dtype=np.float64)
    state = None
    started = time.monotonic()
    replay_steps = 0

    def forward(start, end, hidden):
        if start == 0:
            emb = torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end-1])))
        else:
            emb = model.embedding(x[start-1:end-1])
        return model(emb, hidden)

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, len(x), CHUNK):
        end = min(start + CHUNK, len(x))
        optimizer.zero_grad(set_to_none=True)
        logits, state = forward(start, end, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        bits[start:end] = losses.detach().double().numpy() / math.log(2)
        probs[start:end] = logits.detach().double().softmax(-1).numpy()
        state = detach(state)
        step(losses.mean())
        if replay:
            for _ in range(2):
                # Uniform over all complete 64-target chunks of the scored prefix.
                rstart = int(rng.integers(0, end-CHUNK+1))
                assert rstart + CHUNK <= end
                optimizer.zero_grad(set_to_none=True)
                rlogits, _ = forward(rstart, rstart+CHUNK, None)
                step(F.cross_entropy(rlogits, x[rstart:rstart+CHUNK]))
                replay_steps += 1
        if (start // CHUNK + 1) % progress_every == 0 or end == len(x):
            print(f'{now()} gru{"_r2" if replay else ""} seed={seed} n={end}/{len(x)} bits={bits[:end].mean():.6f} seconds={time.monotonic()-started:.2f}', flush=True)
    assert np.isfinite(bits).all()
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(bits, -np.log2(probs[np.arange(len(x)), x.numpy()]), atol=2e-6)
    return probs, bits, dict(wall_seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()), replay_steps=replay_steps)


def mixture(p, q, x):
    started = time.monotonic()
    weights = np.full(2, 0.5)
    bits = np.empty(len(x))
    for t, symbol in enumerate(x):
        likelihoods = np.array([p[t, symbol], q[t, symbol]])
        likelihood = float(weights @ likelihoods)
        bits[t] = -math.log2(likelihood)
        weights *= likelihoods / likelihood
        weights = (1 - 0.001) * weights + 0.001 / 2
    return bits, time.monotonic()-started


def report(data, out):
    rows = data['runs']
    lines = ['# H21b measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
        f"N = {data['n']}; seeds = {data['seeds']}; CPU threads = 2; population std (ddof=0).", '',
        '| Learner / floor | slow bits/symbol | fast bits/symbol |', '|---|---:|---:|']
    for name in NAMES:
        lines.append('| ' + name + ' | ' + ' | '.join(mean_std([r['bits_per_symbol'] for r in rows if r['world'] == w and r['learner'] == name]) for w in WORLDS) + ' |')
    lines += ['', '| World | Seed | ' + ' | '.join(NAMES) + ' |', '|---|---:|' + '---:|'*len(NAMES)]
    for w in WORLDS:
        for seed in data['seeds']:
            found = {r['learner']: r for r in rows if r['world'] == w and r['seed'] == seed}
            if found:
                lines.append(f'| {w} | {seed} | ' + ' | '.join(f"{found[k]['bits_per_symbol']:.6f}" if k in found else 'NA' for k in NAMES) + ' |')
    lines += ['', 'Per-process bits/symbol, slow world; arithmetic mean over seeds with observations.', '',
        '| Learner | ' + ' | '.join(PROCS) + ' |', '|---|' + '---:|'*len(PROCS)]
    for name in NAMES:
        rs = [r for r in rows if r['world'] == 'slow' and r['learner'] == name]
        vals = [[r.get('by_process', {}).get(p) for r in rs] for p in PROCS]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{np.mean([v for v in vs if v is not None]):.6f}' if any(v is not None for v in vs) else 'NA' for vs in vals) + ' |')
    lines += ['', 'lzma per-process bits: NA; the whole-stream file code has no per-symbol or process attribution.', '',
        'Slow-world match statistics: arithmetic mean over seeds with observations (hit rate excludes seeds with no matches).', '',
        '| Process | Seeds observed | Mean match fraction | Mean match hit rate |', '|---|---:|---:|---:|']
    slow_matches = [r for r in rows if r['world'] == 'slow' and r['learner'] == 'match']
    for p in ['all'] + PROCS:
        stats = [r['match_statistics'][p] for r in slow_matches]
        fractions = [s['fraction'] for s in stats if s['fraction'] is not None]
        hit_rates = [s['hit_rate'] for s in stats if s['hit_rate'] is not None]
        frac = f'{np.mean(fractions):.6f}' if fractions else 'NA'
        hit = f'{np.mean(hit_rates):.6f}' if hit_rates else 'NA'
        lines.append(f'| {p} | {len(fractions)} | {frac} | {hit} |')
    lines += ['',
        '| World | Seed | Process | Symbols | Match fraction | Match hit rate |', '|---|---:|---|---:|---:|---:|']
    for r in rows:
        if r['learner'] == 'match':
            for p, stats in r['match_statistics'].items():
                hit = 'NA' if stats['hit_rate'] is None else f"{stats['hit_rate']:.6f}"
                frac = 'NA' if stats['fraction'] is None else f"{stats['fraction']:.6f}"
                lines.append(f"| {r['world']} | {r['seed']} | {p} | {stats['symbols']} | {frac} | {hit} |")
    lines += ['', '| World | Seed | Learner | Wall seconds |', '|---|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['world']} | {r['seed']} | {r['learner']} | {r['wall_seconds']:.3f} |")
    lines += ['', 'Mixture wall time includes its cortex, match model, and mixture update; components are reused, not rerun.',
        'GRU: H21a imported GRU(256), embedding 32, chunk 64, Adam lr 0.003, clip 1.0; shifted inputs, first embedding zero, carried state detached per chunk.',
        'Replay: 2 separate Adam steps after every scored chunk; independent uniform starts 0 through scored_prefix_length−64 inclusive; replay hidden state zero, preceding symbol input (zero embedding at position 0). Live hidden state is preserved across replay.',
        'Match: preceding 8-symbol key, most recent earlier target, backward extension to 32; bucket confidence (hits+1)/(trials+2); remaining probability uniform over 40 symbols. Updates follow scoring.',
        'Mixtures: initial weights (0.5,0.5); likelihood update then fixed share 0.001 after each scored symbol.',
        'Scores are prequential observation bits; raw history and match table are retained. No store charge is added to the specified score; this is not two-part T.',
        'lzma: preset 9 | PRESET_EXTREME, whole-stream bytes including container overhead. KT: imported h20_world.kt_floors.',
        'Full float64 predictive distributions and symbol code lengths are saved in stream-specific NPZ files.', '']
    lines += [f'Runtime note: {note}' for note in data['notes']]
    if 'total_wall_seconds' in data:
        lines += ['', f"Total run wall seconds: {data['total_wall_seconds']:.3f}."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    report(data, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path('results/h21b') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    n, seeds = (8192, [6]) if args.smoke else (N, [6,7,8])
    clock = time.monotonic()
    data = dict(started=now(), n=n, seeds=seeds, principles=['1: store everything', '2: improve the compressor'],
                torch_version=torch.__version__, notes=[], runs=[])
    drop_fast_replay = False
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        times = {r['learner']: r['wall_seconds']*16 for r in smoke['runs']}
        projected = 6*sum(times[k] for k in ['gru','gru_r2','match','kt_mixture','lzma_9e'])
        data['smoke_projected_total_seconds'] = projected
        if projected > 81*60:
            drop_fast_replay = True
            data['notes'].append('Fast gru_r2 and gru_r2+match omitted: smoke projection exceeds 81-minute planning allowance within 90-minute budget.')
    for world in (['slow'] if args.smoke else list(WORLDS)):
        for seed in seeds:
            text, labels = stream(world, seed, n)
            procs = np.array(labels)
            x = np.array([ALPHA.index(c) for c in text], dtype=np.uint8)
            arrays = dict(symbols=x, processes=procs)
            def add(name, bits, **extra):
                r = dict(world=world, seed=seed, learner=name, n=n, **summarize(bits, procs), **extra)
                data['runs'].append(r)
                arrays[name+'_bits'] = bits
                save(data, out)
                return r
            started = time.monotonic()
            _, kt, _ = kt_floors(text)
            add('kt_mixture', kt, wall_seconds=time.monotonic()-started)
            started = time.monotonic()
            compressed = lzma.compress(text.encode(), preset=9 | lzma.PRESET_EXTREME)
            data['runs'].append(dict(world=world, seed=seed, learner='lzma_9e', n=n,
                bits_per_symbol=8*len(compressed)/n, compressed_bytes=len(compressed), wall_seconds=time.monotonic()-started))
            mp, matched, correct, lengths, mstats = match_predict(x)
            stats = {}
            for p in ['all']+PROCS:
                mask = np.ones(n, dtype=bool) if p == 'all' else procs == p
                count, matches = int(mask.sum()), int(matched[mask].sum())
                stats[p] = dict(symbols=count, matches=matches, hits=int(correct[mask].sum()),
                    fraction=matches/count if count else None,
                    hit_rate=int(correct[mask].sum())/matches if matches else None)
            mr = add('match', -np.log2(mp[np.arange(n),x]), match_statistics=stats, **mstats)
            arrays.update(match_probs=mp, matched=matched, correct=correct, match_lengths=lengths)
            for replay in [False, True]:
                if replay and world == 'fast' and (drop_fast_replay or time.monotonic()-clock > 70*60):
                    if not drop_fast_replay:
                        drop_fast_replay = True
                        data['notes'].append('Remaining fast gru_r2 and mixture omitted after elapsed wall time exceeded 70 minutes.')
                    continue
                name = 'gru_r2' if replay else 'gru'
                gp, bits, timing = cortex(text, seed, replay)
                arrays[name+'_probs'] = gp
                gr = add(name, bits, **timing)
                mbits, elapsed = mixture(gp, mp, x)
                add(name+'+match', mbits, wall_seconds=gr['wall_seconds']+mr['wall_seconds']+elapsed,
                    mixture_only_wall_seconds=elapsed)
            np.savez_compressed(out/f'{world}_{seed}.npz', **arrays)
            print(f'{now()} completed {world} seed={seed}; total seconds={time.monotonic()-clock:.1f}', flush=True)
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()
