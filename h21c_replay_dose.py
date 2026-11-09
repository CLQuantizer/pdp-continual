"""H21c replay dose measurements; principles 1 (history) and 2 (compressor).

Protocol: handoffs/2026-11-10_codex_h21c_replay_dose.md. No verdicts.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h21a_substrate import GRU, CHUNK, N, ALPHA, PROCS, detach, summarize, mean_std, now, stream
from h21b_cortex_store import mixture

VARIANTS = {'gru_r4': (4, 1), 'gru_rb8': (1, 8), 'gru_rb8x2': (2, 8)}
BASELINES = ['gru', 'gru_r2', 'gru_r2+match', 'kt_mixture', 'lzma_9e']
NAMES = BASELINES[:3] + [v for name in VARIANTS for v in (name, name+'+match')] + BASELINES[3:]


def cortex(text, seed, name, progress_every=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRU(256)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor([ALPHA.index(c) for c in text], dtype=torch.long)
    probs = np.empty((len(x), len(ALPHA)), dtype=np.float64)
    bits = np.empty(len(x), dtype=np.float64)
    steps, batch = VARIANTS[name]
    state = None
    replay_steps = 0
    started = time.monotonic()

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
        end = min(start+CHUNK, len(x))
        optimizer.zero_grad(set_to_none=True)
        logits, state = forward(start, end, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        bits[start:end] = losses.detach().double().numpy()/math.log(2)
        probs[start:end] = logits.detach().double().softmax(-1).numpy()
        state = detach(state)
        step(losses.mean())
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            if batch == 1:
                rstart = int(rng.integers(0, end-CHUNK+1))
                rlogits, _ = forward(rstart, rstart+CHUNK, None)
                loss = F.cross_entropy(rlogits, x[rstart:rstart+CHUNK])
            else:
                # Independent uniform draws with replacement, just as H21b.
                starts = torch.from_numpy(rng.integers(0, end-CHUNK+1, size=batch))
                indices = torch.arange(CHUNK)[:, None] + starts[None, :]
                assert int(indices.max()) < end
                emb = model.embedding(x[(indices-1).clamp_min(0)])
                # Every sequence starts with zero hidden state; only global t=0
                # also has a zero input embedding. Other starts use x[t-1].
                emb = emb.masked_fill((indices == 0)[:, :, None], 0)
                outputs, _ = model.gru(emb, None)
                rlogits = model.readout(outputs)
                loss = F.cross_entropy(rlogits.reshape(-1, len(ALPHA)), x[indices].reshape(-1))
            step(loss)
            replay_steps += 1
        if (start//CHUNK+1) % progress_every == 0 or end == len(x):
            print(f'{now()} {name} seed={seed} n={end}/{len(x)} bits={bits[:end].mean():.6f} seconds={time.monotonic()-started:.2f}', flush=True)
    assert np.isfinite(bits).all()
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(bits, -np.log2(probs[np.arange(len(x)), x.numpy()]), atol=2e-6)
    return probs, bits, dict(wall_seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()), replay_steps=replay_steps,
        replay_batch_size=batch, replay_chunks=replay_steps*batch)


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    rows = data['runs']
    lines = ['# H21c measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
        f"Slow world; N = {data['n']}; seeds = {data['seeds']}; CPU threads = 2; population std (ddof=0).", '',
        '| Learner / floor | Mean ± std bits/symbol | ' + ' | '.join(f'Seed {s}' for s in data['seeds']) + ' |',
        '|---|---:|' + '---:|'*len(data['seeds'])]
    for name in NAMES:
        rs = [r for r in rows if r['learner'] == name]
        found = {r['seed']: r for r in rs}
        lines.append('| '+name+' | '+mean_std([r['bits_per_symbol'] for r in rs])+' | '+' | '.join(f"{found[s]['bits_per_symbol']:.6f}" if s in found else 'NA' for s in data['seeds'])+' |')
    lines += ['', 'Per-process bits/symbol; arithmetic mean over seeds with observations.', '',
        '| Learner | '+' | '.join(PROCS)+' |', '|---|'+'---:|'*len(PROCS)]
    for name in NAMES:
        vals = [[r.get('by_process', {}).get(p) for r in rows if r['learner'] == name] for p in PROCS]
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([v for v in vs if v is not None]):.6f}' if any(v is not None for v in vs) else 'NA' for vs in vals)+' |')
    lines += ['', '| Seed | Learner | '+' | '.join(PROCS)+' |', '|---:|---|'+'---:|'*len(PROCS)]
    for r in rows:
        if r['learner'] in BASELINES:
            continue
        lines.append(f"| {r['seed']} | {r['learner']} | "+' | '.join('NA' if r['by_process'][p] is None else f"{r['by_process'][p]:.6f}" for p in PROCS)+' |')
    lines += ['', '| Seed | Learner | Wall seconds |', '|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['learner']} | {r['wall_seconds']:.3f} |")
    lines += ['', 'Baseline scores and timings copied from H21b; no baseline reruns. Match probabilities and timings reused from H21b.',
        'GRU(256), embedding 32, chunk 64, Adam lr 0.003, clip 1.0; H21b shifted inputs and detached carried live state.',
        'Replay follows each scored chunk and its live Adam step. Uniform starts 0 through scored_prefix_length−64 inclusive, with replacement; zero replay hidden state; preceding-symbol input (zero embedding at global position 0). Live state is preserved during replay.',
        'gru_r4: four sequential single-chunk Adam steps. gru_rb8: one Adam step on eight chunks. gru_rb8x2: two such steps. Batched loss is the mean over all 8 × 64 targets.',
        'Mixtures reuse H21b fixed share α = 0.001; wall seconds include cortex, reused match timing, and mixture update.',
        'Scores are prequential observation bits; retained history is unpriced, as specified in H21b. lzma has no per-process attribution.',
        'Grammar G for every row is in the per-process table; plain GRU reference = 1.176889 bits/symbol (H21b).',
        'Full predictive distributions and code lengths are saved in learner-specific NPZ files.', '']
    lines += ['Runtime note: '+note for note in data['notes']]
    if 'total_wall_seconds' in data:
        lines += ['', f"Total run wall seconds: {data['total_wall_seconds']:.3f}."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path('results/h21c') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    source = Path('results/h21b') / ('smoke' if args.smoke else '')
    old = json.loads((source/'results.json').read_text())
    n, seeds = (8192, [6]) if args.smoke else (N, [6, 7, 8])
    assert old['n'] == n
    clock = time.monotonic()
    data = dict(started=now(), n=n, seeds=seeds, principles=['1: store everything', '2: improve the compressor'],
        torch_version=torch.__version__, notes=[], runs=[], baseline_source=str(source/'results.json'))
    for r in old['runs']:
        if r['world'] == 'slow' and r['seed'] in seeds and r['learner'] in BASELINES:
            row = copy.deepcopy(r)
            row['reused_from'] = str(source/'results.json')
            data['runs'].append(row)
    assert len(data['runs']) == len(seeds)*len(BASELINES)
    variants = list(VARIANTS)
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        times = {r['learner']: r['wall_seconds'] for r in smoke['runs'] if r['learner'] in VARIANTS}
        projection = sum(times.values())*16*len(seeds)
        data['smoke_projected_total_seconds'] = projection
        # Reserve 10% for mixtures, serialization, and runtime variability.
        if projection > 81*60:
            variants.remove('gru_r4')
            data['notes'].append('gru_r4 and gru_r4+match omitted first: smoke projection exceeds 81-minute planning allowance within 90-minute budget.')
    save(data, out)
    # Batched variants first, leaving sequential replay as first budget drop.
    order = [v for v in ['gru_rb8', 'gru_rb8x2', 'gru_r4'] if v in variants]
    for name in order:
        for seed in seeds:
            if not args.smoke and name == 'gru_r4':
                smoke_time = times[name]*16
                if time.monotonic()-clock + smoke_time*(len(seeds)-seeds.index(seed)) > 88*60:
                    data['notes'].append(f'Remaining gru_r4 runs omitted before seed {seed}: projected wall exceeds 88-minute planning allowance.')
                    break
            text, labels = stream('slow', seed, n)
            procs = np.array(labels)
            x = np.array([ALPHA.index(c) for c in text], dtype=np.uint8)
            with np.load(source/f'slow_{seed}.npz') as previous:
                assert np.array_equal(x, previous['symbols'])
                assert np.array_equal(procs, previous['processes'])
                mp = previous['match_probs'].copy()
            gp, bits, timing = cortex(text, seed, name)
            data['runs'].append(dict(world='slow', seed=seed, learner=name, n=n, **summarize(bits, procs), **timing))
            mbits, elapsed = mixture(gp, mp, x)
            match_time = next(r['wall_seconds'] for r in old['runs'] if r['world'] == 'slow' and r['seed'] == seed and r['learner'] == 'match')
            data['runs'].append(dict(world='slow', seed=seed, learner=name+'+match', n=n, **summarize(mbits, procs),
                wall_seconds=timing['wall_seconds']+match_time+elapsed, mixture_only_wall_seconds=elapsed))
            np.savez_compressed(out/f'slow_{seed}_{name}.npz', symbols=x, processes=procs, probs=gp, bits=bits, mixture_bits=mbits)
            save(data, out)
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()
