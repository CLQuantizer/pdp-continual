"""H25c: principles 1–4; per-channel signed gradient-momentum progress.
Fixed protocol: handoffs/2026-11-10_codex_h25c_gmc_per_channel.md.
Only this file and results/h25c/* are written. No verdicts or bar changes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h25a_curio_neural import (NeuralCompressor, CH, LEARN, B,
    EVAL, EPS, BETA, glance, World, ALPHA, A, RES, now, mean_std)

NEW_CONDS = ['gmcc', 'gmcc_prop']
CONDS = ['rand', 'rand7', 'err', 'gmc_prop'] + NEW_CONDS
PROTOCOL = [
    'rand/rand7/err rows reused unchanged from H25a; gmc_prop rows reused unchanged from H25b. Historical baseline wall times are excluded from H25c elapsed wall time.',
    'Each channel has an independent initially-zero parameter-shaped momentum, updated only by its live gradient: m_c = 0.9*m_c + 0.1*g.',
    'Before updating momentum: reward = dot(g,m_c)/(norm(g)*norm(m_c)+1e-12) * live pre-learning bits/symbol. First channel glance reward is zero. Signed reward; no absolute value or reward clipping.',
    'g is the live mean-loss gradient after the unchanged global norm clipping at 1.0, exactly as used by the live Adam step. Reward and channel momentum update precede Adam and replay. Replay never updates channel momentum.',
    'No reward normalization, tuning or baseline reruns. Original EMA/controller, including max(value,0)+0.01 for proportional allocation.',
]


@torch.no_grad()
def gmcc_reward(parameters, momentum, bits):
    # Float64 reductions avoid underflow for near-learned channels.
    dot = torch.zeros((), dtype=torch.float64)
    g2 = torch.zeros_like(dot)
    m2 = torch.zeros_like(dot)
    for p, m in zip(parameters, momentum):
        grad = p.grad.detach().to(torch.float64)
        old = m.to(torch.float64)
        dot += (grad*old).sum()
        g2 += grad.square().sum()
        m2 += old.square().sum()
    reward = float(dot/(g2.sqrt()*m2.sqrt()+1e-12))*bits
    for p, m in zip(parameters, momentum):
        m.mul_(0.9).add_(p.grad, alpha=0.1)
    return reward


def run(cond, seed, glances, out, phase):
    assert cond in NEW_CONDS
    started = time.monotonic()
    torch.manual_seed(seed)
    model = NeuralCompressor()
    parameters = list(model.parameters())
    momenta = [[torch.zeros_like(p) for p in parameters] for _ in CH]
    parameter_count = sum(p.numel() for p in parameters)
    momentum_bytes = sum(m.numel()*m.element_size() for ms in momenta for m in ms)
    assert len(momenta) == 9 and momentum_bytes == 9*parameter_count*4
    assert momentum_bytes < 3*1024**3
    print(f'{now()} {cond} s{seed}: parameters={parameter_count}; '
          f'channel momentum={momentum_bytes} bytes ({momentum_bytes/1024**2:.3f} MiB); '
          'fits 3 GiB=True', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world, rng = World(seed), np.random.default_rng(10_000+seed)
    rrng, replay_rng = np.random.default_rng(20_000+seed), np.random.default_rng(30_000+seed)
    hist = torch.empty((len(CH), glances*B), dtype=torch.long)
    lengths = np.zeros(len(CH), dtype=np.int64)
    states = [None]*len(CH)
    reservoirs = [[] for _ in CH]
    values, counts = np.zeros(len(CH)), np.zeros(len(CH), dtype=int)
    log, loss_log, rewards = [], [], []

    def step(loss, channel=None, bits=0.0):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        reward = gmcc_reward(parameters, momenta[channel], bits) if channel is not None else 0.0
        opt.step()
        return reward

    for g in range(glances):
        if g < len(CH):
            ci = g
        elif cond.endswith('_prop'):
            weights = np.maximum(values, 0)+0.01
            ci = int(rng.choice(len(CH), p=weights/weights.sum()))
        elif rng.random() < EPS:
            ci = int(rng.integers(len(CH)))
        else:
            ci = max(range(len(CH)), key=lambda i: (values[i], rng.random()))
        channel = CH[ci]
        x = torch.tensor(glance(world, channel, rng), dtype=torch.long)
        oldlen = int(lengths[ci])
        previous = torch.cat((hist[ci, oldlen-1:oldlen] if oldlen else torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        bits = float(losses.detach().mean())/math.log(2)
        states[ci] = state.detach()
        reward = step(losses.mean(), channel=ci, bits=bits)
        if counts[ci] == 0:
            assert reward == 0.0
        hist[ci, oldlen:oldlen+B] = x
        lengths[ci] += B
        # Every valid 64-symbol segment of any observed channel has equal weight.
        valid = np.maximum(lengths-B+1, 0)
        cumulative = np.cumsum(valid)
        for _ in range(2):
            draws = replay_rng.integers(int(cumulative[-1]), size=8)
            channels = np.searchsorted(cumulative, draws, side='right')
            starts = draws-np.r_[0, cumulative[:-1]][channels]
            indices = torch.tensor(starts)[None, :]+torch.arange(B)[:, None]
            cis = torch.tensor(channels)
            targets = hist[cis[None, :], indices]
            previous = hist[cis[None, :], (indices-1).clamp_min(0)].clone()
            previous[indices == 0] = -1
            replay_logits, _ = model(previous, cis)
            step(F.cross_entropy(replay_logits.reshape(-1, A), targets.reshape(-1)))
        # Algorithm R, over every position, including the short initial contexts.
        for position in range(oldlen, oldlen+B):
            if len(reservoirs[ci]) < RES:
                reservoirs[ci].append(position)
            else:
                k = int(rrng.integers(position+1))
                if k < RES:
                    reservoirs[ci][k] = position
        values[ci] = reward if counts[ci] == 0 else (1-BETA)*values[ci]+BETA*reward
        counts[ci] += 1
        log.append(ci); loss_log.append(bits); rewards.append(reward)
        if (g+1) % 128 == 0:
            print(f'{now()} {phase} {cond} s{seed} T={g+1}/{glances} wall={time.monotonic()-started:.1f}s', flush=True)
    held = {}
    wc = copy.deepcopy(world)
    model.eval()
    with torch.no_grad():
        for ci, channel in enumerate(LEARN):
            seq = torch.tensor([ALPHA.index(s) for s in wc.take(channel, EVAL)])
            previous = torch.cat((hist[ci, lengths[ci]-1:lengths[ci]], seq[:-1]))
            state = states[ci]
            total = 0.0
            for start in range(0, EVAL, B):
                logits, state = model(previous[start:start+B, None], torch.tensor([ci]), state)
                total += float(F.cross_entropy(logits[:, 0], seq[start:start+B], reduction='sum'))/math.log(2)
            held[channel] = total/EVAL
    choices = np.array(log)
    assert counts.sum() == glances and np.all(lengths == counts*B)
    assert all(len(r) == min(RES, int(lengths[i])) for i, r in enumerate(reservoirs))
    assert np.isfinite(loss_log).all() and np.isfinite(list(held.values())).all()
    assert np.isfinite(rewards).all()
    assert np.all(np.abs(rewards) <= np.asarray(loss_log)+1e-9)
    result = dict(phase=phase, cond=cond, seed=seed, glances=glances,
        W=float(np.mean(list(held.values()))), held=held,
        share={c: float(counts[i]/glances) for i, c in enumerate(CH)},
        share_by_quarter=[{c: float(np.mean(q == i)) for i, c in enumerate(CH)} for q in np.array_split(choices, 4)],
        life_bits=float(np.mean(loss_log)), wall_seconds=time.monotonic()-started,
        parameters=parameter_count, momentum_vectors=len(CH),
        momentum_bytes=momentum_bytes, momentum_fits_3gib=True, adam_steps=3*glances,
        replay_segments=16*glances, history_symbols=int(lengths.sum()),
        reservoir_samples=[len(r) for r in reservoirs])
    np.savez_compressed(out/f'{phase}_{cond}_{seed}_{glances}.npz', channels=choices,
                        prequential_bits=loss_log, rewards=rewards,
                        reservoir_positions=np.array(reservoirs, dtype=object))
    print(f'{now()} done {phase} {cond} s{seed} T={glances} W={result["W"]:.6f} wall={result["wall_seconds"]:.1f}s', flush=True)
    return result

def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    lines = ['# H25c measurements', '', f'Started: {data["started"]}. Updated: {now()}.', '',
        'CPU threads: 2. Standard deviation: population (ddof=0).']
    lines += ['', f'Chosen T: {data.get("chosen_T")}. Scoring seeds: {data.get("scoring_seeds", [])}.', '',
              '| Condition | W mean ± std | N share | Z share | Wall seconds (sum) |', '|---|---:|---:|---:|---:|']
    scoring = [r for r in data['runs'] if r['phase'] == 'score']
    for c in CONDS:
        rs = [r for r in scoring if r['cond'] == c]
        if rs:
            lines.append(f'| {c} | {mean_std([r["W"] for r in rs])} | {np.mean([r["share"]["N"] for r in rs]):.6f} | {np.mean([r["share"]["Z"] for r in rs]):.6f} | {sum(r["wall_seconds"] for r in rs):.3f} |')
    for phase in ['smoke', 'pilot', 'score']:
        rs = [r for r in data['runs'] if r['phase'] == phase]
        if not rs:
            continue
        lines += ['', f'{phase}: per-run held-out bits/symbol.', '',
                  '| T | Seed | Condition | W | '+' | '.join(LEARN)+' | Wall seconds |',
                  '|---:|---:|---|---:|'+'---:|'*(len(LEARN)+1)]
        for r in rs:
            lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | {r["W"]:.6f} | '+' | '.join(f'{r["held"][c]:.6f}' for c in LEARN)+f' | {r["wall_seconds"]:.3f} |')
    if scoring:
        lines += ['', 'Mean overall glance shares.', '', '| Condition | '+' | '.join(CH)+' |', '|---|'+'---:|'*len(CH)]
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            if rs:
                lines.append('| '+cond+' | '+' | '.join(f'{np.mean([r["share"][c] for r in rs]):.6f}' for c in CH)+' |')
        lines += ['', 'Mean per-channel held-out bits/symbol.', '', '| Condition | '+' | '.join(LEARN)+' |', '|---|'+'---:|'*len(LEARN)]
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            if rs:
                lines.append('| '+cond+' | '+' | '.join(f'{np.mean([r["held"][c] for r in rs]):.6f}' for c in LEARN)+' |')
        lines += ['', 'Glance shares: per seed, overall and quarter of life.', '', '| Seed | Condition | Period | '+' | '.join(CH)+' |', '|---:|---|---|'+'---:|'*len(CH)]
        for r in scoring:
            for label, shares in [('all', r['share'])]+[(f'Q{i+1}', q) for i, q in enumerate(r['share_by_quarter'])]:
                lines.append(f'| {r["seed"]} | {r["cond"]} | {label} | '+' | '.join(f'{shares[c]:.6f}' for c in CH)+' |')
        lines += ['', 'Mean glance shares by quarter.', '', '| Condition | Quarter | '+' | '.join(CH)+' |', '|---|---:|'+'---:|'*len(CH)]
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            if rs:
                for q in range(4):
                    lines.append(f'| {cond} | {q+1} | '+' | '.join(f'{np.mean([r["share_by_quarter"][q][c] for r in rs]):.6f}' for c in CH)+' |')
        lines += ['', 'Mean first-to-last-quarter glance share changes.', '',
                  '| Condition | Channel | Q1 | Q4 | Q4 − Q1 |', '|---|---|---:|---:|---:|']
        for cond in CONDS:
            rs = [r for r in scoring if r['cond'] == cond]
            if rs:
                for channel in ('C1', 'C7', 'E30', 'E110'):
                    first, last = [np.mean([r['share_by_quarter'][q][channel] for r in rs]) for q in (0, 3)]
                    lines.append(f'| {cond} | {channel} | {first:.6f} | {last:.6f} | {last-first:.6f} |')
        lines += ['', 'Paired W differences.', '', '| Seed | Condition | W − rand | W − rand7 | W − err |', '|---:|---|---:|---:|---:|']
        for seed in data.get('scoring_seeds', []):
            w = {r['cond']: r['W'] for r in scoring if r['seed'] == seed}
            for cond in NEW_CONDS:
                if cond in w:
                    lines.append(f'| {seed} | {cond} | {w[cond]-w["rand"]:.6f} | {w[cond]-w["rand7"]:.6f} | {w[cond]-w["err"]:.6f} |')
    lines += ['', 'Protocol: shared GRU 256; symbol embedding 32; channel embedding 8; Adam 0.003; clipping 1.0; one live step and two replay steps (8 × 64) per glance.',
              'Replay draws uniformly with replacement over all valid channel-local 64-symbol segments, including the current glance, from zero hidden state; preceding symbol supplied except at channel start.',
              'Live states are carried separately per channel and detached before learning; replay and reservoir scoring do not change them.',
              'Reservoir: Algorithm R over every observed channel position; up to 32 preceding symbols and target; zero hidden state and absent-symbol input before context. Current positions added after reward. Progress divided by 128, including partially filled reservoirs.',
              'W: frozen final model, fresh 2048-symbol continuation per learnable channel from a deep-copied world, starting from saved live channel state. History/store unpriced in the specified W metric.',
              'Controller epsilon 0.1; EMA beta 0.3. Proportional controller uses max(value,0)+0.01 without epsilon after the first pass.', '']
    lines += ['', *PROTOCOL, '']
    new = [r for r in data['runs'] if r['cond'] in NEW_CONDS]
    if new:
        r = new[0]
        lines += ['', f'Parameters: {r["parameters"]}. Extra momentum vectors: {r["momentum_vectors"]}. Extra momentum bytes: {r["momentum_bytes"]}. Fits 3 GiB: {int(r["momentum_fits_3gib"])}.']
    reference = data.get('h25b_gmc_N_share')
    if reference is not None and scoring:
        rs = [r for r in scoring if r['cond'] == 'gmcc']
        if rs:
            share = float(np.mean([r['share']['N'] for r in rs]))
            lines += ['', '| Condition | N share | N share change vs H25b gmc |', '|---|---:|---:|',
                      f'| H25b gmc | {reference:.6f} | 0.000000 |',
                      f'| gmcc | {share:.6f} | {share-reference:.6f} |']
    lines += [f'Runtime note: {n}' for n in data['notes']]
    if 'projected_total_seconds_six_seeds' in data:
        lines += ['', f'Six-seed projected total wall seconds: {data["projected_total_seconds_six_seeds"]:.3f}.']
    lines += ['', f'Elapsed wall seconds: {data.get("total_wall_seconds", 0):.3f}.', f'Status: {data["status"]}.']
    if 'wall_seconds_including_smoke' in data:
        lines += ['', f'Smoke wall seconds: {data["smoke_wall_seconds"]:.3f}.',
                  f'Wall seconds including smoke: {data["wall_seconds_including_smoke"]:.3f}.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path('results/h25c') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    clock = time.monotonic()
    sources = {name: Path(f'results/{name}/results.json') for name in ('h25a', 'h25b')}
    baseline = {name: json.loads(path.read_text()) for name, path in sources.items()}
    reused = []
    for name, conditions in [('h25a', ['rand', 'rand7', 'err']), ('h25b', ['gmc_prop'])]:
        reused += [dict(r, source=str(sources[name])) for r in baseline[name]['runs']
                   if r['phase'] == 'score' and r['seed'] in (6, 7, 8) and r['cond'] in conditions]
    assert len(reused) == 12 and all(r['glances'] == 512 for r in reused)
    reference = np.mean([r['share']['N'] for r in baseline['h25b']['runs']
                         if r['phase'] == 'score' and r['cond'] == 'gmc' and r['seed'] in (6, 7, 8)])
    data = dict(started=now(), status='running', principles=[1, 2, 3, 4],
                torch_version=torch.__version__, notes=[], runs=[] if args.smoke else reused,
                baseline_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in sources.items()},
                chosen_T=32 if args.smoke else 512,
                scoring_seeds=[] if args.smoke else [6, 7, 8], protocol=PROTOCOL,
                budget_wall_seconds=5400, h25b_gmc_N_share=float(reference))
    smoke_seconds = 0.0
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        assert smoke['status'] == 'complete'
        assert {(r['cond'], r['seed'], r['glances']) for r in smoke['runs']} == {
            (c, 0, 32) for c in NEW_CONDS}
        smoke_seconds = smoke['total_wall_seconds']
        data['projected_scoring_seconds'] = 512*3*sum(r['wall_seconds']/32 for r in smoke['runs'])
    save(data, out)
    for cond in NEW_CONDS:
        for seed in ([0] if args.smoke else [6, 7, 8]):
            if time.monotonic()-clock+smoke_seconds >= 5400:
                data['status'] = 'budget_exhausted'
                data['total_wall_seconds'] = time.monotonic()-clock
                save(data, out)
                return
            data['runs'].append(run(cond, seed, data['chosen_T'], out, 'smoke' if args.smoke else 'score'))
            data['total_wall_seconds'] = time.monotonic()-clock
            save(data, out)
    data['status'] = 'complete'
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    if not args.smoke:
        data['smoke_wall_seconds'] = smoke_seconds
        data['wall_seconds_including_smoke'] = data['total_wall_seconds']+smoke_seconds
    save(data, out)


if __name__ == '__main__':
    main()
