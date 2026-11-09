"""H25b: principles 1–4; fixed reward ablation of H25a.
Only this file and results/h25b/* are written. No verdicts or bar changes.
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
from h25a_curio_neural import (NeuralCompressor as BaseCompressor, CH, LEARN, B,
    EVAL, EPS, BETA, glance, World, ALPHA, A, RES, CONTEXT, now, mean_std)

NEW_CONDS = ['progc', 'progc_prop', 'gmc', 'gmc_prop']
CONDS = ['rand', 'rand7', 'err', 'prog', 'prog_prop'] + NEW_CONDS
PROTOCOL = [
    'Baseline rows are reused unchanged from results/h25a/results.json; their wall times are historical and excluded from H25b elapsed wall time.',
    'progc: reservoir scores use softmax probabilities clamped to [2**(-8)/40, 1-2**(-8)] then normalized over all 41 symbols. Only reservoir reward scores use this coder; training and held-out W are unchanged.',
    'progc before/after use the identical prior-history reservoir, immediately before/after the live Adam step and before either replay step; current positions enter the reservoir afterwards.',
    'GMC exact formula: r = sum_i abs(g_i * mhat_i / max(sqrt(vhat_i), 0.001*abs(theta_i))) / sqrt(d).',
    'g is the live mean-loss gradient after the unchanged global norm clipping at 1.0; theta is the pre-live-update parameter. mhat=m_t/(1-beta1**t), vhat=v_t/(1-beta2**t) use Adam state immediately after the live step (thus include this gradient). Moments and per-parameter step counters include previous live and replay updates. d counts all parameters. Zero numerator/denominator contributes zero; no Adam epsilon is added to this specified denominator. Reward is computed before replay.',
    'No reward normalization, tuning or baseline reruns. All four new conditions use the original EMA/controller, including max(value,0)+0.01 for proportional allocation.',
]


def coder_probs(logits):
    p = logits.softmax(-1).clamp(2**(-8)/40, 1-2**(-8))
    return p/p.sum(-1, keepdim=True)


class NeuralCompressor(BaseCompressor):
    @torch.no_grad()
    def reservoir_bits(self, ci, positions, history):
        if not positions:
            return 0.0
        positions = np.asarray(positions)
        total = 0.0
        for length in np.unique(np.minimum(positions, CONTEXT)):
            pos = torch.tensor(positions[np.minimum(positions, CONTEXT) == length])
            inputs = torch.full((int(length)+1, len(pos)), -1, dtype=torch.long)
            if length:
                inputs[1:] = history[pos[None, :] - int(length) + torch.arange(int(length))[:, None]]
            logits, _ = self(inputs, torch.full((len(pos),), ci, dtype=torch.long))
            p = coder_probs(logits[-1])
            total += float(-p.gather(1, history[pos, None]).log2().sum())
        return total


@torch.no_grad()
def gmc_reward(model, opt, floors):
    total = torch.zeros((), dtype=torch.float64)
    for p, floor in zip(model.parameters(), floors):
        state = opt.state[p]
        group = opt.param_groups[0]
        beta1, beta2 = group['betas']
        t = float(state['step'])
        mhat = state['exp_avg']/(1-beta1**t)
        vhat = state['exp_avg_sq']/(1-beta2**t)
        denom = torch.maximum(vhat.sqrt(), floor)
        term = torch.where(denom > 0, (p.grad*mhat).abs()/denom, 0.0)
        total += term.sum(dtype=torch.float64)
    return float(total)/math.sqrt(sum(p.numel() for p in model.parameters()))


def run(cond, seed, glances, out, phase):
    assert cond in NEW_CONDS
    started = time.monotonic()
    torch.manual_seed(seed)
    model = NeuralCompressor()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world, rng = World(seed), np.random.default_rng(10_000+seed)
    rrng, replay_rng = np.random.default_rng(20_000+seed), np.random.default_rng(30_000+seed)
    hist = torch.empty((len(CH), glances*B), dtype=torch.long)
    lengths = np.zeros(len(CH), dtype=np.int64)
    states = [None]*len(CH)
    reservoirs = [[] for _ in CH]
    values, counts = np.zeros(len(CH)), np.zeros(len(CH), dtype=int)
    log, loss_log, rewards = [], [], []

    def step(loss, coupling=False):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        floors = [1e-3*p.detach().abs() for p in model.parameters()] if coupling else None
        opt.step()
        return gmc_reward(model, opt, floors) if coupling else 0.0

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
        progress = cond in ('progc', 'progc_prop')
        before = model.reservoir_bits(ci, reservoirs[ci], hist[ci]) if progress else 0.0
        previous = torch.cat((hist[ci, oldlen-1:oldlen] if oldlen else torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        bits = float(losses.detach().mean())/math.log(2)
        states[ci] = state.detach()
        coupling = step(losses.mean(), coupling=cond.startswith('gmc'))
        # Attribute progress only to the live Adam update, before any replay.
        after = model.reservoir_bits(ci, reservoirs[ci], hist[ci]) if progress else 0.0
        reward = (before-after)/RES if progress else coupling
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
    result = dict(phase=phase, cond=cond, seed=seed, glances=glances,
        W=float(np.mean(list(held.values()))), held=held,
        share={c: float(counts[i]/glances) for i, c in enumerate(CH)},
        share_by_quarter=[{c: float(np.mean(q == i)) for i, c in enumerate(CH)} for q in np.array_split(choices, 4)],
        life_bits=float(np.mean(loss_log)), wall_seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()), adam_steps=3*glances,
        replay_segments=16*glances, history_symbols=int(lengths.sum()),
        reservoir_samples=[len(r) for r in reservoirs])
    np.savez_compressed(out/f'{phase}_{cond}_{seed}_{glances}.npz', channels=choices,
                        prequential_bits=loss_log, rewards=rewards,
                        reservoir_positions=np.array(reservoirs, dtype=object))
    print(f'{now()} done {phase} {cond} s{seed} T={glances} W={result["W"]:.6f} wall={result["wall_seconds"]:.1f}s', flush=True)
    return result


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    lines = ['# H25b measurements', '', f'Started: {data["started"]}. Updated: {now()}.', '',
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
    out = Path('results/h25b') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    clock = time.monotonic()
    source = Path('results/h25a/results.json')
    baseline = json.loads(source.read_text())
    reused = [dict(r, source='results/h25a/results.json') for r in baseline['runs']
              if r['phase'] == 'score' and r['seed'] in (6, 7, 8)]
    assert len(reused) == 15 and all(r['glances'] == 512 for r in reused)
    data = dict(started=now(), status='running', principles=[1, 2, 3, 4],
                torch_version=torch.__version__, notes=[], runs=[] if args.smoke else reused,
                baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                chosen_T=32 if args.smoke else 512,
                scoring_seeds=[] if args.smoke else [6, 7, 8], protocol=PROTOCOL,
                budget_wall_seconds=7200, dropped_conditions=[])
    def execute(cond, seed, t, phase):
        data['runs'].append(run(cond, seed, t, out, phase))
        data['total_wall_seconds'] = time.monotonic()-clock
        save(data, out)
    if args.smoke:
        for cond in NEW_CONDS:
            execute(cond, 0, 32, 'smoke')
    else:
        smoke = json.loads((out/'smoke/results.json').read_text())
        rates = {r['cond']: r['wall_seconds']/32 for r in smoke['runs']}
        projected = 512*3*sum(rates.values())
        data['projected_scoring_seconds'] = projected
        active = NEW_CONDS.copy()
        if projected + smoke['total_wall_seconds'] > 7200:
            active = ['progc', 'gmc']
            data['dropped_conditions'] = ['progc_prop', 'gmc_prop']
            data['notes'].append('Proportional variants dropped: smoke-based projected total exceeds 120 minutes.')
        save(data, out)
        # Finish epsilon-greedy variants first if the wall budget tightens.
        for cond in sorted(active, key=lambda c: c.endswith('_prop')):
            for seed in (6, 7, 8):
                elapsed = time.monotonic()-clock + smoke['total_wall_seconds']
                if cond.endswith('_prop') and elapsed + rates[cond]*512 > 7200:
                    data['notes'].append(f'Dropped remaining {cond} seed {seed}: 120-minute wall budget.')
                    data['dropped_conditions'].append(f'{cond}/seed{seed}')
                    continue
                execute(cond, seed, 512, 'score')
    data['status'] = 'complete'
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    if not args.smoke:
        data['smoke_wall_seconds'] = smoke['total_wall_seconds']
        data['wall_seconds_including_smoke'] = data['total_wall_seconds']+smoke['total_wall_seconds']
    save(data, out)


if __name__ == '__main__':
    main()
