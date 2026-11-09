"""H25a; principles 1–4: history, neural compression, progress reward, active choice.
Fixed protocol: handoffs/2026-11-10_codex_h25a_curiosity_neural.md.
Only this script and results/h25a are written. No verdicts or bar changes.
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
from curio import CH, LEARN, B, EVAL, EPS, BETA, glance
from h20_world import World, ALPHA, A
from h21a_substrate import now, mean_std

CONDS = ['rand', 'rand7', 'err', 'prog', 'prog_prop']
RES, CONTEXT = 128, 32


class NeuralCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(A, 32)
        self.channel = nn.Embedding(len(CH), 8)
        self.gru = nn.GRU(40, 256)
        self.readout = nn.Linear(256, A)

    def forward(self, previous, channels, state=None):
        # -1 is an absent preceding observation, not a new alphabet symbol.
        emb = self.embedding(previous.clamp_min(0))
        emb = emb.masked_fill((previous < 0).unsqueeze(-1), 0)
        ch = self.channel(channels).unsqueeze(0).expand(len(previous), -1, -1)
        output, state = self.gru(torch.cat((emb, ch), -1), state)
        return self.readout(output), state

    @torch.no_grad()
    def reservoir_bits(self, ci, positions, history):
        if not positions:
            return 0.0
        # Group shorter initial contexts so padding never changes recurrent state.
        positions = np.asarray(positions)
        total = 0.0
        for length in np.unique(np.minimum(positions, CONTEXT)):
            pos = torch.tensor(positions[np.minimum(positions, CONTEXT) == length])
            inputs = torch.full((int(length)+1, len(pos)), -1, dtype=torch.long)
            if length:
                inputs[1:] = history[pos[None, :] - int(length) + torch.arange(int(length))[:, None]]
            logits, _ = self(inputs, torch.full((len(pos),), ci, dtype=torch.long))
            total += float(F.cross_entropy(logits[-1], history[pos], reduction='sum')) / math.log(2)
        return total


def run(cond, seed, glances, out, phase):
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

    def step(loss):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for g in range(glances):
        if g < len(CH):
            ci = g
        elif cond == 'rand7':
            ci = int(rng.integers(len(LEARN)))
        elif cond == 'prog_prop':
            weights = np.maximum(values, 0)+0.01
            ci = int(rng.choice(len(CH), p=weights/weights.sum()))
        elif cond == 'rand' or rng.random() < EPS:
            ci = int(rng.integers(len(CH)))
        else:
            ci = max(range(len(CH)), key=lambda i: (values[i], rng.random()))
        channel = CH[ci]
        x = torch.tensor(glance(world, channel, rng), dtype=torch.long)
        oldlen = int(lengths[ci])
        progress = cond in ('prog', 'prog_prop')
        before = model.reservoir_bits(ci, reservoirs[ci], hist[ci]) if progress else 0.0
        previous = torch.cat((hist[ci, oldlen-1:oldlen] if oldlen else torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        bits = float(losses.detach().mean())/math.log(2)
        states[ci] = state.detach()
        step(losses.mean())
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
        after = model.reservoir_bits(ci, reservoirs[ci], hist[ci]) if progress else 0.0
        reward = (before-after)/RES if progress else (bits if cond == 'err' else 0.0)
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
    lines = ['# H25a measurements', '', f'Started: {data["started"]}. Updated: {now()}.', '',
        'CPU threads: 2. Standard deviation: population (ddof=0).', '',
        '| Pilot T | rand W | rand7 W | rand − rand7 |', '|---:|---:|---:|---:|']
    for t in (512, 1024):
        rs = [r for r in data['runs'] if r['phase'] == 'pilot' and r['glances'] == t]
        if len(rs) == 6:
            a = np.mean([r['W'] for r in rs if r['cond'] == 'rand'])
            b = np.mean([r['W'] for r in rs if r['cond'] == 'rand7'])
            lines.append(f'| {t} | {a:.6f} | {b:.6f} | {a-b:.6f} |')
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
        lines += ['', 'Paired W differences.', '', '| Seed | prog − rand | prog_prop − rand | prog − err | prog_prop − err |', '|---:|---:|---:|---:|---:|']
        for seed in data.get('scoring_seeds', []):
            w = {r['cond']: r['W'] for r in scoring if r['seed'] == seed}
            if len(w) == 5:
                lines.append(f'| {seed} | {w["prog"]-w["rand"]:.6f} | {w["prog_prop"]-w["rand"]:.6f} | {w["prog"]-w["err"]:.6f} | {w["prog_prop"]-w["err"]:.6f} |')
    lines += ['', 'Protocol: shared GRU 256; symbol embedding 32; channel embedding 8; Adam 0.003; clipping 1.0; one live step and two replay steps (8 × 64) per glance.',
              'Replay draws uniformly with replacement over all valid channel-local 64-symbol segments, including the current glance, from zero hidden state; preceding symbol supplied except at channel start.',
              'Live states are carried separately per channel and detached before learning; replay and reservoir scoring do not change them.',
              'Reservoir: Algorithm R over every observed channel position; up to 32 preceding symbols and target; zero hidden state and absent-symbol input before context. Current positions added after reward. Progress divided by 128, including partially filled reservoirs.',
              'W: frozen final model, fresh 2048-symbol continuation per learnable channel from a deep-copied world, starting from saved live channel state. History/store unpriced in the specified W metric.',
              'Controller epsilon 0.1; EMA beta 0.3. Proportional controller uses max(value,0)+0.01 without epsilon after the first pass.', '']
    lines += [f'Runtime note: {n}' for n in data['notes']]
    if 'projected_total_seconds_six_seeds' in data:
        lines += ['', f'Six-seed projected total wall seconds: {data["projected_total_seconds_six_seeds"]:.3f}.']
    lines += ['', f'Elapsed wall seconds: {data.get("total_wall_seconds", 0):.3f}.', f'Status: {data["status"]}.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path('results/h25a') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    clock = time.monotonic()
    data = dict(started=now(), status='running', principles=[1, 2, 3, 4],
                torch_version=torch.__version__, notes=[], runs=[])
    def execute(cond, seed, t, phase):
        data['runs'].append(run(cond, seed, t, out, phase))
        data['total_wall_seconds'] = time.monotonic()-clock
        save(data, out)
    if args.smoke:
        for cond in CONDS:
            execute(cond, 0, 32, 'smoke')
    else:
        for t in (512, 1024):
            for seed in (0, 1, 2):
                for cond in ('rand', 'rand7'):
                    execute(cond, seed, t, 'pilot')
            rs = [r for r in data['runs'] if r['glances'] == t]
            headroom = np.mean([r['W'] for r in rs if r['cond'] == 'rand'])-np.mean([r['W'] for r in rs if r['cond'] == 'rand7'])
            if headroom >= 0.1:
                data['chosen_T'] = t
                break
        if data.get('chosen_T') is None:
            data['notes'].append('Neither pilot T reached mean headroom 0.1; scoring not run, per protocol.')
        else:
            t = data['chosen_T']
            smoke = json.loads(Path('results/h25a/smoke/results.json').read_text())
            # Use full-size pilot time for baselines; smoke for progress overhead.
            pilot_rate = np.mean([r['wall_seconds']/r['glances'] for r in data['runs']])
            smoke_rate = {r['cond']: r['wall_seconds']/32 for r in smoke['runs']}
            rates = {c: max(pilot_rate, smoke_rate[c]) for c in CONDS}
            projection = time.monotonic()-clock + t*sum(rates.values())*6
            data['projected_total_seconds_six_seeds'] = projection
            seeds = list(range(6, 9 if projection > 7200 else 12))
            data['scoring_seeds'] = seeds
            if len(seeds) == 3:
                data['notes'].append('Scoring reduced to seeds 6–8: projection for six seeds exceeds 120-minute wall budget; original 5-of-6 bar unchanged.')
            save(data, out)
            for seed in seeds:
                for cond in CONDS:
                    execute(cond, seed, t, 'score')
    data['status'] = 'complete'
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()
