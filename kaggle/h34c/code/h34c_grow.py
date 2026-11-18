"""H34c: grow H34b's plain stack one layer at a time.

plain_4 is that stack, all four layers from byte 0. grow_* starts at one
layer. A layer added later has the same initial weights as plain_4's layer,
a zero readout, a zero recurrent state, and a new mixer weight of 1, so the
prediction is unchanged on the chunk where it appears. Freeze stops Adam on
a layer when the next one is added; the readout still runs into the mixer.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h34b_plain_depth as h34b
from h32b_mixer import MIX_LR, GeoMixer

CHUNK, LOG2, RANK = h34b.CHUNK, h34b.LOG2, h34b.RANK
N, DEPTH = 128, 4
N_BYTES, TAIL_START, TAIL_END, SLICE = 180_000, 40_000, 180_000, 20_000
WINDOW, SETTLE_REL = 4_096, 0.005
LIMIT_S, LAYER_LR = 90 * 60, 1e-3
ARMS = ('plain_4', 'grow_4_sched', 'grow_4_settle', 'grow_4_freeze')
SCHEDULE = {
    'plain_4': (),
    'grow_4_sched': (20_000, 40_000, 60_000),
    'grow_4_settle': (),
    'grow_4_freeze': (20_000, 40_000, 60_000),
}
FREEZE_ARMS = frozenset(('grow_4_freeze',))
SETTLE_ARMS = frozenset(('grow_4_settle',))
dump, h33 = h34b.dump, h34b.h33
assert CHUNK == 128 and WINDOW % CHUNK == 0
assert TAIL_START % SLICE == 0 and N_BYTES % SLICE == 0 and TAIL_END == N_BYTES
assert all(a % CHUNK != 0 for a in SCHEDULE['grow_4_sched'])


def build_stack(seed):
    torch.manual_seed(seed)
    return [h34b.Layer(N, 0 if k == 0 else 2 * N + RANK, True, True) for k in range(DEPTH)]


def iter_chunks(n, adds, chunk=CHUNK):
    """Chunks of at most `chunk` bytes. An add byte ends a chunk so the new layer starts there."""
    adds = tuple(adds)
    if list(adds) != sorted(set(adds)):
        raise ValueError(adds)
    pending = [a for a in adds if 0 < a < n]
    pos, i = 0, 0
    while pos < n:
        end = min(pos + chunk, n)
        if i < len(pending) and pos < pending[i] < end:
            end = pending[i]
        if end <= pos:
            raise RuntimeError('chunk did not advance')
        add_after = i < len(pending) and end == pending[i]
        yield pos, end, add_after
        if add_after:
            i += 1
        pos = end
    if i != len(pending):
        raise RuntimeError('missed an add boundary')


def expand_mixer(mixer):
    """Append a mixer weight of 1 and keep Adam moments on the weights that already existed."""
    old = mixer.W
    n = int(old.shape[1])
    fresh = torch.ones(1, n + 1, dtype=old.dtype, device=old.device)
    with torch.no_grad():
        fresh[:, :n].copy_(old.detach())
    state = mixer.opt.state.get(old)
    saved = None
    if state:
        saved = (state['step'].detach().clone(), state['exp_avg'].detach().clone(), state['exp_avg_sq'].detach().clone())
    mixer.W = nn.Parameter(fresh)
    mixer.n_experts = n + 1
    mixer.opt = torch.optim.Adam([mixer.W], lr=MIX_LR)
    if saved is not None:
        step, exp, exp_sq = saved
        exp_full, exp_sq_full = torch.zeros_like(mixer.W), torch.zeros_like(mixer.W)
        with torch.no_grad():
            exp_full[:, :n].copy_(exp)
            exp_sq_full[:, :n].copy_(exp_sq)
        mixer.opt.state[mixer.W] = {'step': step, 'exp_avg': exp_full, 'exp_avg_sq': exp_sq_full}
    return mixer


def _snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _params_equal(module, snap):
    return all(torch.equal(p, snap[name]) for name, p in module.named_parameters())


def _param_max_abs(module, snap):
    return max(float((p - snap[name]).abs().max()) for name, p in module.named_parameters())


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(values.mean())


def run(arm, seed, raw, keep=False, schedule=None):
    if arm not in ARMS:
        raise ValueError(arm)
    if schedule is None:
        schedule = SCHEDULE[arm]
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw, dtype=np.int64))
    n = len(tokens)
    layers = build_stack(seed)
    frozen = [False] * DEPTH
    snapshots = {}
    if arm == 'plain_4':
        if tuple(schedule):
            raise ValueError('plain_4 has no adds')
        active = DEPTH
        added_at = [0] * DEPTH
        states = [m.initial() for m in layers]
        opts = [torch.optim.Adam(m.parameters(), lr=LAYER_LR) for m in layers]
        mixer = GeoMixer(arm, 1, DEPTH, torch.ones(1, DEPTH))
    else:
        active = 1
        added_at = [0]
        states = [layers[0].initial(), None, None, None]
        opts = [torch.optim.Adam(layers[0].parameters(), lr=LAYER_LR), None, None, None]
        mixer = GeoMixer(arm, 1, 1, torch.ones(1, 1))
    bits = np.empty(n, np.float64)
    own = np.full((n, DEPTH), np.nan, np.float64)
    weight_tail = np.zeros(DEPTH)
    weight_tail_n = np.zeros(DEPTH)
    settle_checks = []
    identity_bytes = []
    identity_n = 0
    identity_max_prob = 0.0
    identity_max_bits = 0.0
    identity_max_logit = 0.0
    identity_max_weight_gap = 0.0
    pending_identity = False
    n_short = 0
    wall_at_8192 = None
    frozen_max = 0.0

    def add_layer(at):
        nonlocal active, pending_identity
        if active >= DEPTH:
            raise RuntimeError('depth cap')
        if arm in FREEZE_ARMS:
            for k in range(active):
                if frozen[k]:
                    continue
                snapshots[k] = _snapshot(layers[k])
                layers[k].zero_grad(set_to_none=True)
                frozen[k] = True
        states[active] = layers[active].initial()
        opts[active] = torch.optim.Adam(layers[active].parameters(), lr=LAYER_LR)
        expand_mixer(mixer)
        added_at.append(at)
        active += 1
        pending_identity = True

    def maybe_settle(end):
        if arm not in SETTLE_ARMS or active >= DEPTH or end % WINDOW != 0 or end < 2 * WINDOW:
            return
        k = active - 1
        ready = added_at[k] <= end - 2 * WINDOW
        row = dict(byte=int(end), top_layer=k + 1, ready=bool(ready),
                   prev_bits=None, last_bits=None, relative_improvement=None, added=False)
        if ready:
            prev_w = own[end - 2 * WINDOW:end - WINDOW, k]
            last_w = own[end - WINDOW:end, k]
            if not (np.isfinite(prev_w).all() and np.isfinite(last_w).all()):
                raise FloatingPointError('nonfinite settle window')
            prev, last = float(prev_w.mean()), float(last_w.mean())
            if not (math.isfinite(prev) and math.isfinite(last) and prev > 0):
                raise FloatingPointError(f'settle window prev={prev} last={last}')
            rel = (prev - last) / prev
            row.update(prev_bits=prev, last_bits=last, relative_improvement=rel, added=bool(rel < SETTLE_REL))
        settle_checks.append(row)
        if row['added']:
            add_layer(end)

    for start, end, add_after in iter_chunks(n, schedule):
        if end - start < CHUNK:
            n_short += 1
        target = tokens[start:end]
        outputs, feats, caches = [], [], []
        for k in range(active):
            m = layers[k]
            u = m.inputs(tokens, start, end, feats[-1] if k else None)
            z, f, states[k], cache = m.score(u, tokens, start, states[k])
            outputs.append(z)
            feats.append(f)
            caches.append(cache)
        stacked = torch.stack([z.detach() for z in outputs], 1)
        combined = (stacked * mixer.W[0, :, None]).sum(1)
        losses = F.cross_entropy(combined, target, reduction='none')
        bits[start:end] = losses.detach().double().numpy() / LOG2
        length = end - start
        for k, z in enumerate(outputs):
            own[start:end, k] = F.cross_entropy(z.detach(), target, reduction='none').double().numpy() / LOG2
        if end > TAIL_START and start < TAIL_END:
            take = min(end, TAIL_END) - max(start, TAIL_START)
            weights = mixer.W.detach().numpy()[0]
            for k in range(active):
                weight_tail[k] += float(weights[k]) * take
                weight_tail_n[k] += take
        if pending_identity:
            prev = (stacked[:, :-1] * mixer.W[0, :-1, None]).sum(1)
            prob = float((combined.detach().softmax(-1) - prev.detach().softmax(-1)).abs().max())
            bit_gap = float((losses.detach() - F.cross_entropy(prev, target, reduction='none').detach()).abs().max()) / LOG2
            logit = float(stacked[:, -1].abs().max())
            weight_gap = abs(float(mixer.W.detach()[0, -1]) - 1.0)
            identity_n += 1
            identity_max_prob = max(identity_max_prob, prob)
            identity_max_bits = max(identity_max_bits, bit_gap)
            identity_max_logit = max(identity_max_logit, logit)
            identity_max_weight_gap = max(identity_max_weight_gap, weight_gap)
            identity_bytes.append(start)
            pending_identity = False
        for k in range(active):
            if frozen[k]:
                continue
            g = outputs[k].detach().softmax(-1)
            g[torch.arange(length), target] -= 1
            layers[k].learn(outputs[k], g / length, caches[k])
        mixer.opt.zero_grad(set_to_none=True)
        losses.mean().backward()
        for k in range(active):
            if frozen[k]:
                continue
            nn.utils.clip_grad_norm_(layers[k].parameters(), 1.)
            opts[k].step()
        mixer.opt.step()
        if snapshots:
            for k, snap in snapshots.items():
                if not _params_equal(layers[k], snap):
                    frozen_max = max(frozen_max, _param_max_abs(layers[k], snap))
                    raise RuntimeError(f'frozen layer {k} changed by {frozen_max}')
        if not (np.isfinite(bits[start:end]).all() and np.isfinite(own[start:end, :active]).all() and torch.isfinite(stacked).all()):
            raise FloatingPointError('nonfinite score')
        if end == 8192:
            wall_at_8192 = time.perf_counter() - t0
        if end // 10240 != start // 10240 or end == n:
            print(arm, seed, end, float(bits[:end].mean()), active, time.perf_counter() - t0, flush=True)
        if time.perf_counter() - t0 > LIMIT_S:
            raise TimeoutError('90 minute limit')
        if add_after:
            add_layer(end)
        else:
            maybe_settle(end)

    if pending_identity:
        raise RuntimeError('a layer was added with no bytes left to score')
    if not np.isfinite(bits).all():
        raise FloatingPointError('nonfinite bits')
    if len(added_at) != active or tuple(mixer.W.shape) != (1, active):
        raise RuntimeError(f'mixer width {tuple(mixer.W.shape)} != active {active}')
    slices = [dict(start=i, end=min(i + SLICE, n), bits_per_byte=float(bits[i:i + SLICE].mean()))
              for i in range(0, n, SLICE)]
    diagnostics = []
    final_w = mixer.W.detach().numpy()[0]
    for k in range(active):
        life = own[added_at[k]:, k]
        tail_a = max(added_at[k], TAIL_START)
        tail_b = min(n, TAIL_END)
        diagnostics.append(dict(
            layer=k + 1,
            added_at=int(added_at[k]),
            own_bits_per_byte=float(life.mean()),
            tail_own_bits_per_byte=_mean_or_none(own[tail_a:tail_b, k]) if tail_b > tail_a else None,
            mean_mixer_weight_tail=(float(weight_tail[k] / weight_tail_n[k]) if weight_tail_n[k] else None),
            final_mixer_weight=float(final_w[k]),
            fraction_lambda_lt_half=float((layers[k].lam().abs() < .5).float().mean()),
        ))
    if n >= TAIL_END:
        tail = float(bits[TAIL_START:TAIL_END].mean())
    elif n > TAIL_START:
        tail = float(bits[TAIL_START:].mean())
    else:
        tail = None
    identical = None if not snapshots else frozen_max == 0.0 and all(_params_equal(layers[k], snap) for k, snap in snapshots.items())
    result = dict(
        arm=arm, seed=int(seed), n_bytes=n, bits_per_byte=float(bits.mean()), tail_bits_per_byte=tail,
        tail_start=TAIL_START, tail_end=min(TAIL_END, n) if n > TAIL_START else None,
        slices=slices, layers=diagnostics, added_at=[int(x) for x in added_at],
        settle_checks=settle_checks, identity_bytes=identity_bytes,
        identity_max_probability_abs=(identity_max_prob if identity_n else None),
        identity_max_bits_abs=(identity_max_bits if identity_n else None),
        identity_max_added_logit_abs=(identity_max_logit if identity_n else None),
        identity_max_new_weight_abs_from_1=(identity_max_weight_gap if identity_n else None),
        frozen_parameters_identical=identical,
        frozen_parameter_max_abs=(frozen_max if snapshots else None),
        n_short_chunks=n_short, wall_seconds=time.perf_counter() - t0, wall_at_8192=wall_at_8192,
        live_readout_max_abs=float(max(layers[active - 1].W.weight.detach().abs().max(),
                                       layers[active - 1].V.weight.detach().abs().max())),
    )
    return (result, bits) if keep else result


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ('h34c_grow.py', 'h34b_plain_depth.py', 'h34a2_logit_sum.py', 'h34a_local_depth.py',
             'h26_cued.py', 'h32_mixer.py', 'h32b_mixer.py', 'h33_bank_text.py', 'h20_world.py',
             'h30w_ring.py', 'h30x_trace.py', 'h30y_roots.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _check_mechanisms():
    adds = []
    covered = np.zeros(N_BYTES, np.int8)
    shorts = []
    for start, end, add_after in iter_chunks(N_BYTES, SCHEDULE['grow_4_sched']):
        if end - start > CHUNK or end <= start:
            raise AssertionError((start, end))
        covered[start:end] += 1
        if end - start < CHUNK:
            shorts.append((start, end))
        if add_after:
            adds.append(end)
    if adds != [20_000, 40_000, 60_000] or not covered.all() or covered.max() != 1:
        raise AssertionError((adds, int(covered.min()), int(covered.max()), shorts))
    # Adds at 20/40/60 KB sit 32 bytes into a chunk. After 60 KB the grid is
    # offset by 96, so the tail chunk is 64 bytes (179,936–180,000).
    if shorts != [(19_968, 20_000), (39_968, 40_000), (59_968, 60_000), (179_936, 180_000)]:
        raise AssertionError(shorts)
    plain = [(s, e) for s, e, _ in iter_chunks(8192, ())]
    if plain != [(i, i + 128) for i in range(0, 8192, 128)]:
        raise AssertionError(plain[:3])
    # 0.5% is relative. Equal to 0.5% does not add; worse than the previous window does.
    cases = [(4.0, 3.98, False), (4.0, 3.9800000001, True), (4.0, 4.1, True), (4.0, 3.9, False)]
    for prev, last, expect in cases:
        got = (prev - last) / prev < SETTLE_REL
        if got != expect:
            raise AssertionError((prev, last, got, expect))
    torch.manual_seed(0)
    mixer = GeoMixer('grow', 1, 1, torch.ones(1, 1))
    for i in range(4):
        mixer.opt.zero_grad(set_to_none=True)
        ((mixer.W - 0.2 * (i + 1)) ** 2).sum().backward()
        mixer.opt.step()
    before = mixer.W.detach().clone()
    step = mixer.opt.state[mixer.W]['step'].detach().clone()
    exp = mixer.opt.state[mixer.W]['exp_avg'].detach().clone()
    exp_sq = mixer.opt.state[mixer.W]['exp_avg_sq'].detach().clone()
    twin = GeoMixer('twin', 1, 1, before.clone())
    twin.opt.state[twin.W] = {'step': step.clone(), 'exp_avg': exp.clone(), 'exp_avg_sq': exp_sq.clone()}
    expand_mixer(mixer)
    if mixer.W.shape != (1, 2) or not torch.equal(mixer.W.detach()[:, :1], before) or float(mixer.W.detach()[0, 1]) != 1.0:
        raise AssertionError('expand did not keep the old weight and a new weight of 1')
    if not torch.equal(mixer.opt.state[mixer.W]['step'], step):
        raise AssertionError('Adam step was reset')
    if float(mixer.opt.state[mixer.W]['exp_avg'][0, 1]) != 0.0 or float(mixer.opt.state[mixer.W]['exp_avg_sq'][0, 1]) != 0.0:
        raise AssertionError('new mixer weight inherited moments')
    mixer.opt.zero_grad(set_to_none=True)
    twin.opt.zero_grad(set_to_none=True)
    (mixer.W[0, 0] * 3).backward()
    (twin.W[0, 0] * 3).backward()
    mixer.opt.step()
    twin.opt.step()
    if not torch.equal(mixer.W.detach()[0, 0], twin.W.detach()[0, 0]):
        raise AssertionError((float(mixer.W.detach()[0, 0]), float(twin.W.detach()[0, 0])))
    if float(mixer.W.detach()[0, 1]) != 1.0:
        raise AssertionError('zero-grad new weight moved')
    for seed in (0, 1):
        ours = build_stack(seed)
        ref = h34b.build('plain_4', seed)
        for m, r in zip(ours, ref):
            a, b = m.state_dict(), r.state_dict()
            if list(a) != list(b) or any(not torch.equal(a[k], b[k]) for k in a):
                raise AssertionError(f'init mismatch seed {seed}')


def _prefix_ok(arm, seed, raw, prefix):
    _, short = run(arm, seed, raw[:prefix], keep=True)
    _, long = run(arm, seed, raw[:prefix * 2], keep=True)
    changed = raw[:prefix * 2].copy()
    changed[prefix:] = 255 - changed[prefix:]
    _, altered = run(arm, seed, changed, keep=True)
    same = float(np.abs(short - long[:prefix]).max())
    future = float(np.abs(short - altered[:prefix]).max())
    return same, future


def smoke(out, data):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, n_bytes=N_BYTES, tail_start=TAIL_START, tail_end=TAIL_END,
                   slice_size=SLICE, threads=1, log=[], source_hashes=source_hashes())

    def record(check, **values):
        row = dict(check=check, **values)
        payload['log'].append(row)
        print(json_dumps(row), flush=True)
        dump(out / 'smoke.json', payload)

    try:
        h33.configure(1)
        _check_mechanisms()
        record('mechanisms', chunks='sched boundaries and plain grid', settle='relative < 0.005',
               mixer='expanded Adam matches the unexpanded weight', init='plain_4 state_dict')
        raw, digest = h33.load_checked(data)
        payload['sha256_e10m'] = digest
        for seed in (0, 1):
            ref, rb = h34b.run('plain_4', seed, raw[:256], keep=True)
            ours, ob = run('plain_4', seed, raw[:256], keep=True)
            diff = float(np.abs(rb - ob).max())
            record('fast_gate', seed=seed, n_bytes=256, max_bits_abs=diff, limit=1e-9)
            if diff > 1e-9:
                raise AssertionError(f'fast gate seed {seed} diff {diff}')
        fast = run('grow_4_sched', 0, raw[:384], schedule=(128,))
        record('fast_identity', n_bytes=384, add_at=128, identity_bytes=fast['identity_bytes'],
               max_probability_abs=fast['identity_max_probability_abs'],
               max_bits_abs=fast['identity_max_bits_abs'],
               max_added_logit_abs=fast['identity_max_added_logit_abs'],
               max_weight_gap=fast['identity_max_new_weight_abs_from_1'], limit=1e-9)
        if fast['identity_bytes'] != [128] or fast['identity_max_probability_abs'] > 1e-9 or fast['identity_max_bits_abs'] > 1e-9:
            raise AssertionError('fast identity failed')
        if fast['identity_max_added_logit_abs'] > 1e-9 or fast['identity_max_new_weight_abs_from_1'] != 0.0:
            raise AssertionError('fast identity weight/logit failed')
        frozen = run('grow_4_freeze', 0, raw[:384], schedule=(128,))
        record('fast_freeze', n_bytes=384, identical=frozen['frozen_parameters_identical'],
               max_abs=frozen['frozen_parameter_max_abs'], live_readout_max_abs=frozen['live_readout_max_abs'],
               identity_max_probability_abs=frozen['identity_max_probability_abs'])
        if frozen['frozen_parameters_identical'] is not True or frozen['frozen_parameter_max_abs'] != 0.0:
            raise AssertionError('fast freeze failed')
        if frozen['live_readout_max_abs'] <= 0:
            raise AssertionError('new layer readout did not move')
        for seed in (0, 1):
            ref, rb = h34b.run('plain_4', seed, raw[:8192], keep=True)
            ours, ob = run('plain_4', seed, raw[:8192], keep=True)
            diff = float(np.abs(rb - ob).max())
            record('h34b_equivalence', seed=seed, n_bytes=8192, max_bits_abs=diff, limit=1e-9,
                   reference_bits=ref['bits_per_byte'], bits=ours['bits_per_byte'],
                   wall_seconds=ours['wall_seconds'], wall_at_8192=ours['wall_at_8192'])
            if diff > 1e-9:
                raise AssertionError(f'equivalence seed {seed} diff {diff}')
            payload[f'plain4_seconds_per_byte_seed{seed}'] = ours['wall_seconds'] / 8192
        sched = run('grow_4_sched', 0, raw[:20128])
        record('add_identity', arm='grow_4_sched', n_bytes=20128, identity_bytes=sched['identity_bytes'],
               n_short_chunks=sched['n_short_chunks'], max_probability_abs=sched['identity_max_probability_abs'],
               max_bits_abs=sched['identity_max_bits_abs'], max_added_logit_abs=sched['identity_max_added_logit_abs'],
               max_weight_gap=sched['identity_max_new_weight_abs_from_1'], added_at=sched['added_at'], limit=1e-9)
        if sched['identity_bytes'] != [20_000] or sched['added_at'] != [0, 20_000] or sched['n_short_chunks'] != 1:
            raise AssertionError('sched add boundary failed')
        if sched['identity_max_probability_abs'] > 1e-9 or sched['identity_max_bits_abs'] > 1e-9:
            raise AssertionError('sched identity failed')
        if sched['identity_max_added_logit_abs'] > 1e-9 or sched['identity_max_new_weight_abs_from_1'] != 0.0:
            raise AssertionError('sched logit/weight failed')
        held = run('grow_4_freeze', 0, raw[:20128])
        record('freeze_parameters', arm='grow_4_freeze', n_bytes=20128, added_at=held['added_at'],
               identical=held['frozen_parameters_identical'], max_abs=held['frozen_parameter_max_abs'],
               identity_max_probability_abs=held['identity_max_probability_abs'],
               live_readout_max_abs=held['live_readout_max_abs'])
        if held['added_at'] != [0, 20_000] or held['frozen_parameters_identical'] is not True or held['frozen_parameter_max_abs'] != 0.0:
            raise AssertionError('freeze parameters changed')
        if held['identity_max_probability_abs'] > 1e-9 or held['live_readout_max_abs'] <= 0:
            raise AssertionError('freeze identity or live readout failed')
        settled = run('grow_4_settle', 0, raw[:8320])
        checks = settled['settle_checks']
        record('settle_wiring', n_bytes=8320, added_at=settled['added_at'], checks=checks,
               identity_bytes=settled['identity_bytes'])
        if [c['byte'] for c in checks] != [8192] or checks[0]['top_layer'] != 1 or checks[0]['ready'] is not True:
            raise AssertionError('settle did not check byte 8192')
        rel = checks[0]['relative_improvement']
        if (rel < SETTLE_REL) != checks[0]['added']:
            raise AssertionError('settle threshold disagreed with the add')
        if checks[0]['added']:
            if settled['added_at'] != [0, 8192] or settled['identity_bytes'] != [8192]:
                raise AssertionError('settle add was not at the check byte')
            if settled['identity_max_probability_abs'] > 1e-9:
                raise AssertionError('settle identity failed')
        elif settled['added_at'] != [0] or settled['identity_bytes']:
            raise AssertionError('settle added without the threshold')
        for arm in ('plain_4', 'grow_4_sched'):
            same, future = _prefix_ok(arm, 0, raw, 1024)
            record('prequential_prefix', arm=arm, max_bits_abs=same, changed_suffix_max_bits_abs=future,
                   prefix_bytes=1024, extended_bytes=2048)
            if same != 0.0 or future != 0.0:
                raise AssertionError(f'prequential {arm} same={same} future={future}')
        rate4 = payload['plain4_seconds_per_byte_seed0']
        rate1 = sched['wall_at_8192'] / 8192
        per = (rate4 - rate1) / 3
        rates = {d: rate1 + (d - 1) * per for d in range(1, 5)}
        phased = 20_000 * rates[1] + 20_000 * rates[2] + 20_000 * rates[3] + 120_000 * rates[4]
        estimates = {
            'plain_4': rate4 * N_BYTES,
            'grow_4_sched': phased,
            'grow_4_freeze': phased,
            'grow_4_settle': rate4 * N_BYTES,
        }
        record('wall_estimates', n_bytes=N_BYTES, tail_start=TAIL_START, tail_end=TAIL_END,
               seconds_per_byte=dict(one_layer=rate1, four_layers=rate4),
               estimated_kernel_seconds=estimates, limit=LIMIT_S,
               method=('One thread. plain_4 rate from its 8,192-byte run. One-layer rate from grow_4_sched bytes 0–8,192. '
                       'Depth rates are linear between those two (H34b MAC ratios are linear in depth). '
                       'sched and freeze use 20 KB at 1 layer, 20 KB at 2, 20 KB at 3, 120 KB at 4. '
                       'freeze skips updates, so this is an upper bound. settle is bounded by plain_4 '
                       '(4 layers from byte 0). No Kaggle speedup is assumed.'))
        if max(estimates.values()) > LIMIT_S:
            raise AssertionError(f'wall estimate {max(estimates.values())} exceeds {LIMIT_S}')
        payload['ok'] = True
        dump(out / 'smoke.json', payload)
    except Exception as exc:
        payload['failure'] = repr(exc)
        dump(out / 'smoke.json', payload)
        raise


def json_dumps(row):
    return __import__('json').dumps(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--seed', type=int, choices=(0, 1), default=0)
    p.add_argument('--data', default='data/enwik8/enwik8')
    p.add_argument('--out', default='results/h34c')
    args = p.parse_args()
    h33.configure(1)
    if args.smoke:
        return smoke(args.out, args.data)
    if args.arm is None:
        p.error('--arm is required')
    sm = __import__('json').loads((Path(args.out) / 'smoke.json').read_text())
    if not sm['ok']:
        raise RuntimeError('smoke has not passed')
    if sm['source_hashes'] != source_hashes():
        raise RuntimeError('smoke hashes do not match this source')
    raw, digest = h33.load_checked(args.data)
    result = run(args.arm, args.seed, raw[:sm['n_bytes']])
    result.update(sha256_e10m=digest, kernel_id=f'ezzzzzx/h34c-{args.arm.replace("_", "-")}-s{args.seed}')
    dump(Path(args.out) / f'{args.arm}-s{args.seed}.json', result)


if __name__ == '__main__':
    main()
