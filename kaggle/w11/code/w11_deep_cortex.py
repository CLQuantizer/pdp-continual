"""W11: Own, Wait, See in W10's life (compression principles 2 and 4).

World, stores, actions, pressure-retina constants and W6 scoring are unchanged.
Upper banks concatenate the channel+symbol embedding and the mean of detached,
per-step RMS-normalized [Re(h), Im(h), P1(h)*P2(h)] lower features.
"""
import os
import sys
sys.dont_write_bytecode = True
if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CURIO_LR'] = '0.004'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if os.environ.get('_W11_MAIN') != '1':
        os.environ['_W11_MAIN'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import pickle
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import w10_open_sleep as w10
from h30w_ring import RANK
from h32b_mixer import GeoMixer
from h34a2_logit_sum import normalize
m10, w6_world, w9, retina = w10.m10, w10.w6_world, w10.w9, w10.retina
m7, m8, w4, w6 = w10.m7, w10.m8, w10.w4, w10.w6
ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(132, 144))
CORTICES = ('bank_1', 'deep_4', 'deep_4_open')
L_SCORE, PERIOD = w10.L_SCORE, w10.PERIOD
DATASET = 'ezzzzzx/pdp-code-w11'
CODE_FILES = tuple(dict.fromkeys(('w11_deep_cortex.py', 'h32b_mixer.py',
    'h32_mixer.py', 'm7b_w3.py', 'h34a2_logit_sum.py', 'h34a_local_depth.py', 'h33_bank_text.py', 'h26_cued.py') + w10.CODE_FILES))
dump, night = w10.dump, w10.night
DayNightWorld, RetinaPressure = w10.DayNightWorld, w10.RetinaPressure
eyelid_checksum, eyelid_glance = w10.eyelid_checksum, w10.eyelid_glance
random_sleep_mask = w10.random_sleep_mask
BANK_CLASS = w4.WideCortex
SCORE_SEQUENCES = m7.score_sequences
FEATURE_DIM = 2 * m7.UNITS + RANK


def features(model, h):
    with torch.no_grad():
        z = torch.cat((h.real, h.imag), -1)
        return torch.cat((z, model.P1(z) * model.P2(z)), -1).detach()


def upper_trace(bank, symbols, prev, channel, state, lower):
    """M7 trace_glance with extra, detached input columns. No lower adjoint.

    Embedding and channel traces use only B[:, :EMB]; B's eligibility uses
    all input columns. Loss summation, gradient averaging and clip match M7.
    """
    model, channel_emb = bank.model, bank.channel_emb
    hidden, e_lam, e_b, z, e_ch = state
    length = len(symbols)
    ids = torch.cat((torch.tensor([max(0, int(prev))]), symbols[:-1]))
    emb = model.embedding(ids).detach()
    if prev < 0:
        emb[0] = 0
    emb = emb + channel_emb[channel].detach()
    inputs = torch.cat((emb, lower.detach()), -1)
    with torch.no_grad():
        lam = model.lam().detach()
        injection_b = torch.complex(model.B_re.detach()[:, :m7.EMB], model.B_im.detach()[:, :m7.EMB])
        injected = model.inject(inputs)
        dnu, dth = -torch.exp(model.nu.detach()) * lam, 1j * lam
        hs, els, ebs, zs, ecs = [], [], [], [], []
        for t in range(length):
            e_lam.mul_(lam).add_(hidden)
            e_b.mul_(lam[:, None]).add_(inputs[t])
            z.mul_(lam[:, None])
            if t or prev >= 0:
                z[:, int(symbols[t-1]) if t else int(prev)] += 1
            e_ch.mul_(lam)
            e_ch[channel] += 1
            hidden.mul_(lam).add_(injected[t])
            hs.append(hidden.clone()); els.append(e_lam.clone()); ebs.append(e_b.clone())
            zs.append(z.clone()); ecs.append(e_ch.clone())
        hb, el, eb, ez, ec = map(torch.stack, (hs, els, ebs, zs, ecs))
    hr, hi = hb.real.detach().requires_grad_(), hb.imag.detach().requires_grad_()
    logits = model.logits_from(torch.complex(hr, hi))
    feat = features(model, hb)
    losses = F.cross_entropy(logits, symbols, reduction='none')
    model.zero_grad(set_to_none=True)
    channel_emb.grad = None
    losses.sum().backward()
    with torch.no_grad():
        c = torch.complex(hr.grad, hi.grad).conj()
        model.nu.grad = (c * (el * dnu)).real.sum(0)
        model.theta.grad = (c * (el * dth)).real.sum(0)
        model.B_re.grad = (c[:, :, None] * eb).real.sum(0)
        model.B_im.grad = (c[:, :, None] * (1j * eb)).real.sum(0)
        model.embedding.weight.grad = torch.einsum('tia,ie->ae', c[:, :, None] * ez, injection_b).real
        channel_emb.grad = torch.einsum('tci,ie->ce', c[:, None, :] * ec, injection_b).real
        for p in bank.params:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('missing/nonfinite own gradient')
            p.grad.mul_(1. / length)
    return logits.detach(), feat, (hidden, e_lam, e_b, z, e_ch)


class DeepCortex:
    def __init__(self, n_ch=w4.N_CH, model_seed=m7.MODEL_SEED, depth=4, opened=False):
        self.layers = [BANK_CLASS(n_ch, model_seed=model_seed)]
        for k in range(1, depth):
            bank = BANK_CLASS(n_ch, model_seed=model_seed + k)
            # Same bank width, roots, readout, embedding, optimizer; more input columns.
            dim = m7.EMB + FEATURE_DIM
            generator = torch.Generator().manual_seed(70_000 + model_seed + k)
            bank.model.B_re = nn.Parameter(torch.randn(m7.UNITS, dim, generator=generator) / math.sqrt(dim))
            bank.model.B_im = nn.Parameter(torch.randn(m7.UNITS, dim, generator=generator) / math.sqrt(dim))
            bank.params = list(bank.model.parameters()) + [bank.channel_emb]
            bank.opt = torch.optim.Adam(bank.params, lr=m7.LR)
            self.layers.append(bank)
        self.depth = depth
        self.mixer = GeoMixer('w11', 1, depth, torch.ones(1, depth) if opened else
                              torch.tensor([[1.] + [0.] * (depth - 1)]))
        self.state = self.zero_state()
        self.prev_symbol = -1
        # M7 scoring entry points carry a model and channel table. The router
        # recognizes this facade and runs every bank, including the mixer.
        self.model = self
        self.channel_emb = self.layers[0].channel_emb
        self.last_logits = None
        self.last_own_logits = None

    def zero_state(self):
        states = [bank.zero_state() for bank in self.layers]
        for state, bank in zip(states[1:], self.layers[1:]):
            state[2].resize_(m7.UNITS, bank.model.B_re.shape[1]).zero_()
        return states

    def learn_glance(self, symbols, channel, prev, state=None, commit=True):
        tokens = torch.as_tensor(symbols, dtype=torch.long)
        use = self.state if state is None else state
        outputs, feats, new_states = [], [], []
        for k, bank in enumerate(self.layers):
            if k == 0:
                capture = []
                original = bank.model.logits_from
                def observe(h):
                    logits = original(h)
                    capture.append((logits.detach().clone(), features(bank.model, h)))
                    return logits
                # Original M7 gradient computation and operation order, verbatim.
                with patch.object(bank.model, 'logits_from', observe), patch.object(m7, 'N_CH', bank.n_ch):
                    _, new = m7.trace_glance(bank.model, bank.channel_emb, tokens, prev, channel, use[k], mean=True)
                logits, feat = capture[0]
            else:
                lower = torch.stack([normalize(f) for f in feats]).mean(0) if k > 1 else normalize(feats[0])
                logits, feat, new = upper_trace(bank, tokens, prev, channel, use[k], lower)
            outputs.append(logits); feats.append(feat); new_states.append(new)
        stacked = torch.stack(outputs, 1).detach()
        combined = (stacked * self.mixer.W[0, :, None]).sum(1)
        losses = F.cross_entropy(combined, tokens, reduction='none')
        self.last_logits = combined.detach().clone()
        self.last_own_logits = stacked.detach().clone()
        if self.depth > 1:
            self.mixer.opt.zero_grad(set_to_none=True)
            losses.mean().backward()
        for bank in self.layers:
            nn.utils.clip_grad_norm_(bank.params, m7.CLIP)
            bank.opt.step()
        if self.depth > 1:
            self.mixer.opt.step()
        if commit:
            self.state, self.prev_symbol = new_states, int(tokens[-1])
        bits = losses.detach().double().numpy() / m7.LOG2
        if not np.isfinite(bits).all() or not torch.isfinite(self.mixer.W).all():
            raise FloatingPointError('nonfinite mixture')
        return bits

    def clone_live(self):
        return [m7.clone_state(s) for s in self.state], self.prev_symbol

    def live_matches(self, snap):
        return self.prev_symbol == snap[1] and all(m7.states_equal(a, b) for a, b in zip(self.state, snap[0]))

    def clone_params(self):
        return [p.detach().clone() for bank in self.layers for p in bank.params] + [self.mixer.W.detach().clone()]

    def param_delta(self, snap):
        return max(float((a-b).abs().max()) for a, b in zip(self.clone_params(), snap))

    def score_sequences(self, symbols, prev, channels, return_steps=False):
        batch, length = symbols.shape
        if not symbols.numel():
            return torch.zeros(0, dtype=torch.float64)
        ids = torch.cat((prev.clamp(min=0)[:, None], symbols[:, :-1]), 1)
        with torch.no_grad():
            embeds = []
            for bank in self.layers:
                emb = bank.model.embedding(ids)
                emb[prev < 0, 0] = 0
                embeds.append(emb + bank.channel_emb[channels][:, None, :])
            hs = [torch.zeros(batch, m7.UNITS, dtype=torch.complex64) for _ in self.layers]
            lams = [bank.model.lam() for bank in self.layers]
            # Precompute symbol injections as M7 does; upper feature injection is causal.
            injections = [torch.complex(e @ b.model.B_re[:, :m7.EMB].T,
                                       e @ b.model.B_im[:, :m7.EMB].T) for e, b in zip(embeds, self.layers)]
            total = torch.zeros(batch, dtype=torch.float64)
            step_losses = []
            for t in range(length):
                feats, outputs = [], []
                for k, bank in enumerate(self.layers):
                    inj = injections[k][:, t]
                    if k:
                        lower = normalize(feats[0]) if k == 1 else torch.stack([normalize(f) for f in feats]).mean(0)
                        # One concatenated matmul, matching learning's input definition.
                        x = torch.cat((embeds[k][:, t], lower), -1)
                        inj = bank.model.inject(x)
                    hs[k] = lams[k] * hs[k] + inj
                    outputs.append(bank.model.logits_from(hs[k]))
                    feats.append(features(bank.model, hs[k]))
                mixed = (torch.stack(outputs, 1) * self.mixer.W[0, :, None]).sum(1)
                step = F.cross_entropy(mixed, symbols[:, t], reduction='none').double()
                total += step
                if return_steps:
                    step_losses.append(step)
        return torch.stack(step_losses, 1) / m7.LOG2 if return_steps else total / m7.LOG2


def score_sequences(model, channel_emb, symbols, prev, channels):
    if isinstance(model, DeepCortex):
        return model.score_sequences(symbols, prev, channels)
    return SCORE_SEQUENCES(model, channel_emb, symbols, prev, channels)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}

class TrackedLife(w6.OpenLife):
    def __init__(self, *args, cortex='bank_1', depth=None, **kwargs):
        with patch.object(w4, 'WideCortex', BANK_CLASS if cortex == 'bank_1' else
                          lambda n_ch, model_seed: DeepCortex(n_ch, model_seed,
                              depth=depth or 4, opened=cortex == 'deep_4_open')):
            super().__init__(*args, **kwargs)
        self.glance_log = []

    def _learn(self, symbols, channel, prev, state=None, commit=True):
        bits = super()._learn(symbols, channel, prev, state, commit)
        self.glance_log.append(dict(channel=int(channel), prev=int(prev), wake=bool(commit),
            preq=bool(self.count_preq), bits=float(bits.sum()), n=len(bits),
            sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        return bits


def weights(cortex):
    return cortex.mixer.W.detach()[0].tolist() if isinstance(cortex, DeepCortex) else [1.]


def _run_life(seed, cortex, L=L_SCORE, night_enabled=True, trace=False, check_eyelid=False,
              depth=None, checkpoint=None, resume=None, stop_at=None):
    arm = 'pressure_retina' 
    if cortex not in CORTICES:
        raise ValueError(arm)
    m10._threads()
    start = time.perf_counter()
    world = DayNightWorld(seed, night_enabled)
    names = list(m10.IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    life = TrackedLife(seed, world, names, index, m10.N_RING, w4.N_CH, list(m10.w4_world.LEARN14), cortex=cortex, depth=depth)
    rng = np.random.default_rng(1_000 + seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    values, counts, last_end = [0.] * 18, [0] * 18, [None] * 18
    v_max = None
    unvisited = set(range(18))
    plan, target, total, n_dec = [], None, 0., 0
    visit_pending = None
    channels, records, traces, sleep_visits = [], [], [], []
    points = set(w6.checkpoints_of(L))
    clock = {'eval': 0.}
    prev_cp = 0
    controller = RetinaPressure()
    eyelid_checks, input_bits = [], []
    random_mask, random_starts = random_sleep_mask(seed, L)
    input_rooms, choices = [], []
    sleep_left, t = 0, 0
    snap, checked = None, False
    sleep_learned = 0

    mixer_history, mixer_sum, mixer_n = [], np.zeros(len(weights(life.cortex))), 0
    previous_wall = 0.
    if resume is not None:
        with Path(resume).open('rb') as f:
            saved = pickle.load(f)
        assert saved['source_hashes'] == source_hashes()
        assert saved['config'] == (seed, cortex, L, night_enabled, depth)
        world = saved['state']['world']
        names = saved['state']['names']
        index = saved['state']['index']
        life = saved['state']['life']
        rng = saved['state']['rng']
        choice_rng = saved['state']['choice_rng']
        values = saved['state']['values']
        counts = saved['state']['counts']
        last_end = saved['state']['last_end']
        v_max = saved['state']['v_max']
        unvisited = saved['state']['unvisited']
        plan = saved['state']['plan']
        target = saved['state']['target']
        total = saved['state']['total']
        n_dec = saved['state']['n_dec']
        visit_pending = saved['state']['visit_pending']
        channels = saved['state']['channels']
        records = saved['state']['records']
        traces = saved['state']['traces']
        sleep_visits = saved['state']['sleep_visits']
        points = saved['state']['points']
        clock = saved['state']['clock']
        prev_cp = saved['state']['prev_cp']
        controller = saved['state']['controller']
        eyelid_checks = saved['state']['eyelid_checks']
        input_bits = saved['state']['input_bits']
        random_mask = saved['state']['random_mask']
        random_starts = saved['state']['random_starts']
        input_rooms = saved['state']['input_rooms']
        choices = saved['state']['choices']
        sleep_left = saved['state']['sleep_left']
        t = saved['state']['t']
        snap = saved['state']['snap']
        checked = saved['state']['checked']
        sleep_learned = saved['state']['sleep_learned']
        mixer_history = saved['state']['mixer_history']
        mixer_sum = saved['state']['mixer_sum']
        mixer_n = saved['state']['mixer_n']
        previous_wall = saved['state']['previous_wall']
        torch.set_rng_state(saved['torch_rng'])
    limit = L if stop_at is None else int(stop_at)
    assert t <= limit <= L

    def finish_room():
        nonlocal v_max, visit_pending
        if n_dec:
            reward = total / n_dec
            m7.ema(values, counts, target, reward)
            v_max = reward if v_max is None else max(v_max, reward)
            last_end[target] = t
            visit_pending = w4._close_visit(life, visit_pending)

    while t < limit:
        mixer_sum += np.asarray(weights(life.cortex))
        mixer_n += 1
        if sleep_left == 0:
            want = (controller.wants_sleep() if arm == 'pressure_retina' else
                    t in random_starts if arm == 'fixed_random' else
                    night(t) if arm == 'fixed_night' else False)
            if want:
                sleep_left = min(8, L - t)
                sleep_visits.append({'t': t, 'length': sleep_left})
                visit_pending = w4._close_visit(life, visit_pending)
                if not checked:
                    snap = life.begin_sleep_visit()
                    sleep_learned = 0
        sleeping = sleep_left > 0
        # Trace reports controller state BEFORE this decision and AFTER selection.
        if trace and 180 <= t <= 260:
            rec = dict(t=t, night=int(night(t)), S=controller.S, r_bar=controller.w,
                       m_t=controller.m, asleep=int(sleeping), forced=int(controller.forced))
            traces.append(rec)
            print('TRACE ' + json.dumps(rec), flush=True)
        input_rooms.append(dict(t=t, slot=world.room, kind=world.kind[world.room], asleep=int(sleeping)))
        if sleeping:
            world.t = t
            control_state = (values, counts, last_end, v_max, unvisited, plan, target,
                             total, n_dec, visit_pending, choice_rng)
            before_hash = eyelid_checksum(life, controller, control_state) if check_eyelid else None
            signal = eyelid_glance(world, life, controller, index, rng) if arm == 'pressure_retina' else None
            if check_eyelid:
                after_hash = eyelid_checksum(life, controller, control_state)
                assert before_hash == after_hash, 'eyelid changed learner/controller state'
                eyelid_checks.append(dict(t=t, before=before_hash, after=after_hash, intact=True))
            sleep_learned += int(life.sleep_decision())
            channels.append('sleep')
            controller.update(True)
            sleep_left -= 1
            if sleep_left == 0 and snap is not None:
                delta = life.end_sleep_visit(snap)
                if sleep_learned and delta <= 0:
                    raise RuntimeError('sleep learned without changing parameters')
                snap, checked = None, True
        else:
            if not plan:
                target, _, _ = m8.choose_room(world.room, unvisited, values, counts,
                                             last_end, v_max, t, world.rooms, choice_rng, True)
                choices.append(dict(t=t, target=target))
                plan = m10.w4_world.shortest(world.room, target, 18) + ['stay'] * 8
                total, n_dec = 0., 0
            action = plan.pop(0)
            world.t = t
            learnable_stay = action == 'stay' and world.is_learnable(world.room)
            _, channel, symbols = world.step(action, rng)
            signal = retina.retina_bits(symbols)
            life.count_preq = learnable_stay
            total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            life.count_preq = False
            n_dec += 1
            controller.update(False, signal)
            if channel != 'corridor':
                visit_pending = channel
            channels.append(channel)
        input_bits.append(dict(t=t, r=signal, r_bar=controller.w,
                               m_t=controller.m, S=controller.S, asleep=int(sleeping),
                               forced=int(controller.forced)))
        t += 1
        if t % 512 == 0:
            mixer_history.append(dict(decision=t, mean_weights=(mixer_sum / mixer_n).tolist(),
                                      weights=weights(life.cortex)))
            mixer_sum[:] = 0
            mixer_n = 0
        world.maybe_birth(t)
        if t in points:
            records.append(w6.take_checkpoint(life, t, prev_cp, clock))
            prev_cp = t
        if not sleeping and not plan:
            finish_room()
            n_dec = 0
    segment_wall = time.perf_counter() - start
    previous_wall += segment_wall
    if checkpoint is not None:
        state = {'world': world, 'names': names, 'index': index, 'life': life, 'rng': rng, 'choice_rng': choice_rng, 'values': values, 'counts': counts, 'last_end': last_end, 'v_max': v_max, 'unvisited': unvisited, 'plan': plan, 'target': target, 'total': total, 'n_dec': n_dec, 'visit_pending': visit_pending, 'channels': channels, 'records': records, 'traces': traces, 'sleep_visits': sleep_visits, 'points': points, 'clock': clock, 'prev_cp': prev_cp, 'controller': controller, 'eyelid_checks': eyelid_checks, 'input_bits': input_bits, 'random_mask': random_mask, 'random_starts': random_starts, 'input_rooms': input_rooms, 'choices': choices, 'sleep_left': sleep_left, 't': t, 'snap': snap, 'checked': checked, 'sleep_learned': sleep_learned, 'mixer_history': mixer_history, 'mixer_sum': mixer_sum, 'mixer_n': mixer_n, 'previous_wall': previous_wall}
        saved = dict(source_hashes=source_hashes(), config=(seed, cortex, L, night_enabled, depth),
                     state=state, torch_rng=torch.get_rng_state())
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            pickle.dump(saved, f, protocol=5)
        temp.replace(path)
    if t < L:
        return dict(seed=seed, cortex=cortex, decision=t, complete=False,
                    segment_wall_s=segment_wall, wall_s=previous_wall)
    if n_dec:
        finish_room()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    scores = [r['open'] for r in records]
    slept = sum(c == 'sleep' for c in channels)
    night_slept = sum(c == 'sleep' and night(i) for i, c in enumerate(channels))
    assert world.birth_log == w6_world.expected_birth_log(seed, PERIOD, L)
    assert not life.exam.ids & life.replay.ids and life.exam_drawn == 0
    assert len(channels) == L and len(records) == 8
    row = dict(seed=seed, arm=arm, L=L, period=128, night_enabled=night_enabled,
               T_open=float(np.mean(scores)), final=scores[-1], checkpoints=records,
               sleep_share=slept / L, sleep_decisions=slept, night_sleep_decisions=night_slept,
               night_sleep_share=night_slept / slept if slept else None,
               sleep_visits=sleep_visits, random_starts=[],
               birth_log=world.birth_log, sleep_checks=life.sleep_checks,
               finite=all(math.isfinite(v) for cp in records for v in [cp['open']] + [r['bits'] for r in cp['rooms']]),
               wall_s=previous_wall, segment_wall_s=segment_wall, cortex=cortex)
    if arm == 'fixed_random':
        assert np.array_equal(random_mask, np.array(channels) == 'sleep')
    if arm == 'fixed_night':
        assert all((c == 'sleep') == night(i) for i,c in enumerate(channels))
    row['input_retina'] = input_bits if arm == 'pressure_retina' else []
    row['input_rooms'] = input_rooms
    row['choices'] = choices
    row['preq_bits'], row['preq_symbols'] = life.preq_bits, life.preq_symbols
    row['learner_checksum'] = eyelid_checksum(life, SimpleNamespace(), ())
    day_sleeps = [r for r in input_rooms if r['asleep'] and not night(r['t'])]
    row['day_sleep_decisions'] = len(day_sleeps)
    row['day_noise_sleep_decisions'] = sum(r['kind'] == 'N' for r in day_sleeps)
    row['day_noise_sleep_share'] = row['day_noise_sleep_decisions'] / len(day_sleeps) if day_sleeps else None
    row['random_starts'] = random_starts if arm == 'fixed_random' else []
    row['eyelid_checks'] = eyelid_checks
    if trace:
        row['trace'] = traces
    if not row['finite'] or not math.isfinite(row['T_open']):
        raise RuntimeError('nonfinite score')
    row['glance_log'] = life.glance_log
    row['preq_bps'] = life.preq_bits / life.preq_symbols if life.preq_symbols else None
    row['mixer_history'] = mixer_history
    row['initial_mixer_weights'] = ([1.] if cortex == 'bank_1' else
        [1.] * (depth or 4) if cortex == 'deep_4_open' else [1.] + [0.] * ((depth or 4)-1))
    row['final_mixer_weights'] = weights(life.cortex)
    row['complete'] = True
    return row


def run_life(*args, **kwargs):
    with patch.object(m7, 'score_sequences', score_sequences):
        return _run_life(*args, **kwargs)


def load_w10_summary():
    spec = importlib.util.spec_from_file_location('w10_summary', ROOT / 'kaggle/w10/summarize.py')
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    return summary


def smoke(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, workers=1, L=L_SCORE, source_hashes=source_hashes(), checks=[])
    start = time.perf_counter()
    def record(name, **values):
        payload['checks'].append(dict(check=name, **values))
        print(json.dumps(payload['checks'][-1]), flush=True)
        dump(out / 'smoke.json', payload)
    try:
        m10._threads()
        # Reference is the unmodified W10 loop and original cortex.
        ref_log = []
        original_learn = w6.OpenLife._learn
        def observe_reference(life, symbols, channel, prev, state=None, commit=True):
            bits = original_learn(life, symbols, channel, prev, state, commit)
            ref_log.append(dict(channel=int(channel), prev=int(prev), wake=bool(commit),
                preq=bool(life.count_preq), bits=float(bits.sum()), n=len(bits),
                sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
            return bits
        with patch.object(w6.OpenLife, '_learn', observe_reference):
            reference = w10.run_life(120, 'pressure_retina', 512, check_eyelid=True)
        bank = run_life(120, 'bank_1', 512, check_eyelid=True)
        thin = run_life(120, 'deep_4', 512, depth=1, check_eyelid=True)
        for field in ('checkpoints', 'preq_bits', 'preq_symbols', 'choices', 'sleep_visits', 'input_retina', 'input_rooms'):
            assert reference[field] == bank[field] == thin[field], field
        assert ref_log == bank['glance_log'] == thin['glance_log']
        record('W10_and_depth1_exact', seed=120, decisions=512, glances=len(bank['glance_log']),
               max_bits_abs=0, choices_exact=True, sleeps_exact=True)
        symbols = np.arange(m7.CHUNK, dtype=np.int64) % len(m7.ALPHA)
        single, deep = DeepCortex(depth=1), DeepCortex()
        sb = single.learn_glance(symbols, 0, -1)
        db = deep.learn_glance(symbols, 0, -1)
        assert torch.equal(deep.last_logits, deep.last_own_logits[:, 0])
        assert np.array_equal(sb, db)
        assert all(torch.equal(a, b) for a, b in zip(single.layers[0].params, deep.layers[0].params))
        assert m7.states_equal(single.state[0], deep.state[0])
        record('Wait_and_layer1_isolation', max_logits_abs=0, max_bits_abs=0, max_param_abs=0)
        # Prefix and future-suffix checks on every cortex; replay cannot mutate wake traces.
        for cortex in CORTICES:
            a = BANK_CLASS(w4.N_CH) if cortex == 'bank_1' else DeepCortex(opened=cortex.endswith('open'))
            b = BANK_CLASS(w4.N_CH) if cortex == 'bank_1' else DeepCortex(opened=cortex.endswith('open'))
            ab = a.learn_glance(symbols, 0, -1)
            bb = b.learn_glance(symbols, 0, -1)
            assert np.array_equal(ab, bb)
            live = a.clone_live()
            a.learn_glance(symbols[::-1].copy(), 1, 3, state=a.zero_state(), commit=False)
            assert a.live_matches(live)
            # A later, changed glance cannot change the committed prefix.
            b.learn_glance((symbols + 7) % len(m7.ALPHA), 1, int(symbols[-1]))
            assert np.array_equal(ab, bb)
            if isinstance(a, DeepCortex):
                future = np.concatenate((symbols, symbols))[None, :]
                changed = future.copy()
                changed[:,64:] = (changed[:,64:] + 11) % len(m7.ALPHA)
                x = a.score_sequences(torch.as_tensor(future), torch.tensor([-1]), torch.tensor([0]), True)
                y = a.score_sequences(torch.as_tensor(changed), torch.tensor([-1]), torch.tensor([0]), True)
                z = a.score_sequences(torch.as_tensor(future[:,:64]), torch.tensor([-1]), torch.tensor([0]), True)
                assert torch.equal(x[:,:64], y[:,:64]) and torch.equal(x[:,:64], z)
            record('prequential_and_replay', cortex=cortex, max_prefix_bits_abs=0, wake_intact=True)
        timed = run_life(120, 'deep_4', 512, check_eyelid=True)
        load_w10_summary().validate_retina(timed, {r['t'] for r in timed['input_rooms'] if r['asleep']})
        assert timed['eyelid_checks'] and all(r['intact'] for r in timed['eyelid_checks'])
        assert timed['preq_symbols'] == sum(g['n'] for g in timed['glance_log'] if g['preq'])
        assert timed['preq_bits'] == sum(g['bits'] for g in timed['glance_log'] if g['preq'])
        record('eyelid_controller_preq', decisions=len(timed['eyelid_checks']), intact=True,
               preq_symbols=timed['preq_symbols'], finite=timed['finite'])
        # Exact serialization at a boundary inside a sleep/room visit, not just a score boundary.
        cp = out / 'smoke-checkpoint.pkl'
        partial = run_life(120, 'deep_4', 512, checkpoint=cp, stop_at=253, check_eyelid=True)
        resumed = run_life(120, 'deep_4', 512, resume=cp, check_eyelid=True)
        for field in ('checkpoints','preq_bits','preq_symbols','choices','sleep_visits','input_retina',
                      'input_rooms','glance_log','mixer_history','final_mixer_weights'):
            assert timed[field] == resumed[field], 'resume ' + field
        # W10's pickle-byte checksum detects mutation within an eyelid glance.
        # Its object memoization is not invariant across checkpoint loading.
        assert [(r['t'], r['intact']) for r in timed['eyelid_checks']] == [
            (r['t'], r['intact']) for r in resumed['eyelid_checks']]
        assert all(r['intact'] and r['before'] == r['after'] for r in resumed['eyelid_checks'])
        check_spec = importlib.util.spec_from_file_location('w11_checkpoint_check', ROOT / 'kaggle/w11/check_checkpoint.py')
        checkpoint_check = importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(checkpoint_check)
        record('checkpoint_values', **checkpoint_check.check(out))
        cp.unlink()
        record('exact_resume', split=253, complete=partial['complete'], max_bits_abs=0,
               pickle_checksum_equal=timed['learner_checksum'] == resumed['learner_checksum'])
        opened = run_life(120, 'deep_4_open', 512)
        # Bound late-life work after reservoirs fill, rather than assuming
        # early L512 scoring costs stay constant as stores grow.
        late, estimates, parts, segment_estimates = {}, {}, {}, {}
        for cortex in CORTICES:
            obj = BANK_CLASS(w4.N_CH) if cortex == 'bank_1' else DeepCortex(opened=cortex.endswith('open'))
            tick = time.perf_counter()
            obj.learn_glance(symbols, 0, -1)
            learn_s = time.perf_counter() - tick
            def rate(batch, length):
                tokens = torch.arange(batch*length).reshape(batch,length) % len(m7.ALPHA)
                tick = time.perf_counter()
                score_sequences(obj.model, obj.channel_emb, tokens,
                                torch.full((batch,),-1), torch.arange(batch) % w4.N_CH)
                return time.perf_counter()-tick
            one, room, replay, exam = rate(1,64), rate(m7.R,64), rate(18*m7.R,64), rate(18,2048)
            # Every decision is bounded by the more expensive of a wake or
            # replay decision. Wake allows two progress exams, an insertion,
            # and a visit snapshot; replay rescans the full reservoir.
            per_decision = learn_s + max(3*room+one, replay)
            estimates[cortex] = 30 + 2*(L_SCORE*per_decision + 8*exam)
            for count in range(1,65):
                size = math.ceil(L_SCORE/count)
                exams = min(8, math.ceil(size/512)+1)
                bound = 30 + 2*(size*per_decision + exams*exam)
                if bound <= 5100:
                    parts[cortex], segment_estimates[cortex] = count, bound
                    break
            else:
                raise RuntimeError('no safe split within 64 exact parts')
            late[cortex] = dict(learn_s=learn_s, score_one_s=one, score_room32_s=room,
                score_replay576_s=replay, score_exam18x2048_s=exam, per_decision_bound_s=per_decision)
        payload['estimate'] = dict(kernel_wall_s=estimates, parts=parts,
            segment_wall_s=segment_estimates, late_reservoir_timing=late,
            limit_s=5100, margin=2., startup_s=30., workers=1)
        record('wall_estimate', **payload['estimate'])
        for name, row in [('bank_1', bank), ('deep_4_depth1', thin), ('deep_4', timed), ('deep_4_open', opened)]:
            dump(out / f'smoke-{name}.json', row)
        assert payload['source_hashes'] == source_hashes(), 'source changed during smoke'
        payload['ok'] = True
    except BaseException as exc:
        payload['error'] = repr(exc)
        raise
    finally:
        payload['wall_s'] = time.perf_counter() - start
        dump(out / 'smoke.json', payload)
    return payload


def kernel_main(seed, cortex, part, parts, kernel_id, out='/kaggle/working'):
    assert seed in SEEDS and cortex in CORTICES and 1 <= part <= parts <= 64
    slug = f'w11-{cortex.replace("_", "-")}-s{seed}'
    assert kernel_id == 'ezzzzzx/' + slug + (f'-p{part}' if parts > 1 else '')
    out = Path(out)
    name = f'{cortex}-s{seed}'
    resume = None
    if part > 1:
        hits = list(Path('/kaggle/input').rglob(f'{name}-p{part-1}.pkl'))
        assert len(hits) == 1, 'expected unique exact checkpoint from previous kernel part'
        resume = hits[0]
    end = L_SCORE if part == parts else L_SCORE * part // parts
    cp = out / f'{name}-p{part}.pkl' if part < parts else None
    start = time.perf_counter()
    row = run_life(seed, cortex, checkpoint=cp, resume=resume, stop_at=end)
    payload = dict(seed=seed, cortex=cortex, kernel_id=kernel_id, part=part, parts=parts,
                   workers=1, runs=[row], code_sha256=source_hashes(), wall_s=time.perf_counter()-start)
    if cp:
        payload['checkpoint_sha256'] = hashlib.sha256(cp.read_bytes()).hexdigest()
        payload['checkpoint_file'] = cp.name
    dump(out / f'{name}-p{part}.json', payload)
    print(json.dumps(dict(seed=seed, cortex=cortex, part=part, wall_s=payload['wall_s'],
                          T_open=row.get('T_open'), decision=end)), flush=True)


def stage(payload):
    assert payload['ok'] and payload['source_hashes'] == source_hashes()
    base = ROOT / 'kaggle/w11'
    code = base / 'code'
    code.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    dump(code / 'dataset-metadata.json', dict(title='pdp code w11', id=DATASET, licenses=[{'name':'other'}]))
    dump(base / 'code-sha256.json', source_hashes())
    manifest = []
    for seed in SEEDS:
        for cortex in CORTICES:
            parts = payload['estimate']['parts'][cortex]
            previous = None
            for part in range(1, parts + 1):
                slug = f'w11-{cortex.replace("_", "-")}-s{seed}' + (f'-p{part}' if parts > 1 else '')
                kid = 'ezzzzzx/' + slug
                folder = base / slug.removeprefix('w11-')
                folder.mkdir(parents=True, exist_ok=True)
                dump(folder / 'kernel-metadata.json', dict(id=kid, title=slug, code_file='kernel.py',
                    language='python', kernel_type='script', is_private=True, enable_gpu=False,
                    enable_tpu=False, enable_internet=False, dataset_sources=[DATASET],
                    competition_sources=[], kernel_sources=[previous] if previous else [], model_sources=[]))
                src = f'''import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
if os.environ.get('_W11_KERNEL') != '1':
    os.environ['_W11_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = list(Path('/kaggle/input').rglob('w11_deep_cortex.py'))
    assert len(hits) == 1, 'expected one W11 code dataset'
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w11_deep_cortex as w11
    w11.kernel_main({seed}, {cortex!r}, {part}, {parts}, {kid!r})
'''
                (folder / 'kernel.py').write_text(src)
                manifest.append(dict(seed=seed, cortex=cortex, part=part, parts=parts,
                    folder=folder.name, kernel_id=kid, arms=['pressure_retina'],
                    depends_on=previous, output=f'{cortex}-s{seed}-p{part}.json'))
                previous = kid
    dump(base / 'manifest.json', manifest)
    print(f'STAGED {len(manifest)} private CPU kernels', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('smoke','stage'))
    p.add_argument('--out', type=Path, default=ROOT / 'results/w11')
    args = p.parse_args()
    payload = smoke(args.out) if args.command == 'smoke' else json.loads((args.out / 'smoke.json').read_text())
    stage(payload)


if __name__ == '__main__':
    main()
