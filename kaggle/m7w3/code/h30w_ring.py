"""H30w: frequency bank vs propagating ring vs GRU. Numbers only. No verdict.

Online prequential protocol matches h21a: score a chunk of 64 under the current
parameters, one Adam step on that chunk, carry the state forward detached.
Input at each step is the previous symbol. The first input embedding is zero.

Ring (amended): ρ = exp(−exp(ν)), g = sigmoid(γ),
a = ρ g exp(iθ), b = ρ (1−g) exp(iφ), so |a| + |b| = ρ < 1.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from h20_world import ALPHA, PROCS, stream

CHUNK = 64
EMB = 32
UNITS = 256
HIDDEN = 166
RANK = 64
LR = 1e-3
CLIP = 1.0
H20_N = 262_144
ECHO_N = 131_072
SMOKE_N = 4_096
TIME_N = 8_192
KERNEL_LIMIT_S = 90 * 60
BIN_EDGES = [i / 10 for i in range(11)]
MODELS = ('gru', 'bank', 'ring')
ECHO_DS = (32, 128)
LOG2 = math.log(2.0)


def configure_threads(n):
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


configure_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))


def py(value):
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(py(obj), indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def count_params(model):
    total = 0
    for param in model.parameters():
        if param.is_complex():
            raise RuntimeError('complex parameter would under-count degrees of freedom')
        total += param.numel()
    return int(total)


def encode_text(text):
    table = np.full(128, -1, dtype=np.int64)
    for i, char in enumerate(ALPHA):
        table[ord(char)] = i
    raw = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
    tokens = table[raw]
    if np.any(tokens < 0):
        raise RuntimeError('symbol outside the H20 alphabet')
    return tokens


def echo_stream(seed, delay, n):
    """Fresh block of `delay` uniform symbols, then the same block again."""
    rng = np.random.default_rng(100 + seed)
    symbols = np.empty(n, dtype=np.int64)
    fresh = np.empty(n, dtype=bool)
    pos = 0
    while pos < n:
        block = rng.integers(0, 16, size=delay)
        take = min(delay, n - pos)
        symbols[pos:pos + take] = block[:take]
        fresh[pos:pos + take] = True
        pos += take
        if pos >= n:
            break
        take = min(delay, n - pos)
        symbols[pos:pos + take] = block[:take]
        fresh[pos:pos + take] = False
        pos += take
    return symbols, fresh


class GRUModel(nn.Module):
    def __init__(self, alphabet):
        super().__init__()
        self.embedding = nn.Embedding(alphabet, EMB)
        self.gru = nn.GRU(EMB, HIDDEN)
        self.readout = nn.Linear(HIDDEN, alphabet)

    def forward_chunk(self, emb, state):
        y, state = self.gru(emb[:, None, :], state)
        return self.readout(y[:, 0]), state


class Bank(nn.Module):
    """h_t = λ ⊙ h_{t-1} + B x_t, λ = exp(−exp(ν) + iθ). Order-2 readout."""

    def __init__(self, alphabet):
        super().__init__()
        self.embedding = nn.Embedding(alphabet, EMB)
        mag = torch.empty(UNITS).uniform_(0.9, 0.999)
        theta = torch.empty(UNITS).uniform_(0.0, math.pi)
        nu = torch.log(-torch.log(mag))
        self.nu = nn.Parameter(nu)
        self.theta = nn.Parameter(theta)
        scale = 1.0 / math.sqrt(EMB)
        self.B_re = nn.Parameter(torch.randn(UNITS, EMB) * scale)
        self.B_im = nn.Parameter(torch.randn(UNITS, EMB) * scale)
        self.W = nn.Linear(2 * UNITS, alphabet, bias=False)
        self.P1 = nn.Linear(2 * UNITS, RANK, bias=False)
        self.P2 = nn.Linear(2 * UNITS, RANK, bias=False)
        self.V = nn.Linear(RANK, alphabet, bias=False)

    def lam(self):
        return torch.exp(-torch.exp(self.nu) + 1j * self.theta)

    def inject(self, emb):
        return torch.complex(emb @ self.B_re.T, emb @ self.B_im.T)

    def logits_from(self, hidden):
        z = torch.cat((hidden.real, hidden.imag), dim=-1)
        quad = self.P1(z) * self.P2(z)
        return self.W(z) + self.V(quad)

    def forward_chunk(self, emb, state):
        h = emb.new_zeros(UNITS, dtype=torch.complex64) if state is None else state
        lam = self.lam()
        inj = self.inject(emb)
        hs = []
        for t in range(emb.shape[0]):
            h = lam * h + inj[t]
            hs.append(h)
        return self.logits_from(torch.stack(hs)), h


def ring_step(h, a, b, inj):
    """One ring update. roll(+1)[i] = h[i-1], so unit 0 reads unit 255.

    The right-hand side reads h before the write, so the neighbour term is h at t−1.
    """
    return a * h + b * torch.roll(h, 1) + inj


class Ring(Bank):
    """h_i,t = a_i h_i,t−1 + b_i h_{i−1,t−1} + (B x_t)_i.

    ρ = exp(−exp(ν)) < 1, g = sigmoid(γ), a = ρ g e^(iθ), b = ρ (1−g) e^(iφ).
    Then |a| + |b| = ρ < 1 for every finite parameter value.
    """

    def __init__(self, alphabet):
        super().__init__(alphabet)
        # Bank already drew ρ into ν and θ into theta, same ranges as the ring spec.
        gate = torch.empty(UNITS).uniform_(0.2, 0.8)
        phi = torch.empty(UNITS).uniform_(0.0, math.pi)
        self.gamma = nn.Parameter(torch.logit(gate))
        self.phi = nn.Parameter(phi)

    def coupling(self):
        rho = torch.exp(-torch.exp(self.nu))
        gate = torch.sigmoid(self.gamma)
        # polar sets the modulus directly, so |a| = ρ g and |b| = ρ (1−g).
        a = torch.polar(rho * gate, self.theta)
        b = torch.polar(rho * (1.0 - gate), self.phi)
        return a, b

    def forward_chunk(self, emb, state):
        h = emb.new_zeros(UNITS, dtype=torch.complex64) if state is None else state
        a, b = self.coupling()
        inj = self.inject(emb)
        hs = []
        for t in range(emb.shape[0]):
            h = ring_step(h, a, b, inj[t])
            hs.append(h)
        return self.logits_from(torch.stack(hs)), h


def build(name, alphabet, model_seed):
    torch.manual_seed(model_seed)
    if name == 'gru':
        model = GRUModel(alphabet)
    elif name == 'bank':
        model = Bank(alphabet)
    elif name == 'ring':
        model = Ring(alphabet)
    else:
        raise ValueError(name)
    check_init(model)
    return model


def check_init(model):
    if not hasattr(model, 'nu'):
        return
    with torch.no_grad():
        mag = torch.exp(-torch.exp(model.nu))
        theta = model.theta
        if float(mag.min()) < 0.9 - 1e-6 or float(mag.max()) > 0.999 + 1e-6:
            raise RuntimeError(f'|λ| init outside [0.9, 0.999]: {float(mag.min())} {float(mag.max())}')
        if float(theta.min()) < -1e-6 or float(theta.max()) > math.pi + 1e-6:
            raise RuntimeError(f'θ init outside [0, π]: {float(theta.min())} {float(theta.max())}')
        if hasattr(model, 'gamma'):
            gate = torch.sigmoid(model.gamma)
            phi = model.phi
            if float(gate.min()) < 0.2 - 1e-5 or float(gate.max()) > 0.8 + 1e-5:
                raise RuntimeError(f'g init outside [0.2, 0.8]: {float(gate.min())} {float(gate.max())}')
            if float(phi.min()) < -1e-6 or float(phi.max()) > math.pi + 1e-6:
                raise RuntimeError(f'φ init outside [0, π]: {float(phi.min())} {float(phi.max())}')


def chunk_embedding(model, tokens, start, end):
    if start == 0:
        if end == 1:
            return model.embedding.weight.new_zeros(1, EMB)
        prev = model.embedding(tokens[:end - 1])
        zero = prev.new_zeros(1, EMB)
        return torch.cat((zero, prev), dim=0)
    return model.embedding(tokens[start - 1:end - 1])


def train(model, tokens, label, progress_every):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    n = int(tokens_t.shape[0])
    bits = np.empty(n, dtype=np.float64)
    state = None
    clock = time_counter()
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        optimizer.zero_grad(set_to_none=True)
        emb = chunk_embedding(model, tokens_t, start, end)
        logits, state = model.forward_chunk(emb, state)
        losses = F.cross_entropy(logits, tokens_t[start:end], reduction='none')
        scored = losses.detach().double().numpy() / LOG2
        if not np.all(np.isfinite(scored)):
            raise FloatingPointError(f'{label}: nonfinite bits at {start}')
        bits[start:end] = scored
        state = state.detach()
        if state.requires_grad:
            raise RuntimeError(f'{label}: carried state still requires grad')
        losses.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        done = (start // CHUNK + 1)
        if done % progress_every == 0 or end == n:
            print(
                f'{label} n={end}/{n} bits={bits[:end].mean():.6f} seconds={time_counter() - clock:.2f}',
                flush=True,
            )
    return bits, time_counter() - clock


def time_counter():
    return __import__('time').perf_counter()


def ring_mags(model):
    with torch.no_grad():
        a, b = model.coupling()
        mag_a = a.abs().detach().cpu().numpy()
        mag_b = b.abs().detach().cpu().numpy()
    return mag_hist(mag_a, 'a'), mag_hist(mag_b, 'b')


def mag_hist(mag, name):
    if not np.all(np.isfinite(mag)):
        raise FloatingPointError(f'| {name} | is nonfinite')
    counts, edges = np.histogram(mag, bins=BIN_EDGES)
    outside = int(np.sum((mag < 0.0) | (mag > 1.0)))
    if int(counts.sum()) + outside != int(mag.size):
        raise RuntimeError(f'histogram count mismatch for |{name}|')
    return dict(
        name=name,
        counts=[int(c) for c in counts],
        bin_edges=[float(e) for e in edges],
        n=int(mag.size),
        n_outside=outside,
        min=float(mag.min()),
        max=float(mag.max()),
        mean=float(mag.mean()),
    )


def run_learner(name, alphabet, tokens, model_seed, label, progress_every):
    model = build(name, alphabet, model_seed)
    parameters = count_params(model)
    before = None
    if name == 'ring':
        before = {key: getattr(model, key).detach().clone() for key in ('nu', 'gamma', 'theta', 'phi')}
    bits, seconds = train(model, tokens, label, progress_every)
    out = dict(
        model=name,
        alphabet=alphabet,
        model_seed=model_seed,
        parameters=parameters,
        n=int(len(tokens)),
        wall_seconds=float(seconds),
        bits_per_symbol=float(bits.mean()),
        bits=bits,
    )
    if name == 'ring':
        hist_a, hist_b = ring_mags(model)
        delta = 0.0
        with torch.no_grad():
            for key, old in before.items():
                delta = max(delta, float((getattr(model, key).detach() - old).abs().max()))
        out['mag_a'] = hist_a
        out['mag_b'] = hist_b
        out['recurrence_max_abs_delta'] = delta
    del model
    return out


def h20_record(raw, procs):
    by = {}
    counts = {}
    for name in PROCS:
        mask = procs == name
        counts[name] = int(mask.sum())
        by[name] = float(raw['bits'][mask].mean()) if counts[name] else None
    both = (procs == 'C1') | (procs == 'C7')
    record = {k: v for k, v in raw.items() if k != 'bits'}
    record.update(
        stream='h20_slow',
        by_process=by,
        process_counts=counts,
        c1_c7=float(raw['bits'][both].mean()) if int(both.sum()) else None,
        c1_c7_count=int(both.sum()),
    )
    return record


def echo_record(raw, fresh, delay, seed):
    copy = ~fresh
    record = {k: v for k, v in raw.items() if k != 'bits'}
    record.update(
        stream='echo',
        D=int(delay),
        seed=int(seed),
        fresh=float(raw['bits'][fresh].mean()) if int(fresh.sum()) else None,
        copy=float(raw['bits'][copy].mean()) if int(copy.sum()) else None,
        fresh_count=int(fresh.sum()),
        copy_count=int(copy.sum()),
    )
    return record


def ring_transport_check():
    """Hand-set transport: a_i = 0, b_i = 1 (g = 0, ρ = 1, θ = φ = 0).

    Set directly, not through the stable parametrisation, which cannot reach ρ = 1.
    Real injection only at unit 0. This is a check, not the learned init.
    """
    steps = 256
    values = torch.arange(1, steps + 1, dtype=torch.float32)
    h = torch.zeros(UNITS, dtype=torch.complex64)
    a = torch.zeros(UNITS, dtype=torch.complex64)
    b = torch.ones(UNITS, dtype=torch.complex64)
    for t in range(steps):
        inj = torch.zeros(UNITS, dtype=torch.complex64)
        inj[0] = values[t]
        h = ring_step(h, a, b, inj)
    lines = ['ring_check handset a=0 b=1 injection=unit0_real steps=256']
    ok = True
    for k in (0, 1, 5, 200):
        expected = float(values[steps - 1 - k])
        held = h[k]
        match = abs(float(held.real) - expected) < 1e-5 and abs(float(held.imag)) < 1e-5
        ok = ok and match
        lines.append(
            f'ring_check k={k} held_real={float(held.real):.6f} held_imag={float(held.imag):.6f} '
            f'input_k_steps_earlier={expected:.6f} match={match}'
        )
    return ok, lines


def prequential_check():
    """Bits of chunk c stay put when the stream after that chunk is altered."""
    rng = np.random.default_rng(12345)
    n = CHUNK * 8
    original = rng.integers(0, 16, size=n)
    altered = original.copy()
    c = 2
    end = (c + 1) * CHUNK
    altered[end:] = rng.integers(0, 16, size=n - end)
    if np.array_equal(original, altered):
        raise RuntimeError('prequential fixture did not alter the suffix')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = run_learner('bank', 16, original, 0, 'preq_original', progress_every=10_000)
        second = run_learner('bank', 16, altered, 0, 'preq_altered', progress_every=10_000)
    finally:
        torch.set_num_threads(threads)
    delta = float(np.max(np.abs(first['bits'][:end] - second['bits'][:end])))
    suffix = float(np.max(np.abs(first['bits'][end:] - second['bits'][end:])))
    ok = delta == 0.0
    line = (
        f'prequential_check model=bank chunk={c} end={end} max_abs_delta_through_chunk={delta:.8e} '
        f'max_abs_delta_after={suffix:.8e} unchanged={ok}'
    )
    return ok, line


def echo_shape_check():
    lines = []
    ok = True
    for delay in ECHO_DS:
        tokens, fresh = echo_stream(0, delay, ECHO_N)
        period = 2 * delay
        copy_matches = np.array_equal(tokens[:delay], tokens[delay:period])
        aligned = bool(fresh[:delay].all()) and not bool(fresh[delay:period].any())
        balanced = int(fresh.sum()) == ECHO_N // 2
        this = copy_matches and aligned and balanced and tokens.shape == (ECHO_N,)
        ok = ok and this
        lines.append(
            f'echo_check D={delay} n={ECHO_N} fresh_count={int(fresh.sum())} '
            f'copy_matches={copy_matches} aligned={aligned} ok={this}'
        )
    return ok, lines


def finite_runs():
    """4,096 symbols of H20 slow and of echo at both D. Every model."""
    lines = []
    records = []
    text, labels = stream('slow', 6, SMOKE_N)
    tokens = encode_text(text)
    procs = np.asarray(labels)
    if len(tokens) != SMOKE_N or len(procs) != SMOKE_N:
        raise RuntimeError('H20 smoke stream length')
    text2, labels2 = stream('slow', 6, SMOKE_N)
    repro = text2 == text and labels2 == labels
    lines.append(f'h20_repro seed=6 n={SMOKE_N} match={repro}')
    if not repro:
        raise RuntimeError('H20 stream is not deterministic')
    for name in MODELS:
        raw = run_learner(name, len(ALPHA), tokens, 0, f'smoke_h20_{name}', progress_every=10_000)
        finite = bool(np.isfinite(raw['bits']).all())
        lines.append(
            f'smoke_finite stream=h20_slow seed=6 model={name} n={SMOKE_N} '
            f'parameters={raw["parameters"]} bits={raw["bits_per_symbol"]:.6f} '
            f'wall_seconds={raw["wall_seconds"]:.3f} finite={finite}'
        )
        if name == 'ring':
            lines.append(f'smoke_ring_recurrence_max_abs_delta={raw["recurrence_max_abs_delta"]:.8e}')
            if raw['recurrence_max_abs_delta'] == 0.0:
                raise RuntimeError('ring recurrence parameters did not move; gradient did not reach the recurrence')
        if not finite:
            raise FloatingPointError(name)
        records.append(h20_record(raw, procs))
    for delay in ECHO_DS:
        echo_tokens, fresh = echo_stream(0, delay, SMOKE_N)
        for name in MODELS:
            raw = run_learner(name, 16, echo_tokens, 0, f'smoke_echo_D{delay}_{name}', progress_every=10_000)
            finite = bool(np.isfinite(raw['bits']).all())
            lines.append(
                f'smoke_finite stream=echo D={delay} seed=0 model={name} n={SMOKE_N} '
                f'parameters={raw["parameters"]} bits={raw["bits_per_symbol"]:.6f} '
                f'fresh={float(raw["bits"][fresh].mean()):.6f} '
                f'copy={float(raw["bits"][~fresh].mean()):.6f} '
                f'wall_seconds={raw["wall_seconds"]:.3f} finite={finite}'
            )
            if not finite:
                raise FloatingPointError(name)
            records.append(echo_record(raw, fresh, delay, 0))
    return records, lines


def time_models():
    text, _labels = stream('slow', 6, TIME_N)
    tokens = encode_text(text)
    lines = []
    seconds = {}
    for name in MODELS:
        raw = run_learner(name, len(ALPHA), tokens, 0, f'time_{name}', progress_every=10_000)
        if not np.isfinite(raw['bits']).all():
            raise FloatingPointError(f'timing run nonfinite: {name}')
        seconds[name] = float(raw['wall_seconds'])
        lines.append(
            f'time_8192 model={name} seconds={seconds[name]:.3f} '
            f'bits={raw["bits_per_symbol"]:.6f} threads={torch.get_num_threads()}'
        )
    symbols = H20_N + ECHO_N * len(ECHO_DS)
    scale = symbols / TIME_N
    per = {name: seconds[name] * scale for name in MODELS}
    total = float(sum(per.values()))
    lines.append(
        f'kernel_estimate symbols_per_model={symbols} scale={scale:.6f} '
        + ' '.join(f'{name}_seconds={per[name]:.1f}' for name in MODELS)
        + f' total_seconds={total:.1f} limit_seconds={KERNEL_LIMIT_S}'
    )
    return seconds, per, total, lines


def stability_check():
    """At init, max_i (|a_i| + |b_i|) < 1 for every model seed. Both alphabets are built."""
    lines = []
    ok = True
    for alphabet in (len(ALPHA), 16):
        for seed in (0, 1, 2):
            model = build('ring', alphabet, seed)
            with torch.no_grad():
                rho = torch.exp(-torch.exp(model.nu))
                gate = torch.sigmoid(model.gamma)
                a, b = model.coupling()
                summed = a.abs() + b.abs()
                mx = float(summed.max())
                this = mx < 1.0
                lines.append(
                    f'stability alphabet={alphabet} model_seed={seed} '
                    f'max_abs_a_plus_abs_b={mx:.8f} min_abs_a_plus_abs_b={float(summed.min()):.8f} '
                    f'rho_min={float(rho.min()):.6f} rho_max={float(rho.max()):.6f} '
                    f'g_min={float(gate.min()):.6f} g_max={float(gate.max()):.6f} '
                    f'theta_min={float(model.theta.min()):.6f} theta_max={float(model.theta.max()):.6f} '
                    f'phi_min={float(model.phi.min()):.6f} phi_max={float(model.phi.max()):.6f} '
                    f'stable={str(this).lower()}'
                )
            ok = ok and this
            del model
    return ok, lines


def write_smoke_files(out_dir, payload):
    dump(out_dir / 'smoke.json', payload)
    dump(out_dir / 'results.json', payload)
    (out_dir / 'smoke.log').write_text('\n'.join(payload['log']) + '\n')
    body = ['# H30w measurements', '']
    if payload.get('stopped'):
        body.append(f"Stopped before kernels. {payload.get('reason', '')}")
        body.append('')
    else:
        body.append('Smoke passed. Scored kernels are separate from this file until assemble.')
        body.append('')
    body += ['## Smoke test', '', '```', *payload['log'], '```', '']
    (out_dir / 'RESULTS.md').write_text('\n'.join(body) + '\n')


def smoke(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    print(f'torch={torch.__version__} threads={torch.get_num_threads()} dtype=float32', flush=True)
    lines.append(f'torch={torch.__version__} threads={torch.get_num_threads()} dtype=float32')
    for alphabet, tag in ((len(ALPHA), 'h20'), (16, 'echo')):
        for name in MODELS:
            model = build(name, alphabet, 0)
            n_params = count_params(model)
            line = f'parameters model={name} alphabet={alphabet} stream={tag} count={n_params}'
            print(line, flush=True)
            lines.append(line)
            if tag == 'h20' and hasattr(model, 'nu'):
                with torch.no_grad():
                    mag = torch.exp(-torch.exp(model.nu))
                    lines.append(
                        f'init_{name} abs_eig_min={float(mag.min()):.6f} abs_eig_max={float(mag.max()):.6f} '
                        f'theta_min={float(model.theta.min()):.6f} theta_max={float(model.theta.max()):.6f}'
                    )
                    print(lines[-1], flush=True)
                    if hasattr(model, 'gamma'):
                        gate = torch.sigmoid(model.gamma)
                        a, b = model.coupling()
                        summed = a.abs() + b.abs()
                        lines.append(
                            f'init_ring g_min={float(gate.min()):.6f} g_max={float(gate.max()):.6f} '
                            f'phi_min={float(model.phi.min()):.6f} phi_max={float(model.phi.max()):.6f} '
                            f'max_abs_a_plus_abs_b={float(summed.max()):.8f}'
                        )
                        print(lines[-1], flush=True)
            del model
    ok, echo_lines = echo_shape_check()
    for line in echo_lines:
        print(line, flush=True)
    lines.extend(echo_lines)
    if not ok:
        raise RuntimeError('echo stream check failed')
    records, finite_lines = finite_runs()
    for line in finite_lines:
        print(line, flush=True)
    lines.extend(finite_lines)
    ok, ring_lines = ring_transport_check()
    for line in ring_lines:
        print(line, flush=True)
    lines.extend(ring_lines)
    if not ok:
        raise RuntimeError('ring transport check failed')
    ok, preq_line = prequential_check()
    print(preq_line, flush=True)
    lines.append(preq_line)
    if not ok:
        raise RuntimeError('prequential check failed')
    seconds, per_model, total, time_lines = time_models()
    for line in time_lines:
        print(line, flush=True)
    lines.extend(time_lines)
    ok, stable_lines = stability_check()
    for line in stable_lines:
        print(line, flush=True)
    lines.extend(stable_lines)
    over = total > KERNEL_LIMIT_S
    reason = ''
    if not ok:
        reason = 'stability check failed: max (|a| + |b|) is not < 1'
    elif over:
        reason = f'kernel estimate {total:.1f}s exceeds {KERNEL_LIMIT_S}s'
    payload = dict(
        torch=torch.__version__,
        threads=int(torch.get_num_threads()),
        dtype='float32',
        gru_hidden=HIDDEN,
        stopped=bool(reason),
        reason=reason,
        kernels_pushed=False,
        records=[{k: v for k, v in record.items() if k != 'bits'} for record in records],
        time_8192_seconds=seconds,
        kernel_estimate_seconds_per_model=per_model,
        kernel_estimate_seconds=total,
        kernel_limit_seconds=KERNEL_LIMIT_S,
        log=lines,
    )
    write_smoke_files(out_dir, payload)
    if reason:
        print(f'SMOKE_FAIL {reason}. Kernels not pushed.', flush=True)
        raise SystemExit(2)
    print('smoke_ok', flush=True)
    return payload


def kernel_main(out, seed_index, kernel_id):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if seed_index not in (0, 1, 2):
        raise SystemExit(f'seed_index {seed_index} is outside 0, 1, 2')
    world_seed = 6 + seed_index
    echo_seed = seed_index
    model_seed = seed_index
    clock = time_counter()
    payload = dict(
        kernel_id=kernel_id,
        seed_index=seed_index,
        world_seed=world_seed,
        echo_seed=echo_seed,
        model_seed=model_seed,
        torch=torch.__version__,
        threads=int(torch.get_num_threads()),
        runs=[],
    )
    dump(out / 'results.json', payload)
    text, labels = stream('slow', world_seed, H20_N)
    tokens = encode_text(text)
    procs = np.asarray(labels)
    if len(tokens) != H20_N:
        raise RuntimeError('H20 length')
    for name in MODELS:
        label = f'{kernel_id} h20_s{world_seed} {name}'
        raw = run_learner(name, len(ALPHA), tokens, model_seed, label, progress_every=256)
        record = h20_record(raw, procs)
        record.update(kernel_id=kernel_id, world_seed=world_seed, model_seed=model_seed)
        payload['runs'].append(record)
        payload['wall_seconds'] = time_counter() - clock
        dump(out / 'results.json', payload)
        print(
            f'done {label} bits={record["bits_per_symbol"]:.6f} c1_c7={record["c1_c7"]} '
            f'seconds={record["wall_seconds"]:.2f}',
            flush=True,
        )
    for delay in ECHO_DS:
        echo_tokens, fresh = echo_stream(echo_seed, delay, ECHO_N)
        for name in MODELS:
            label = f'{kernel_id} echo_D{delay}_s{echo_seed} {name}'
            raw = run_learner(name, 16, echo_tokens, model_seed, label, progress_every=256)
            record = echo_record(raw, fresh, delay, echo_seed)
            record.update(kernel_id=kernel_id, model_seed=model_seed)
            payload['runs'].append(record)
            payload['wall_seconds'] = time_counter() - clock
            dump(out / 'results.json', payload)
            print(
                f'done {label} fresh={record["fresh"]:.6f} copy={record["copy"]:.6f} '
                f'seconds={record["wall_seconds"]:.2f}',
                flush=True,
            )
    payload['wall_seconds'] = time_counter() - clock
    payload['finished'] = True
    dump(out / 'results.json', payload)
    print(f'kernel_done {kernel_id} seconds={payload["wall_seconds"]:.2f}', flush=True)
    return payload


def mean_std(values):
    values = [v for v in values if v is not None]
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(array.mean()), std=float(array.std(ddof=0)), n=int(array.size))


def fmt(stats, digits=6):
    if not stats or stats['mean'] is None:
        return 'NA'
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def cell(value, digits=6):
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def index_runs(runs):
    h20 = [r for r in runs if r['stream'] == 'h20_slow']
    echo = [r for r in runs if r['stream'] == 'echo']
    return h20, echo


def pick(runs, **wanted):
    found = []
    for run in runs:
        if all(run.get(key) == value for key, value in wanted.items()):
            found.append(run)
    if len(found) != 1:
        raise RuntimeError(f'expected one run for {wanted}, found {len(found)}')
    return found[0]


def gap_block(per_seed):
    """per_seed: list of dicts with ring, bank, gru."""
    rows = []
    for row in per_seed:
        ring, bank, gru = row['ring'], row['bank'], row['gru']
        if None in (ring, bank, gru):
            rows.append(dict(row, ring_minus_bank=None, min_bank_ring_minus_gru=None))
        else:
            rows.append(dict(
                row,
                ring_minus_bank=float(ring - bank),
                min_bank_ring_minus_gru=float(min(bank, ring) - gru),
            ))
    return dict(
        per_seed=rows,
        ring_minus_bank=mean_std([r['ring_minus_bank'] for r in rows]),
        min_bank_ring_minus_gru=mean_std([r['min_bank_ring_minus_gru'] for r in rows]),
    )


def assemble(smoke_payload, kernel_payloads, out_dir):
    runs = []
    kernel_ids = []
    kernel_wall = {}
    for payload in kernel_payloads:
        if not payload.get('finished'):
            raise RuntimeError(f'kernel {payload.get("kernel_id")} did not finish')
        kernel_ids.append(payload['kernel_id'])
        kernel_wall[payload['kernel_id']] = payload.get('wall_seconds')
        runs.extend(payload['runs'])
    h20, echo = index_runs(runs)
    summary = dict(h20={}, echo={}, gaps={})
    for name in MODELS:
        group = [pick(h20, model=name, world_seed=seed) for seed in (6, 7, 8)]
        summary['h20'][name] = dict(
            bits_per_symbol=mean_std([g['bits_per_symbol'] for g in group]),
            by_process={
                proc: mean_std([g['by_process'][proc] for g in group]) for proc in PROCS
            },
            c1_c7=mean_std([g['c1_c7'] for g in group]),
            per_seed=[
                dict(
                    world_seed=g['world_seed'],
                    model_seed=g['model_seed'],
                    bits_per_symbol=g['bits_per_symbol'],
                    by_process=g['by_process'],
                    process_counts=g['process_counts'],
                    c1_c7=g['c1_c7'],
                    c1_c7_count=g['c1_c7_count'],
                    wall_seconds=g['wall_seconds'],
                    parameters=g['parameters'],
                    kernel_id=g['kernel_id'],
                )
                for g in group
            ],
        )
        if name == 'ring':
            summary['h20'][name]['mag_a'] = [g['mag_a'] for g in group]
            summary['h20'][name]['mag_b'] = [g['mag_b'] for g in group]
    c1_rows = []
    for seed in (6, 7, 8):
        c1_rows.append(dict(
            world_seed=seed,
            ring=pick(h20, model='ring', world_seed=seed)['c1_c7'],
            bank=pick(h20, model='bank', world_seed=seed)['c1_c7'],
            gru=pick(h20, model='gru', world_seed=seed)['c1_c7'],
        ))
    summary['gaps']['h20_c1_c7'] = gap_block(c1_rows)
    for delay in ECHO_DS:
        summary['echo'][str(delay)] = {}
        for name in MODELS:
            group = [pick(echo, model=name, D=delay, seed=seed) for seed in (0, 1, 2)]
            summary['echo'][str(delay)][name] = dict(
                fresh=mean_std([g['fresh'] for g in group]),
                copy=mean_std([g['copy'] for g in group]),
                bits_per_symbol=mean_std([g['bits_per_symbol'] for g in group]),
                per_seed=[
                    dict(
                        seed=g['seed'],
                        model_seed=g['model_seed'],
                        fresh=g['fresh'],
                        copy=g['copy'],
                        fresh_count=g['fresh_count'],
                        copy_count=g['copy_count'],
                        bits_per_symbol=g['bits_per_symbol'],
                        wall_seconds=g['wall_seconds'],
                        parameters=g['parameters'],
                        kernel_id=g['kernel_id'],
                    )
                    for g in group
                ],
            )
            if name == 'ring':
                summary['echo'][str(delay)][name]['mag_a'] = [g['mag_a'] for g in group]
                summary['echo'][str(delay)][name]['mag_b'] = [g['mag_b'] for g in group]
        copy_rows = []
        for seed in (0, 1, 2):
            copy_rows.append(dict(
                seed=seed,
                D=delay,
                ring=pick(echo, model='ring', D=delay, seed=seed)['copy'],
                bank=pick(echo, model='bank', D=delay, seed=seed)['copy'],
                gru=pick(echo, model='gru', D=delay, seed=seed)['copy'],
            ))
        summary['gaps'][f'echo_copy_D{delay}'] = gap_block(copy_rows)
    parameters = {}
    for run in runs:
        parameters.setdefault(run['model'], {})[run['alphabet']] = run['parameters']
    result = dict(
        principle='2: improve the compressor',
        n_h20=H20_N,
        n_echo=ECHO_N,
        chunk=CHUNK,
        lr=LR,
        clip=CLIP,
        embedding=EMB,
        units=UNITS,
        gru_hidden=HIDDEN,
        order2_rank=RANK,
        std='population ddof=0',
        parameters=parameters,
        kernel_ids=kernel_ids,
        kernel_wall_seconds=kernel_wall,
        smoke=smoke_payload,
        summary=summary,
        runs=runs,
    )
    out_dir = Path(out_dir)
    dump(out_dir / 'results.json', result)
    (out_dir / 'RESULTS.md').write_text(render(result))
    return result


def render(result):
    lines = [
        '# H30w measurements',
        '',
        f"H20 slow n = {result['n_h20']}. Echo n = {result['n_echo']}. Chunk = {result['chunk']}. "
        f"Adam lr = {result['lr']}. Clip = {result['clip']}. Embedding = {result['embedding']}. "
        f"Bank and ring units = {result['units']}. GRU hidden = {result['gru_hidden']}. "
        f"Order-2 maps 512 to {result['order2_rank']}. Population std, ddof = 0.",
        '',
        'Protocol: score each chunk under current parameters (−log2 p), one Adam step on the mean '
        'cross-entropy of that chunk, carry the state forward detached. No replay and no store. '
        'Input is the previous symbol. First input embedding is zero. Float32. '
        'Bank: λ = exp(−exp(ν) + iθ), |λ| initial uniform on [0.9, 0.999], θ uniform on [0, π]. '
        'Ring: ρ = exp(−exp(ν)), g = sigmoid(γ), a = ρ g exp(iθ), b = ρ (1−g) exp(iφ), '
        'so |a| + |b| = ρ < 1. ρ initial uniform on [0.9, 0.999], g uniform on [0.2, 0.8], '
        'θ and φ uniform on [0, π]. '
        'B real and imag entries initial Normal(0, 1/sqrt(32)). Bank and ring readouts have no bias. '
        'GRU is torch.nn.GRU hidden 166 plus a linear readout, pytorch default initialisation, biases on. '
        'H20 world seed 6, 7, 8 uses model seed 0, 1, 2. Echo uses the stream seed as the model seed.',
        '',
        '## Parameter counts',
        '',
        '| Model | Alphabet | Parameters |',
        '|---|---:|---:|',
    ]
    for name in MODELS:
        for alphabet in sorted(result['parameters'][name]):
            lines.append(f"| {name} | {alphabet} | {result['parameters'][name][alphabet]} |")
    lines += ['', '## H20 slow, bits/symbol', '',
              '| Model | mean ± std | seed 6 | seed 7 | seed 8 |',
              '|---|---:|---:|---:|---:|']
    summary = result['summary']
    for name in MODELS:
        seeds = summary['h20'][name]['per_seed']
        cells = ' | '.join(cell(row['bits_per_symbol']) for row in seeds)
        lines.append(f"| {name} | {fmt(summary['h20'][name]['bits_per_symbol'])} | {cells} |")
    lines += ['', '## H20 per process, bits/symbol, mean ± std over seeds', '',
              '| Model | ' + ' | '.join(PROCS) + ' |',
              '|---|' + '---:|' * len(PROCS)]
    for name in MODELS:
        cells = ' | '.join(fmt(summary['h20'][name]['by_process'][proc]) for proc in PROCS)
        lines.append(f'| {name} | {cells} |')
    lines += ['', '## H20 per seed per process, bits/symbol', '',
              '| Model | Seed | ' + ' | '.join(PROCS) + ' |',
              '|---|---:|' + '---:|' * len(PROCS)]
    for name in MODELS:
        for row in summary['h20'][name]['per_seed']:
            cells = ' | '.join(cell(row['by_process'][proc]) for proc in PROCS)
            lines.append(f"| {name} | {row['world_seed']} | {cells} |")
    lines += ['', '## H20 process counts', '',
              '| Seed | ' + ' | '.join(PROCS) + ' |',
              '|---:|' + '---:|' * len(PROCS)]
    for row in summary['h20']['gru']['per_seed']:
        cells = ' | '.join(str(row['process_counts'][proc]) for proc in PROCS)
        lines.append(f"| {row['world_seed']} | {cells} |")
    lines += ['', '## H20 C1+C7 combined, bits/symbol', '',
              '| Model | mean ± std | seed 6 | seed 7 | seed 8 | count 6 | count 7 | count 8 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        rows = summary['h20'][name]['per_seed']
        cells = ' | '.join(cell(row['c1_c7']) for row in rows)
        counts = ' | '.join(str(row['c1_c7_count']) for row in rows)
        lines.append(f"| {name} | {fmt(summary['h20'][name]['c1_c7'])} | {cells} | {counts} |")
    gap = summary['gaps']['h20_c1_c7']
    lines += ['', '## Gaps on H20 C1+C7 combined', '',
              '| Seed | ring | bank | gru | ring − bank | min(bank, ring) − gru |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in gap['per_seed']:
        lines.append(
            f"| {row['world_seed']} | {cell(row['ring'])} | {cell(row['bank'])} | {cell(row['gru'])} | "
            f"{cell(row['ring_minus_bank'])} | {cell(row['min_bank_ring_minus_gru'])} |"
        )
    lines.append(
        f"| mean ± std |  |  |  | {fmt(gap['ring_minus_bank'])} | {fmt(gap['min_bank_ring_minus_gru'])} |"
    )
    for delay in ECHO_DS:
        block = summary['echo'][str(delay)]
        lines += ['', f'## Echo D = {delay}, bits/symbol', '',
                  '| Model | half | mean ± std | seed 0 | seed 1 | seed 2 |',
                  '|---|---|---:|---:|---:|---:|']
        for name in MODELS:
            for half in ('fresh', 'copy'):
                cells = ' | '.join(cell(row[half]) for row in block[name]['per_seed'])
                lines.append(f"| {name} | {half} | {fmt(block[name][half])} | {cells} |")
        lines += ['', f'## Echo D = {delay}, totals, bits/symbol', '',
                  '| Model | mean ± std | seed 0 | seed 1 | seed 2 |',
                  '|---|---:|---:|---:|---:|']
        for name in MODELS:
            cells = ' | '.join(cell(row['bits_per_symbol']) for row in block[name]['per_seed'])
            lines.append(f"| {name} | {fmt(block[name]['bits_per_symbol'])} | {cells} |")
        egap = summary['gaps'][f'echo_copy_D{delay}']
        lines += ['', f'## Gaps on echo D = {delay} copy halves', '',
                  '| Seed | ring | bank | gru | ring − bank | min(bank, ring) − gru |',
                  '|---:|---:|---:|---:|---:|---:|']
        for row in egap['per_seed']:
            lines.append(
                f"| {row['seed']} | {cell(row['ring'])} | {cell(row['bank'])} | {cell(row['gru'])} | "
                f"{cell(row['ring_minus_bank'])} | {cell(row['min_bank_ring_minus_gru'])} |"
            )
        lines.append(
            f"| mean ± std |  |  |  | {fmt(egap['ring_minus_bank'])} | {fmt(egap['min_bank_ring_minus_gru'])} |"
        )
    lines += ['', '## Ring |a| and |b| histograms', '',
              'Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are the learned magnitudes at the end of the run.',
              '',
              '| Run | quantity | ' + ' | '.join(f'[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})' for i in range(9))
              + f' | [{BIN_EDGES[9]:.1f},{BIN_EDGES[10]:.1f}] | min | max | mean |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in result['runs']:
        if run['model'] != 'ring':
            continue
        if run['stream'] == 'h20_slow':
            tag = f"h20 seed {run['world_seed']}"
        else:
            tag = f"echo D={run['D']} seed {run['seed']}"
        for key in ('mag_a', 'mag_b'):
            hist = run[key]
            counts = ' | '.join(str(c) for c in hist['counts'])
            lines.append(
                f"| {tag} | |{hist['name']}| | {counts} | {hist['min']:.6f} | {hist['max']:.6f} | {hist['mean']:.6f} |"
            )
    lines += ['', '## Wall seconds', '',
              '| Kernel | Stream | Seed | Model | Seconds |',
              '|---|---|---:|---|---:|']
    for run in result['runs']:
        if run['stream'] == 'h20_slow':
            lines.append(
                f"| {run['kernel_id']} | h20_slow | {run['world_seed']} | {run['model']} | {run['wall_seconds']:.3f} |"
            )
        else:
            lines.append(
                f"| {run['kernel_id']} | echo D={run['D']} | {run['seed']} | {run['model']} | {run['wall_seconds']:.3f} |"
            )
    lines += ['', '| Kernel | Total seconds |', '|---|---:|']
    for kernel_id in result['kernel_ids']:
        lines.append(f"| {kernel_id} | {result['kernel_wall_seconds'][kernel_id]:.3f} |")
    lines += ['', '## Kernel ids', '']
    lines += [f'- {kernel_id}' for kernel_id in result['kernel_ids']]
    lines += ['', '## Smoke test', '']
    lines += ['```']
    lines += result['smoke']['log']
    lines += ['```', '']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--out', default='results/h30w')
    parser.add_argument('--kernel-dir', action='append', default=[])
    args = parser.parse_args()
    if args.smoke:
        smoke(args.out)
        return
    if args.assemble:
        out = Path(args.out)
        smoke_payload = json.loads((out / 'smoke.json').read_text())
        payloads = [json.loads((Path(path) / 'results.json').read_text()) for path in args.kernel_dir]
        assemble(smoke_payload, payloads, out)
        return
    raise SystemExit('pass --smoke or --assemble')


if __name__ == '__main__':
    main()
