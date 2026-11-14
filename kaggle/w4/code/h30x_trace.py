"""H30x: diagonal bank, eligibility traces vs BPTT-64. Numbers only. No verdict.

The bank is h30w_ring.Bank. bptt64 calls h30w_ring.train. The trace arm uses the
same chunks, Adam, and score-before-step order. Its recurrent gradient is the
exact diagonal eligibility trace (no BPTT). Embedding is trained in both arms.
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
from h30w_ring import (
    CHUNK,
    CLIP,
    EMB,
    LR,
    RANK,
    UNITS,
    build,
    chunk_embedding,
    count_params,
    echo_stream,
    encode_text,
    train,
)

ECHO_DS = (32, 128, 512)
H20_N = 262_144
ECHO_N = 131_072
SMOKE_N = 4_096
TIME_N = 8_192
GRAD_N = 300
KERNEL_LIMIT_S = 90 * 60
BIN_EDGES = [i / 10 for i in range(11)]
ARMS = ('bptt64', 'trace')
LOG2 = math.log(2.0)
REL_LIMIT = 1e-4
BITS_LIMIT = 1e-6
SPEC_GRADS = ('nu', 'theta', 'B_re', 'B_im', 'W.weight', 'P1.weight', 'P2.weight', 'V.weight')


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


def cdtype_of(model):
    if model.nu.dtype == torch.float64:
        return torch.complex128
    if model.nu.dtype == torch.float32:
        return torch.complex64
    raise RuntimeError(f'unsupported dtype {model.nu.dtype}')


def init_state(model):
    cdtype = cdtype_of(model)
    alphabet = int(model.embedding.num_embeddings)
    h = torch.zeros(UNITS, dtype=cdtype)
    e_lam = torch.zeros(UNITS, dtype=cdtype)
    e_b = torch.zeros(UNITS, EMB, dtype=cdtype)
    z = torch.zeros(UNITS, alphabet, dtype=cdtype)
    return (h, e_lam, e_b, z)


def _finite_complex(tensor):
    return bool(torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all())


def trace_chunk(model, tokens_t, start, end, state, mean):
    """Advance one chunk. Grads are the sum of per-step losses, or that sum / length.

    Eligibility, carried in `state` and never reset by the caller:
      e^λ_t = λ e^λ_{t-1} + h_{t-1}
      E^B_t = λ E^B_{t-1} + x_t
      z_{a,t} = λ z_{a,t-1} + 1{symbol_{t-1} = a}   (no addition at t = 0, where x = 0)
    δ = ∂L/∂u + i ∂L/∂v from the readout at h_t only. For a holomorphic sensitivity
    e, dL/dp = Re(conj(δ) e). λ = exp(−exp(ν) + iθ), so
    ∂h/∂ν = e^λ (−exp(ν) λ) and ∂h/∂θ = e^λ (i λ).
    ∂h_i/∂Emb[a, j] = B_ij z_{i,a}.
    """
    h, e_lam, e_b, z = state
    if h.dtype != cdtype_of(model):
        raise RuntimeError('trace state dtype does not match the model')
    length = end - start
    if length <= 0:
        raise RuntimeError('empty chunk')
    emb = chunk_embedding(model, tokens_t, start, end).detach()
    cdtype = h.dtype
    with torch.no_grad():
        lam = model.lam().detach()
        injection_b = torch.complex(model.B_re.detach(), model.B_im.detach())
        inj = model.inject(emb)
        dlam_dnu = -torch.exp(model.nu.detach()) * lam
        dlam_dth = 1j * lam
        h_buf = torch.empty(length, UNITS, dtype=cdtype)
        e_buf = torch.empty(length, UNITS, dtype=cdtype)
        b_buf = torch.empty(length, UNITS, EMB, dtype=cdtype)
        z_buf = torch.empty(length, UNITS, z.shape[1], dtype=cdtype)
        lam_col = lam[:, None]
        for t in range(length):
            e_lam.mul_(lam).add_(h)
            e_b.mul_(lam_col).add_(emb[t])
            z.mul_(lam_col)
            global_t = start + t
            if global_t > 0:
                z[:, int(tokens_t[global_t - 1])] += 1
            h.mul_(lam).add_(inj[t])
            h_buf[t].copy_(h)
            e_buf[t].copy_(e_lam)
            b_buf[t].copy_(e_b)
            z_buf[t].copy_(z)
    hr = h_buf.real.detach().requires_grad_(True)
    hi = h_buf.imag.detach().requires_grad_(True)
    logits = model.logits_from(torch.complex(hr, hi))
    losses = F.cross_entropy(logits, tokens_t[start:end], reduction='none')
    model.zero_grad(set_to_none=True)
    losses.sum().backward()
    if hr.grad is None or hi.grad is None:
        raise RuntimeError('readout did not produce a hidden-state gradient')
    delta = torch.complex(hr.grad.detach(), hi.grad.detach())
    with torch.no_grad():
        c = torch.conj(delta)
        model.nu.grad = (c * (e_buf * dlam_dnu)).real.sum(dim=0)
        model.theta.grad = (c * (e_buf * dlam_dth)).real.sum(dim=0)
        cc = c[:, :, None]
        model.B_re.grad = (cc * b_buf).real.sum(dim=0)
        model.B_im.grad = (cc * (1j * b_buf)).real.sum(dim=0)
        coeff = c[:, :, None] * z_buf
        model.embedding.weight.grad = torch.einsum('tia,ie->ae', coeff, injection_b).real
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f'{name} has no gradient')
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError(f'{name} gradient is nonfinite')
        if mean:
            scale = 1.0 / float(length)
            for param in model.parameters():
                param.grad.mul_(scale)
    for tensor in (h, e_lam, e_b, z):
        if tensor.requires_grad or not _finite_complex(tensor):
            raise FloatingPointError('trace state is nonfinite or still on the graph')
    return losses.detach(), (h, e_lam, e_b, z)


def train_trace(model, tokens, label, progress_every):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    n = int(tokens_t.shape[0])
    bits = np.empty(n, dtype=np.float64)
    state = init_state(model)
    clock = time_counter()
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        losses, state = trace_chunk(model, tokens_t, start, end, state, mean=True)
        scored = losses.double().numpy() / LOG2
        if not np.all(np.isfinite(scored)):
            raise FloatingPointError(f'{label}: nonfinite bits at {start}')
        bits[start:end] = scored
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        done = (start // CHUNK + 1)
        if done % progress_every == 0 or end == n:
            print(
                f'{label} n={end}/{n} bits={bits[:end].mean():.6f} seconds={time_counter() - clock:.2f}',
                flush=True,
            )
    return bits, time_counter() - clock


def train_bptt64(model, tokens, label, progress_every):
    return train(model, tokens, label, progress_every)


def time_counter():
    return __import__('time').perf_counter()


def lam_hist(model):
    with torch.no_grad():
        mag = model.lam().abs().detach().double().cpu().numpy()
    if not np.all(np.isfinite(mag)):
        raise FloatingPointError('|λ| is nonfinite')
    counts, edges = np.histogram(mag, bins=BIN_EDGES)
    outside = int(np.sum((mag < 0.0) | (mag > 1.0)))
    if int(counts.sum()) + outside != int(mag.size):
        raise RuntimeError('histogram count mismatch for |λ|')
    return dict(
        counts=[int(c) for c in counts],
        bin_edges=[float(e) for e in edges],
        n=int(mag.size),
        n_gt_0_99=int(np.sum(mag > 0.99)),
        n_outside=outside,
        min=float(mag.min()),
        max=float(mag.max()),
        mean=float(mag.mean()),
    )


def run_arm(arm, alphabet, tokens, model_seed, label, progress_every):
    if arm not in ARMS:
        raise ValueError(arm)
    model = build('bank', alphabet, model_seed)
    parameters = count_params(model)
    if arm == 'bptt64':
        bits, seconds = train_bptt64(model, tokens, label, progress_every)
    else:
        bits, seconds = train_trace(model, tokens, label, progress_every)
    hist = lam_hist(model)
    record = dict(
        arm=arm,
        model='bank',
        alphabet=int(alphabet),
        model_seed=int(model_seed),
        parameters=int(parameters),
        n=int(len(tokens)),
        wall_seconds=float(seconds),
        bits_per_symbol=float(bits.mean()),
        embedding_frozen=False,
        lam=hist,
        bits=bits,
    )
    del model
    return record


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
    n = int(raw['bits'].shape[0])
    quarter = (3 * n) // 4
    copy_q = copy[quarter:]
    record = {k: v for k, v in raw.items() if k != 'bits'}
    record.update(
        stream='echo',
        D=int(delay),
        seed=int(seed),
        fresh=float(raw['bits'][fresh].mean()) if int(fresh.sum()) else None,
        copy=float(raw['bits'][copy].mean()) if int(copy.sum()) else None,
        fresh_count=int(fresh.sum()),
        copy_count=int(copy.sum()),
        copy_last_quarter=float(raw['bits'][quarter:][copy_q].mean()) if int(copy_q.sum()) else None,
        copy_last_quarter_count=int(copy_q.sum()),
        last_quarter_start=int(quarter),
    )
    return record


def grad_pair(name, trace_g, bptt_g):
    diff = (trace_g - bptt_g).abs()
    max_abs = float(diff.max())
    bptt_max = float(bptt_g.abs().max())
    denom = max(bptt_max, 1e-30)
    rel = max_abs / denom
    elem = diff / bptt_g.abs().clamp(min=1e-12)
    return dict(
        name=name,
        bptt_norm=float(bptt_g.norm()),
        trace_norm=float(trace_g.norm()),
        max_abs=max_abs,
        max_rel=float(rel),
        max_elem_rel=float(elem.max()),
    )


def bptt_sum_grads(model, tokens):
    """Summed cross-entropy over the whole sequence. No detach, no chunking."""
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    n = int(tokens_t.shape[0])
    emb = chunk_embedding(model, tokens_t, 0, n)
    lam = model.lam()
    inj = model.inject(emb)
    h = emb.new_zeros(UNITS, dtype=cdtype_of(model))
    hs = []
    for t in range(n):
        h = lam * h + inj[t]
        hs.append(h)
    logits = model.logits_from(torch.stack(hs))
    loss = F.cross_entropy(logits, tokens_t, reduction='sum')
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    return float(loss.detach()), grads


def trace_sum_grads(model, tokens):
    """Same summed loss, via the chunked trace rule with parameters frozen."""
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    n = int(tokens_t.shape[0])
    state = init_state(model)
    acc = None
    total = 0.0
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        losses, state = trace_chunk(model, tokens_t, start, end, state, mean=False)
        total += float(losses.sum())
        if acc is None:
            acc = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
        else:
            for name, param in model.named_parameters():
                acc[name] = acc[name] + param.grad.detach()
    return total, acc


def gradient_check():
    tokens, _fresh = echo_stream(0, 128, GRAD_N)
    model = build('bank', 16, 0)
    model.double()
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loss_b, grads_b = bptt_sum_grads(model, tokens)
    model.load_state_dict(saved)
    loss_t, grads_t = trace_sum_grads(model, tokens)
    rows = []
    missing = [name for name in list(SPEC_GRADS) + ['embedding.weight'] if name not in grads_b]
    if missing:
        raise RuntimeError(f'gradient check missing {missing}')
    for name in list(SPEC_GRADS) + ['embedding.weight']:
        rows.append(grad_pair(name, grads_t[name], grads_b[name]))
    spec = [row for row in rows if row['name'] in SPEC_GRADS]
    max_rel = max(row['max_rel'] for row in spec)
    max_elem = max(row['max_elem_rel'] for row in spec)
    emb = next(row for row in rows if row['name'] == 'embedding.weight')
    loss_abs = abs(loss_b - loss_t)
    ok = max_rel <= REL_LIMIT and max_elem <= REL_LIMIT and emb['max_rel'] <= REL_LIMIT and loss_abs <= 1e-8
    return dict(
        ok=bool(ok),
        stream='echo',
        D=128,
        seed=0,
        n=GRAD_N,
        chunk=CHUNK,
        dtype='float64',
        loss_bptt=loss_b,
        loss_trace=loss_t,
        loss_abs=float(loss_abs),
        max_rel_spec=float(max_rel),
        max_elem_rel_spec=float(max_elem),
        embedding_max_rel=float(emb['max_rel']),
        rows=rows,
    )


def echo_shape_check():
    lines = []
    ok = True
    for delay in ECHO_DS:
        tokens, fresh = echo_stream(0, delay, ECHO_N)
        period = 2 * delay
        copy_matches = bool(np.array_equal(tokens[:delay], tokens[delay:period]))
        aligned = bool(fresh[:delay].all()) and not bool(fresh[delay:period].any())
        balanced = int(fresh.sum()) == ECHO_N // 2
        quarter = (3 * ECHO_N) // 4
        quarter_copy = int((~fresh[quarter:]).sum())
        expected_quarter = ECHO_N // 8
        this = copy_matches and aligned and balanced and quarter_copy == expected_quarter
        ok = ok and this
        lines.append(
            f'echo_check D={delay} n={ECHO_N} fresh_count={int(fresh.sum())} '
            f'copy_matches={copy_matches} aligned={aligned} '
            f'last_quarter_copy_count={quarter_copy} ok={this}'
        )
    return ok, lines


def bptt_match_check():
    tokens, _fresh = echo_stream(0, 32, SMOKE_N)
    ref = run_arm('bptt64', 16, tokens, 0, 'h30w_bank_echo_D32', progress_every=10_000)
    arm = run_arm('bptt64', 16, tokens, 0, 'bptt64_echo_D32', progress_every=10_000)
    delta = float(np.max(np.abs(ref['bits'] - arm['bits'])))
    mean_delta = abs(float(ref['bits_per_symbol'] - arm['bits_per_symbol']))
    ok = delta <= BITS_LIMIT
    return dict(
        ok=bool(ok),
        stream='echo',
        D=32,
        seed=0,
        n=SMOKE_N,
        h30w_bits=float(ref['bits_per_symbol']),
        bptt64_bits=float(arm['bits_per_symbol']),
        max_abs_delta=delta,
        mean_abs_delta=float(mean_delta),
        fresh_h30w=float(ref['bits'][_fresh].mean()),
        copy_h30w=float(ref['bits'][~_fresh].mean()),
    )


def finite_runs():
    lines = []
    for delay in ECHO_DS:
        tokens, fresh = echo_stream(0, delay, SMOKE_N)
        for arm in ARMS:
            raw = run_arm(arm, 16, tokens, 0, f'smoke_echo_D{delay}_{arm}', progress_every=10_000)
            finite = bool(np.isfinite(raw['bits']).all())
            lines.append(
                f'smoke_finite stream=echo D={delay} seed=0 arm={arm} n={SMOKE_N} '
                f'parameters={raw["parameters"]} bits={raw["bits_per_symbol"]:.6f} '
                f'fresh={float(raw["bits"][fresh].mean()):.6f} '
                f'copy={float(raw["bits"][~fresh].mean()):.6f} '
                f'wall_seconds={raw["wall_seconds"]:.3f} finite={finite}'
            )
            if not finite:
                raise FloatingPointError(f'{arm} echo D={delay}')
    text, labels = stream('slow', 6, SMOKE_N)
    tokens = encode_text(text)
    procs = np.asarray(labels)
    if len(tokens) != SMOKE_N or len(procs) != SMOKE_N:
        raise RuntimeError('H20 smoke stream length')
    text2, labels2 = stream('slow', 6, SMOKE_N)
    if text2 != text or labels2 != labels:
        raise RuntimeError('H20 stream is not deterministic')
    lines.append(f'h20_repro seed=6 n={SMOKE_N} match=True')
    for arm in ARMS:
        raw = run_arm(arm, len(ALPHA), tokens, 0, f'smoke_h20_{arm}', progress_every=10_000)
        finite = bool(np.isfinite(raw['bits']).all())
        record = h20_record(raw, procs)
        c1 = 'NA' if record['c1_c7'] is None else f'{record["c1_c7"]:.6f}'
        lines.append(
            f'smoke_finite stream=h20_slow seed=6 arm={arm} n={SMOKE_N} '
            f'parameters={raw["parameters"]} bits={raw["bits_per_symbol"]:.6f} '
            f'c1_c7={c1} c1_c7_count={record["c1_c7_count"]} '
            f'wall_seconds={raw["wall_seconds"]:.3f} finite={finite}'
        )
        if not finite:
            raise FloatingPointError(f'{arm} h20')
    return lines


def time_arms():
    text, _labels = stream('slow', 6, TIME_N)
    tokens = encode_text(text)
    lines = []
    seconds = {}
    for arm in ARMS:
        raw = run_arm(arm, len(ALPHA), tokens, 0, f'time_{arm}', progress_every=10_000)
        if not np.isfinite(raw['bits']).all():
            raise FloatingPointError(f'timing run nonfinite: {arm}')
        seconds[arm] = float(raw['wall_seconds'])
        lines.append(
            f'time_8192 arm={arm} stream=h20_slow seed=6 seconds={seconds[arm]:.3f} '
            f'bits={raw["bits_per_symbol"]:.6f} threads={torch.get_num_threads()}'
        )
    symbols_one_arm = H20_N + ECHO_N * len(ECHO_DS)
    scale = symbols_one_arm / TIME_N
    per_arm = {arm: seconds[arm] * scale for arm in ARMS}
    total = float(sum(per_arm.values()))
    lines.append(
        f'kernel_estimate symbols_per_arm={symbols_one_arm} scale={scale:.6f} '
        + ' '.join(f'{arm}_seconds={per_arm[arm]:.1f}' for arm in ARMS)
        + f' total_seconds={total:.1f} limit_seconds={KERNEL_LIMIT_S}'
    )
    return seconds, per_arm, total, lines


def write_smoke_files(out_dir, payload):
    dump(out_dir / 'smoke.json', payload)
    dump(out_dir / 'results.json', payload)
    (out_dir / 'smoke.log').write_text('\n'.join(payload['log']) + '\n')
    body = ['# H30x measurements', '']
    if payload.get('stopped'):
        body.append(f"Stopped before kernels. {payload.get('reason', '')}")
    else:
        body.append('Smoke checks met. Scored kernels are separate from this file until assemble.')
    body.append('')
    body += ['## Smoke test', '', '```', *payload['log'], '```', '']
    (out_dir / 'RESULTS.md').write_text('\n'.join(body) + '\n')


def smoke(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    reason = ''
    grad = None
    match = None
    seconds = None
    per_arm = None
    total = None
    try:
        log(f'torch={torch.__version__} threads={torch.get_num_threads()} train_dtype=float32 grad_check_dtype=float64')
        log('embedding_frozen=False bptt64_impl=h30w_ring.train')
        for alphabet, tag in ((len(ALPHA), 'h20'), (16, 'echo')):
            model = build('bank', alphabet, 0)
            n_params = count_params(model)
            with torch.no_grad():
                mag = torch.exp(-torch.exp(model.nu))
            line = (
                f'parameters arm=bank alphabet={alphabet} stream={tag} count={n_params} '
                f'abs_lam_min={float(mag.min()):.6f} abs_lam_max={float(mag.max()):.6f}'
            )
            log(line)
            del model
        ok, shape_lines = echo_shape_check()
        for line in shape_lines:
            log(line)
        if not ok:
            reason = 'echo stream check failed'
        if not reason:
            grad = gradient_check()
            log(
                f'trace_grad_check D=128 seed=0 n={GRAD_N} dtype=float64 chunk={CHUNK} '
                f'loss_bptt={grad["loss_bptt"]:.8f} loss_trace={grad["loss_trace"]:.8f} '
                f'loss_abs={grad["loss_abs"]:.3e}'
            )
            for row in grad['rows']:
                log(
                    f'trace_grad name={row["name"]} bptt_norm={row["bptt_norm"]:.8e} '
                    f'trace_norm={row["trace_norm"]:.8e} max_abs={row["max_abs"]:.8e} '
                    f'max_rel={row["max_rel"]:.8e} max_elem_rel={row["max_elem_rel"]:.8e}'
                )
            log(
                f'trace_grad_summary max_rel_spec={grad["max_rel_spec"]:.8e} '
                f'max_elem_rel_spec={grad["max_elem_rel_spec"]:.8e} '
                f'embedding_max_rel={grad["embedding_max_rel"]:.8e} '
                f'limit={REL_LIMIT:.1e} ok={grad["ok"]}'
            )
            if not grad['ok']:
                reason = (
                    f'trace gradient check failed: max_rel={grad["max_rel_spec"]:.3e} '
                    f'elem={grad["max_elem_rel_spec"]:.3e} embedding={grad["embedding_max_rel"]:.3e} '
                    f'loss_abs={grad["loss_abs"]:.3e}'
                )
        if not reason:
            match = bptt_match_check()
            log(
                f'bptt64_match stream=echo D=32 seed=0 n={SMOKE_N} '
                f'h30w_bits={match["h30w_bits"]:.8f} bptt64_bits={match["bptt64_bits"]:.8f} '
                f'max_abs_delta={match["max_abs_delta"]:.3e} mean_abs_delta={match["mean_abs_delta"]:.3e} '
                f'limit={BITS_LIMIT:.1e} ok={match["ok"]}'
            )
            if not match['ok']:
                reason = (
                    f'bptt64 does not match h30w bank bits: max_abs_delta={match["max_abs_delta"]:.3e}'
                )
        if not reason:
            for line in finite_runs():
                log(line)
        if not reason:
            seconds, per_arm, total, time_lines = time_arms()
            for line in time_lines:
                log(line)
            if total > KERNEL_LIMIT_S:
                reason = f'kernel estimate {total:.1f}s exceeds {KERNEL_LIMIT_S}s'
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        log(f'exception {reason}')
    payload = dict(
        torch=torch.__version__,
        threads=int(torch.get_num_threads()),
        dtype='float32',
        grad_check_dtype='float64',
        embedding_frozen=False,
        bptt64_impl='h30w_ring.train',
        stopped=bool(reason),
        reason=reason,
        kernels_pushed=False,
        grad_check=grad,
        bptt_match=match,
        time_8192_seconds=seconds,
        kernel_estimate_seconds_per_arm=per_arm,
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
        embedding_frozen=False,
        dtype='float32',
        runs=[],
        finished=False,
    )
    dump(out / 'results.json', payload)
    text, labels = stream('slow', world_seed, H20_N)
    tokens = encode_text(text)
    procs = np.asarray(labels)
    if len(tokens) != H20_N or len(procs) != H20_N:
        raise RuntimeError('H20 length')
    jobs = [('h20_slow', None, tokens, procs)]
    for delay in ECHO_DS:
        echo_tokens, fresh = echo_stream(echo_seed, delay, ECHO_N)
        jobs.append(('echo', delay, echo_tokens, fresh))
    for stream_name, delay, stream_tokens, extra in jobs:
        alphabet = len(ALPHA) if stream_name == 'h20_slow' else 16
        for arm in ARMS:
            if stream_name == 'h20_slow':
                label = f'{kernel_id} h20_s{world_seed} {arm}'
            else:
                label = f'{kernel_id} echo_D{delay}_s{echo_seed} {arm}'
            raw = run_arm(arm, alphabet, stream_tokens, model_seed, label, progress_every=256)
            if stream_name == 'h20_slow':
                record = h20_record(raw, extra)
                record.update(kernel_id=kernel_id, world_seed=world_seed, model_seed=model_seed)
            else:
                record = echo_record(raw, extra, delay, echo_seed)
                record.update(kernel_id=kernel_id, model_seed=model_seed)
            payload['runs'].append(record)
            payload['wall_seconds'] = time_counter() - clock
            dump(out / 'results.json', payload)
            if stream_name == 'h20_slow':
                print(
                    f'done {label} bits={record["bits_per_symbol"]:.6f} c1_c7={record["c1_c7"]:.6f} '
                    f'seconds={record["wall_seconds"]:.2f}',
                    flush=True,
                )
            else:
                print(
                    f'done {label} fresh={record["fresh"]:.6f} copy={record["copy"]:.6f} '
                    f'copy_last_quarter={record["copy_last_quarter"]:.6f} '
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
    if not stats or stats.get('mean') is None:
        return 'NA'
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def cell(value, digits=6):
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def pick(runs, **wanted):
    found = [run for run in runs if all(run.get(key) == value for key, value in wanted.items())]
    if len(found) != 1:
        raise RuntimeError(f'expected one run for {wanted}, found {len(found)}')
    return found[0]


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
    if [payload['seed_index'] for payload in kernel_payloads] != [0, 1, 2] and sorted(
        payload['seed_index'] for payload in kernel_payloads
    ) != [0, 1, 2]:
        raise RuntimeError('expected seed indexes 0, 1, 2')
    h20 = [run for run in runs if run['stream'] == 'h20_slow']
    echo = [run for run in runs if run['stream'] == 'echo']
    summary = dict(h20={}, echo={}, gaps={})
    for arm in ARMS:
        group = [pick(h20, arm=arm, world_seed=seed) for seed in (6, 7, 8)]
        summary['h20'][arm] = dict(
            bits_per_symbol=mean_std([row['bits_per_symbol'] for row in group]),
            by_process={
                proc: mean_std([row['by_process'][proc] for row in group]) for proc in PROCS
            },
            c1_c7=mean_std([row['c1_c7'] for row in group]),
            per_seed=[
                dict(
                    world_seed=row['world_seed'],
                    model_seed=row['model_seed'],
                    bits_per_symbol=row['bits_per_symbol'],
                    by_process=row['by_process'],
                    process_counts=row['process_counts'],
                    c1_c7=row['c1_c7'],
                    c1_c7_count=row['c1_c7_count'],
                    wall_seconds=row['wall_seconds'],
                    parameters=row['parameters'],
                    kernel_id=row['kernel_id'],
                    lam=row['lam'],
                )
                for row in group
            ],
        )
    c1_rows = []
    for seed in (6, 7, 8):
        trace_v = pick(h20, arm='trace', world_seed=seed)['c1_c7']
        bptt_v = pick(h20, arm='bptt64', world_seed=seed)['c1_c7']
        c1_rows.append(dict(
            world_seed=seed,
            trace=trace_v,
            bptt64=bptt_v,
            trace_minus_bptt64=float(trace_v - bptt_v),
        ))
    summary['gaps']['h20_c1_c7'] = dict(
        per_seed=c1_rows,
        trace_minus_bptt64=mean_std([row['trace_minus_bptt64'] for row in c1_rows]),
    )
    for delay in ECHO_DS:
        summary['echo'][str(delay)] = {}
        for arm in ARMS:
            group = [pick(echo, arm=arm, D=delay, seed=seed) for seed in (0, 1, 2)]
            summary['echo'][str(delay)][arm] = dict(
                fresh=mean_std([row['fresh'] for row in group]),
                copy=mean_std([row['copy'] for row in group]),
                copy_last_quarter=mean_std([row['copy_last_quarter'] for row in group]),
                bits_per_symbol=mean_std([row['bits_per_symbol'] for row in group]),
                per_seed=[
                    dict(
                        seed=row['seed'],
                        model_seed=row['model_seed'],
                        fresh=row['fresh'],
                        copy=row['copy'],
                        fresh_count=row['fresh_count'],
                        copy_count=row['copy_count'],
                        copy_last_quarter=row['copy_last_quarter'],
                        copy_last_quarter_count=row['copy_last_quarter_count'],
                        bits_per_symbol=row['bits_per_symbol'],
                        wall_seconds=row['wall_seconds'],
                        parameters=row['parameters'],
                        kernel_id=row['kernel_id'],
                        lam=row['lam'],
                    )
                    for row in group
                ],
            )
        copy_rows = []
        for seed in (0, 1, 2):
            trace_v = pick(echo, arm='trace', D=delay, seed=seed)['copy']
            bptt_v = pick(echo, arm='bptt64', D=delay, seed=seed)['copy']
            copy_rows.append(dict(
                seed=seed,
                D=delay,
                trace=trace_v,
                bptt64=bptt_v,
                trace_minus_bptt64=float(trace_v - bptt_v),
            ))
        summary['gaps'][f'echo_copy_D{delay}'] = dict(
            per_seed=copy_rows,
            trace_minus_bptt64=mean_std([row['trace_minus_bptt64'] for row in copy_rows]),
        )
    parameters = {}
    for run in runs:
        parameters.setdefault(str(run['alphabet']), run['parameters'])
    result = dict(
        principle='2: improve the compressor',
        embedding_frozen=False,
        n_h20=H20_N,
        n_echo=ECHO_N,
        echo_D=list(ECHO_DS),
        chunk=CHUNK,
        lr=LR,
        clip=CLIP,
        embedding=EMB,
        units=UNITS,
        order2_rank=RANK,
        std='population ddof=0',
        arms=list(ARMS),
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


def _ordered(runs):
    def key(run):
        stream_i = 0 if run['stream'] == 'h20_slow' else 1
        seed = run.get('world_seed', run.get('seed'))
        delay = run.get('D', 0)
        arm_i = 0 if run['arm'] == 'bptt64' else 1
        return (stream_i, delay, seed, arm_i)

    return sorted(runs, key=key)


def render(result):
    lines = [
        '# H30x measurements',
        '',
        f"H20 slow n = {result['n_h20']}. Echo n = {result['n_echo']}, D in {result['echo_D']}. "
        f"Chunk = {result['chunk']}. Adam lr = {result['lr']}. Clip = {result['clip']}. "
        f"Embedding = {result['embedding']}. Units = {result['units']}. "
        f"Order-2 maps 512 to {result['order2_rank']}. Population std, ddof = 0. "
        f"Embedding frozen = {str(result['embedding_frozen']).lower()}.",
        '',
        'Protocol: one bank, two learning rules. The bank is h30w_ring.Bank, N = 256, '
        'h_t = λ ⊙ h_{t−1} + B x_t, λ = exp(−exp(ν) + iθ). |λ| initial uniform on [0.9, 0.999], '
        'θ uniform on [0, π]. B real and imag entries initial Normal(0, 1/sqrt(32)). '
        'Readout is order-2 with no bias: W, P1, P2, V. Input is the previous symbol. '
        'The first input embedding is zero. Float32. '
        'bptt64 is h30w_ring.train: score a chunk of 64 under the current parameters, '
        'one Adam step on the mean cross-entropy, carry the state forward detached. '
        'trace uses the same chunks, the same Adam, and the same score-before-step order. '
        'Readout gradients are autograd of the chunk losses through W, P1, P2, V with h detached. '
        'Recurrent gradients use complex eligibility traces carried across chunks and never reset: '
        'e^λ_t = λ e^λ_{t−1} + h_{t−1}, E^B_t = λ E^B_{t−1} + x_t, and '
        'z_{a,t} = λ z_{a,t−1} + 1{symbol_{t−1} = a} (no addition on the first step). '
        'δ_t = ∂L_t/∂h_t from the step-t readout. The real parameter gradient is the real part of '
        'conj(δ) times the sensitivity, chained from λ to ν and θ. '
        'The embedding gradient uses z and B. Traces use the current λ and are not recomputed after the Adam step. '
        'Summed chunk gradients are divided by the chunk length. '
        'H20 world seeds 6, 7, 8 use model seeds 0, 1, 2. Echo uses the stream seed as the model seed. '
        'Echo uniform rate is 4 bits on both halves. Copy last quarter is the mean over copy symbols '
        'in the last quarter of the stream.',
        '',
        '## Parameter counts',
        '',
        '| Alphabet | Parameters |',
        '|---:|---:|',
    ]
    for alphabet in sorted(result['parameters'], key=lambda value: int(value)):
        lines.append(f"| {alphabet} | {result['parameters'][alphabet]} |")
    summary = result['summary']
    lines += [
        '',
        '## H20 slow, bits/symbol',
        '',
        '| Arm | mean ± std | seed 6 | seed 7 | seed 8 |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        seeds = summary['h20'][arm]['per_seed']
        cells = ' | '.join(cell(row['bits_per_symbol']) for row in seeds)
        lines.append(f"| {arm} | {fmt(summary['h20'][arm]['bits_per_symbol'])} | {cells} |")
    lines += [
        '',
        '## H20 per process, bits/symbol, mean ± std over seeds',
        '',
        '| Arm | ' + ' | '.join(PROCS) + ' |',
        '|---|' + '---:|' * len(PROCS),
    ]
    for arm in ARMS:
        cells = ' | '.join(fmt(summary['h20'][arm]['by_process'][proc]) for proc in PROCS)
        lines.append(f'| {arm} | {cells} |')
    lines += [
        '',
        '## H20 per seed per process, bits/symbol',
        '',
        '| Arm | Seed | ' + ' | '.join(PROCS) + ' |',
        '|---|---:|' + '---:|' * len(PROCS),
    ]
    for arm in ARMS:
        for row in summary['h20'][arm]['per_seed']:
            cells = ' | '.join(cell(row['by_process'][proc]) for proc in PROCS)
            lines.append(f"| {arm} | {row['world_seed']} | {cells} |")
    lines += [
        '',
        '## H20 process counts',
        '',
        '| Seed | ' + ' | '.join(PROCS) + ' |',
        '|---:|' + '---:|' * len(PROCS),
    ]
    for row in summary['h20']['bptt64']['per_seed']:
        cells = ' | '.join(str(row['process_counts'][proc]) for proc in PROCS)
        lines.append(f"| {row['world_seed']} | {cells} |")
    lines += [
        '',
        '## H20 C1+C7 combined, bits/symbol',
        '',
        '| Arm | mean ± std | seed 6 | seed 7 | seed 8 | count 6 | count 7 | count 8 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        rows = summary['h20'][arm]['per_seed']
        cells = ' | '.join(cell(row['c1_c7']) for row in rows)
        counts = ' | '.join(str(row['c1_c7_count']) for row in rows)
        lines.append(f"| {arm} | {fmt(summary['h20'][arm]['c1_c7'])} | {cells} | {counts} |")
    gap = summary['gaps']['h20_c1_c7']
    lines += [
        '',
        '## Gaps on H20 C1+C7 combined, trace − bptt64',
        '',
        '| Seed | trace | bptt64 | trace − bptt64 |',
        '|---:|---:|---:|---:|',
    ]
    for row in gap['per_seed']:
        lines.append(
            f"| {row['world_seed']} | {cell(row['trace'])} | {cell(row['bptt64'])} | "
            f"{cell(row['trace_minus_bptt64'])} |"
        )
    lines.append(f"| mean ± std |  |  | {fmt(gap['trace_minus_bptt64'])} |")
    for delay in result['echo_D']:
        block = summary['echo'][str(delay)]
        lines += [
            '',
            f'## Echo D = {delay}, bits/symbol',
            '',
            '| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |',
            '|---|---|---:|---:|---:|---:|',
        ]
        for arm in ARMS:
            for half in ('fresh', 'copy'):
                cells = ' | '.join(cell(row[half]) for row in block[arm]['per_seed'])
                lines.append(f"| {arm} | {half} | {fmt(block[arm][half])} | {cells} |")
        lines += [
            '',
            f'## Echo D = {delay}, copy bits in the last quarter of the stream',
            '',
            '| Arm | mean ± std | seed 0 | seed 1 | seed 2 | count |',
            '|---|---:|---:|---:|---:|---:|',
        ]
        for arm in ARMS:
            rows = block[arm]['per_seed']
            cells = ' | '.join(cell(row['copy_last_quarter']) for row in rows)
            count = rows[0]['copy_last_quarter_count']
            lines.append(f"| {arm} | {fmt(block[arm]['copy_last_quarter'])} | {cells} | {count} |")
        lines += [
            '',
            f'## Echo D = {delay}, totals, bits/symbol',
            '',
            '| Arm | mean ± std | seed 0 | seed 1 | seed 2 |',
            '|---|---:|---:|---:|---:|',
        ]
        for arm in ARMS:
            cells = ' | '.join(cell(row['bits_per_symbol']) for row in block[arm]['per_seed'])
            lines.append(f"| {arm} | {fmt(block[arm]['bits_per_symbol'])} | {cells} |")
        egap = summary['gaps'][f'echo_copy_D{delay}']
        lines += [
            '',
            f'## Gaps on echo D = {delay} copy halves, trace − bptt64',
            '',
            '| Seed | trace | bptt64 | trace − bptt64 |',
            '|---:|---:|---:|---:|',
        ]
        for row in egap['per_seed']:
            lines.append(
                f"| {row['seed']} | {cell(row['trace'])} | {cell(row['bptt64'])} | "
                f"{cell(row['trace_minus_bptt64'])} |"
            )
        lines.append(f"| mean ± std |  |  | {fmt(egap['trace_minus_bptt64'])} |")
    lines += [
        '',
        '## Learned |λ|',
        '',
        'Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are |λ| at the end of the run. '
        'n(|λ| > 0.99) is a separate count, not an extra bin.',
        '',
        '| Run | ' + ' | '.join(f'[{BIN_EDGES[i]:.1f},{BIN_EDGES[i + 1]:.1f})' for i in range(9))
        + f' | [{BIN_EDGES[9]:.1f},{BIN_EDGES[10]:.1f}] | n(|λ|>0.99) | min | max | mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for run in _ordered(result['runs']):
        if run['stream'] == 'h20_slow':
            tag = f"h20 seed {run['world_seed']} {run['arm']}"
        else:
            tag = f"echo D={run['D']} seed {run['seed']} {run['arm']}"
        hist = run['lam']
        counts = ' | '.join(str(c) for c in hist['counts'])
        lines.append(
            f"| {tag} | {counts} | {hist['n_gt_0_99']} | {hist['min']:.6f} | "
            f"{hist['max']:.6f} | {hist['mean']:.6f} |"
        )
    lines += [
        '',
        '## Wall seconds',
        '',
        '| Kernel | Stream | Seed | Arm | Seconds |',
        '|---|---|---:|---|---:|',
    ]
    for run in _ordered(result['runs']):
        if run['stream'] == 'h20_slow':
            lines.append(
                f"| {run['kernel_id']} | h20_slow | {run['world_seed']} | {run['arm']} | "
                f"{run['wall_seconds']:.3f} |"
            )
        else:
            lines.append(
                f"| {run['kernel_id']} | echo D={run['D']} | {run['seed']} | {run['arm']} | "
                f"{run['wall_seconds']:.3f} |"
            )
    lines += ['', '| Kernel | Total seconds |', '|---|---:|']
    for kernel_id in result['kernel_ids']:
        lines.append(f"| {kernel_id} | {result['kernel_wall_seconds'][kernel_id]:.3f} |")
    lines += ['', '## Kernel ids', '']
    lines += [f'- {kernel_id}' for kernel_id in result['kernel_ids']]
    smoke = result['smoke']
    lines += [
        '',
        '## Smoke test',
        '',
        f"Embedding frozen = {str(smoke.get('embedding_frozen', False)).lower()}. "
        f"Train dtype = {smoke.get('dtype')}. Gradient check dtype = {smoke.get('grad_check_dtype')}.",
        '',
        '```',
        *smoke['log'],
        '```',
        '',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--out', default='results/h30x')
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
