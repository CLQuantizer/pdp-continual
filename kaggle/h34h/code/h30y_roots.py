"""H30y: diagonal bank initialised on the roots of unity. Numbers only. No verdict.

All three arms call h30x_trace.train_trace. trace_rand is the H30x bank.
roots_learn and roots_frozen start at θ_k = 2πk/256 and |λ| = 0.98.
roots_frozen does not step ν or θ.
"""
import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn

from h20_world import ALPHA, PROCS, stream
from h30w_ring import (
    CHUNK,
    CLIP,
    EMB,
    LR,
    RANK,
    UNITS,
    build,
    count_params,
    echo_stream,
    encode_text,
)
from h30x_trace import echo_record as _echo_record_h30x
from h30x_trace import h20_record, lam_hist, run_arm as h30x_run_arm, train_trace

ECHO_DS = (32, 128, 512)
H20_N = 262_144
ECHO_N = 131_072
SMOKE_N = 4_096
TIME_N = 8_192
DELAY_N = 600
KERNEL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
# Amended delay bar: relative error ≤ 0.02. Alias weight is 0.98**256 ≈ 0.006.
DELAY_SMALL = 0.02
BIN_EDGES = [i / 10 for i in range(11)]
ARMS = ('trace_rand', 'roots_learn', 'roots_frozen')
ROOTS_ARMS = ('roots_learn', 'roots_frozen')
DELAYS = (1, 32, 128, 255, 256)
RHO = 0.98


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


def time_counter():
    return __import__('time').perf_counter()


def apply_roots(model):
    """θ_k = 2πk/N, ν = log(−log 0.98), so |λ_k| = exp(−exp(ν))."""
    with torch.no_grad():
        k = torch.arange(UNITS, dtype=torch.float64)
        theta = (2.0 * math.pi) * k / float(UNITS)
        nu = math.log(-math.log(RHO))
        model.theta.copy_(theta.to(dtype=model.theta.dtype))
        model.nu.fill_(nu)
    return model


def build_arm(arm, alphabet, model_seed):
    if arm not in ARMS:
        raise ValueError(arm)
    model = build('bank', alphabet, model_seed)
    if model.nu.dtype != torch.float32 or model.theta.dtype != torch.float32:
        raise RuntimeError(f'bank dtype is {model.nu.dtype}, expected float32')
    if arm == 'trace_rand':
        return model
    apply_roots(model)
    if arm == 'roots_frozen':
        model.nu.requires_grad_(False)
        model.theta.requires_grad_(False)
    return model


@contextmanager
def learning_parameters_only():
    """train_trace steps every parameter that has a gradient.

    It writes ν.grad and θ.grad directly. Adam applies those even when
    requires_grad is false, and the clip norm would include them. Skip both.
    """
    original_adam = torch.optim.Adam
    original_clip = nn.utils.clip_grad_norm_

    def adam_learnable(params, lr):
        chosen = [p for p in params if p.requires_grad]
        if not chosen:
            raise RuntimeError('Adam received no learnable parameters')
        return original_adam(chosen, lr=lr)

    def clip_learnable(parameters, max_norm, *args, **kwargs):
        chosen = [p for p in parameters if p.requires_grad]
        return original_clip(chosen, max_norm, *args, **kwargs)

    torch.optim.Adam = adam_learnable
    nn.utils.clip_grad_norm_ = clip_learnable
    try:
        yield
    finally:
        torch.optim.Adam = original_adam
        nn.utils.clip_grad_norm_ = original_clip


def train_arm(arm, model, tokens, label, progress_every):
    if arm == 'roots_frozen':
        with learning_parameters_only():
            return train_trace(model, tokens, label, progress_every)
    if arm in ('trace_rand', 'roots_learn'):
        return train_trace(model, tokens, label, progress_every)
    raise ValueError(arm)


def run_arm(arm, alphabet, tokens, model_seed, label, progress_every):
    model = build_arm(arm, alphabet, model_seed)
    init_abs_min = None
    init_abs_max = None
    theta0 = None
    nu0 = None
    if arm in ROOTS_ARMS:
        with torch.no_grad():
            mag = model.lam().abs()
            init_abs_min = float(mag.min())
            init_abs_max = float(mag.max())
            theta0 = model.theta.detach().clone()
            nu0 = model.nu.detach().clone()
    parameters = count_params(model)
    bits, seconds = train_arm(arm, model, tokens, label, progress_every)
    hist = lam_hist(model)
    theta_mae = None
    if theta0 is not None:
        with torch.no_grad():
            dtheta = (model.theta.detach() - theta0).abs()
            dnu = (model.nu.detach() - nu0).abs()
            theta_mae = float(dtheta.mean())
            if arm == 'roots_frozen' and (float(dnu.max()) != 0.0 or float(dtheta.max()) != 0.0):
                raise RuntimeError(
                    f'frozen λ moved: max |Δν|={float(dnu.max())} max |Δθ|={float(dtheta.max())}'
                )
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
        lambda_frozen=arm == 'roots_frozen',
        init_abs_lam_min=init_abs_min,
        init_abs_lam_max=init_abs_max,
        theta_abs_mean_delta=theta_mae,
        lam=hist,
        bits=bits,
    )
    del model
    return record


def echo_record(raw, fresh, delay, seed):
    record = _echo_record_h30x(raw, fresh, delay, seed)
    bits = raw['bits']
    n = int(bits.shape[0])
    quarter = (3 * n) // 4
    fresh_q = fresh[quarter:]
    record['fresh_last_quarter'] = float(bits[quarter:][fresh_q].mean()) if int(fresh_q.sum()) else None
    record['fresh_last_quarter_count'] = int(fresh_q.sum())
    return record


def check_echo(tokens, fresh, delay):
    period = 2 * delay
    if tokens.shape[0] < period:
        raise RuntimeError('echo stream shorter than one period')
    copy_matches = bool(np.array_equal(tokens[:delay], tokens[delay:period]))
    aligned = bool(fresh[:delay].all()) and not bool(fresh[delay:period].any())
    balanced = int(fresh.sum()) * 2 == int(tokens.shape[0])
    if not (copy_matches and aligned and balanced):
        raise RuntimeError(f'echo D={delay} shape check failed')


def rel_l2(est, true):
    est = np.asarray(est, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    denom = float(np.linalg.norm(true))
    if denom == 0.0 or not np.isfinite(denom):
        raise RuntimeError('delay target norm is zero or nonfinite')
    err = float(np.linalg.norm(est - true) / denom)
    if not np.isfinite(err):
        raise RuntimeError('delay relative error is nonfinite')
    return err


def delay_check():
    """Hand-set B, roots λ frozen. Not used by any scored run."""
    alphabet = 16
    model = build('bank', alphabet, 0)
    apply_roots(model)
    generator = torch.Generator()
    generator.manual_seed(12345)
    c = torch.randn(EMB, generator=generator, dtype=model.B_re.dtype)
    rng = np.random.default_rng(0)
    symbols = torch.as_tensor(rng.integers(0, alphabet, size=DELAY_N), dtype=torch.long)
    with torch.no_grad():
        model.B_re.copy_(c.unsqueeze(0).expand(UNITS, EMB).contiguous())
        model.B_im.zero_()
        emb = model.embedding(symbols)
        injected = model.inject(emb).detach().cpu().numpy().astype(np.complex128)
        lam = model.lam().detach().cpu().numpy().astype(np.complex128)
        mag = np.abs(lam)
    spread = float(np.max(np.abs(injected - injected[:, :1])))
    s = injected[:, 0].real.copy()
    imag_inj = float(np.linalg.norm(injected[:, 0].imag) / np.linalg.norm(s))
    h = np.zeros(UNITS, dtype=np.complex128)
    states = np.empty((DELAY_N, UNITS), dtype=np.complex128)
    for t in range(DELAY_N):
        h = lam * h + injected[t]
        states[t] = h
    rows = []
    est = {}
    for delay in DELAYS:
        weight = lam ** (-delay) / float(UNITS)
        recon = (states * weight[None, :]).sum(axis=1)
        real = recon.real
        est[delay] = real
        full = rel_l2(real[delay:], s[:DELAY_N - delay])
        hi = min(DELAY_N, delay + UNITS)
        first = None
        if delay < UNITS and hi > delay:
            first = rel_l2(real[delay:hi], s[:hi - delay])
        imag_rel = float(np.linalg.norm(recon.imag) / np.linalg.norm(s))
        rows.append(dict(
            d=int(delay),
            rel=full,
            first_period_rel=first,
            imag_rel=imag_rel,
        ))
    scale = lam ** (-UNITS)
    scale_mean = complex(np.mean(scale))
    scale_spread = float(np.max(np.abs(scale - scale_mean)) / max(abs(scale_mean), 1e-30))
    weight0 = np.ones(UNITS, dtype=np.complex128) / float(UNITS)
    est0 = (states * weight0[None, :]).sum(axis=1)
    alias_rel = rel_l2(est[UNITS], (scale_mean * est0).real)
    rho_n = float(RHO ** UNITS)
    full_small = all(row['rel'] <= DELAY_SMALL for row in rows if row['d'] < UNITS)
    alias_holds = alias_rel <= DELAY_SMALL and rows[-1]['rel'] > DELAY_SMALL
    ok = bool(full_small and alias_holds and spread <= 1e-5 and imag_inj <= 1e-5)
    return dict(
        ok=ok,
        n=DELAY_N,
        rho=RHO,
        rho_N=rho_n,
        small_limit=DELAY_SMALL,
        inject_mode_spread=spread,
        inject_imag_rel=imag_inj,
        abs_lam_min=float(mag.min()),
        abs_lam_max=float(mag.max()),
        scale_spread=scale_spread,
        alias_with_lag0_rel=alias_rel,
        full_stream_small=bool(full_small),
        alias_with_lag0=bool(alias_holds),
        rows=rows,
        c_seed=12345,
        symbol_seed=0,
    )


def match_check():
    tokens, _fresh = echo_stream(0, 32, SMOKE_N)
    ref = h30x_run_arm('trace', 16, tokens, 0, 'h30x_trace_echo_D32', progress_every=10_000)
    arm = run_arm('trace_rand', 16, tokens, 0, 'trace_rand_echo_D32', progress_every=10_000)
    delta = float(np.max(np.abs(ref['bits'] - arm['bits'])))
    ok = bool(np.isfinite(delta) and delta <= BITS_LIMIT)
    return dict(
        ok=ok,
        stream='echo',
        D=32,
        seed=0,
        n=SMOKE_N,
        h30x_bits=float(ref['bits_per_symbol']),
        trace_rand_bits=float(arm['bits_per_symbol']),
        max_abs_delta=delta,
    )


def finite_runs():
    lines = []
    init_printed = set()
    for delay in ECHO_DS:
        tokens, fresh = echo_stream(0, delay, SMOKE_N)
        check_echo(tokens, fresh, delay)
        for arm in ARMS:
            raw = run_arm(arm, 16, tokens, 0, f'smoke_echo_D{delay}_{arm}', progress_every=10_000)
            finite = bool(np.isfinite(raw['bits']).all())
            if arm in ROOTS_ARMS and arm not in init_printed:
                lines.append(
                    f'roots_init arm={arm} abs_lam_min={raw["init_abs_lam_min"]:.8f} '
                    f'abs_lam_max={raw["init_abs_lam_max"]:.8f}'
                )
                init_printed.add(arm)
            elif arm in ROOTS_ARMS:
                lines.append(
                    f'roots_init arm={arm} stream=echo D={delay} '
                    f'abs_lam_max={raw["init_abs_lam_max"]:.8f}'
                )
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
        if arm in ROOTS_ARMS:
            lines.append(
                f'roots_init arm={arm} stream=h20_slow abs_lam_max={raw["init_abs_lam_max"]:.8f}'
            )
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
    body = ['# H30y measurements', '']
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
    matched = None
    delay = None
    seconds = None
    per_arm = None
    total = None
    try:
        log(
            f'torch={torch.__version__} threads={torch.get_num_threads()} '
            f'train_dtype=float32 trace_impl=h30x_trace.train_trace'
        )
        log(f'embedding_frozen=False roots_abs_lam={RHO} roots_theta=2*pi*k/256')
        matched = match_check()
        log(
            f'trace_rand_match stream=echo D=32 seed=0 n={SMOKE_N} '
            f'h30x_bits={matched["h30x_bits"]:.8f} trace_rand_bits={matched["trace_rand_bits"]:.8f} '
            f'max_abs_delta={matched["max_abs_delta"]:.3e} limit={BITS_LIMIT:.1e} ok={matched["ok"]}'
        )
        if not matched['ok']:
            reason = (
                f'trace_rand does not match h30x trace: max_abs_delta={matched["max_abs_delta"]:.3e}'
            )
        if not reason:
            delay = delay_check()
            log(
                f'delay_setup n={delay["n"]} rho={delay["rho"]} rho_N={delay["rho_N"]:.6f} '
                f'abs_lam_min={delay["abs_lam_min"]:.8f} abs_lam_max={delay["abs_lam_max"]:.8f} '
                f'inject_mode_spread={delay["inject_mode_spread"]:.3e} '
                f'inject_imag_rel={delay["inject_imag_rel"]:.3e}'
            )
            for row in delay['rows']:
                first = 'NA' if row['first_period_rel'] is None else f'{row["first_period_rel"]:.3e}'
                log(
                    f'delay d={row["d"]} rel={row["rel"]:.6e} first_period_rel={first} '
                    f'imag_rel={row["imag_rel"]:.3e}'
                )
            log(
                f'delay_alias d=256 alias_with_lag0_rel={delay["alias_with_lag0_rel"]:.6e} '
                f'scale_spread={delay["scale_spread"]:.3e} '
                f'alias_with_lag0={delay["alias_with_lag0"]}'
            )
            log(
                f'delay_summary small_limit={delay["small_limit"]:.1e} '
                f'full_stream_small={delay["full_stream_small"]} '
                f'alias_with_lag0={delay["alias_with_lag0"]} ok={delay["ok"]}'
            )
            if not delay['ok']:
                shown = ' '.join(
                    f'd={row["d"]}:{row["rel"]:.3e}' for row in delay['rows']
                )
                reason = (
                    'delay check failed: full-stream relative errors are not small for every lag '
                    f'below 256 ({shown}); small_limit={DELAY_SMALL:.1e}; '
                    f'alias_with_lag0={delay["alias_with_lag0"]} '
                    f'alias_rel={delay["alias_with_lag0_rel"]:.3e}'
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
        embedding_frozen=False,
        trace_impl='h30x_trace.train_trace',
        stopped=bool(reason),
        reason=reason,
        kernels_pushed=False,
        trace_rand_match=matched,
        delay_check=delay,
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
        check_echo(echo_tokens, fresh, delay)
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
                if record['c1_c7'] is None:
                    raise RuntimeError(f'{label} has no C1 or C7 symbols')
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
    if sorted(payload['seed_index'] for payload in kernel_payloads) != [0, 1, 2]:
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
                    theta_abs_mean_delta=row.get('theta_abs_mean_delta'),
                    init_abs_lam_max=row.get('init_abs_lam_max'),
                )
                for row in group
            ],
        )
    c1_rows = []
    for seed in (6, 7, 8):
        base = pick(h20, arm='trace_rand', world_seed=seed)['c1_c7']
        learned = pick(h20, arm='roots_learn', world_seed=seed)['c1_c7']
        frozen = pick(h20, arm='roots_frozen', world_seed=seed)['c1_c7']
        c1_rows.append(dict(
            world_seed=seed,
            trace_rand=base,
            roots_learn=learned,
            roots_frozen=frozen,
            roots_learn_minus_trace_rand=float(learned - base),
            roots_frozen_minus_trace_rand=float(frozen - base),
        ))
    summary['gaps']['h20_c1_c7'] = dict(
        per_seed=c1_rows,
        roots_learn_minus_trace_rand=mean_std(
            [row['roots_learn_minus_trace_rand'] for row in c1_rows]
        ),
        roots_frozen_minus_trace_rand=mean_std(
            [row['roots_frozen_minus_trace_rand'] for row in c1_rows]
        ),
    )
    for delay in ECHO_DS:
        summary['echo'][str(delay)] = {}
        for arm in ARMS:
            group = [pick(echo, arm=arm, D=delay, seed=seed) for seed in (0, 1, 2)]
            summary['echo'][str(delay)][arm] = dict(
                fresh=mean_std([row['fresh'] for row in group]),
                copy=mean_std([row['copy'] for row in group]),
                fresh_last_quarter=mean_std([row['fresh_last_quarter'] for row in group]),
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
                        fresh_last_quarter=row['fresh_last_quarter'],
                        fresh_last_quarter_count=row['fresh_last_quarter_count'],
                        copy_last_quarter=row['copy_last_quarter'],
                        copy_last_quarter_count=row['copy_last_quarter_count'],
                        bits_per_symbol=row['bits_per_symbol'],
                        wall_seconds=row['wall_seconds'],
                        parameters=row['parameters'],
                        kernel_id=row['kernel_id'],
                        lam=row['lam'],
                        theta_abs_mean_delta=row.get('theta_abs_mean_delta'),
                        init_abs_lam_max=row.get('init_abs_lam_max'),
                    )
                    for row in group
                ],
            )
        copy_rows = []
        for seed in (0, 1, 2):
            base = pick(echo, arm='trace_rand', D=delay, seed=seed)['copy']
            learned = pick(echo, arm='roots_learn', D=delay, seed=seed)['copy']
            frozen = pick(echo, arm='roots_frozen', D=delay, seed=seed)['copy']
            copy_rows.append(dict(
                seed=seed,
                D=delay,
                trace_rand=base,
                roots_learn=learned,
                roots_frozen=frozen,
                roots_learn_minus_trace_rand=float(learned - base),
                roots_frozen_minus_trace_rand=float(frozen - base),
            ))
        summary['gaps'][f'echo_copy_D{delay}'] = dict(
            per_seed=copy_rows,
            roots_learn_minus_trace_rand=mean_std(
                [row['roots_learn_minus_trace_rand'] for row in copy_rows]
            ),
            roots_frozen_minus_trace_rand=mean_std(
                [row['roots_frozen_minus_trace_rand'] for row in copy_rows]
            ),
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
        roots_abs_lam=RHO,
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
        arm_i = ARMS.index(run['arm'])
        return (stream_i, delay, seed, arm_i)

    return sorted(runs, key=key)


def render(result):
    lines = [
        '# H30y measurements',
        '',
        f"H20 slow n = {result['n_h20']}. Echo n = {result['n_echo']}, D in {result['echo_D']}. "
        f"Chunk = {result['chunk']}. Adam lr = {result['lr']}. Clip = {result['clip']}. "
        f"Embedding = {result['embedding']}. Units = {result['units']}. "
        f"Order-2 maps 512 to {result['order2_rank']}. Population std, ddof = 0. "
        f"Embedding frozen = {str(result['embedding_frozen']).lower()}.",
        '',
        'Protocol: the trace rule is h30x_trace.train_trace, unchanged. '
        'The bank is h30w_ring.Bank, N = 256, h_t = λ ⊙ h_{t−1} + B x_t, '
        'λ = exp(−exp(ν) + iθ). Order-2 readout with no bias: W, P1, P2, V. '
        'Input is the previous symbol. The first input embedding is zero. Float32. '
        'Chunks of 64, score before the Adam step, lr 1e-3, clip 1.0. '
        'trace_rand uses the H30x init: |λ| uniform on [0.9, 0.999], θ uniform on [0, π]. '
        'roots_learn and roots_frozen use θ_k = 2πk/256 and |λ_k| = 0.98, '
        'ν_k = log(−log 0.98). roots_learn steps ν and θ. '
        'roots_frozen does not step ν or θ; B, the readout and the embedding do. '
        'H20 world seeds 6, 7, 8 use model seeds 0, 1, 2. Echo uses the stream seed as the model seed. '
        'Echo uniform rate is 4 bits on both halves. '
        'Last quarter is symbols from index 3n/4 to the end.',
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
    for row in summary['h20']['trace_rand']['per_seed']:
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
        '## Gaps on H20 C1+C7 combined, roots − trace_rand',
        '',
        '| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for row in gap['per_seed']:
        lines.append(
            f"| {row['world_seed']} | {cell(row['roots_learn'])} | {cell(row['roots_frozen'])} | "
            f"{cell(row['trace_rand'])} | {cell(row['roots_learn_minus_trace_rand'])} | "
            f"{cell(row['roots_frozen_minus_trace_rand'])} |"
        )
    lines.append(
        f"| mean ± std |  |  |  | {fmt(gap['roots_learn_minus_trace_rand'])} | "
        f"{fmt(gap['roots_frozen_minus_trace_rand'])} |"
    )
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
            f'## Echo D = {delay}, last quarter, bits/symbol',
            '',
            '| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 | count |',
            '|---|---|---:|---:|---:|---:|---:|',
        ]
        for arm in ARMS:
            rows = block[arm]['per_seed']
            for half, count_key in (
                ('fresh', 'fresh_last_quarter_count'),
                ('copy', 'copy_last_quarter_count'),
            ):
                cells = ' | '.join(cell(row[f'{half}_last_quarter']) for row in rows)
                lines.append(
                    f"| {arm} | {half} | {fmt(block[arm][f'{half}_last_quarter'])} | "
                    f"{cells} | {rows[0][count_key]} |"
                )
        egap = summary['gaps'][f'echo_copy_D{delay}']
        lines += [
            '',
            f'## Gaps on echo D = {delay} copy halves, roots − trace_rand',
            '',
            '| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |',
            '|---:|---:|---:|---:|---:|---:|',
        ]
        for row in egap['per_seed']:
            lines.append(
                f"| {row['seed']} | {cell(row['roots_learn'])} | {cell(row['roots_frozen'])} | "
                f"{cell(row['trace_rand'])} | {cell(row['roots_learn_minus_trace_rand'])} | "
                f"{cell(row['roots_frozen_minus_trace_rand'])} |"
            )
        lines.append(
            f"| mean ± std |  |  |  | {fmt(egap['roots_learn_minus_trace_rand'])} | "
            f"{fmt(egap['roots_frozen_minus_trace_rand'])} |"
        )
    lines += [
        '',
        '## roots_learn |λ|',
        '',
        'Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are |λ| at the end of the run. '
        'n(|λ| > 0.99) is a separate count, not an extra bin. '
        'mean |Δθ| is the mean absolute change in θ from the roots init.',
        '',
        '| Run | ' + ' | '.join(f'[{BIN_EDGES[i]:.1f},{BIN_EDGES[i + 1]:.1f})' for i in range(9))
        + f' | [{BIN_EDGES[9]:.1f},{BIN_EDGES[10]:.1f}] | n(|λ|>0.99) | min | max | mean | mean |Δθ| |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for run in _ordered(result['runs']):
        if run['arm'] != 'roots_learn':
            continue
        if run['stream'] == 'h20_slow':
            tag = f"h20 seed {run['world_seed']}"
        else:
            tag = f"echo D={run['D']} seed {run['seed']}"
        hist = run['lam']
        counts = ' | '.join(str(c) for c in hist['counts'])
        lines.append(
            f"| {tag} | {counts} | {hist['n_gt_0_99']} | {hist['min']:.6f} | "
            f"{hist['max']:.6f} | {hist['mean']:.6f} | {run['theta_abs_mean_delta']:.6f} |"
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
        f"Train dtype = {smoke.get('dtype')}. Trace = {smoke.get('trace_impl')}.",
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
    parser.add_argument('--out', default='results/h30y')
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
