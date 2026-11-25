"""H34j: improve the living compressor on a longer, unchanged enwik8 stream.

Only binds stream length, tail boundary and the existing arm constructors to
H34g's unchanged checkpointed loop; imported module globals are never mutated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import types
import numpy as np
import torch
import h34i_generality as i
import h34h_dense as h

g = i.g
h33, dump = g.h33, g.dump
ARMS = ('plain_1', 'zgate_6', 'dense_6', 'dense_8')
N_DATA, TAIL_START, SLICE = 1_000_000, 200_000, 100_000
ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {'h34j_long.py': sha(Path(__file__).read_bytes()), **i.source_hashes()}


def config(arm):
    return h.config(arm) if arm == 'dense_8' else i.config(arm)


def build(arm, seed):
    return h.build(arm, seed) if arm == 'dense_8' else i.build(arm, seed)


def initial_gate(arm):
    return h.initial_gate(arm) if arm == 'dense_8' else i.initial_gate(arm)


def run(arm, seed, raw, keep=False, slice_size=SLICE, **kwargs):
    assert arm in ARMS
    bindings = dict(g.run.__globals__, build=build, initial_gate=initial_gate,
                    source_hashes=source_hashes, TAIL_START=TAIL_START)
    runner = types.FunctionType(g.run.__code__, bindings, g.run.__name__, g.run.__defaults__)
    runner.__kwdefaults__ = g.run.__kwdefaults__
    return runner(arm, seed, raw, keep, slice_size, **kwargs)


def strip_wall(value):
    return {k: v for k, v in value.items() if 'wall' not in k}


def smoke(out, data):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw, digest = h33.load_checked(data)
    raw = raw[:N_DATA]; assert len(raw) == N_DATA
    payload = dict(ok=False, n_bytes=N_DATA, tail_start=TAIL_START, threads=torch.get_num_threads(),
                   source_hashes=source_hashes(), sha256_e10m=digest, raw_sha256=sha(raw.tobytes()), log=[])
    def record(check, **values):
        row = dict(check=check, **values); payload['log'].append(row)
        dump(out/'smoke.json', payload); print(json.dumps(row), flush=True)
    timings = {a: [] for a in ARMS}
    try:
        assert torch.get_num_threads() == 1
        for arm in ARMS:
            for seed in (0, 1):
                whole, split = out/'whole.pt', out/'split.pt'
                result, bits = run(arm, seed, raw[:8192], keep=True, checkpoint=whole)
                reference = h if arm.startswith('dense') else i
                ref, rb = reference.run(arm, seed, raw[:8192], keep=True)
                assert np.array_equal(bits, rb)
                record('reference_equivalence', arm=arm, seed=seed, n_bytes=8192,
                       reference=reference.__name__, bits_per_byte=result['bits_per_byte'],
                       reference_bits_per_byte=ref['bits_per_byte'], max_bits_abs=0.)
                part = run(arm, seed, raw[:8192], checkpoint=split, stop_at=4096)
                resumed, bb = run(arm, seed, raw[:8192], keep=True, resume=split, checkpoint=split)
                assert np.array_equal(bits, bb) and h.equal(strip_wall(result), strip_wall(resumed))
                a, b = [torch.load(p, weights_only=False) for p in (whole, split)]
                assert h.equal(strip_wall(a), strip_wall(b))
                record('checkpoint_equivalence', arm=arm, seed=seed, n_bytes=8192,
                       split_at=4096, offset=part['offset'], max_bits_abs=0., state_exact=True,
                       wall_seconds=result['wall_seconds'])
                timings[arm].append(max(result['wall_seconds'], resumed['wall_seconds'])/8192)
                whole.unlink(); split.unlink()
        estimates = {a: max(ts)*N_DATA for a, ts in timings.items()}
        # 25% runtime reserve; aligned boundaries preserve every Adam update.
        counts = {a: max(1, math.ceil(seconds*1.25/4800)) for a, seconds in estimates.items()}
        boundaries = {a: [((N_DATA*k//count)//128)*128 for k in range(1,count)] + [N_DATA]
                      for a, count in counts.items()}
        payload.update(part_counts=counts, boundaries=boundaries)
        record('wall_plan', estimated_full_seconds=estimates, part_counts=counts, boundaries=boundaries,
               estimated_part_seconds={a: estimates[a]/counts[a] for a in ARMS}, limit_seconds=5100,
               runtime_reserve=1.25)
        payload['ok'] = True; dump(out/'smoke.json', payload)
    except BaseException as exc:
        payload['failure'] = repr(exc); dump(out/'smoke.json', payload); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true')
    p.add_argument('--data', default='data/enwik8/enwik8'); p.add_argument('--out', default='results/h34j')
    args = p.parse_args(); h33.configure(1)
    if not args.smoke: p.error('--smoke required; scored runs use staged kernels')
    smoke(args.out, args.data)
