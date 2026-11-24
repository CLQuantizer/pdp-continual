"""H34i: improve the living compressor; unchanged local depth on new streams.

Bind H34g's exact training/checkpoint loop without mutating imported modules.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import types
import numpy as np
import torch
import h34b_plain_depth as b
import h34f_zero_gate as f
import h34g_depth as g
import h34h_dense as h

ROOT = Path(__file__).resolve().parent
ARMS = ('plain_1', 'plain_6', 'zgate_6', 'dense_6', 'dense_6_ones')
STREAMS = ('c4', 'bible', 'mnist')
N_DATA, TAIL_START, CHUNK, EMB, RANK = g.N_DATA, g.TAIL_START, g.CHUNK, g.EMB, g.RANK
h33, dump = g.h33, g.dump


def sha(data):
    return hashlib.sha256(data).hexdigest()


def config(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    depth = int(arm.split('_')[1])
    ones = arm.startswith('plain') or arm.endswith('_ones')
    return dict(depth=depth, dense=arm.startswith('dense'), n=128, rank=RANK,
                emb=EMB, chunk=CHUNK, dtype='float32', learning='trace', loss='own',
                optimizer='Adam', lr=1e-3, clip=1., normalization='per-step',
                mixer_initial=[1.] * depth if ones else [1.] + [0.] * (depth-1))


def build(arm, seed):
    torch.manual_seed(seed)
    return [g.Layer(128, 0 if k == 0 else 256+RANK, True, True)
            for k in range(config(arm)['depth'])]


def initial_gate(arm):
    return torch.tensor([config(arm)['mixer_initial']], dtype=torch.float32)


def source_hashes():
    return {'h34i_generality.py': sha(Path(__file__).read_bytes()), **h.source_hashes()}


def run(arm, seed, raw, keep=False, slice_size=20_000, **kwargs):
    config(arm)
    bindings = dict(g.run.__globals__, build=build, initial_gate=initial_gate,
                    source_hashes=source_hashes)
    runner = types.FunctionType(g.run.__code__, bindings, g.run.__name__, g.run.__defaults__)
    runner.__kwdefaults__ = g.run.__kwdefaults__
    return runner(arm, seed, raw, keep, slice_size, **kwargs)


def prepare_data():
    """Only local reads. Keep original source provenance and exact stream hashes."""
    dest = ROOT/'data/h34i'
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    for stream, source in (('c4', 'data/h35c/c4_val.bin'), ('bible', 'data/pg10.txt'),
                           ('mnist', 'data/train-images-idx3-ubyte.gz')):
        content = (ROOT/source).read_bytes()
        if stream == 'mnist':
            decoded = gzip.decompress(content)
            header = struct.unpack('>4I', decoded[:16])
            assert header == (2051, 60000, 28, 28)
            assert len(decoded) == 16 + 60000*784
            raw = decoded[16:16+N_DATA]
        else:
            raw = content[:N_DATA]
        assert len(raw) == N_DATA
        row = dict(source=source, source_sha256=sha(content), sha256=sha(raw), n_bytes=len(raw),
                   file=stream+'.bin', start=0, end=N_DATA)
        if stream == 'mnist':
            row.update(idx_header=list(header), skip_header_bytes=16, complete_images=229,
                       final_image_bytes=464, order='image index, row, column')
        if stream == 'c4':
            manifest = json.loads((ROOT/'data/h35c/c4_val_manifest.json').read_text())
            assert manifest['sha256'] == sha(content) and manifest['n_bytes'] == len(content)
            row['source_manifest'] = manifest
        (dest/row['file']).write_bytes(raw)
        rows[stream] = row
    dump(dest/'manifest.json', rows)
    return rows


def load_stream(stream, folder=None, manifest=None):
    folder = Path(folder) if folder is not None else ROOT/'data/h34i'
    manifest = manifest or json.loads((folder/'manifest.json').read_text())
    content = (folder/manifest[stream]['file']).read_bytes()
    assert len(content) == N_DATA == manifest[stream]['n_bytes']
    assert sha(content) == manifest[stream]['sha256']
    return np.frombuffer(content, dtype=np.uint8).copy()


def smoke(out, enwik8):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, n_bytes=N_DATA, threads=torch.get_num_threads(),
                   source_hashes=source_hashes(), streams=prepare_data(), log=[])
    def record(check, **values):
        row = dict(check=check, **values)
        payload['log'].append(row)
        print(json.dumps(row), flush=True)
        dump(out/'smoke.json', payload)
    try:
        assert torch.get_num_threads() == 1
        raw, digest = h33.load_checked(enwik8)
        payload['sha256_enwik8'] = digest
        timings = {a: [] for a in ARMS}
        references = {'plain_1': (b, f), 'plain_6': (b, f), 'zgate_6': (f, g),
                      'dense_6': (g, h), 'dense_6_ones': (h,)}
        for seed in (0, 1):
            for arm in ARMS:
                r, bits = run(arm, seed, raw[:8192], keep=True)
                assert np.isfinite(bits).all()
                timings[arm].append(r['wall_seconds']/8192)
                for reference in references[arm]:
                    ref, refbits = reference.run(arm, seed, raw[:8192], keep=True)
                    assert np.array_equal(bits, refbits)
                    record('enwik8_equivalence', arm=arm, seed=seed, reference=reference.__name__,
                           n_bytes=8192, max_bits_abs=0., bits_per_byte=r['bits_per_byte'],
                           reference_bits=ref['bits_per_byte'], wall_seconds=r['wall_seconds'])
        for stream in STREAMS:
            new = load_stream(stream)
            if stream == 'mnist':
                preview = [' '.join(f'{v:02x}' for v in row) for row in new[:784].reshape(28, 28)]
            else:
                preview = ''.join(chr(v) if 32 <= v <= 126 else '.' for v in new[:200])
            record('stream_identity', stream=stream, **payload['streams'][stream], preview=preview)
            for seed in (0, 1):
                r, bits = run('plain_1', seed, new[:8192], keep=True)
                assert np.isfinite(bits).all()
                timings['plain_1'].append(r['wall_seconds']/8192)
                record('new_stream_plain_1', stream=stream, seed=seed, n_bytes=8192,
                       bits_per_byte=r['bits_per_byte'], wall_seconds=r['wall_seconds'])
        # All arms: prefix invariant under length extension and changed future bytes.
        new = load_stream('c4')
        for arm in ARMS:
            for seed in (0, 1):
                _, short = run(arm, seed, new[:1024], keep=True)
                _, longer = run(arm, seed, new[:2048], keep=True)
                altered = new[:2048].copy(); altered[1024:] = 255-altered[1024:]
                _, changed = run(arm, seed, altered, keep=True)
                assert np.array_equal(short, longer[:1024]) and np.array_equal(short, changed[:1024])
                record('prequential_prefix', stream='c4', arm=arm, seed=seed, n_bytes=1024,
                       max_bits_abs=0., changed_suffix_max_bits_abs=0.)
        # Exercise aligned and actual 90 KB-style unaligned boundaries for every arm.
        for arm in ARMS:
            for seed in (0, 1):
                whole, split = out/'whole.pt', out/'split.pt'
                r, bits = run(arm, seed, new[:2048], keep=True, checkpoint=whole)
                for boundary in (1024, 1040):
                    part = run(arm, seed, new[:2048], stop_at=boundary, checkpoint=split)
                    resumed, rb = run(arm, seed, new[:2048], keep=True, resume=split, checkpoint=split)
                    assert np.array_equal(bits, rb)
                    strip = lambda row: {k:v for k,v in row.items() if 'wall' not in k}
                    assert h.equal(strip(r), strip(resumed))
                    a, c = [torch.load(p, weights_only=False) for p in (whole, split)]
                    # Uninitialized trailing arrays in intermediate checkpoints are never scored.
                    assert h.equal(strip(a), strip(c))
                    record('checkpoint_equivalence', stream='c4', arm=arm, seed=seed, n_bytes=2048,
                           split_at=boundary, committed_bytes=part['offset'],
                           pending_bytes=part['pending_bytes'], max_bits_abs=0., state_exact=True)
                whole.unlink(); split.unlink()
        estimates = {a:max(ts)*N_DATA for a, ts in timings.items()}
        split_arms = [a for a, seconds in estimates.items() if seconds > 5100]
        segments = {a:v/(2 if a in split_arms else 1) for a,v in estimates.items()}
        payload['split_arms'] = split_arms
        record('wall_estimates', estimated_kernel_seconds=estimates, estimated_segment_seconds=segments,
               split_arms=split_arms, limit=5100, n_bytes=N_DATA)
        assert max(segments.values()) <= 5100
        payload['ok'] = True
        dump(out/'smoke.json', payload)
    except BaseException as exc:
        payload['failure'] = repr(exc); dump(out/'smoke.json', payload)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-data', action='store_true')
    p.add_argument('--arm', choices=ARMS); p.add_argument('--stream', choices=STREAMS)
    p.add_argument('--seed', type=int, choices=(0, 1), default=0)
    p.add_argument('--enwik8', default='data/enwik8/enwik8')
    p.add_argument('--out', default='results/h34i')
    p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--stop-at', type=int)
    args = p.parse_args(); h33.configure(1)
    if args.smoke:
        return smoke(args.out, args.enwik8)
    if args.prepare_data:
        prepare_data(); return
    if args.arm is None or args.stream is None:
        p.error('--arm and --stream required')
    sm = json.loads((Path(args.out)/'smoke.json').read_text())
    assert sm['ok'] and sm['source_hashes'] == source_hashes()
    raw = load_stream(args.stream, manifest=sm['streams'])
    r = run(args.arm, args.seed, raw, checkpoint=args.checkpoint, resume=args.resume, stop_at=args.stop_at)
    r.update(stream=args.stream, raw_sha256=sm['streams'][args.stream]['sha256'])
    dump(Path(args.out)/f'{args.stream}-{args.arm}-s{args.seed}.json', r)


if __name__ == '__main__':
    main()
