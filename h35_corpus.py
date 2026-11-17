"""H35 corpus: clean enwik8 documents plus five junk kinds.

`build(seed)` is deterministic in the corpus seed. The pool order is the only
shuffle. Labels stay beside the documents for the oracle and for reporting.
Every other policy is meant to ignore them. Does not commit.
"""
import hashlib
from pathlib import Path

import numpy as np

import h26_cued as h26

DOC = 4096
N_CLEAN = 1600
N_EACH_JUNK = 320
N_JUNK_KINDS = 5
N_POOL = N_CLEAN + N_EACH_JUNK * N_JUNK_KINDS
CLEAN_END = N_CLEAN * DOC
JUNK_LO = CLEAN_END
JUNK_HI = 8_900_000
TEMPLATE_LO = 8_900_000
TEMPLATE_HI = 8_900_300
TEMPLATE = TEMPLATE_HI - TEMPLATE_LO
REF_LO = 8_900_300
REF_HI = 9_000_300
TEST_LO = 9_000_300
TEST_HI = 9_262_444
N_TEST = 64
N_BLOCKS = (JUNK_HI - JUNK_LO) // DOC
KINDS = (
    'clean',
    'J1_random',
    'J2_corrupt',
    'J3_dup',
    'J4_boiler',
    'J5_shuffle',
)
DATA_DIR = Path('data/h35')


def _check_spans():
    if CLEAN_END != 6_553_600:
        raise AssertionError(CLEAN_END)
    if N_BLOCKS < N_EACH_JUNK:
        raise AssertionError(f'only {N_BLOCKS} aligned junk blocks')
    if JUNK_LO + N_BLOCKS * DOC > JUNK_HI:
        raise AssertionError('junk block walks out of the source span')
    if TEMPLATE != 300:
        raise AssertionError(TEMPLATE)
    if REF_HI - REF_LO != 100_000:
        raise AssertionError(REF_HI - REF_LO)
    if TEST_HI - TEST_LO != N_TEST * DOC:
        raise AssertionError(TEST_HI - TEST_LO)
    if N_POOL != 3200:
        raise AssertionError(N_POOL)


def load_raw(path='data/enwik8/enwik8'):
    raw = h26.load_enwik8(path)
    digest = h26.sha256_arr(raw[:h26.E10M])
    if digest != h26.E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {h26.E10M_SHA256}')
    return raw, digest


def _junk_blocks(rng, raw):
    """320 aligned blocks from [JUNK_LO, JUNK_HI), drawn with replacement."""
    picks = rng.integers(0, N_BLOCKS, size=N_EACH_JUNK)
    starts = JUNK_LO + picks.astype(np.int64) * DOC
    out = np.empty((N_EACH_JUNK, DOC), dtype=np.uint8)
    for i, start in enumerate(starts):
        out[i] = raw[int(start):int(start) + DOC]
    return out


def _j2_corrupt(rng, raw):
    docs = _junk_blocks(rng, raw)
    for i in range(N_EACH_JUNK):
        mask = rng.random(DOC) < 0.30
        n = int(mask.sum())
        if n:
            docs[i, mask] = rng.integers(0, 256, size=n, dtype=np.uint8)
    return docs


def _j3_dup(rng, clean):
    choice = rng.choice(N_CLEAN, size=16, replace=False)
    docs = np.repeat(clean[choice], 20, axis=0)
    if docs.shape != (N_EACH_JUNK, DOC):
        raise AssertionError(docs.shape)
    for i, src in enumerate(choice):
        for rep in range(20):
            if not np.array_equal(docs[i * 20 + rep], clean[int(src)]):
                raise AssertionError('J3 copy is not the clean document')
    return docs


def _j4_boiler(rng, raw):
    template = raw[TEMPLATE_LO:TEMPLATE_HI]
    if template.shape != (TEMPLATE,):
        raise AssertionError(template.shape)
    docs = np.empty((N_EACH_JUNK, DOC), dtype=np.uint8)
    for i in range(N_EACH_JUNK):
        pos = 0
        while pos < DOC:
            number = f'{int(rng.integers(0, 1_000_000)):06d}'.encode('ascii')
            piece = template.copy()
            piece[100:106] = np.frombuffer(number, dtype=np.uint8)
            take = min(TEMPLATE, DOC - pos)
            docs[i, pos:pos + take] = piece[:take]
            pos += take
    return docs


def _j5_shuffle(rng, raw):
    docs = _junk_blocks(rng, raw)
    for i in range(N_EACH_JUNK):
        parts = docs[i].tobytes().split(b' ')
        order = rng.permutation(len(parts))
        joined = b' '.join(parts[int(j)] for j in order)
        if len(joined) >= DOC:
            docs[i] = np.frombuffer(joined[:DOC], dtype=np.uint8)
        else:
            pad = joined + b' ' * (DOC - len(joined))
            docs[i] = np.frombuffer(pad, dtype=np.uint8)
    return docs


def build(seed, raw=None, data_path='data/enwik8/enwik8'):
    """Return pool (3200, 4096), labels (3200,), test (64, 4096)."""
    _check_spans()
    seed = int(seed)
    if raw is None:
        raw, _digest = load_raw(data_path)
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.shape != (h26.FULL,):
        raise AssertionError(f'enwik8 shape {raw.shape}')
    clean = raw[:CLEAN_END].reshape(N_CLEAN, DOC).copy()
    rng = np.random.default_rng(seed + 500)
    j1 = rng.integers(0, 256, size=(N_EACH_JUNK, DOC), dtype=np.uint8)
    j2 = _j2_corrupt(rng, raw)
    j3 = _j3_dup(rng, clean)
    j4 = _j4_boiler(rng, raw)
    j5 = _j5_shuffle(rng, raw)
    parts = (clean, j1, j2, j3, j4, j5)
    if any(p.shape != (n, DOC) for p, n in zip(
            parts, (N_CLEAN, N_EACH_JUNK, N_EACH_JUNK, N_EACH_JUNK, N_EACH_JUNK, N_EACH_JUNK))):
        raise AssertionError('junk block shape')
    pool = np.concatenate(parts, axis=0)
    labels = np.array(
        ['clean'] * N_CLEAN
        + ['J1_random'] * N_EACH_JUNK
        + ['J2_corrupt'] * N_EACH_JUNK
        + ['J3_dup'] * N_EACH_JUNK
        + ['J4_boiler'] * N_EACH_JUNK
        + ['J5_shuffle'] * N_EACH_JUNK,
        dtype='<U16')
    order = np.random.default_rng(seed + 600).permutation(N_POOL)
    pool = pool[order].copy()
    labels = labels[order].copy()
    test = raw[TEST_LO:TEST_HI].reshape(N_TEST, DOC).copy()
    if pool.shape != (N_POOL, DOC) or labels.shape != (N_POOL,) or test.shape != (N_TEST, DOC):
        raise AssertionError('build shape')
    return pool, labels, test


def reference_bytes(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    return raw[REF_LO:REF_HI].copy()


def save(seed, pool, labels, test, reference, directory=DATA_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'seed{int(seed)}.npz'
    # np.savez_compressed appends ".npz" unless the name already ends with it.
    tmp = directory / f'seed{int(seed)}.tmp.npz'
    np.savez_compressed(
        tmp,
        pool=np.asarray(pool, dtype=np.uint8),
        labels=np.asarray(labels),
        test=np.asarray(test, dtype=np.uint8),
        reference=np.asarray(reference, dtype=np.uint8),
        seed=np.int64(seed))
    tmp.replace(path)
    return path


def corpus_sha256(pool, labels, test):
    h = hashlib.sha256()
    h.update(np.asarray(pool, dtype=np.uint8).tobytes())
    h.update(np.asarray(labels).tobytes())
    h.update(np.asarray(test, dtype=np.uint8).tobytes())
    return h.hexdigest()
