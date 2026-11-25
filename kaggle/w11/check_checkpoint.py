"""Compare checkpoint values independently of pickle's object memoization."""
import pickle
from collections import deque
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch


def equal_values(a, b, seen=None, path='state'):
    if seen is None:
        seen = {}
    assert type(a) is type(b), (path, type(a), type(b))
    pair = (id(a), id(b))
    if pair in seen:
        return
    seen[pair] = (a, b)
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.requires_grad == b.requires_grad, path
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, np.random.Generator):
        equal_values(a.bit_generator.state, b.bit_generator.state, seen, path + '.rng_state')
    elif isinstance(a, dict):
        assert len(a) == len(b), path
        for i, ((ka, va), (kb, vb)) in enumerate(zip(a.items(), b.items())):
            equal_values(ka, kb, seen, path + f'.key{i}')
            equal_values(va, vb, seen, path + f'[{ka if isinstance(ka, str) else i}]')
    elif isinstance(a, (list, tuple, deque)):
        assert len(a) == len(b), path
        if isinstance(a, deque):
            assert a.maxlen == b.maxlen, path
        for i, (x, y) in enumerate(zip(a, b)):
            equal_values(x, y, seen, path + f'[{i}]')
    elif isinstance(a, (str, bytes, int, float, complex, bool, type(None), set, frozenset, np.generic)):
        assert a == b, path
    elif isinstance(a, type):
        assert a is b, path
    elif hasattr(a, '__getstate__') and a.__getstate__() is not None:
        # Optimizers expose defaults, ordered parameter groups and moment state;
        # modules expose parameters and buffers. Transient .grad is overwritten
        # by zero_grad at the next own-loss calculation, as in BankCortex.
        equal_values(a.__getstate__(), b.__getstate__(), seen, path + '.__state__')
    elif hasattr(a, '__dict__'):
        equal_values(vars(a), vars(b), seen, path + '.__dict__')
    else:
        assert a == b, path


def check(out):
    import w11_deep_cortex as w
    w.m10._threads()
    out = Path(out)
    path = out / 'smoke-value-checkpoint.pkl'
    captured = {}
    original = pickle.dump
    def observe(saved, *args, **kwargs):
        if isinstance(saved, dict) and 'source_hashes' in saved and 'config' in saved:
            captured['saved'] = saved
        return original(saved, *args, **kwargs)
    with patch.object(pickle, 'dump', observe):
        row = w.run_life(120, 'deep_4', 512, checkpoint=path, stop_at=13)
    assert row['decision'] == 13 and not row['complete']
    live = captured['saved']
    with path.open('rb') as f:
        restored = pickle.load(f)
    equal_values(live, restored)
    a, b = live['state']['life'].cortex, restored['state']['life'].cortex
    tokens = np.arange(w.m7.CHUNK, dtype=np.int64) % len(w.m7.ALPHA)
    for commit in (True, False):
        sa, sb = (None, None) if commit else (a.zero_state(), b.zero_state())
        x = a.learn_glance(tokens, 0, a.prev_symbol, state=sa, commit=commit)
        y = b.learn_glance(tokens, 0, b.prev_symbol, state=sb, commit=commit)
        assert np.array_equal(x, y)
        equal_values(a, b)
    path.unlink()
    return dict(split=13, checkpoint_values_exact=True, wake_update_exact=True,
                replay_update_exact=True, max_bits_abs=0)


if __name__ == '__main__':
    import hashlib
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import w11_deep_cortex as w
    result = check(Path('results/w11'))
    payload = dict(result=result, source_hashes=w.source_hashes(),
                   checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    w.dump(Path('results/w11/checkpoint-values.json'), payload)
    print(json.dumps(result), flush=True)
