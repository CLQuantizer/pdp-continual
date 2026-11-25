"""Continue the interrupted, hash-checked W11 smoke without repeating passed checks."""
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_W11_FINISH') != '1':
    os.environ['_W11_FINISH'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import math
import time
import numpy as np
import torch
import w11_deep_cortex as w


def main():
    out = w.ROOT / 'results/w11'
    payload = json.loads((out / 'smoke.json').read_text())
    assert payload['source_hashes'] == w.source_hashes(), 'changed source; full smoke required'
    required = ['W10_and_depth1_exact', 'Wait_and_layer1_isolation',
                'prequential_and_replay', 'prequential_and_replay',
                'prequential_and_replay', 'eyelid_controller_preq']
    assert [r['check'] for r in payload['checks'][:6]] == required
    assert [r['cortex'] for r in payload['checks'][2:5]] == list(w.CORTICES)
    if payload['ok']:
        w.stage(payload)
        return
    start = time.perf_counter()
    previous_wall = payload.get('wall_s', 0.)
    cached_open = (out / 'smoke-deep_4_open.json').exists()
    def record(name, **values):
        row = dict(check=name, **values)
        payload['checks'].append(row)
        print(json.dumps(row), flush=True)
        w.dump(out / 'smoke.json', payload)
    def life(name, cortex, **kwargs):
        path = out / ('smoke-' + name + '.json')
        if path.exists():
            return json.loads(path.read_text())
        print('START ' + name, flush=True)
        row = w.run_life(120, cortex, 512, **kwargs)
        w.dump(path, row)
        print('DONE ' + name, flush=True)
        return row
    try:
        w.m10._threads()
        assert torch.get_num_threads() == 1
        if not any(r['check'] == 'exact_resume' for r in payload['checks']):
            timed = life('deep_4', 'deep_4', check_eyelid=True)
            cp = out / 'smoke-checkpoint.pkl'
            if not cp.exists():
                print('START checkpoint-253', flush=True)
                partial = w.run_life(120, 'deep_4', 512, checkpoint=cp, stop_at=253, check_eyelid=True)
                assert partial['complete'] is False and partial['decision'] == 253
            resumed = life('deep_4-resumed', 'deep_4', resume=cp, check_eyelid=True)
            for field in ('checkpoints', 'preq_bits', 'preq_symbols', 'choices', 'sleep_visits',
                          'input_retina', 'input_rooms', 'glance_log', 'mixer_history',
                          'final_mixer_weights'):
                assert timed[field] == resumed[field], 'resume ' + field
            assert [(r['t'], r['intact']) for r in timed['eyelid_checks']] == [
                (r['t'], r['intact']) for r in resumed['eyelid_checks']]
            assert all(r['intact'] and r['before'] == r['after'] for r in resumed['eyelid_checks'])
            from check_checkpoint import check
            cached_check = out / 'checkpoint-values.json'
            if cached_check.exists():
                import hashlib
                cached = json.loads(cached_check.read_text())
                assert cached['source_hashes'] == w.source_hashes()
                assert cached['checker_sha256'] == hashlib.sha256(Path(__file__).with_name('check_checkpoint.py').read_bytes()).hexdigest()
                checked_values = cached['result']
            else:
                checked_values = check(out)
            record('checkpoint_values', **checked_values)
            record('exact_resume', split=253, complete=False, max_bits_abs=0,
                   pickle_checksum_equal=timed['learner_checksum'] == resumed['learner_checksum'])
            cp.unlink()
        opened = life('deep_4_open', 'deep_4_open')
        if cached_open and not payload.get('open_wall_accounted', False):
            previous_wall += opened['wall_s']
        payload['open_wall_accounted'] = True
        if 'estimate' not in payload:
            symbols = np.arange(w.m7.CHUNK, dtype=np.int64) % len(w.m7.ALPHA)
            late, estimates, parts, segments = {}, {}, {}, {}
            for cortex in w.CORTICES:
                print('START timing ' + cortex, flush=True)
                obj = w.BANK_CLASS(w.w4.N_CH) if cortex == 'bank_1' else w.DeepCortex(opened=cortex.endswith('open'))
                tick = time.perf_counter()
                obj.learn_glance(symbols, 0, -1)
                learn_s = time.perf_counter() - tick
                def rate(batch, length):
                    tokens = torch.arange(batch*length).reshape(batch, length) % len(w.m7.ALPHA)
                    tick = time.perf_counter()
                    w.score_sequences(obj.model, obj.channel_emb, tokens,
                                      torch.full((batch,), -1), torch.arange(batch) % w.w4.N_CH)
                    return time.perf_counter() - tick
                one, room, replay, exam = rate(1, 64), rate(w.m7.R, 64), rate(18*w.m7.R, 64), rate(18, 2048)
                per_decision = learn_s + max(3*room+one, replay)
                estimates[cortex] = 30 + 2*(w.L_SCORE*per_decision + 8*exam)
                for count in range(1, 65):
                    size = math.ceil(w.L_SCORE/count)
                    exams = min(8, math.ceil(size/512)+1)
                    bound = 30 + 2*(size*per_decision + exams*exam)
                    if bound <= 5100:
                        parts[cortex], segments[cortex] = count, bound
                        break
                else:
                    raise RuntimeError('no safe split within 64 exact parts')
                late[cortex] = dict(learn_s=learn_s, score_one_s=one, score_room32_s=room,
                    score_replay576_s=replay, score_exam18x2048_s=exam, per_decision_bound_s=per_decision)
            payload['estimate'] = dict(kernel_wall_s=estimates, parts=parts,
                segment_wall_s=segments, late_reservoir_timing=late,
                limit_s=5100, margin=2., startup_s=30., workers=1)
            record('wall_estimate', **payload['estimate'])
        assert payload['source_hashes'] == w.source_hashes()
        payload.pop('error', None)
        payload['ok'] = True
    except BaseException as exc:
        payload['error'] = repr(exc)
        raise
    finally:
        payload['wall_s'] = previous_wall + time.perf_counter() - start
        payload['continuation_wall_s'] = time.perf_counter() - start
        w.dump(out / 'smoke.json', payload)
    w.stage(payload)


if __name__ == '__main__':
    main()
