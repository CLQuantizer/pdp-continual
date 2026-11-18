"""Kaggle entry points for H35c. Smoke, then scored arms.

Autocast is fp16. A T4 has no bf16 tensor cores. Evaluation is called only
after an arm's training loop returns, and the selection guard counts any
eval call that happens while a peek is in progress.
"""
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h35c_web as h


def find_input(name):
    root = Path('/kaggle/input')
    if not root.exists():
        root = h.DATA
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'{name}: {hits}')
    return hits[0]


def data_root():
    return find_input('pool.bin').parent


def cls_dir():
    return find_input('model.safetensors').parent


def device():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda')


def capture_report(pool, capture):
    scores = np.asarray(capture['scores'], dtype=np.float64)
    ids = np.asarray(capture['ids'], dtype=np.int64)
    order = np.argsort(-scores, kind='stable')

    def show(j):
        doc = int(ids[j])
        return {
            'id': doc,
            'score': float(scores[j]),
            'text': pool.preview(doc, 200).decode('utf-8', errors='replace'),
        }

    pct = {str(q): float(np.percentile(scores, q)) for q in (0, 10, 25, 50, 75, 90, 100)}
    return {
        'n': int(len(scores)),
        'mean': float(scores.mean()),
        'std': float(scores.std(ddof=0)),
        'percentiles': pct,
        'highest': [show(j) for j in order[:3]],
        'lowest': [show(j) for j in order[-3:]],
    }


def delta_steps(flop, per_step_tokens, steps):
    """Extra train steps matching measured prog_snap scoring FLOPs.

    per_step_tokens is the smoke run's mean forward-tokens per training step
    (current model plus snapshot). Peek forwards are charged at the measured
    batch-32 context-512 forward cost per token.
    """
    fwd = flop['measured_forward_batch32'] or flop['analytical_forward_batch32']
    step = flop['measured_train_step_batch32'] or (3.0 * fwd)
    per_token = fwd / (h.BATCH * h.CTX)
    select = steps * per_step_tokens * per_token
    return int(math.ceil(select / step)), float(select), float(step), float(per_token)


def survival(pool, masks, ppl, cls):
    n = pool.n
    ppl_ids = h.ppl_keep_ids(ppl)
    cls_ids = h.cls_keep_ids(np.arange(n), cls)
    pipe_ids = h.cls_keep_ids(masks['pipe_before_cls'], cls)
    rows = {
        'heur': masks['heur'],
        'dedup': masks['dedup'],
        'ppl': ppl_ids,
        'cls': cls_ids,
        'pipeline': pipe_ids,
    }
    out = {'n_docs': n, 'pool_sequences': int(pool.nseq.sum()), 'need_sequences': h.S_DEFAULT * h.BATCH}
    for name, ids in rows.items():
        out[name] = {
            'docs': int(len(ids)),
            'rate': float(len(ids) / n),
            'sequences': h.sequence_supply(pool, ids),
        }
    out['pipe_before_cls'] = {
        'docs': int(len(masks['pipe_before_cls'])),
        'sequences': h.sequence_supply(pool, masks['pipe_before_cls']),
    }
    return out


def run_smoke(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    pool = h.Pool(data_root())
    masks = h.load_masks(data_root())
    gpu = torch.cuda.get_device_name(0)
    print(f'gpu {gpu}', flush=True)
    t_all = time.perf_counter()
    result = {
        'gpu': gpu,
        'autocast': 'fp16',
        'classifier_dtype': 'fp32',
        'grad_clip': h.CLIP,
        'init': 'N(0, 0.02), residual out and down scaled by 0.02/sqrt(2*layers)',
        'params': {size: h.ByteLM(**h.SIZES[size]).n_params() for size in h.SIZES},
        'ok': False,
    }
    flops = {}
    for size in ('small', 'medium'):
        flops[size] = h.measure_flops(size, dev)
        print(size, flops[size], flush=True)
    result['flops'] = flops
    timing = {}
    schedule = (
        ('small', 'rand', 8),
        ('small', 'high_loss', 4),
        ('small', 'rho', 4),
        ('small', 'prog_snap', 4),
        ('medium', 'rand', 4),
        ('medium', 'high_loss', 3),
        ('medium', 'rho', 3),
        ('medium', 'prog_snap', 3),
    )
    for size, kind, n in schedule:
        sec = h.time_steps(size, dev, kind, n, pool)
        timing[f'{size}/{kind}'] = sec
        print(f'time {size} {kind} {sec:.4f} s/step', flush=True)
    result['seconds_per_step'] = timing
    train_est = 200 * timing['small/rand'] + 401 * timing['small/prog_snap'] + 1000 * timing['small/rand']
    result['smoke_train_est_s'] = train_est
    print(f'smoke train estimate {train_est:.0f}s', flush=True)
    if train_est > 2.5 * 3600:
        result['stop'] = f'smoke training estimate {train_est / 3600:.2f}h exceeds 2.5h'
        result['wall_s'] = time.perf_counter() - t_all
        h.dump(out / 'smoke.json', result)
        print(result['stop'], flush=True)
        return
    model, opt, scaler = h.make_model('small', 0, dev)
    order = h.shuffled(np.arange(pool.n), 0)
    rand = h.run_offline(model, opt, scaler, pool, order, 200, dev, log_every=20, tag='smoke_rand')
    fell = rand['losses'][-1][1] < rand['losses'][0][1]
    del model, opt, scaler
    torch.cuda.empty_cache()
    result['rand_200'] = {'losses': rand['losses'], 'wall_s': rand['wall_s'], 'loss_fell': bool(fell)}
    result['eval_during_selection'] = h.GUARD.eval_during_selection
    if not fell or not all(math.isfinite(v) for _, v in rand['losses']):
        result['ok'] = False
        result['stop'] = 'rand loss did not fall over 200 steps'
        result['wall_s'] = time.perf_counter() - t_all
        h.dump(out / 'smoke.json', result)
        print(result['stop'], flush=True)
        return
    model, opt, scaler = h.make_model('small', 0, dev)
    prog = h.run_online(
        model, opt, scaler, pool, 'prog_snap', 401, 0, dev, capture_step=400, log_every=50)
    if prog.get('capture') is None or h.GUARD.eval_during_selection != 0:
        result['ok'] = False
        result['stop'] = 'prog_snap capture missing or eval ran during selection'
        result['prog_draws'] = prog.get('n_draws')
        result['wall_s'] = time.perf_counter() - t_all
        h.dump(out / 'smoke.json', result)
        print(result['stop'], flush=True)
        return
    result['prog_snap_401'] = {
        'losses': prog['losses'],
        'wall_s': prog['wall_s'],
        'wall_select_s': prog['wall_select_s'],
        'n_draws': prog['n_draws'],
        'n_scored': prog['n_scored'],
        'scored_tokens': prog['scored_tokens'],
        'snap_tokens': prog['snap_tokens'],
        'steps_completed': prog['steps_completed'],
        'capture': capture_report(pool, prog['capture']),
    }
    del model, opt, scaler
    torch.cuda.empty_cache()
    result['wall_s'] = time.perf_counter() - t_all
    h.dump(out / 'smoke.json', result)
    raw = (data_root() / 'wiki_ref.bin').read_bytes()
    ref, ref_info = h.train_reference(dev, raw)
    torch.save(ref.state_dict(), out / 'ref.pt')
    ppl, ppl_tokens = h.score_documents_bpb(ref, pool, dev)
    np.save(out / 'ppl_scores.npy', ppl)
    del ref
    torch.cuda.empty_cache()
    cls_scores, cls_wall, cls_per_doc = h.score_classifier(pool, cls_dir(), dev)
    np.save(out / 'cls_scores.npy', cls_scores)
    mean_peek = float(np.minimum(pool.lengths, h.PEEK).mean())
    done = max(1, int(prog['steps_completed']))
    measured_tokens = int(prog['scored_tokens']) + int(prog['snap_tokens'])
    per_step_tokens = measured_tokens / done
    formula_tokens = 2 * h.K_CAND * mean_peek
    deltas = {}
    for size in ('small', 'medium'):
        delta, select, step, per_token = delta_steps(flops[size], per_step_tokens, h.S_DEFAULT)
        formula_select = h.S_DEFAULT * formula_tokens * per_token
        deltas[size] = {
            'delta': delta,
            'selection_flops_at_S': select,
            'formula_selection_flops_at_S': float(formula_select),
            'formula_delta': int(math.ceil(formula_select / step)),
            'train_step_flops': step,
            'forward_flops_per_token': per_token,
            'measured_tokens_per_step': per_step_tokens,
            'formula_tokens_per_step': formula_tokens,
            'draws_per_step': prog['n_draws'] / done,
        }
    result.update({
        'ok': True,
        'ref': ref_info,
        'ppl_tokens': int(ppl_tokens),
        'cls_wall_s': cls_wall,
        'cls_flops_per_doc': cls_per_doc,
        'mean_peek_bytes': mean_peek,
        'delta': deltas,
        'survival': survival(pool, masks, ppl, cls_scores),
        'eval_during_selection': h.GUARD.eval_during_selection,
        'prequential': (
            'Selection scores a candidate with the current weights, and for '
            'prog_snap also with the stored snapshot. No evaluation bytes are read.'
        ),
        'wall_s': time.perf_counter() - t_all,
    })
    h.dump(out / 'smoke.json', result)
    print(f'smoke ok wall_s {result["wall_s"]:.1f}', flush=True)


def load_ref(dev, scores_root):
    ref, _, _ = h.make_model('small', 0, dev)
    state = torch.load(Path(scores_root) / 'ref.pt', map_location=dev)
    ref.load_state_dict(state)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def run_scored(out_dir, arms, size, seed, steps, delta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    root = data_root()
    pool = h.Pool(root)
    masks = h.load_masks(root)
    scores_root = find_input('ppl_scores.npy').parent
    scores = {
        'ppl': np.load(scores_root / 'ppl_scores.npy'),
        'cls': np.load(scores_root / 'cls_scores.npy'),
    }
    ref = load_ref(dev, scores_root) if 'rho' in arms else None
    c4, wiki = h.load_eval(root)
    flops = h.measure_flops(size, dev)
    rows = []
    for arm in arms:
        print(f'start {size} seed {seed} {arm}', flush=True)
        row = h.run_one(arm, size, seed, steps, pool, masks, scores, ref, dev, c4, wiki, delta=delta)
        fwd = flops['measured_forward_batch32'] or flops['analytical_forward_batch32']
        step_f = flops['measured_train_step_batch32'] or (3.0 * fwd)
        per_token = fwd / (h.BATCH * h.CTX)
        row['train_flops'] = row['steps_completed'] * step_f
        row['forward_flops_per_token'] = per_token
        row['train_step_flops'] = step_f
        rows.append(row)
        h.dump(out / 'result.json', {'gpu': torch.cuda.get_device_name(0), 'rows': rows, 'flops': flops})
        print(f'done {arm} c4 {row.get("c4_bpb")}', flush=True)
    h.dump(out / 'result.json', {'gpu': torch.cuda.get_device_name(0), 'rows': rows, 'flops': flops, 'done': True})


def main():
    mode = os.environ.get('H35C_MODE', 'smoke')
    out = os.environ.get('H35C_OUT', '/kaggle/working')
    if mode == 'smoke':
        run_smoke(out)
    elif mode == 'scored':
        run_scored(
            out,
            arms=os.environ['H35C_ARMS'].split(','),
            size=os.environ['H35C_SIZE'],
            seed=int(os.environ['H35C_SEED']),
            steps=int(os.environ['H35C_STEPS']),
            delta=int(os.environ.get('H35C_DELTA', '0')),
        )
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
