"""Build results/h35c/RESULTS.md and results.json from kernel outputs.

Numbers only. The pre-registered bars are copied, not reinterpreted.
A size whose mean G_pipe is below 0.02 makes the bars void.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/pdp')
sys.path.insert(0, str(ROOT))
import h35c_web as h

OUT = ROOT / 'results' / 'h35c'
DATA = ROOT / 'data' / 'h35c'
ARMS = h.ARMS
HEAD = ('rand', 'pipeline', 'prog_snap', 'rand_plus')


def load_json(path):
    return json.loads(Path(path).read_text())


def collect_rows():
    rows = []
    kernels = OUT / 'kernels'
    if not kernels.is_dir():
        return rows
    for result in sorted(kernels.glob('*/result.json')):
        body = load_json(result)
        for row in body.get('rows') or []:
            row = dict(row)
            row['kernel_dir'] = result.parent.name
            rows.append(row)
    return rows


def usable(row):
    arm = row.get('arm')
    if row.get('c4_bpb') is None:
        return False
    if arm in h.ONLINE:
        return row.get('peek_bytes') == h.PEEK and row.get('k_cand') == h.K_CAND
    if arm == 'rand_plus':
        return 'sequences_reread' in row
    return True


def index_rows(rows):
    out = {}
    for row in rows:
        if not usable(row):
            continue
        out[(row['size'], int(row['seed']), row['arm'])] = row
    return out


def fmt(value, digits=6):
    if value is None:
        return ''
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f'{float(value):.{digits}f}'


def mean(xs):
    xs = list(xs)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def pipeline_survivors(cls_scores):
    masks = h.load_masks(DATA)
    ids = h.cls_keep_ids(masks['pipe_before_cls'], cls_scores)
    return set(int(i) for i in ids), masks


def preview(pool, doc_id):
    return pool.preview(int(doc_id), 200).decode('utf-8', errors='replace')


def selection_flops(row, smoke, small_step, small_per_token):
    """Per-arm selection FLOPs. Shared passes are included here and were run once."""
    arm = row['arm']
    per_token = float(row['forward_flops_per_token'])
    n_pool = int(smoke['survival']['n_docs'])
    ref_train = 1000.0 * small_step
    ppl_score = float(smoke['ppl_tokens']) * small_per_token
    cls_per = smoke.get('cls_flops_per_doc')
    cls_full = None if cls_per is None else float(cls_per) * n_pool
    pipe_n = int(smoke['survival']['pipe_before_cls']['docs'])
    cls_pipe = None if cls_per is None else float(cls_per) * pipe_n
    scored = float(row.get('scored_tokens') or 0) * per_token
    snap = float(row.get('snap_tokens') or 0) * per_token
    ref_fwd = float(row.get('ref_tokens') or 0) * small_per_token
    if arm in ('rand', 'rand_plus', 'heur', 'dedup'):
        return 0.0
    if arm == 'ppl':
        return ref_train + ppl_score
    if arm == 'cls':
        return cls_full
    if arm == 'pipeline':
        return cls_pipe
    if arm == 'high_loss':
        return scored
    if arm == 'rho':
        return ref_train + scored + ref_fwd
    if arm == 'prog_snap':
        return scored + snap
    raise KeyError(arm)


def shared_block(smoke, small_step, small_per_token):
    n_pool = int(smoke['survival']['n_docs'])
    cls_per = smoke.get('cls_flops_per_doc')
    return {
        'ref_train_flops': 1000.0 * small_step,
        'ppl_score_flops': float(smoke['ppl_tokens']) * small_per_token,
        'ppl_tokens': int(smoke['ppl_tokens']),
        'cls_flops_per_doc': cls_per,
        'cls_full_pool_flops': None if cls_per is None else float(cls_per) * n_pool,
        'cls_pipeline_docs': int(smoke['survival']['pipe_before_cls']['docs']),
        'cls_pipeline_flops': None if cls_per is None else float(cls_per) * int(smoke['survival']['pipe_before_cls']['docs']),
        'minhash_wall_s': float(np.load(DATA / 'filters_cpu.npz')['minhash_wall_s']),
        'heur_wall_s': float(np.load(DATA / 'filters_cpu.npz')['heur_wall_s']),
        'ref_wall_s': smoke['ref']['wall_s'],
        'cls_wall_s': smoke['cls_wall_s'],
        'note': 'shared passes ran once; smoke kernel holds their GPU wall; MinHash FLOPs are 0',
    }


def scoring_tables(idx, plan):
    seeds = plan['seeds']
    cells = []
    for size, seed_list in seeds.items():
        for seed in seed_list:
            vals = {}
            missing = []
            for arm in HEAD:
                row = idx.get((size, seed, arm))
                if row is None or row.get('c4_bpb') is None:
                    missing.append(arm)
                else:
                    vals[arm] = float(row['c4_bpb'])
            cell = {'size': size, 'seed': int(seed), 'missing': missing}
            if not missing:
                g_pipe = vals['rand'] - vals['pipeline']
                g_prog = vals['rand'] - vals['prog_snap']
                cell.update({
                    'rand': vals['rand'],
                    'pipeline': vals['pipeline'],
                    'prog_snap': vals['prog_snap'],
                    'rand_plus': vals['rand_plus'],
                    'G_pipe': g_pipe,
                    'G_prog': g_prog,
                    'G_prog_over_G_pipe': None if g_pipe == 0 else g_prog / g_pipe,
                    'prog_snap_minus_rand_plus': vals['prog_snap'] - vals['rand_plus'],
                    'bar1': int(g_prog >= 0.5 * g_pipe),
                    'bar2': int((vals['prog_snap'] - vals['rand_plus']) <= -0.01),
                })
            cells.append(cell)
    by_size = {}
    for size in seeds:
        owned = [c for c in cells if c['size'] == size and not c['missing']]
        g_pipe = mean(c['G_pipe'] for c in owned)
        g_prog = mean(c['G_prog'] for c in owned)
        by_size[size] = {
            'n': len(owned),
            'G_pipe_mean': g_pipe,
            'G_prog_mean': g_prog,
            'pre': None if g_pipe is None else int(g_pipe >= 0.02),
        }
    pre_ok = all(by_size[s]['pre'] == 1 for s in seeds)
    small = by_size.get('small', {})
    medium = by_size.get('medium', {})
    bar3 = None
    if small.get('G_prog_mean') is not None and medium.get('G_prog_mean') is not None:
        bar3 = int(medium['G_prog_mean'] >= 0.8 * small['G_prog_mean'])
    ratio = None
    if small.get('G_prog_mean') not in (None, 0) and medium.get('G_prog_mean') is not None:
        ratio = medium['G_prog_mean'] / small['G_prog_mean']
    observed = [c for c in cells if not c['missing']]
    summary = {
        'precondition': 1 if pre_ok else 0,
        'bars': 'void' if not pre_ok else 'scored',
        'by_size': by_size,
        'G_prog_medium_over_small': ratio,
        'bar3': None if not pre_ok else bar3,
        'bar1_count': None if not pre_ok else sum(c['bar1'] for c in observed),
        'bar1_n': None if not pre_ok else len(observed),
        'bar2_count': None if not pre_ok else sum(c['bar2'] for c in observed),
        'bar2_n': None if not pre_ok else len(observed),
        'bar3_count': None if not pre_ok or bar3 is None else bar3,
        'bar3_n': None if not pre_ok or bar3 is None else 1,
    }
    return cells, summary


def markdown(payload):
    plan = payload['plan']
    lines = ['## Scoring table', '']
    lines.append(
        'C4-val bits/byte. G_pipe = rand − pipeline. G_prog = rand − prog_snap. '
        'pre is 1 when mean G_pipe ≥ 0.02 at that size. '
        'bar1 is 1 when G_prog ≥ 0.5 · G_pipe. bar2 is 1 when prog_snap − rand_plus ≤ −0.01. '
        'bar3 is 1 when G_prog(medium) ≥ 0.8 · G_prog(small), means over seeds. '
        'If pre is 0 at either size, bars are void.'
    )
    lines.append('')
    lines.append('| size | seed | rand | pipeline | prog_snap | rand_plus | G_pipe | G_prog | G_prog/G_pipe | prog_snap−rand_plus | bar1 | bar2 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for cell in payload['cells']:
        if cell['missing']:
            lines.append(f"| {cell['size']} | {cell['seed']} | missing {','.join(cell['missing'])} |")
            continue
        void = payload['summary']['bars'] == 'void'
        b1 = '' if void else str(cell['bar1'])
        b2 = '' if void else str(cell['bar2'])
        lines.append(
            f"| {cell['size']} | {cell['seed']} | {fmt(cell['rand'])} | {fmt(cell['pipeline'])} | "
            f"{fmt(cell['prog_snap'])} | {fmt(cell['rand_plus'])} | {fmt(cell['G_pipe'])} | "
            f"{fmt(cell['G_prog'])} | {fmt(cell['G_prog_over_G_pipe'])} | "
            f"{fmt(cell['prog_snap_minus_rand_plus'])} | {b1} | {b2} |"
        )
    s = payload['summary']
    lines.append('')
    for size, row in s['by_size'].items():
        lines.append(
            f"{size} mean G_pipe {fmt(row['G_pipe_mean'])} mean G_prog {fmt(row['G_prog_mean'])} pre {row['pre']} n {row['n']}"
        )
    lines.append(
        f"precondition {s['precondition']}. bars {s['bars']}. "
        f"G_prog(medium)/G_prog(small) {fmt(s['G_prog_medium_over_small'])}."
    )
    if s['bars'] == 'void':
        lines.append('bar1 void. bar2 void. bar3 void.')
    else:
        lines.append(
            f"bar1 count {s['bar1_count']}/{s['bar1_n']}. "
            f"bar2 count {s['bar2_count']}/{s['bar2_n']}. "
            f"bar3 count {s['bar3_count']}/{s['bar3_n']}."
        )
    lines.append('')
    lines.append('# H35c web curiosity')
    lines.append('')
    lines.append(
        f"S = {plan['steps']}. seeds = {json.dumps(plan['seeds'])}. cuts = {json.dumps(plan['cuts'])}."
    )
    lines.append(
        f"autocast = {payload['smoke'].get('autocast')}. classifier = {payload['smoke'].get('classifier_dtype')}. "
        f"grad_clip = {payload['smoke'].get('grad_clip')}. init = {payload['smoke'].get('init')}."
    )
    lines.append(
        f"online peek bytes = {h.PEEK}. online candidates = {h.K_CAND}. rand_plus max epochs = 2."
    )
    lines.append(
        f"enwik8 first 10MB sha256 = `{payload['cpu_preview']['wiki_ref']['enwik8_e10m_sha256']}`."
    )
    lines.append(
        f"pool sha256 = `{payload['cpu_preview']['pool']['sha256']}`."
    )
    lines.append('')
    lines.append('## C4-val and WikiText bits/byte')
    lines.append('')
    lines.append('| arm | size | seed | c4_bpb | wiki_bpb | steps |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for row in payload['bpb']:
        lines.append(
            f"| {row['arm']} | {row['size']} | {row['seed']} | {fmt(row['c4_bpb'])} | "
            f"{fmt(row['wiki_bpb'])} | {row['steps_completed']} |"
        )
    lines.append('')
    lines.append('## rand_plus sequences re-read')
    lines.append('')
    lines.append('| size | seed | sequences_reread | epochs |')
    lines.append('|---|---:|---:|---:|')
    for row in payload['reread']:
        lines.append(
            f"| {row['size']} | {row['seed']} | {row['sequences_reread']} | {row['epochs']} |"
        )
    lines.append('')
    lines.append('## Compute')
    lines.append('')
    lines.append(
        'train_flops is completed steps times the measured train-step FLOPs at that size. '
        'select_flops counts scoring forwards, reference training, and classifier inference. '
        + payload['shared']['note'] + '.'
    )
    lines.append('')
    lines.append('| arm | size | seed | train_flops | select_flops | select_pct_of_train | wall_s | wall_select_s | wall_eval_s |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for row in payload['compute']:
        lines.append(
            f"| {row['arm']} | {row['size']} | {row['seed']} | {fmt(row['train_flops'], 3)} | "
            f"{fmt(row['select_flops'], 3)} | {fmt(row['select_pct'], 4)} | {fmt(row['wall_s'], 3)} | "
            f"{fmt(row['wall_select_s'], 3)} | {fmt(row['wall_eval_s'], 3)} |"
        )
    sh = payload['shared']
    lines.append('')
    lines.append(
        f"ref_train_flops {fmt(sh['ref_train_flops'], 3)}. ppl_score_flops {fmt(sh['ppl_score_flops'], 3)}. "
        f"ppl_tokens {sh['ppl_tokens']}. cls_full_pool_flops {fmt(sh['cls_full_pool_flops'], 3)}. "
        f"cls_pipeline_flops {fmt(sh['cls_pipeline_flops'], 3)} on {sh['cls_pipeline_docs']} docs. "
        f"heur_wall_s {fmt(sh['heur_wall_s'], 3)}. minhash_wall_s {fmt(sh['minhash_wall_s'], 3)}. "
        f"ref_wall_s {fmt(sh['ref_wall_s'], 3)}. cls_wall_s {fmt(sh['cls_wall_s'], 3)}."
    )
    lines.append('')
    lines.append('## Filter survival')
    lines.append('')
    lines.append('| filter | docs | rate | sequences |')
    lines.append('|---|---:|---:|---:|')
    surv = payload['survival']
    n_docs = surv.get('n_docs') or 0
    for name, row in surv.items():
        if name in ('n_docs', 'pool_sequences', 'need_sequences') or not isinstance(row, dict):
            continue
        rate = row.get('rate')
        if rate is None and n_docs and row.get('docs') is not None:
            rate = row['docs'] / n_docs
        lines.append(f"| {name} | {row['docs']} | {fmt(rate, 6)} | {row['sequences']} |")
    lines.append('')
    lines.append(
        f"pool docs {surv['n_docs']} sequences {surv['pool_sequences']} need {surv['need_sequences']}."
    )
    lines.append('')
    lines.append('## Overlap with pipeline survivors')
    lines.append('')
    lines.append('| arm | size | seed | selected | shared_with_pipeline |')
    lines.append('|---|---|---:|---:|---:|')
    for row in payload['overlap']:
        lines.append(
            f"| {row['arm']} | {row['size']} | {row['seed']} | {row['n_selected']} | {row['shared']} |"
        )
    lines.append('')
    lines.append('## prog_snap examples')
    lines.append('')
    lines.append('First selected documents, then the 10 documents rejected most often. Preview is the first 200 bytes.')
    lines.append('')
    for block in payload['examples']:
        lines.append(f"### {block['size']} seed {block['seed']}")
        lines.append('')
        lines.append('| kind | rank | id | count | preview |')
        lines.append('|---|---:|---:|---:|---|')
        for kind in ('first', 'rejected'):
            for item in block[kind]:
                text = item['preview'].replace('|', '\\|').replace('\n', '\\n')
                count = item.get('count', '')
                lines.append(f"| {kind} | {item['rank']} | {item['id']} | {count} | {text} |")
        lines.append('')
    lines.append('## Smoke')
    lines.append('')
    lines.append('```')
    lines.append(payload['smoke_block'].rstrip())
    lines.append('```')
    lines.append('')
    lines.append('## Kernels')
    lines.append('')
    lines.append('| id | state | session_h | est_h | attempts |')
    lines.append('|---|---|---:|---:|---:|')
    for row in payload['kernels']:
        lines.append(
            f"| {row['id']} | {row['state']} | {fmt(row.get('session_h'), 4)} | "
            f"{fmt(row.get('est_h'), 4)} | {row.get('attempts', '')} |"
        )
    lines.append('')
    lines.append(
        f"gpu_hours {fmt(payload['gpu_hours'], 4)}. "
        f"plan_hours {fmt(payload['plan'].get('hours'), 4)}. "
        f"longest_arm_h {fmt((payload['plan'].get('longest_s') or 0) / 3600, 4)}."
    )
    lines.append('')
    return '\n'.join(lines)


def smoke_block(smoke, cpu):
    rand = smoke.get('rand_200') or {}
    prog = smoke.get('prog_snap_401') or {}
    cap = prog.get('capture') or {}
    lines = []
    lines.append(f"gpu {smoke.get('gpu')}")
    lines.append(f"ok {int(bool(smoke.get('ok')))} wall_s {smoke.get('wall_s')} session_s {smoke.get('session_s')}")
    lines.append(f"eval_during_selection {smoke.get('eval_during_selection')}")
    lines.append(f"loss_fell {int(bool(rand.get('loss_fell')))} rand_wall_s {rand.get('wall_s')}")
    losses = rand.get('losses') or []
    if losses:
        lines.append(f"rand_first_step {losses[0][0]} rand_first_bpb {losses[0][1]}")
        lines.append(f"rand_last_step {losses[-1][0]} rand_last_bpb {losses[-1][1]}")
    lines.append(
        f"prog_steps {prog.get('steps_completed')} draws {prog.get('n_draws')} "
        f"scored_tokens {prog.get('scored_tokens')} snap_tokens {prog.get('snap_tokens')}"
    )
    if cap:
        lines.append(
            f"capture_n {cap.get('n')} mean {cap.get('mean')} std {cap.get('std')} "
            f"percentiles {json.dumps(cap.get('percentiles'))}"
        )
        for label in ('highest', 'lowest'):
            for item in cap.get(label) or []:
                text = item['text'].replace('\n', '\\n')
                lines.append(f"capture_{label} id {item['id']} score {item['score']} {text}")
    lines.append(f"delta {json.dumps(smoke.get('delta'))}")
    lines.append(f"seconds_per_step {json.dumps(smoke.get('seconds_per_step'))}")
    if smoke.get('seconds_per_step_raw'):
        lines.append(f"seconds_per_step_raw {json.dumps(smoke.get('seconds_per_step_raw'))}")
    if smoke.get('timing_correction'):
        lines.append(f"timing_correction {json.dumps(smoke.get('timing_correction'))}")
    lines.append(smoke.get('prequential') or '')
    lines.append(
        f"cpu_preview_seed {cpu['preview_seed']} pool_docs {cpu['pool']['n_docs']} "
        f"pool_bytes {cpu['pool']['n_bytes']}"
    )
    for doc in cpu['docs']:
        text = doc['preview'].replace('\n', '\\n')
        lines.append(f"cpu_doc id {doc['pool_id']} line {doc['line']} bytes {doc['n_bytes']} {text}")
    filt = cpu['filters']
    lines.append(
        f"heur {filt['heur_docs']}/{cpu['pool']['n_docs']} seq {filt['heur_sequences']} "
        f"dedup {filt['dedup_docs']} seq {filt['dedup_sequences']} "
        f"pipe_before_cls {filt['pipe_before_cls_docs']} seq {filt['pipe_before_cls_sequences']}"
    )
    return '\n'.join(lines)


def main():
    plan = load_json(OUT / 'plan.json')
    smoke = load_json(OUT / 'smoke.json')
    cpu = load_json(OUT / 'cpu_preview.json')
    rows = collect_rows()
    idx = index_rows(rows)
    expected = []
    for size, seed_list in plan['seeds'].items():
        for seed in seed_list:
            for arm in ARMS:
                expected.append((size, int(seed), arm))
    missing = [key for key in expected if key not in idx or idx[key].get('c4_bpb') is None]
    if missing:
        raise SystemExit(f'missing rows {missing[:8]} count {len(missing)}')
    small_flop = smoke['flops']['small']
    small_fwd = small_flop['measured_forward_batch32'] or small_flop['analytical_forward_batch32']
    small_step = small_flop['measured_train_step_batch32'] or (3.0 * small_fwd)
    small_per_token = small_fwd / (h.BATCH * h.CTX)
    cells, summary = scoring_tables(idx, plan)
    bpb = []
    compute = []
    overlap = []
    pool = h.Pool(DATA)
    survivors, _masks = pipeline_survivors(np.load(OUT / 'smoke-kernel' / 'cls_scores.npy'))
    for size, seed, arm in expected:
        row = idx[(size, seed, arm)]
        bpb.append({
            'arm': arm, 'size': size, 'seed': seed,
            'c4_bpb': row['c4_bpb'], 'wiki_bpb': row['wiki_bpb'],
            'steps_completed': row['steps_completed'],
            'sequences_reread': row.get('sequences_reread'),
            'epochs': row.get('epochs'),
        })
        select = selection_flops(row, smoke, small_step, small_per_token)
        train = float(row['train_flops'])
        compute.append({
            'arm': arm, 'size': size, 'seed': seed,
            'train_flops': train, 'select_flops': select,
            'select_pct': None if train == 0 else 100.0 * select / train,
            'wall_s': row.get('wall_s'),
            'wall_select_s': row.get('wall_select_s'),
            'wall_eval_s': row.get('wall_eval_s'),
            'n_draws': row.get('n_draws'),
            'steps_completed': row['steps_completed'],
        })
        selected = [int(i) for i in row.get('selected_ids') or []]
        overlap.append({
            'arm': arm, 'size': size, 'seed': seed,
            'n_selected': len(selected),
            'shared': len(set(selected) & survivors),
        })
    examples = []
    for size, seed_list in plan['seeds'].items():
        for seed in seed_list:
            row = idx[(size, int(seed), 'prog_snap')]
            first = []
            for rank, doc_id in enumerate(row.get('selected_ids') or [], start=1):
                if rank > 10:
                    break
                first.append({'rank': rank, 'id': int(doc_id), 'preview': preview(pool, doc_id)})
            rejected = []
            for rank, item in enumerate(row.get('rejected_top') or [], start=1):
                rejected.append({
                    'rank': rank, 'id': int(item['id']), 'count': int(item['count']),
                    'preview': preview(pool, item['id']),
                })
            examples.append({'size': size, 'seed': int(seed), 'first': first, 'rejected': rejected})
    reread = []
    for size, seed_list in plan['seeds'].items():
        for seed in seed_list:
            row = idx[(size, int(seed), 'rand_plus')]
            reread.append({
                'size': size, 'seed': int(seed),
                'sequences_reread': int(row.get('sequences_reread') or 0),
                'epochs': int(row.get('epochs') or 0),
            })
    runs = []
    if (OUT / 'runs.json').is_file():
        runs = load_json(OUT / 'runs.json')
    kernels = [{
        'id': smoke.get('kernel'),
        'state': 'COMPLETE' if smoke.get('ok') else 'SMOKE_NOT_OK',
        'session_h': None if smoke.get('session_s') is None else smoke['session_s'] / 3600,
        'est_h': None,
        'attempts': smoke.get('attempts'),
    }]
    gpu_hours = kernels[0]['session_h'] or 0.0
    cold_path = OUT / 'smoke-cold.json'
    if cold_path.is_file():
        cold = load_json(cold_path)
        cold_h = None if cold.get('session_s') is None else cold['session_s'] / 3600
        kernels.append({
            'id': cold.get('kernel') or 'ezzzzzx/h35c-smoke',
            'state': 'COLD_GATE',
            'session_h': cold_h,
            'est_h': None,
            'attempts': cold.get('attempts'),
        })
        if cold_h:
            gpu_hours += cold_h
    for row in runs:
        session_h = None if row.get('session_s') is None else row['session_s'] / 3600
        kernels.append({
            'id': row['id'], 'state': row['state'], 'session_h': session_h,
            'est_h': None if row.get('est_s') is None else row['est_s'] / 3600,
            'attempts': row.get('attempts'),
        })
        if session_h:
            gpu_hours += session_h
    payload = {
        'plan': {k: plan[k] for k in ('steps', 'seeds', 'cuts', 'hours', 'ok', 'longest_s') if k in plan},
        'smoke': smoke,
        'cpu_preview': cpu,
        'cells': cells,
        'summary': summary,
        'bpb': bpb,
        'compute': compute,
        'shared': shared_block(smoke, small_step, small_per_token),
        'survival': smoke['survival'],
        'overlap': overlap,
        'examples': examples,
        'reread': reread,
        'kernels': kernels,
        'gpu_hours': gpu_hours,
        'smoke_block': smoke_block(smoke, cpu),
    }
    # Drop bulky smoke fields from results.json (capture text is in the markdown).
    slim_smoke = {
        k: smoke[k] for k in smoke
        if k not in ('prog_snap_401',)
    }
    if 'prog_snap_401' in smoke:
        prog = dict(smoke['prog_snap_401'])
        cap = prog.get('capture') or {}
        prog['capture'] = {
            'n': cap.get('n'), 'mean': cap.get('mean'), 'std': cap.get('std'),
            'percentiles': cap.get('percentiles'),
            'highest_ids': [x['id'] for x in cap.get('highest') or []],
            'lowest_ids': [x['id'] for x in cap.get('lowest') or []],
        }
        slim_smoke['prog_snap_401'] = prog
    out_json = {
        'plan': payload['plan'],
        'summary': summary,
        'cells': cells,
        'bpb': bpb,
        'compute': compute,
        'shared': payload['shared'],
        'survival': smoke['survival'],
        'overlap': overlap,
        'examples': examples,
        'reread': reread,
        'kernels': kernels,
        'gpu_hours': gpu_hours,
        'smoke': slim_smoke,
        'fixed': {
            'autocast': smoke.get('autocast'),
            'classifier_dtype': smoke.get('classifier_dtype'),
            'grad_clip': smoke.get('grad_clip'),
            'init': smoke.get('init'),
            'context': h.CTX,
            'peek_bytes': h.PEEK,
            'k_cand': h.K_CAND,
            'rand_plus_epochs': 2,
            'batch': h.BATCH,
            'lr': h.LR,
            'warmup': h.WARMUP,
            'weight_decay': h.WD,
        },
    }
    (OUT / 'results.json').write_text(json.dumps(out_json, indent=2) + '\n')
    text = markdown(payload)
    (OUT / 'RESULTS.md').write_text(text)
    print(f'wrote RESULTS.md bytes {len(text)} rows {len(bpb)} gpu_hours {gpu_hours:.4f}')


if __name__ == '__main__':
    main()
