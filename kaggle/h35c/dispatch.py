"""Upload H35c data and run private GPU kernels in the foreground.

Account running kernels stay <= 5. Each kernel is under 3 GPU hours.
The whole experiment, including the smoke kernel, stays within 14 GPU hours,
and the account keeps 10 GPU hours in reserve. A failed kernel is retried once.
"""
import csv
import fcntl
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/pdp')
BASE = ROOT / 'kaggle' / 'h35c'
OUT = ROOT / 'results' / 'h35c'
CLI = str(ROOT / '.venv/bin/kaggle')
LOCK = OUT / 'dispatch.lock'
DATA_ID = 'ezzzzzx/pdp-h35c-data'
CLS_ID = 'ezzzzzx/pdp-h35c-cls'
CODE_ID = 'ezzzzzx/pdp-code-h35c'
SCORE_ID = 'ezzzzzx/pdp-h35c-scores'
ACCOUNT_CAP = 5
GPU_BUDGET_H = 14.0
GPU_RESERVE_H = 10.0
KERNEL_LIMIT_S = 3 * 3600
PACK_LIMIT_S = 2.6 * 3600
# One of our kernels at a time. A second queued GPU session cannot be cancelled
# from this CLI, and it would hold a slot the other experiments need.
INFLIGHT_CAP = 1
STATUS_CACHE = OUT / 'status-cache.json'
SMOKE_EST_H = 2.0


def load_token():
    if os.environ.get('KAGGLE_API_TOKEN'):
        return
    for raw in (ROOT / '.env').read_text().splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip() != 'KAGGLE_API_TOKEN':
            continue
        value = value.strip().strip('"').strip("'")
        if not value:
            raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN'] = value
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args, allow_failure=False, timeout=300):
    """Retry a rate limit. Callers that allow failure get the error back."""
    token = os.environ.get('KAGGLE_API_TOKEN') or '\0'
    pause = 10
    while True:
        last = (1, '')
        for attempt in range(4):
            proc = subprocess.run(
                [CLI, *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout)
            text = proc.stdout.replace(token, '[REDACTED]')
            low = text.lower()
            last = (proc.returncode, text)
            if proc.returncode and ('429' in low or 'too many' in low or 'rate limit' in low):
                time.sleep(2 + attempt * 3)
                continue
            if proc.returncode and not allow_failure:
                raise RuntimeError(f'kaggle {args[:3]} failed: {text[-2000:]}')
            return proc.returncode, text
        if allow_failure:
            return last
        what = ' '.join(str(part) for part in args[:2])
        print(f'kaggle paused {pause}s {what}', flush=True)
        time.sleep(pause)
        pause = min(180, pause * 2)


def quota_gpu():
    _, text = cli('quota')
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == 'GPU' and len(parts) >= 3:
            used = float(parts[1].rstrip('h'))
            remaining = float(parts[2].rstrip('h'))
            return used, remaining, text.strip()
    raise RuntimeError(f'cannot parse quota: {text}')


def assert_budget(est_h, committed_h):
    used, remaining, text = quota_gpu()
    print(
        f'quota used {used:.2f}h remaining {remaining:.2f}h '
        f'est {est_h:.2f}h committed {committed_h:.2f}h',
        flush=True)
    print(text, flush=True)
    if remaining - est_h < GPU_RESERVE_H - 1e-9:
        raise RuntimeError(
            f'reserve would be breached: remaining {remaining:.2f}h est {est_h:.2f}h')
    if committed_h + est_h > GPU_BUDGET_H + 1e-9:
        raise RuntimeError(
            f'H35c budget would be breached: {committed_h + est_h:.2f}h')
    return used, remaining


def accounted_hours():
    """GPU hours already spent, from the smoke files and scored runs.json.

    A restarted dispatcher counts finished kernels instead of the in-memory total.
    """
    total = 0.0
    kernels = []
    for which, path in (('smoke-cold', OUT / 'smoke-cold.json'), ('smoke', OUT / 'smoke.json')):
        if not path.is_file():
            continue
        body = json.loads(path.read_text())
        sec = float(body.get('session_s') or 0.0)
        total += sec / 3600.0
        kernels.append({
            'id': body.get('kernel'),
            'session_s': sec,
            'ok': bool(body.get('ok')),
            'which': which,
        })
    runs = []
    if (OUT / 'runs.json').is_file():
        runs = json.loads((OUT / 'runs.json').read_text())
    for row in runs:
        sec = float(row.get('session_s') or 0.0)
        total += sec / 3600.0
        kernels.append({
            'id': row.get('id'),
            'session_s': sec,
            'ok': row.get('state') == 'COMPLETE',
            'which': 'scored',
            'state': row.get('state'),
        })
    return total, kernels


def write_committed(total, kernels):
    (OUT / 'committed.json').write_text(
        json.dumps({'committed_h': total, 'kernels': kernels}) + '\n')


def list_kernel_stamps():
    stamps = {}
    token = None
    while True:
        args = ['kernels', 'list', '--mine', '--page-size', '100', '--csv', '--sort-by', 'dateRun']
        if token:
            args += ['--page-token', token]
        code, text = cli(*args, allow_failure=True)
        if code != 0:
            raise RuntimeError(f'kernel list failed: {text[-1000:]}')
        lines = text.splitlines()
        tokens = [s.split('=', 1)[1].strip() for s in lines if s.startswith('Next Page Token =')]
        body = '\n'.join(s for s in lines if not s.startswith('Next Page Token ='))
        if body.strip() and body.strip() != 'Not found':
            rows = list(csv.DictReader(io.StringIO(body)))
            if rows and 'ref' not in rows[0]:
                raise RuntimeError('unrecognized kernel listing')
            for row in rows:
                stamps[row['ref']] = row.get('lastRunTime') or ''
        if not tokens:
            break
        if token == tokens[0]:
            raise RuntimeError('repeated kernel page token')
        token = tokens[0]
    return stamps


def status_of(kid):
    _, text = cli('kernels', 'status', kid, allow_failure=True)
    low = text.lower()
    if 'has status' not in low:
        return 'UNKNOWN', text.strip()
    status = low.split('has status', 1)[-1].strip().splitlines()[0]
    if 'complete' in status:
        return 'COMPLETE', text.strip()
    if 'cancel' in status:
        return 'CANCELLED', text.strip()
    if 'error' in status or 'fail' in status:
        return 'ERROR', text.strip()
    if 'queue' in status or 'pending' in status:
        return 'QUEUED', text.strip()
    if 'running' in status:
        return 'RUNNING', text.strip()
    return 'UNKNOWN', text.strip()


def load_status_cache():
    if not STATUS_CACHE.is_file():
        return {}
    try:
        data = json.loads(STATUS_CACHE.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def account_active():
    """Running and queued kernels. Completed rows are cached by lastRunTime."""
    stamps = list_kernel_stamps()
    cache = load_status_cache()
    active = []
    checked = 0
    changed = False
    for ref, stamp in stamps.items():
        cached = cache.get(ref)
        if isinstance(cached, dict) and cached.get('lastRunTime') == stamp and cached.get('kind') in ('complete', 'failed'):
            continue
        checked += 1
        if checked == 1 or checked % 25 == 0:
            print(f'account status {checked} {ref}', flush=True)
        state = 'UNKNOWN'
        text = ''
        for _attempt in range(3):
            state, text = status_of(ref)
            if state != 'UNKNOWN':
                break
            time.sleep(3)
        if state in ('COMPLETE', 'ERROR', 'CANCELLED'):
            cache[ref] = {'lastRunTime': stamp, 'kind': 'complete' if state == 'COMPLETE' else 'failed'}
            changed = True
            store_status_cache(cache)
        elif state in ('RUNNING', 'QUEUED'):
            active.append(f'{ref} {state}')
            if isinstance(cached, dict):
                cache.pop(ref, None)
                changed = True
                store_status_cache(cache)
        else:
            low = text.lower()
            if '429' in low or 'too many' in low or 'rate limit' in low:
                raise RuntimeError(f'rate limited kernels status {ref}')
            raise RuntimeError(f'unknown status for {ref}: {state}')
    print(f'account slot check listed {len(stamps)} statused {checked} active {len(active)}', flush=True)
    return active


def store_status_cache(cache):
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS_CACHE.write_text(json.dumps(cache) + '\n')


def wait_slot():
    while True:
        try:
            running = account_active()
        except RuntimeError as exc:
            if 'rate limited' not in str(exc):
                raise
            print(f'slot check paused {exc}', flush=True)
            time.sleep(120)
            continue
        print(f'slots {len(running)}/{ACCOUNT_CAP} {running}', flush=True)
        if len(running) < ACCOUNT_CAP:
            return len(running)
        time.sleep(60)


def dataset_exists(dataset_id):
    code, text = cli('datasets', 'status', dataset_id, allow_failure=True)
    return code == 0 and 'NotFound' not in text and '404' not in text


def upload_dir(folder, dataset_id, message):
    if dataset_exists(dataset_id):
        cli('datasets', 'version', '-p', str(folder), '-m', message, timeout=7200)
    else:
        cli('datasets', 'create', '-p', str(folder), timeout=7200)
    print(f'uploaded {dataset_id}', flush=True)


def hardlink_tree(files, dest):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for src in files:
        src = Path(src)
        target = dest / src.name
        try:
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)


def meta(dataset_id, title):
    return {'id': dataset_id, 'title': title, 'licenses': [{'name': 'other'}]}


def write_kernel(folder, kid, sources, source):
    folder.mkdir(parents=True, exist_ok=True)
    body = {
        'id': kid,
        'title': kid.split('/')[1],
        'code_file': 'kernel.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': sources,
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }
    (folder / 'kernel-metadata.json').write_text(json.dumps(body, indent=2) + '\n')
    (folder / 'kernel.py').write_text(source)


def kernel_py(env):
    lines = ['import os, sys', 'from pathlib import Path']
    for key, value in env.items():
        lines.append(f'os.environ[{key!r}] = {str(value)!r}')
    lines.append("code = sorted(Path('/kaggle/input').rglob('h35c_run.py'))")
    lines.append('assert len(code) == 1, code')
    lines.append('sys.path.insert(0, str(code[0].parent))')
    lines.append('import h35c_run')
    lines.append('h35c_run.main()')
    lines.append('')
    return '\n'.join(lines)


def kernel_log(kid):
    code, text = cli('kernels', 'logs', kid, allow_failure=True, timeout=120)
    if code != 0:
        return ''
    return text


def current_state(kid):
    """RUNNING, QUEUED, COMPLETE, ERROR, CANCELLED, ABSENT, or RATE."""
    state, text = status_of(kid)
    low = text.lower()
    if state == 'UNKNOWN' and ('429' in low or 'too many' in low or 'rate limit' in low):
        return 'RATE'
    if state == 'UNKNOWN' and 'has status' not in low:
        return 'ABSENT'
    return state


def push_and_wait(folder, kid, est_h, committed_h):
    assert_budget(est_h, committed_h)
    while True:
        existing = current_state(kid)
        if existing in ('RUNNING', 'QUEUED', 'COMPLETE'):
            # A restarted dispatcher must not open a second session.
            print(f'resume {kid} {existing}', flush=True)
            break
        if existing == 'RATE':
            print(f'resume status paused {kid}', flush=True)
            time.sleep(60)
            continue
        wait_slot()
        running = account_active()
        if len(running) >= ACCOUNT_CAP:
            wait_slot()
        existing = current_state(kid)
        if existing in ('RUNNING', 'QUEUED', 'COMPLETE'):
            print(f'resume {kid} {existing}', flush=True)
            break
        if existing == 'RATE':
            print(f'resume status paused {kid}', flush=True)
            time.sleep(60)
            continue
        cli('kernels', 'push', '-p', str(folder), timeout=300)
        print(f'pushed {kid}', flush=True)
        existing = 'UNKNOWN'
        break
    state = existing
    t_run = None
    last_print = 0
    last_tail = ''
    last_change = time.time()
    last_log_check = 0
    last_stall = 0
    announced_over = False
    while True:
        state, text = status_of(kid)
        now = time.time()
        if state == 'RUNNING' and t_run is None:
            t_run = now
        if state != getattr(push_and_wait, 'prev', None) or now - last_print > 300:
            print(f'{kid} {state}', flush=True)
            push_and_wait.prev = state
            last_print = now
        if t_run is not None and now - t_run > KERNEL_LIMIT_S and not announced_over:
            announced_over = True
            print(f'OVER_3H {kid} running {now - t_run:.0f}s', flush=True)
        if now - last_log_check > 120 and state in ('RUNNING', 'QUEUED'):
            last_log_check = now
            log = kernel_log(kid)
            tail = ''
            for line in reversed(log.splitlines()):
                if line.strip():
                    tail = line.strip()[:240]
                    break
            if tail and tail != last_tail:
                last_tail = tail
                last_change = now
                print(f'progress {kid} {tail}', flush=True)
            elif state == 'RUNNING' and now - last_change > 1200 and now - last_stall > 300:
                last_stall = now
                print(f'STALL {kid} no log change for {now - last_change:.0f}s', flush=True)
        if state in ('COMPLETE', 'ERROR', 'CANCELLED'):
            session = None if t_run is None else now - t_run
            return state, session, text
        time.sleep(45)


def download(kid, dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    cli('kernels', 'output', kid, '-p', str(dest), timeout=600)


def arm_seconds(timing, size, arm, steps, delta):
    if arm in ('rand', 'heur', 'dedup', 'ppl', 'cls', 'pipeline', 'rand_plus'):
        n = steps + (delta if arm == 'rand_plus' else 0)
        return n * timing[f'{size}/rand']
    return steps * timing[f'{size}/{arm}']


def experiment_seconds(timing, delta, steps, seeds):
    arms = (
        'rand', 'heur', 'dedup', 'ppl', 'cls', 'pipeline',
        'high_loss', 'rho', 'prog_snap', 'rand_plus',
    )
    total = 0.0
    per = {}
    for size, seed_list in seeds.items():
        for seed in seed_list:
            for arm in arms:
                sec = arm_seconds(timing, size, arm, steps, delta[size]['delta'])
                # Eval is a few dozen forwards. Add 8% so a kernel is not packed to the edge.
                sec *= 1.08
                per[f'{size}/s{seed}/{arm}'] = sec
                total += sec
    return total, per


def pack(per, steps):
    """Bin-pack arms of one size and seed under PACK_LIMIT_S."""
    jobs = []
    groups = {}
    for key, sec in per.items():
        size, seed, arm = key.split('/')
        groups.setdefault((size, seed), []).append((arm, sec))
    for (size, seed), items in sorted(groups.items()):
        items = sorted(items, key=lambda x: -x[1])
        bins = []
        for arm, sec in items:
            if sec > KERNEL_LIMIT_S:
                raise RuntimeError(f'{size} {seed} {arm} estimates {sec/3600:.2f}h')
            placed = False
            for bucket in bins:
                if bucket['seconds'] + sec <= PACK_LIMIT_S:
                    bucket['arms'].append(arm)
                    bucket['seconds'] += sec
                    placed = True
                    break
            if not placed:
                bins.append({'arms': [arm], 'seconds': sec})
        for i, bucket in enumerate(bins):
            tag = f'{size[0]}{seed[-1]}{i}'
            jobs.append({
                'tag': tag,
                'id': f'ezzzzzx/h35c-{tag}',
                'size': size,
                'seed': int(seed[-1]),
                'arms': bucket['arms'],
                'est_s': bucket['seconds'],
                'steps': steps,
            })
    return jobs


def correct_cold_timings(smoke):
    """Drop small-model step times that still include CUDA compilation.

    A warmed train step is faster than a train step plus scoring. When the
    small rand or high_loss average exceeds the warmed prog_snap step, replace
    it with the medium time scaled by the warmed small/medium prog_snap ratio.
    """
    timing = dict(smoke.get('seconds_per_step') or {})
    needed = (
        'small/rand', 'small/high_loss', 'small/prog_snap',
        'medium/rand', 'medium/high_loss', 'medium/prog_snap',
    )
    if any(key not in timing or timing[key] <= 0 for key in needed):
        return smoke
    ratio = timing['small/prog_snap'] / timing['medium/prog_snap']
    raw = dict(timing)
    replaced = {}
    for arm in ('rand', 'high_loss'):
        key = f'small/{arm}'
        if timing[key] > timing['small/prog_snap']:
            used = timing[f'medium/{arm}'] * ratio
            replaced[key] = {'measured': timing[key], 'used': used}
            timing[key] = used
    if not replaced:
        return smoke
    smoke['seconds_per_step_raw'] = raw
    smoke['seconds_per_step'] = timing
    smoke['timing_correction'] = {
        'prog_ratio': ratio,
        'replaced': replaced,
    }
    return smoke


def choose_plan(smoke):
    """Full plan, then the two allowed cuts. A single arm over 3h does not fit."""
    timing = smoke['seconds_per_step']
    base_delta = smoke['delta']
    supply = smoke['survival']
    need = supply['need_sequences']
    short = [name for name in ('heur', 'dedup', 'ppl', 'cls', 'pipeline')
             if supply[name]['sequences'] < need]
    if short:
        return {
            'ok': False, 'short_arms': short, 'steps': 3000,
            'seeds': {'small': [0, 1], 'medium': [0, 1]}, 'cuts': [],
            'hours': None, 'per': {}, 'delta': {},
            'stop': f'sequence supply below {need}: {short}',
        }
    smoke_h = max(float(smoke['wall_s']), float(smoke.get('session_s') or 0)) / 3600
    candidates = (
        (3000, {'small': [0, 1], 'medium': [0, 1]}, []),
        (3000, {'small': [0], 'medium': [0, 1]}, ['small seeds to 1']),
        (2000, {'small': [0], 'medium': [0, 1]}, ['small seeds to 1', 'S to 2000']),
    )
    last = None
    for steps, seeds, cuts in candidates:
        scaled = {
            size: dict(base_delta[size], delta=math.ceil(base_delta[size]['delta'] * steps / 3000))
            for size in base_delta
        }
        total, per = experiment_seconds(timing, scaled, steps, seeds)
        hours = total / 3600 + smoke_h
        longest = max(per.values()) if per else 0.0
        ok = hours <= GPU_BUDGET_H and longest <= KERNEL_LIMIT_S
        last = {
            'ok': ok, 'steps': steps, 'seeds': seeds, 'cuts': list(cuts),
            'hours': hours, 'per': per, 'delta': scaled, 'short_arms': [],
            'longest_s': longest, 'smoke_h': smoke_h,
        }
        if ok:
            return last
    last['stop'] = (
        f'estimated {last["hours"]:.2f} GPU hours, longest arm '
        f'{last["longest_s"] / 3600:.2f}h after the allowed cuts'
    )
    return last


def prepare_uploads():
    data = ROOT / 'data' / 'h35c'
    files = [
        'pool.bin', 'pool_offsets.npy', 'pool_manifest.json', 'pool_meta.jsonl',
        'c4_val.bin', 'c4_val_offsets.npy', 'c4_val_manifest.json',
        'wikitext103_test.bin', 'wikitext_manifest.json',
        'wiki_ref.bin', 'wiki_ref_manifest.json',
        'filters_cpu.npz', 'filters_cpu.json',
    ]
    folder = BASE / 'data'
    hardlink_tree([data / name for name in files], folder)
    (folder / 'dataset-metadata.json').write_text(
        json.dumps(meta(DATA_ID, 'pdp h35c data'), indent=2) + '\n')
    cls = BASE / 'cls'
    hardlink_tree(sorted((data / 'cls').iterdir()), cls)
    (cls / 'dataset-metadata.json').write_text(
        json.dumps(meta(CLS_ID, 'pdp h35c fineweb edu'), indent=2) + '\n')
    code = BASE / 'code'
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    for name in ('h35c_web.py', 'h35c_run.py'):
        shutil.copy2(ROOT / name, code / name)
    (code / 'dataset-metadata.json').write_text(
        json.dumps(meta(CODE_ID, 'pdp code h35c'), indent=2) + '\n')
    return folder, cls, code


def write_cpu_preview():
    """Pool size, three documents, and the CPU filter rates. Seed 0 is recorded."""
    import numpy as np

    data = ROOT / 'data' / 'h35c'
    offsets = np.load(data / 'pool_offsets.npy')
    blob = np.memmap(data / 'pool.bin', dtype=np.uint8, mode='r')
    n = int(len(offsets) - 1)
    rng = np.random.default_rng(0)
    pick = [int(i) for i in rng.choice(n, size=3, replace=False)]
    wanted = set(pick)
    meta_by_id = {}
    with (data / 'pool_meta.jsonl').open() as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get('pool_id') in wanted:
                meta_by_id[obj['pool_id']] = obj
                if len(meta_by_id) == len(wanted):
                    break
    docs = []
    for doc_id in pick:
        a = int(offsets[doc_id])
        b = int(offsets[doc_id + 1])
        info = meta_by_id.get(doc_id, {})
        docs.append({
            'pool_id': doc_id,
            'shard': info.get('shard'),
            'line': info.get('line'),
            'n_bytes': int(b - a),
            'preview': bytes(blob[a:a + 200]).decode('utf-8', errors='replace'),
        })
    out = {
        'preview_seed': 0,
        'pool': json.loads((data / 'pool_manifest.json').read_text()),
        'c4_val': json.loads((data / 'c4_val_manifest.json').read_text()),
        'wikitext': json.loads((data / 'wikitext_manifest.json').read_text()),
        'wiki_ref': json.loads((data / 'wiki_ref_manifest.json').read_text()),
        'filters': json.loads((data / 'filters_cpu.json').read_text()),
        'docs': docs,
        'wikitext_head': bytes(np.memmap(data / 'wikitext103_test.bin', dtype=np.uint8, mode='r')[:40]).decode('utf-8', errors='replace'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cpu_preview.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    filt = out['filters']
    print(
        f"cpu preview docs {out['pool']['n_docs']} bytes {out['pool']['n_bytes']} "
        f"heur {filt['heur_docs']} dedup {filt['dedup_docs']} "
        f"pipe_before_cls {filt['pipe_before_cls_docs']} "
        f"pipe_seqs {filt['pipe_before_cls_sequences']}",
        flush=True)
    return out


def save_plan(plan):
    saved = {
        'ok': plan['ok'],
        'steps': plan['steps'],
        'seeds': plan['seeds'],
        'cuts': plan['cuts'],
        'hours': plan['hours'],
        'short_arms': plan['short_arms'],
        'delta': plan['delta'],
        'per': plan['per'],
        'longest_s': plan.get('longest_s'),
        'smoke_h': plan.get('smoke_h'),
    }
    if 'stop' in plan:
        saved['stop'] = plan['stop']
    (OUT / 'plan.json').write_text(json.dumps(saved, indent=2) + '\n')
    brief = {k: saved[k] for k in ('ok', 'steps', 'seeds', 'cuts', 'hours', 'short_arms', 'longest_s')}
    if 'stop' in saved:
        brief['stop'] = saved['stop']
    print('PLAN ' + json.dumps(brief), flush=True)
    return saved


def run_kernel_once(folder, kid, est_h, committed_h):
    """Push one kernel. A Python traceback is not retried; other failures are."""
    state, session, text = push_and_wait(folder, kid, est_h, committed_h)
    log = kernel_log(kid)
    if state == 'COMPLETE':
        return state, session, log, 1
    (OUT / 'failed-logs').mkdir(parents=True, exist_ok=True)
    slug = kid.split('/')[-1]
    (OUT / 'failed-logs' / f'{slug}-1.log').write_text(log or text)
    if 'Traceback (most recent call last)' in log:
        print(f'code error in {kid}; not retrying the same kernel', flush=True)
        return state, session, log, 1
    print(f'retry {kid} after {state}', flush=True)
    spent_h = (session or 0.0) / 3600.0
    state2, session2, text2 = push_and_wait(folder, kid, est_h, committed_h + spent_h)
    log = kernel_log(kid)
    if state2 != 'COMPLETE':
        (OUT / 'failed-logs' / f'{slug}-2.log').write_text(log or text2)
    spent = (session or 0.0) + (session2 or 0.0)
    return state2, spent if spent else None, log, 2


def phase_smoke(committed_h, upload='all'):
    write_cpu_preview()
    data_dir, cls_dir, code_dir = prepare_uploads()
    if upload == 'all':
        upload_dir(data_dir, DATA_ID, 'h35c pool eval filters')
        upload_dir(cls_dir, CLS_ID, 'fineweb-edu-classifier')
    upload_dir(code_dir, CODE_ID, 'h35c code')
    smoke_id = 'ezzzzzx/h35c-smoke'
    smoke_dir = BASE / 'smoke'
    write_kernel(
        smoke_dir, smoke_id, [DATA_ID, CLS_ID, CODE_ID],
        kernel_py({'H35C_MODE': 'smoke'}))
    state, session, log, attempts = run_kernel_once(smoke_dir, smoke_id, SMOKE_EST_H, committed_h)
    dest = OUT / 'smoke-kernel'
    try:
        download(smoke_id, dest)
    except Exception as exc:
        print(f'download smoke failed: {type(exc).__name__}', flush=True)
        dest.mkdir(parents=True, exist_ok=True)
    (dest / 'kernel.log').write_text(log)
    if state != 'COMPLETE':
        raise RuntimeError(f'smoke kernel {state}')
    smoke_path = dest / 'smoke.json'
    if not smoke_path.is_file():
        raise RuntimeError('smoke kernel completed without smoke.json')
    smoke = json.loads(smoke_path.read_text())
    smoke['kernel'] = smoke_id
    smoke['session_s'] = session
    smoke['attempts'] = attempts
    (OUT / 'smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
    if not smoke.get('ok'):
        print('SMOKE_DONE ' + json.dumps({'ok': False, 'stop': smoke.get('stop'), 'kernel': smoke_id}), flush=True)
        return smoke, None
    smoke = correct_cold_timings(smoke)
    (OUT / 'smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
    plan = choose_plan(smoke)
    save_plan(plan)
    print('SMOKE_DONE ' + json.dumps({'ok': True, 'plan_ok': plan['ok'], 'kernel': smoke_id}), flush=True)
    return smoke, plan


def archive_stale_kernels():
    """Move kernel outputs that predate the 128-byte peek and K=96 amendment."""
    kernels = OUT / 'kernels'
    if not kernels.is_dir():
        return
    stale_root = OUT / 'stale-peek512'
    for result in list(kernels.glob('*/result.json')):
        body = json.loads(result.read_text())
        drop = False
        for row in body.get('rows') or []:
            arm = row.get('arm')
            if arm in ('high_loss', 'rho', 'prog_snap'):
                if row.get('peek_bytes') != 128 or row.get('k_cand') != 96:
                    drop = True
            if arm == 'rand_plus' and 'sequences_reread' not in row:
                drop = True
        if not drop:
            continue
        dest = stale_root / result.parent.name
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(result.parent), str(dest))
        print(f'archived stale {result.parent.name}', flush=True)


def measured_mean_peek(limit):
    import numpy as np
    offsets = np.load(ROOT / 'data' / 'h35c' / 'pool_offsets.npy')
    lengths = np.diff(offsets.astype(np.int64))
    return float(np.minimum(lengths, limit).mean()), int(lengths.min()), int(lengths.max())


def amend_smoke_for_plan(smoke):
    """Recompute delta and wall estimates for 128-byte peeks and 96 candidates.

    The smoke kernel measured K=128 and ~506-byte peeks. Those timings stay in
    smoke.json. This returns a copy for choose_plan only.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    import h35c_run
    import h35c_web as web

    mean_peek, min_len, max_len = measured_mean_peek(web.PEEK)
    old_peek = float(smoke['mean_peek_bytes'])
    draws = float(smoke['delta']['small']['draws_per_step'])
    ratio = (web.K_CAND / 128.0) * (mean_peek / old_peek)
    tokens = 2.0 * web.K_CAND * mean_peek * draws
    safety = 1.5
    delta = {}
    for size in ('small', 'medium'):
        steps_delta, select, step, per_token = h35c_run.delta_steps(
            smoke['flops'][size], tokens, web.S_DEFAULT)
        delta[size] = {
            'delta': int(steps_delta),
            'selection_flops_at_S': float(select),
            'train_step_flops': float(step),
            'forward_flops_per_token': float(per_token),
            'formula_tokens_per_step': float(tokens),
            'draws_per_step': draws,
            'peek_bytes': int(web.PEEK),
            'k_cand': int(web.K_CAND),
        }
    timing = {}
    for key, measured in smoke['seconds_per_step'].items():
        size, arm = key.split('/')
        base = float(smoke['seconds_per_step'][f'{size}/rand'])
        if arm == 'rand':
            timing[key] = float(measured)
        else:
            excess = max(0.0, float(measured) - base)
            timing[key] = base + excess * ratio * safety
    revised = dict(smoke)
    revised['delta'] = delta
    revised['seconds_per_step'] = timing
    revised['mean_peek_bytes'] = mean_peek
    note = {
        'peek_bytes': int(web.PEEK),
        'k_cand': int(web.K_CAND),
        'old_mean_peek_bytes': old_peek,
        'new_mean_peek_bytes': mean_peek,
        'min_doc_bytes': min_len,
        'max_doc_bytes': max_len,
        'token_ratio': ratio,
        'wall_safety_on_selection': safety,
        'draws_per_step': draws,
        'rand_plus_epochs': 2,
        'note': (
            'Delta is prog_snap scoring FLOPs at K=96 and 128-byte peeks, charged at the '
            'smoke kernel forward cost per token, divided by one train step, rounded up. '
            'Online wall estimates keep the measured train step and scale the excess by '
            'that token ratio times 1.5. Completed rand runs stay valid. The smoke kernel '
            'was not re-run.'
        ),
    }
    return revised, note


def write_amended_plan():
    smoke = json.loads((OUT / 'smoke.json').read_text())
    if not smoke.get('ok'):
        raise RuntimeError('smoke is not ok')
    revised, note = amend_smoke_for_plan(smoke)
    plan = choose_plan(revised)
    pool_seq = int(smoke['survival']['pool_sequences'])
    supply = {}
    for size, body in (plan.get('delta') or {}).items():
        need = (int(plan['steps']) + int(body['delta'])) * 32
        supply[size] = {
            'need_sequences': need,
            'pool_sequences': pool_seq,
            'epochs_needed': 1 if need <= pool_seq else 2,
            'fits_two_epochs': need <= pool_seq * 2,
            'sequences_reread_if_full': max(0, need - pool_seq),
        }
        if need > pool_seq * 2:
            plan['ok'] = False
            plan['stop'] = (
                f'{size} rand_plus needs {need} sequences; two epochs supply {pool_seq * 2}'
            )
    plan['amendment'] = note
    plan['rand_plus_supply'] = supply
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    brief = {
        'ok': plan.get('ok'),
        'steps': plan.get('steps'),
        'seeds': plan.get('seeds'),
        'cuts': plan.get('cuts'),
        'hours': plan.get('hours'),
        'longest_s': plan.get('longest_s'),
        'delta': {k: int(v['delta']) for k, v in (plan.get('delta') or {}).items()},
        'supply': supply,
        'stop': plan.get('stop'),
    }
    print('PLAN ' + json.dumps(brief), flush=True)
    return plan


def phase_scored(committed_h):
    plan = json.loads((OUT / 'plan.json').read_text())
    if not plan.get('ok'):
        print('SCORED_DONE ' + json.dumps({'ok': False, 'stop': plan.get('stop')}), flush=True)
        return
    amend = plan.get('amendment') or {}
    if amend.get('peek_bytes') != 128 or amend.get('k_cand') != 96:
        raise RuntimeError('plan.json is not the 128-byte peek, K=96 plan')
    print('upload amended code', flush=True)
    _, _, code_dir = prepare_uploads()
    upload_dir(code_dir, CODE_ID, 'h35c peek 128 K 96 second epoch')
    archive_stale_kernels()
    committed_h, history = accounted_hours()
    write_committed(committed_h, history)
    smoke_dest = OUT / 'smoke-kernel'
    score_dir = BASE / 'scores'
    hardlink_tree(
        [smoke_dest / 'ppl_scores.npy', smoke_dest / 'cls_scores.npy', smoke_dest / 'ref.pt'],
        score_dir)
    (score_dir / 'dataset-metadata.json').write_text(
        json.dumps(meta(SCORE_ID, 'pdp h35c scores'), indent=2) + '\n')
    upload_dir(score_dir, SCORE_ID, 'ppl cls ref')
    jobs = pack(plan['per'], plan['steps'])
    (OUT / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print(f'jobs {len(jobs)} est_hours {plan["hours"]:.2f} inflight {INFLIGHT_CAP}', flush=True)
    runs = []
    if (OUT / 'runs.json').is_file():
        runs = json.loads((OUT / 'runs.json').read_text())
    done_ids = {row['id'] for row in runs if row.get('state') == 'COMPLETE'}
    for job in jobs:
        if job['id'] in done_ids:
            print(f'skip complete {job["id"]}', flush=True)
            continue
        prev = next((row for row in runs if row.get('id') == job['id']), None)
        if prev and prev.get('state') != 'COMPLETE' and int(prev.get('attempts') or 0) >= 2:
            raise RuntimeError(f'{job["id"]} already failed {prev.get("attempts")} attempts')
        est_h = job['est_s'] / 3600
        committed_h, history = accounted_hours()
        if committed_h + est_h > GPU_BUDGET_H + 1e-9:
            print(f'STOP budget {committed_h:.2f}+{est_h:.2f} > {GPU_BUDGET_H}', flush=True)
            write_committed(committed_h, history)
            print('SCORED_DONE ' + json.dumps({
                'ok': False, 'stop': 'budget', 'committed_h': committed_h, 'next_h': est_h,
            }), flush=True)
            return
        delta = plan['delta'][job['size']]['delta']
        folder = BASE / job['tag']
        write_kernel(
            folder, job['id'], [DATA_ID, CODE_ID, SCORE_ID],
            kernel_py({
                'H35C_MODE': 'scored',
                'H35C_ARMS': ','.join(job['arms']),
                'H35C_SIZE': job['size'],
                'H35C_SEED': str(job['seed']),
                'H35C_STEPS': str(plan['steps']),
                'H35C_DELTA': str(delta),
            }))
        state, session, log, attempts = run_kernel_once(folder, job['id'], est_h, committed_h)
        kdest = OUT / 'kernels' / job['tag']
        try:
            download(job['id'], kdest)
        except Exception as exc:
            print(f'download {job["id"]} failed: {type(exc).__name__}', flush=True)
            kdest.mkdir(parents=True, exist_ok=True)
        (kdest / 'kernel.log').write_text(log)
        prev_sec = float(prev.get('session_s') or 0.0) if prev else 0.0
        prev_attempts = int(prev.get('attempts') or 0) if prev else 0
        row = {
            **job,
            'state': state,
            'session_s': prev_sec + (session or 0.0),
            'attempts': prev_attempts + attempts,
        }
        runs = [r for r in runs if r.get('id') != job['id']]
        runs.append(row)
        (OUT / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
        committed_h, history = accounted_hours()
        write_committed(committed_h, history)
        if state != 'COMPLETE':
            raise RuntimeError(f'{job["id"]} {state}')
    committed_h, history = accounted_hours()
    write_committed(committed_h, history)
    print('SCORED_DONE ' + json.dumps({'kernels_done': len(runs), 'committed_h': committed_h}), flush=True)


def selftest():
    def smoke_for(small_rand, small_on, medium_rand, medium_on, delta_s, delta_m, seq=200000, wall=1000):
        timing = {}
        for size, rand_t, on_t in (
            ('small', small_rand, small_on), ('medium', medium_rand, medium_on),
        ):
            timing[f'{size}/rand'] = rand_t
            for arm in ('high_loss', 'rho', 'prog_snap'):
                timing[f'{size}/{arm}'] = on_t
        survival = {'need_sequences': 96000}
        for name in ('heur', 'dedup', 'ppl', 'cls', 'pipeline'):
            survival[name] = {'sequences': seq}
        return {
            'wall_s': wall, 'session_s': wall,
            'seconds_per_step': timing,
            'delta': {
                'small': {'delta': delta_s},
                'medium': {'delta': delta_m},
            },
            'survival': survival,
        }

    fast = choose_plan(smoke_for(0.05, 0.2, 0.08, 0.3, 100, 150))
    assert fast['ok'] and fast['steps'] == 3000 and fast['cuts'] == [], fast
    short = choose_plan(smoke_for(0.05, 0.2, 0.08, 0.3, 100, 150, seq=1000))
    assert short['ok'] is False and 'pipeline' in short['short_arms'], short
    cut = choose_plan(smoke_for(0.4, 0.4, 0.5, 0.5, 2000, 2000, wall=360))
    assert cut['ok'] and cut['cuts'] == ['small seeds to 1'] and cut['steps'] == 3000, cut
    long_arm = choose_plan(smoke_for(0.2, 1.5, 1.2, 4.0, 8000, 8000, wall=60))
    assert long_arm['steps'] == 2000 and long_arm['seeds']['small'] == [0], long_arm
    assert long_arm['longest_s'] <= KERNEL_LIMIT_S if long_arm['ok'] else True
    print('selftest ok', json.dumps({
        'fast_h': round(fast['hours'], 3),
        'cut_h': round(cut['hours'], 3),
        'long_ok': long_arm['ok'],
        'long_h': round(long_arm['hours'], 3),
        'long_arm_h': round(long_arm['longest_s'] / 3600, 3),
        'long_cuts': long_arm['cuts'],
    }))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if phase == 'selftest':
        selftest()
        return
    if phase == 'plan':
        write_amended_plan()
        return
    if phase not in ('smoke', 'smoke-code', 'scored', 'all'):
        raise SystemExit(f'unknown phase {phase}')
    OUT.mkdir(parents=True, exist_ok=True)
    load_token()
    lock_f = open(LOCK, 'w')
    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    committed_h = 0.0
    history = []
    if (OUT / 'committed.json').is_file():
        prior = json.loads((OUT / 'committed.json').read_text())
        committed_h = float(prior.get('committed_h') or 0.0)
        history = list(prior.get('kernels') or [])
    if phase in ('smoke', 'smoke-code', 'all'):
        smoke, plan = phase_smoke(committed_h, upload='code' if phase == 'smoke-code' else 'all')
        session = smoke.get('session_s') or 0
        committed_h += session / 3600
        history.append({'id': smoke.get('kernel'), 'session_s': session, 'ok': bool(smoke.get('ok'))})
        (OUT / 'committed.json').write_text(json.dumps({'committed_h': committed_h, 'kernels': history}) + '\n')
        if phase in ('smoke', 'smoke-code') or not smoke.get('ok') or plan is None or not plan.get('ok'):
            return
    if phase == 'scored':
        if (OUT / 'committed.json').is_file():
            committed_h = json.loads((OUT / 'committed.json').read_text()).get('committed_h') or 0.0
        phase_scored(committed_h)
    elif phase == 'all':
        phase_scored(committed_h)


if __name__ == '__main__':
    main()
