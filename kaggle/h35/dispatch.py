"""Resumable foreground H35 dispatcher.

Two private CPU kernels per seed, because prog_peek needs the whole machine:
  ezzzzzx/h35-s<c>    rand, oracle, high_loss, rho   workers=4 peek_workers=1
  ezzzzzx/h35-s<c>-p  prog_peek                      workers=1 peek_workers=4

Account cap: total running kernels <= 5. While the H34a2 dispatcher is alive
it holds up to 3 of those slots, so this dispatcher leaves them. A failed
kernel is pushed once more. Does not commit. Never prints the API token.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'kaggle' / 'h35'
OUT = ROOT / 'results' / 'h35'
STATE = OUT / 'dispatch.json'
CLI = str(ROOT / '.venv' / 'bin' / 'kaggle')
FILES = (
    'h26_cued.py',
    'h32_mixer.py',
    'h32b_mixer.py',
    'h35_corpus.py',
    'h35_data_diet.py',
)
CODE_ID = 'ezzzzzx/pdp-code-h35'
DATA_ID = 'ezzzzzx/enwik8'
ACCOUNT_CAP = 5
H34A2_CAP = 3


def job_specs():
    specs = []
    for seed in range(6):
        specs.append(dict(
            tag=f's{seed}',
            id=f'ezzzzzx/h35-s{seed}',
            seed=seed,
            policies=['rand', 'oracle', 'high_loss', 'rho'],
            workers=4,
            peek_workers=1,
        ))
        specs.append(dict(
            tag=f's{seed}-p',
            id=f'ezzzzzx/h35-s{seed}-p',
            seed=seed,
            policies=['prog_peek'],
            workers=1,
            peek_workers=4,
        ))
    return specs


def load_token():
    if os.environ.get('KAGGLE_API_TOKEN'):
        return
    for raw in (ROOT / '.env').read_text().splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        if key.strip() != 'KAGGLE_API_TOKEN':
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not val:
            raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN'] = val
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args, allow_failure=False, timeout=180):
    proc = subprocess.run(
        [CLI, *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    token = os.environ.get('KAGGLE_API_TOKEN') or '\0'
    output = proc.stdout.replace(token, '[REDACTED]')
    cli.last_code = proc.returncode
    if proc.returncode and not allow_failure:
        raise RuntimeError(f'Kaggle command failed: {args[:2]}\n{output}')
    return output


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def kernel_source(spec):
    return f'''"""Private CPU H35 seed {spec["seed"]} policies {spec["policies"]}."""
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

code = sorted(Path("/kaggle/input").rglob("h35_data_diet.py"))
assert len(code) == 1, code
os.environ["PYTHONPATH"] = str(code[0].parent)
sys.path.insert(0, str(code[0].parent))
import h35_data_diet as h

# Kaggle re-executes this file in spawn children. The guard keeps that
# re-exec from starting another pool.
if __name__ == "__main__":
    h.kernel_main(
        {spec["seed"]},
        "/kaggle/working",
        {spec["id"]!r},
        policies={spec["policies"]!r},
        budget=800,
        workers={spec["workers"]},
        peek_workers={spec["peek_workers"]},
    )
'''


def prepare():
    code = BASE / 'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    manifest = BASE / 'code-sha256.json'
    if STATE.exists() and not manifest.is_file():
        raise RuntimeError('Dispatch state exists without its code manifest')
    if STATE.exists() and json.loads(manifest.read_text()) != hashes:
        raise RuntimeError('Source changed since dispatch preparation; refusing mixed code versions')
    for name in FILES:
        shutil.copyfile(ROOT / name, code / name)
    (code / 'dataset-metadata.json').write_text(json.dumps(dict(
        title='pdp code h35',
        id=CODE_ID,
        licenses=[dict(name='other')],
    ), indent=2) + '\n')
    manifest.write_text(json.dumps(hashes, indent=2) + '\n')
    for spec in job_specs():
        folder = BASE / spec['tag']
        folder.mkdir(parents=True, exist_ok=True)
        meta = dict(
            id=spec['id'],
            title=spec['id'].split('/', 1)[1],
            code_file='kernel.py',
            language='python',
            kernel_type='script',
            is_private=True,
            enable_gpu=False,
            enable_tpu=False,
            enable_internet=False,
            dataset_sources=[DATA_ID, CODE_ID],
            competition_sources=[],
            kernel_sources=[],
            model_sources=[],
        )
        (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        (folder / 'kernel.py').write_text(kernel_source(spec))


def status_kind(report):
    low = report.lower().split('has status', 1)[-1].splitlines()[0]
    if 'complete' in low:
        return 'complete'
    if any(x in low for x in ('error', 'cancel', 'fail')):
        return 'failed'
    if any(x in low for x in ('running', 'queued', 'pending')):
        return 'active'
    raise RuntimeError('Unknown remote status; refusing to submit more work: ' + report)


CACHE = OUT / 'kernel-status-cache.json'


def load_cache():
    if not CACHE.is_file():
        return {}
    data = json.loads(CACHE.read_text())
    return data if isinstance(data, dict) else {}


def save_cache(cache):
    tmp = CACHE.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache, indent=2) + '\n')
    tmp.replace(CACHE)


def listed_kernels():
    report = cli('kernels', 'list', '--mine', '--sort-by', 'dateRun', '--page-size', '100', '--format', 'json')
    start, end = report.find('['), report.rfind(']')
    if start < 0 or end < start:
        raise RuntimeError('kernel list was not a JSON array')
    data = json.loads(report[start:end + 1])
    if not isinstance(data, list) or not data or not all(isinstance(row, dict) and row.get('ref') for row in data):
        raise RuntimeError('kernel list returned no rows')
    return [(row['ref'], row.get('lastRunTime')) for row in data]


def other_active_ids():
    found = []
    for path in (ROOT / 'results').glob('*/dispatch.json'):
        if path.parent == OUT:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        jobs = data.get('jobs') if isinstance(data, dict) else None
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job.get('status') not in ('active', 'submitting'):
                continue
            kid = job.get('id') or job.get('kernel_id')
            if isinstance(kid, str):
                found.append(kid)
    return found


def dispatcher_alive(fragment):
    proc = Path('/proc')
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        if fragment in cmd and 'dispatch.py' in cmd:
            return True
    return False


def account_running(own_active):
    """Kernel refs that occupy a slot (running, queued, or pending)."""
    cache = load_cache()
    rows = listed_kernels()
    candidates = []
    seen = set()
    for ref, stamp in rows:
        seen.add(ref)
        cached = cache.get(ref)
        if isinstance(cached, dict) and cached.get('lastRunTime') == stamp and cached.get('kind') in ('complete', 'failed'):
            continue
        candidates.append((ref, stamp))
    for ref in list(own_active) + other_active_ids():
        if ref not in seen and ref not in {item[0] for item in candidates}:
            candidates.append((ref, None))
    running = set(own_active)
    print(f'account slot check: {len(candidates)} statuses, {len(own_active)} own active', flush=True)
    for ref, stamp in candidates:
        if ref in own_active:
            continue
        report = ''
        for attempt in range(4):
            report = cli('kernels', 'status', ref, allow_failure=True)
            low = report.lower()
            if 'has status' in low or any(x in low for x in ('403', '404', 'not found', "don't have access", 'forbidden')):
                break
            print('status retry ' + ref, flush=True)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('Refusing to push while a kernel status is unknown: ' + ref)
        low = report.lower()
        if 'has status' not in low:
            if any(x in low for x in ('403', '404', 'not found', "don't have access", 'forbidden')):
                print('unreadable kernel, not counted ' + ref, flush=True)
                continue
            raise RuntimeError('Refusing to push while a kernel status is unknown: ' + ref)
        kind = status_kind(report)
        if stamp is not None and kind in ('complete', 'failed'):
            cache[ref] = dict(kind=kind, lastRunTime=stamp)
        if kind == 'active':
            running.add(ref)
    save_cache(cache)
    return running


def slots_used(running):
    """Running kernels, plus the slots H34a2's live dispatcher will fill."""
    used = set(running)
    if not dispatcher_alive('kaggle/h34a2/dispatch.py'):
        return len(used)
    h34 = {ref for ref in used if 'h34a2' in ref}
    return len(used) + max(0, H34A2_CAP - len(h34))


def ensure_dataset(state):
    if state.get('dataset_ready'):
        return
    report = cli('datasets', 'status', CODE_ID, allow_failure=True)
    low = report.lower()
    exists = any(x in low for x in ('ready', 'pending', 'processing'))
    if not exists:
        # datasets create defaults to private. Never pass --public.
        created = cli('datasets', 'create', '-p', str(BASE / 'code'), allow_failure=True)
        print(created, flush=True)
        if cli.last_code != 0:
            # A dataset created by an earlier attempt is updated in place.
            print(cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', 'h35 code'), flush=True)
    else:
        print(cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', 'h35 code'), flush=True)
    while True:
        report = cli('datasets', 'status', CODE_ID, allow_failure=True)
        low = report.lower()
        print(report.strip(), flush=True)
        if 'ready' in low:
            break
        if 'has status' not in low and any(x in low for x in ('403', '404', 'forbidden', "don't have access")):
            time.sleep(30)
            continue
        if any(x in low for x in ('error', 'fail')):
            raise RuntimeError('Code dataset processing failed')
        time.sleep(30)
    state['dataset_ready'] = True
    save_state(state)


def validate_result(job, smoke):
    dest = OUT / 'kernels' / job['tag']
    dest.mkdir(parents=True, exist_ok=True)
    print(cli('kernels', 'output', job['id'], '-p', str(dest), timeout=300), flush=True)
    path = dest / 'results.json'
    if not path.is_file():
        raise RuntimeError(f'{job["id"]} produced no results.json')
    payload = json.loads(path.read_text())
    if payload.get('kernel_id') != job['id'] or payload.get('seed') != job['seed']:
        raise RuntimeError(f'{job["id"]} result identity mismatch')
    if payload.get('budget') != 800:
        raise RuntimeError(f'{job["id"]} budget {payload.get("budget")}')
    got = set(payload.get('policies', {}))
    if got != set(job['policies']):
        raise RuntimeError(f'{job["id"]} policies {sorted(got)}')
    for policy, run in payload['policies'].items():
        if run.get('sha256_e10m') != smoke['sha256_e10m']:
            raise RuntimeError(f'{job["id"]} {policy} sha mismatch')
        if run.get('kernel_id') != job['id'] or run.get('seed') != job['seed'] or run.get('policy') != policy:
            raise RuntimeError(f'{job["id"]} {policy} identity mismatch')
        if run.get('budget') != 800:
            raise RuntimeError(f'{job["id"]} {policy} budget')
        docs = [row['docs'] for row in run['checkpoints']]
        if docs != [100, 200, 300, 400, 500, 600, 700, 800]:
            raise RuntimeError(f'{job["id"]} {policy} checkpoints {docs}')
        mean = sum(row['bits_per_byte'] for row in run['checkpoints']) / 8.0
        if abs(mean - run['T_clean']) > 1e-9:
            raise RuntimeError(f'{job["id"]} {policy} T_clean mismatch')
        if abs(run['final_bits_per_byte'] - run['checkpoints'][-1]['bits_per_byte']) > 1e-12:
            raise RuntimeError(f'{job["id"]} {policy} final mismatch')
        counts = run['kind_counts']
        if sum(counts.values()) != 800:
            raise RuntimeError(f'{job["id"]} {policy} kind count')
        if policy == 'oracle' and counts.get('clean') != 800:
            raise RuntimeError(f'{job["id"]} oracle read junk')
        if abs(sum(run['kind_share'].values()) - 1.0) > 1e-9:
            raise RuntimeError(f'{job["id"]} {policy} shares')
        if len(run['kinds_read']) != 800:
            raise RuntimeError(f'{job["id"]} {policy} kinds_read')
    walls = [run['wall_seconds'] for run in payload['policies'].values()]
    job['wall_seconds'] = max(walls)
    return payload


def assemble(state, smoke):
    import h35_data_diet as h

    by_seed = {str(seed): {} for seed in range(6)}
    kernel_rows = []
    for job in state['jobs']:
        payload = json.loads((OUT / 'kernels' / job['tag'] / 'results.json').read_text())
        for policy, run in payload['policies'].items():
            by_seed[str(job['seed'])][policy] = run
        kernel_rows.append(dict(
            id=job['id'],
            wall_seconds=job.get('wall_seconds'),
            slot_seconds=job.get('slot_seconds'),
            status=job['status'],
            tag=job['tag'],
            attempts=job['attempts'],
        ))
    for seed in range(6):
        missing = [p for p in h.POLICIES if p not in by_seed[str(seed)]]
        if missing:
            raise RuntimeError(f'seed {seed} missing {missing}')
    body = dict(
        seeds=list(range(6)),
        policies=list(h.POLICIES),
        budget=800,
        sha256_e10m=smoke['sha256_e10m'],
        by_seed=by_seed,
        smoke_lines=smoke.get('lines', []),
        kernels=kernel_rows,
        prog_peek_B100_seconds=smoke.get('prog_peek_B100_seconds'),
        prog_peek_B800_estimate_seconds=smoke.get('prog_peek_B800_estimate_seconds'),
    )
    h.write_results(body, str(OUT / 'RESULTS.md'), str(OUT / 'results.json'))
    print('wrote results/h35/RESULTS.md', flush=True)


def fresh_state():
    jobs = []
    for spec in job_specs():
        jobs.append(dict(
            tag=spec['tag'],
            id=spec['id'],
            seed=spec['seed'],
            policies=spec['policies'],
            workers=spec['workers'],
            peek_workers=spec['peek_workers'],
            status='pending',
            attempts=0,
            pushed_at=None,
            wall_seconds=None,
            slot_seconds=None,
        ))
    return dict(dataset_ready=False, jobs=jobs)


def dispatch():
    load_token()
    smoke = json.loads((OUT / 'smoke.json').read_text())
    if not smoke.get('ok'):
        raise RuntimeError('smoke did not pass')
    if not smoke.get('sha256_e10m'):
        raise RuntimeError('smoke has no sha256')
    state = json.loads(STATE.read_text()) if STATE.exists() else fresh_state()
    expected = {spec['tag']: spec['id'] for spec in job_specs()}
    got = {job['tag']: job['id'] for job in state['jobs']}
    if got != expected or len(state['jobs']) != 12:
        raise RuntimeError('dispatch.json jobs do not match the 12 H35 kernels')
    if any(job['status'] not in ('pending', 'submitting', 'active', 'complete', 'failed') for job in state['jobs']):
        raise RuntimeError('bad job status')
    if any(not 0 <= job['attempts'] <= 2 for job in state['jobs']):
        raise RuntimeError('bad attempt count')
    save_state(state)
    ensure_dataset(state)
    while True:
        for job in state['jobs']:
            if job['status'] == 'submitting':
                raise RuntimeError(
                    f"Ambiguous push for {job['id']}: inspect remote status, then set "
                    'dispatch.json status to active if accepted or pending if not')
            if job['status'] != 'active':
                continue
            report = cli('kernels', 'status', job['id'])
            print(report.strip(), flush=True)
            (OUT / f"{job['tag']}.status.txt").write_text(report)
            kind = status_kind(report)
            if kind == 'complete':
                validate_result(job, smoke)
                job['status'] = 'complete'
                if job.get('pushed_at'):
                    job['slot_seconds'] = time.time() - float(job['pushed_at'])
            elif kind == 'failed':
                dest = OUT / 'failed' / f"{job['tag']}-attempt{job['attempts']}"
                dest.mkdir(parents=True, exist_ok=True)
                print(cli('kernels', 'output', job['id'], '-p', str(dest), allow_failure=True, timeout=300), flush=True)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
            save_state(state)
        if any(job['status'] == 'failed' for job in state['jobs']):
            raise RuntimeError('Kernel exhausted its single retry')
        own_active = {job['id'] for job in state['jobs'] if job['status'] == 'active'}
        pending = [job for job in state['jobs'] if job['status'] == 'pending']
        if pending:
            running = account_running(own_active)
            room = ACCOUNT_CAP - slots_used(running)
            print(
                f'running={len(running)} reserved_slots={slots_used(running)} room={room} '
                f'own={len(own_active)} pending={len(pending)}',
                flush=True)
            if room <= 0:
                print('account cap 5; holding pushes', flush=True)
            elif pending:
                job = pending[0]
                if job['attempts'] >= 2:
                    raise RuntimeError('Retry limit reached')
                job.update(status='submitting', attempts=job['attempts'] + 1)
                save_state(state)
                print(cli('kernels', 'push', '-p', str(BASE / job['tag'])).strip(), flush=True)
                job['status'] = 'active'
                job['pushed_at'] = time.time()
                save_state(state)
        if all(job['status'] == 'complete' for job in state['jobs']):
            break
        time.sleep(30)
    assemble(state, smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'dispatch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        if args.prepare_only:
            print('prepared kaggle/h35', flush=True)
            return
        dispatch()


if __name__ == '__main__':
    main()
