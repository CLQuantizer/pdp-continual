"""Finish independent checks after a fail-fast smoke, preserving failed checks.

No model changes, no network. Refuses stale evidence or unexpected failure.
"""
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h34b_plain_depth as h


def main():
    out=ROOT/'results/h34b'
    payload=json.loads((out/'smoke.json').read_text())
    assert payload['source_hashes']==h.source_hashes()
    assert not payload['ok'] and payload['failure']=='AssertionError()'
    log=payload['log']
    assert [r['seed'] for r in log if r['check']=='h34a2_equivalence']==[0,1]
    assert all(r['max_bits_abs']==0 for r in log if r['check'] in ('h34a2_equivalence','prequential_prefix'))
    assert any(r['check']=='fixed_mixer' and r['max_change']==0 for r in log)
    assert [r['arm'] for r in log if r['check']=='stability_and_timing']==['plain_6']
    h.h33.configure(1)
    raw,digest=h.h33.load_checked(str(ROOT/'data/enwik8/enwik8'))
    assert digest==payload['sha256_e10m']
    result,bits=h.run('wide_6',0,raw[:16384],keep=True,slice_size=2048)
    row=dict(check='stability_and_timing',arm='wide_6',n_bytes=len(bits),
             first_2048_bits=float(bits[:2048].mean()),wall_seconds=result['wall_seconds'],
             estimate_250kb=result['wall_seconds']*250000/len(bits),limit_bits=9)
    log.append(row); print(json.dumps(row),flush=True)
    timing=[r for r in log if r['check']=='stability_and_timing']
    rates={r['arm']:r['wall_seconds']/r['n_bytes'] for r in timing}
    n=200000 if max(rates.values())*250000>5100 else 250000
    costs={a:h.compute_account(a)['macs_per_byte'] for a in h.ARMS}
    estimates={a:rates['plain_6']*n*costs[a]/costs['plain_6'] for a in h.ARMS}
    estimates['wide_6']=rates['wide_6']*n
    row=dict(check='wall_estimates',n_bytes=n,tail_start=50000,tail_end=n,
             estimated_kernel_seconds=estimates,limit=5100,
             method='Measured 16,384-byte one-thread rates for plain_6 and wide_6; remaining arms MAC-scaled from plain_6; no assumed Kaggle speedup')
    log.append(row); print(json.dumps(row),flush=True)
    failures=[f"{r['arm']} first 2048 bytes {r['first_2048_bits']:.9f} > 9 bits/byte" for r in timing if r['first_2048_bits']>9]
    failures += [f'{a} estimated {v:.3f} > 5100 seconds at {n} bytes' for a,v in estimates.items() if v>5100]
    payload.update(n_bytes=n,ok=not failures,failures=failures,
                   failure='; '.join(failures),completed_after_failfast=True)
    h.dump(out/'smoke.json',payload)
    print(json.dumps(dict(ok=payload['ok'],n_bytes=n,failures=failures)),flush=True)
    return 1 if failures else 0

if __name__=='__main__': sys.exit(main())
