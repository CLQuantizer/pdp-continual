"""Exercise report arithmetic and provenance with synthetic, unscored fixtures."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import dispatch as d
import summarize as s
h=d.h


def main():
    manifest,fingerprint=d.prepare()
    jobs=d.job_plan(manifest)
    values={'plain_1':(4.,4.),'zgate_6':(3.8,3.95),
            'dense_6':(3.77,3.92),'dense_8':(3.75,3.915)}
    with tempfile.TemporaryDirectory(dir=d.OUT,prefix='summary-') as tmp:
        out=Path(tmp)
        d.dump(out/'smoke.json',json.loads((d.OUT/'smoke.json').read_text()))
        s.assemble(out)
        empty=json.loads((out/'results.json').read_text())
        assert len(empty['missing'])==len(jobs) and not empty['runs']
        assert all(r['seeds_evaluated']==r['seeds_meeting']==0 and r['mean_gap'] is None for r in empty['scoring'])
        fixtures={}
        for job in jobs:
            folder=out/'kernels'/job['tag']; folder.mkdir(parents=True)
            checkpoint=job['tag'].encode(); (folder/'checkpoint.pt').write_bytes(checkpoint)
            r=dict(code_fingerprint=fingerprint,kernel_id=job['id'],arm=job['arm'],seed=job['seed'],
                n_bytes=h.N_DATA,part=job['part'],start=job['start'],end=job['end'],
                raw_sha256=manifest['raw_sha256'],sha256_e10m=manifest['sha256_e10m'],
                segment_wall_seconds=1.,kernel_wall_seconds=2.,wall_seconds=float(job['part']),
                checkpoint_sha256=h.sha(checkpoint),
                resumed_checkpoint_sha256=fixtures[job['parent_tag']]['checkpoint_sha256'] if job['parent_tag'] else None)
            if job['end']<h.N_DATA:
                r.update(bytes_seen=job['end'],offset=job['end'],pending_bytes=0)
            else:
                depth=int(job['arm'].split('_')[1]); value=values[job['arm']][job['seed']]
                slices=[dict(start=start,end=start+h.SLICE,bits_per_byte=value) for start in range(0,h.N_DATA,h.SLICE)]
                r.update(tail_start=h.TAIL_START,initial_mixer_weights=h.initial_gate(job['arm'])[0].tolist(),
                    slices=slices,bits_per_byte=value,tail_bits_per_byte=value,first_200kb_bits_per_byte=value,
                    mixer_slices=[dict(start=x['start'],end=x['end'],mean_weights=[1./depth]*depth) for x in slices],
                    layers=[dict(own_bits_per_byte=value,tail_own_bits_per_byte=value,
                        slices=[dict(start=x['start'],end=x['end'],own_bits_per_byte=value) for x in slices]) for _ in range(depth)])
            fixtures[job['tag']]=r; d.dump(folder/'results.json',r)
        s.assemble(out)
        result=json.loads((out/'results.json').read_text())
        assert len(result['runs'])==8 and len(result['parts'])==len(jobs) and not result['missing']
        assert [x['seeds_meeting'] for x in result['scoring']]==[1,2,1]*2
        for row in result['scoring']:
            expected=[values[row['left']][seed]-values[row['right']][seed] for seed in (0,1)]
            assert row['seed_gaps']==expected and row['mean_gap']==sum(expected)/2
        last=jobs[-1]; path=out/'kernels'/last['tag']/'results.json'; original=fixtures[last['tag']]
        rejected=0
        for changes in ({'code_fingerprint':'wrong'},{'checkpoint_sha256':'wrong'},
                        {'resumed_checkpoint_sha256':'wrong'},{'bits_per_byte':99.},
                        {'kernel_wall_seconds':5101.}):
            d.dump(path,dict(original,**changes))
            try: s.assemble(out)
            except AssertionError: rejected+=1
            else: raise AssertionError('Corrupt fixture accepted: '+str(changes))
        d.dump(path,original)
        parent=next(j for j in jobs if j['tag']==last['parent_tag'])
        (out/'kernels'/parent['tag']/'results.json').unlink()
        try: s.assemble(out)
        except AssertionError: rejected+=1
        else: raise AssertionError('Missing parent accepted')
    report=dict(ok=True,code_fingerprint=fingerprint,synthetic_lives=8,
                synthetic_parts=len(jobs),invalid_outputs_rejected=rejected,empty_report_checked=True)
    d.dump(d.OUT/'summary-checks.json',report); print(json.dumps(report))

if __name__=='__main__': main()
