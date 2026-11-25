"""Finish the interrupted final smoke cell, retaining validated prior evidence."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h34j_long as h


def main():
    h.h33.configure(1)
    out = h.ROOT/'results/h34j'
    payload = json.loads((out/'smoke.json').read_text())
    assert payload['source_hashes'] == h.source_hashes()
    assert (payload['n_bytes'], payload['tail_start'], payload['threads']) == (h.N_DATA,h.TAIL_START,1)
    raw,digest = h.h33.load_checked(str(h.ROOT/'data/enwik8/enwik8'))
    raw = raw[:h.N_DATA]
    assert digest == payload['sha256_e10m'] and h.sha(raw.tobytes()) == payload['raw_sha256']
    if payload['ok']:
        print('Smoke already complete for current sources and stream',flush=True)
        return
    expected = [(a,s,c) for a in h.ARMS for s in (0,1)
                for c in ('reference_equivalence','checkpoint_equivalence')]
    assert [(r['arm'],r['seed'],r['check']) for r in payload['log']] == expected[:len(payload['log'])]
    assert len(payload['log']) in (14,15,16)
    for r in payload['log']:
        assert r['max_bits_abs'] == 0 and r['n_bytes'] == 8192
        if r['check']=='checkpoint_equivalence':
            assert r['state_exact'] and r['split_at']==r['offset']==4096
    def record(check, **values):
        row = dict(check=check, **values)
        payload['log'].append(row)
        h.dump(out/'smoke.json',payload)
        print(json.dumps(row),flush=True)
    if len(payload['log']) < 16:
        whole,split = out/'whole.pt',out/'split.pt'
        a = h.torch.load(whole,weights_only=False)
        assert (a['arm'],a['seed'],a['offset'],a['n_bytes']) == ('dense_8',1,8192,8192)
        assert a['source_hashes'] == h.source_hashes()
        result,bits = h.run('dense_8',1,raw[:8192],keep=True,resume=whole)
        if len(payload['log']) == 14:
            ref,rb = h.h.run('dense_8',1,raw[:8192],keep=True)
            assert h.np.array_equal(bits,rb)
            record('reference_equivalence',arm='dense_8',seed=1,n_bytes=8192,
                   reference='h34h_dense',bits_per_byte=result['bits_per_byte'],
                   reference_bits_per_byte=ref['bits_per_byte'],max_bits_abs=0.)
        part = h.run('dense_8',1,raw[:8192],checkpoint=split,stop_at=4096)
        resumed,bb = h.run('dense_8',1,raw[:8192],keep=True,resume=split,checkpoint=split)
        assert h.np.array_equal(bits,bb) and h.h.equal(h.strip_wall(result),h.strip_wall(resumed))
        b = h.torch.load(split,weights_only=False)
        assert h.h.equal(h.strip_wall(a),h.strip_wall(b))
        record('checkpoint_equivalence',arm='dense_8',seed=1,n_bytes=8192,split_at=4096,
               offset=part['offset'],max_bits_abs=0.,state_exact=True,wall_seconds=a['wall_seconds'])
        whole.unlink(); split.unlink()
    estimates = {arm:max(r['wall_seconds'] for r in payload['log']
                 if r['arm']==arm and r['check']=='checkpoint_equivalence')/8192*h.N_DATA for arm in h.ARMS}
    counts = {a:max(1,math.ceil(t*1.25/4800)) for a,t in estimates.items()}
    boundaries = {a:[((h.N_DATA*k//n)//128)*128 for k in range(1,n)]+[h.N_DATA] for a,n in counts.items()}
    payload.update(part_counts=counts,boundaries=boundaries)
    record('wall_plan',estimated_full_seconds=estimates,part_counts=counts,boundaries=boundaries,
           estimated_part_seconds={a:estimates[a]/counts[a] for a in h.ARMS},limit_seconds=5100,runtime_reserve=1.25)
    payload['ok'] = True
    payload.pop('failure',None)
    h.dump(out/'smoke.json',payload)

if __name__=='__main__': main()
