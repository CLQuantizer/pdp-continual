"""H35b2: recent compression progress, principles 3 and 4.

Exact sliding document lag (not periodic refresh). Selection uses the first
512 bytes, reset hidden state, and no labels or test data. Baselines delegate
to H35b unchanged. No network operations in this module.
"""
import argparse
from collections import Counter, deque
import copy
import hashlib
import json
from pathlib import Path
import time

import h35b_neural_diet as h
import numpy as np
import torch

POLICIES = ('rand', 'oracle', 'rho', 'prog_snap_5', 'prog_snap_20')
SEEDS = h.SEEDS
dump = h.dump


class Snapshot:
    frozen_totals = h.Learner.frozen_totals

    def __init__(self, live, documents):
        self.documents = documents
        self.model = copy.deepcopy(live.model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def checksum(self):
        digest = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


class SnapshotHistory:
    def __init__(self, live, lag):
        self.lag = lag
        self.states = deque([Snapshot(live, 0)], maxlen=lag+1)

    def record(self, live, documents):
        assert documents == self.states[-1].documents + 1
        self.states.append(Snapshot(live, documents))

    def past(self, documents):
        snap = self.states[0]
        assert snap.documents == max(0, documents-self.lag)
        return snap


def snapshot_scores(live, snapshot, peeks):
    return (snapshot.frozen_totals(peeks)-live.frozen_totals(peeks))/512.


def run_policy(seed, policy, budget, pool, labels, test, reference=None, checkpoint=25):
    if policy in ('rand', 'oracle', 'rho'):
        return h.run_policy(seed, policy, budget, pool, labels, test, reference, checkpoint)
    if policy not in POLICIES:
        raise ValueError(policy)
    clock = time.perf_counter()
    live = h.Learner(seed)
    lag = int(policy.rsplit('_', 1)[1])
    history = SnapshotHistory(live, lag)
    rng = np.random.default_rng(1_000+seed)
    unread = np.ones(len(pool), dtype=bool)
    selected, checkpoints, sources, life_bits = [], [], [], 0.
    for count in range(1, budget+1):
        candidates = rng.choice(np.flatnonzero(unread), size=h.K, replace=False)
        snap = history.past(count-1)
        sources.append(snap.documents)
        scores = snapshot_scores(live, snap, pool[candidates, :512])
        assert np.isfinite(scores).all()
        idx = int(candidates[int(np.argmax(scores))])
        life_bits += live.read(pool[idx])
        unread[idx] = False
        selected.append(idx)
        history.record(live, count)
        if count % checkpoint == 0:
            value = live.test(test)
            checkpoints.append(dict(documents=count, clean_bpb=value))
            print(f'{policy} s{seed} B{count} clean_bpb={value:.9f} wall_s={time.perf_counter()-clock:.1f}', flush=True)
    kinds = Counter(str(labels[idx]) for idx in selected)
    return dict(seed=seed, policy=policy, budget=budget, pool_documents=len(pool),
                checkpoint_every=checkpoint, checkpoints=checkpoints,
                T_clean=float(np.mean([p['clean_bpb'] for p in checkpoints])) if checkpoints else None,
                final=checkpoints[-1]['clean_bpb'] if checkpoints and checkpoints[-1]['documents']==budget else None,
                kinds_read={k:kinds[k] for k in h.corpus.KINDS},
                shares={k:kinds[k]/budget for k in h.corpus.KINDS}, selected=selected,
                life_bits=life_bits, wall_seconds=time.perf_counter()-clock,
                snapshot_lag=lag, snapshot_documents=sources, peek_bytes=512, candidates=h.K)


def kernel_main(seed, out, kernel_id, policies=POLICIES, data='data/enwik8/enwik8', fingerprint=None):
    h.configure()
    clock = time.perf_counter()
    raw, pool, labels, test, digest = h.load(seed, data)
    ref = h.make_reference(seed, raw) if 'rho' in policies else None
    setup = time.perf_counter()-clock
    runs = []
    for policy in policies:
        runs.append(run_policy(seed, policy, 200, pool, labels, test, ref))
        dump(Path(out)/'results.json', dict(seed=seed, kernel_id=kernel_id, runs=runs,
             policies=list(policies), sha256_e10m=digest, code_fingerprint=fingerprint,
             reference_setup_seconds=setup, wall_seconds=time.perf_counter()-clock))
    return runs


def smoke(data, out):
    h.configure()
    report = dict(ok=False, threads=torch.get_num_threads(), B8={})
    def save():
        dump(Path(out)/'smoke.json', report)
    try:
        raw, pool, labels, test, digest = h.load(6, data)
        report['sha256_e10m'] = digest
        # Cross both lag boundaries on small real prefixes, checking that old
        # weights remain immutable while live parameters and Adam state change.
        report['snapshot_checks'] = []
        for lag in (5,20):
            live = h.Learner(6)
            history = SnapshotHistory(live, lag)
            initial = history.states[0]
            initial_hash = initial.checksum()
            known = {0: initial_hash}
            for t in range(23):
                snap = history.past(t)
                before = live.checksum()
                old = snap.checksum()
                scores = snapshot_scores(live, snap, pool[:h.K,:512])
                assert live.checksum() == before and snap.checksum() == old
                assert old == known[max(0,t-lag)]
                if t == 0:
                    assert np.array_equal(scores, np.zeros(h.K))
                assert all(not p.requires_grad for p in snap.model.parameters())
                assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(snap.model.parameters(),live.model.parameters()))
                live.read(pool[t,:128])
                history.record(live,t+1)
                known[t+1] = history.states[-1].checksum()
                assert initial.checksum() == initial_hash and snap.checksum() == old
            report['snapshot_checks'].append(dict(lag=lag, documents=23, initial_before=initial_hash,
                initial_after=initial.checksum(), final_source=history.past(23).documents,
                lag_exact=True, independent_storage=True, frozen=True, scoring_live_unchanged=True))
            save()
        # Fresh same-process H35b comparisons, plus the recorded original smoke.
        original = json.loads(Path('results/h35b/smoke.json').read_text())
        report['baseline_reproduction'] = {}
        for policy in ('rand','oracle'):
            expected = h.run_policy(6,policy,8,pool,labels,test,checkpoint=8)
            actual = run_policy(6,policy,8,pool,labels,test,checkpoint=8)
            delta = actual['T_clean']-expected['T_clean']
            recorded = original['B8'][policy]['T_clean']
            assert abs(delta) <= 1e-9 and abs(actual['T_clean']-recorded) <= 1e-9
            assert actual['selected'] == expected['selected'] == original['B8'][policy]['selected']
            report['baseline_reproduction'][policy] = dict(h35b=expected['T_clean'], h35b2=actual['T_clean'],
                recorded_h35b=recorded, delta=delta, recorded_delta=actual['T_clean']-recorded)
            report['B8'][policy] = actual
            save()
        clock = time.perf_counter()
        ref = h.make_reference(6,raw)
        report['reference_seconds'] = time.perf_counter()-clock
        save()
        for policy in POLICIES[2:]:
            row = run_policy(6,policy,8,pool,labels,test,ref,checkpoint=8)
            assert np.isfinite(row['T_clean']) and sum(row['kinds_read'].values())==8
            report['B8'][policy] = row
            print('B8', policy, json.dumps(row['kinds_read']), flush=True)
            save()
        clock = time.perf_counter()
        h.Learner(6).test(test)
        test_seconds = time.perf_counter()-clock
        report['frozen_test_seconds'] = test_seconds
        report['estimated_policy_minutes'] = {p: ((r['wall_seconds']-test_seconds)*25+test_seconds*8)/60
                                               for p,r in report['B8'].items()}
        report['estimated_kernel_minutes'] = sum(report['estimated_policy_minutes'].values())+report['reference_seconds']/60
        report['ok'] = True
        save()
        print('SMOKE', json.dumps({k:report[k] for k in ('ok','baseline_reproduction','estimated_kernel_minutes')}), flush=True)
    except Exception as exc:
        report['failure'] = str(exc)
        save()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=SEEDS, default=6)
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    p.add_argument('--data', default='data/enwik8/enwik8')
    p.add_argument('--out', default='results/h35b2')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        smoke(args.data, args.out)
    else:
        kernel_main(args.seed,args.out,f'ezzzzzx/h35b2-s{args.seed}',args.policies,args.data)

if __name__ == '__main__':
    main()
