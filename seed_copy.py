"""Copy-first twin retrieval + error attribution (loop 07:00 UTC+8 1 Nov, pre-registered before running).

Context: copy_prune.py (05:00) and majority_twin.py (06:00) both leave the copy store ~15-30 ticks slower than the move
   store m0.3n4 even though copies are made as early as moves (57.6 vs 56.4 ticks). Neither the retrieval integrator
   nor the class cap is the cause. Reading the code: a new pair starts with w = {o: t, c: t}, and ties go to the
   original, so a fresh copy is masked and the original keeps answering the OLD label until the copy earns its own
   labelled hit. The move store answers the new label the moment it moves. The copy is only made after H in-radius
   contradictions on that prototype, so the evidence already favours the new label at creation.
   Wills et al. 2005 / Jezek et al. 2011: after a cue switch the newly cued map dominates retrieval at once, the old
   one only intrudes; the default state of a freshly formed alternative is "on".
H12: 's' (seeded): same storage and pruning as c{P}, but a new pair starts with the copy winning (w_o = t-1, w_c = t);
   later wins as before (last win). Variants m0.3n4c600s, m0.3n4c300s, m0.3c600s; controls z5, m0.3n4, m0.3n4c600.
Diagnostic (all variants): every wrong store answer on a swapped-class arrival in the first 300 ticks after the swap is
   attributed to the prototype that answered: 'orig' = original of a live pair (split by whether its copy had won yet:
   'orig_nowin' / 'orig_won'), 'unpaired' = ordinary prototype still carrying the old label, 'other' = anything else.
Selection on dev fresh seeds 12-29: among s variants, fewest committed pre-swap remaps (s49f streams), ties lower ans_lat.
Prediction on held-out fresh seeds 84-101 (18 streams, never used): selected variant committed pre-swap remaps <= 4 with
   no stream > 2; mean ans_lat <= z5 - 8; mean swap300 >= m0.3n4 - 2; pre_acc >= z5 - 0.3; dev seed 26 committed <= 4.
   Kill: any bar missed. (Null = the masked-at-birth copy is not the delay; errors come from unpaired old-label
   prototypes the move store has already relabelled, and the diagnostic will show 'unpaired' dominating.)
Usage: OMP_NUM_THREADS=1 python3 seed_copy.py
"""
import json, os
from collections import Counter
from multiprocessing import Pool
import numpy as np
import life as LF
import guard_remap as G
import copy_prune as CP
import remap_probe as RP
from data import load_mnist

OD = "results/seed_copy"
NV = ["z5", "m0.3n4", "m0.3n4c600", "m0.3n4c600s", "m0.3n4c300s", "m0.3c600s"]
CAND = [v for v in NV if v.endswith("s")]


class SStore(CP.CPStore):
    def __init__(self, *a, V="z5", **kw):
        self.seeded = V.endswith("s")
        super().__init__(*a, V=V[:-1] if self.seeded else V, **kw)

    def copy(self, s, j, y):
        super().copy(s, j, y)
        p = self.pairs[-1]; p["nc"] = 0
        if self.seeded:
            p["w"]["o"] = self.t - 1

    def observe(self, x, y):
        if self.P is None:
            return super().observe(x, y)
        before = {id(p): p["w"]["c"] for p in self.pairs}
        super().observe(x, y)
        for p in self.pairs:
            if id(p) in before and p["w"]["c"] != before[id(p)]:
                p["nc"] = p.get("nc", 0) + 1

    def who(self, x):
        """Category of the prototype that answers x."""
        m = self.masked(); best = (None, None, np.inf)
        for c in self.mu:
            d = ((self.mu[c] - x) ** 2).sum(1)
            for i in range(len(d)):
                if (c, i) not in m and d[i] < best[2]:
                    best = (c, i, d[i])
        c, i, _ = best; h = self.hist[c][i]
        for p in self.pairs:
            if p["o"][0] == c and p["o"][1] is h:
                return "orig_won" if p.get("nc", 0) else "orig_nowin"
            if p["c"][0] == c and p["c"][1] is h:
                return "copy"
        return "unpaired"


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **G.FRESH[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: SStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in NV}
    rows = {v: [] for v in NV}; errs = {v: Counter() for v in NV}
    sw_cls = set(w.get("swap", ()))
    for t in range(ticks):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        sw = world.swap_at
        for v, st in sts.items():
            pred = CP.ans(st, x)[0]
            rows[v].append((t, c, y, pred))
            if sw is not None and sw <= t < sw + 300 and c in sw_cls and pred != y:
                if st.P is None:
                    errs[v]["unpaired" if pred != -1 else "other"] += 1
                else:
                    errs[v][st.who(x[0])] += 1
            st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    stats = {}
    for v, st in sts.items():
        R = np.array(rows[v])
        pre = R[R[:, 0] < sw]; post = R[(R[:, 0] >= sw) & np.isin(R[:, 1], list(sw_cls))]
        p300 = post[post[:, 0] < sw + 300]
        ok = post[:, 3] == post[:, 2]; al = 2000
        for i in range(len(ok) - 2):
            if ok[i] and ok[i + 1] and ok[i + 2]:
                al = int(post[i + 2, 0] - sw); break
        comm = [e for e in st.commit if e[0] < sw] if st.P is not None else [e for e in st.log if e[0] < sw]
        stats[v] = dict(commit=len(comm), first_commit=comm[:4],
                        copies=sum(e[0] < sw for e in st.copies) if st.P is not None else len(comm),
                        post_events=sum(sw <= e[0] < sw + 300 for e in (st.copies if st.P is not None else st.log)),
                        ans_lat=al, swap300=float((p300[:, 3] == p300[:, 2]).mean() * 100),
                        pre_acc=float((pre[:, 3] == pre[:, 2]).mean() * 100), errs=dict(errs[v]))
    print(f"{wk} seed={seed} {json.dumps({v: (s['commit'], s['ans_lat'], s['errs']) for v, s in stats.items()})}", flush=True)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    seeds = list(range(12, 30)) + list(range(84, 102))
    jobs = [(wk, s) for wk in G.FRESH for s in seeds]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 84)
    M = lambda cells, v, f: float(np.mean([fr[c][v][f] for c in cells]))
    S = lambda cells: [c for c in cells if c[0] == "s49f"]
    sel = min(CAND, key=lambda v: (M(S(dev), v, "commit"), M(dev, v, "ans_lat")))
    md = ["# Copy-first twin retrieval + error attribution (`seed_copy.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29", dev), ("Held-out fresh seeds 84-101", test)):
        md += [f"\n## {name} (18 streams, 36 runs)\n",
               "| Store | Committed false remaps (sum) | Streams with any | Max per stream | ans_lat s49f | ans_lat bg002f | swap300 s49f | swap300 bg002f | pre_acc | copies/moves in first 300 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for v in NV:
            cm = [fr[c][v]["commit"] for c in S(cells)]
            sub = lambda wk: [c for c in cells if c[0] == wk]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {sum(cm)} | {sum(x > 0 for x in cm)} | {max(cm)} | "
                      f"{M(sub('s49f'), v, 'ans_lat'):.1f} | {M(sub('bg002f'), v, 'ans_lat'):.1f} | {M(sub('s49f'), v, 'swap300'):.1f} | "
                      f"{M(sub('bg002f'), v, 'swap300'):.1f} | {M(cells, v, 'pre_acc'):.2f} | {M(cells, v, 'post_events'):.2f} |")
        md += ["\nWrong answers on swapped-class arrivals in the first 300 ticks, per run, by answering prototype:\n",
               "| Store | total | orig_nowin | orig_won | unpaired | copy | other |", "|---|---|---|---|---|---|---|"]
        for v in NV:
            E = Counter()
            for c in cells:
                E.update(fr[c][v]["errs"])
            n = len(cells)
            md.append(f"| {v} | {sum(E.values())/n:.1f} | " + " | ".join(f"{E[k]/n:.1f}" for k in ("orig_nowin", "orig_won", "unpaired", "copy", "other")) + " |")
    md.append("\n## Streams with any committed pre-swap remap (s49f)\n")
    for c in sorted(S(list(fr))):
        bad = {v: fr[c][v] for v in NV if fr[c][v]["commit"]}
        if bad:
            md.append(f"- seed {c[1]}: " + "; ".join(f"{v} {b['commit']} {b['first_commit']}" for v, b in bad.items()))
    tc = [fr[c][sel]["commit"] for c in S(test)]
    verdict = dict(commit_total=int(sum(tc)), commit_max=int(max(tc)),
                   ans_lat=M(test, sel, "ans_lat") - M(test, "z5", "ans_lat"),
                   swap300=M(test, sel, "swap300") - M(test, "m0.3n4", "swap300"),
                   pre_acc=M(test, sel, "pre_acc") - M(test, "z5", "pre_acc"),
                   seed26=int(fr[("s49f", 26)][sel]["commit"]))
    passed = (verdict["commit_total"] <= 4 and verdict["commit_max"] <= 2 and verdict["ans_lat"] <= -8
              and verdict["swap300"] >= -2 and verdict["pre_acc"] >= -0.3 and verdict["seed26"] <= 4)
    md.append(f"\n## Verdict H12 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
