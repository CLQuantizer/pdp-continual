"""Class contagion for the move store (loop 09:00 UTC+8 1 Nov, pre-registered before running).

Context: in every store most wrong answers on swapped-class arrivals in the first 300 ticks come from prototypes that
   never change label (move store m0.3n4 ~87/run 'unpaired' on seeds 102-119). Each prototype must collect its own
   H=2 in-radius contradictions (and the class K=2 hot, N=4 evidence floor) before it moves, so a swap is learned one
   prototype at a time even after the class-level decision has already been made once.
   McClelland, McNaughton & O'Reilly 1995 (CLS) / Kumaran, Hassabis & McClelland 2016: once the hippocampus has
   committed to a new episode-level fact, the update should generalise across the items that share structure, not wait
   for each item's own evidence; in Mamba terms (Gu & Dao 2023) a selective step Delta(x) large on a surprising input
   resets the whole state channel, not one entry. A committed remap s->y is that large Delta for class s.
H14: contagion 'x{T}': after any committed move s->y at time t0, for T=600 ticks any further prototype of class s that
   receives a single in-radius, rival-passing contradiction labelled y moves to y at once (K=1, H=1, no N floor; the
   presence check still applies; class s keeps >= 1 prototype). 'x{T}a' drops the radius requirement (nearest
   prototype of class s, any distance). Base is the move store m0.3n4. Controls z5, m0.3n4, m0.3c600s.
Selection on dev fresh seeds 12-29: among x variants, highest mean swap300, ties lower ans_lat.
Prediction on held-out fresh seeds 120-137 (18 streams, never used): selected variant swap300 >= m0.3n4 + 2 (pooled
   s49f/bg002f), committed pre-swap remap events (s49f, contagion moves included) <= 2 x m0.3n4's and <= m0.3n4 + 4,
   ans_lat <= m0.3n4 + 2, pre_acc >= z5 - 0.3. Kill: any bar missed. (Null = unpaired errors are prototypes that never
   receive an in-radius contradiction in the first 300 ticks, so a 1-contradiction rule cannot reach them.)
Usage: OMP_NUM_THREADS=1 python3 contagion.py
"""
import json, os
from collections import Counter
from multiprocessing import Pool
import numpy as np
import life as LF
import guard_remap as G
import copy_prune as CP
import presence_bg as PB
import seed_copy as SC
from data import load_mnist

OD = "results/contagion"
NV = ["z5", "m0.3n4", "m0.3c600s", "m0.3n4x600", "m0.3n4x600a"]
CAND = ["m0.3n4x600", "m0.3n4x600a"]


class XStore(SC.SStore):
    def __init__(self, *a, V="z5", **kw):
        self.XT, self.anyd = None, False
        if "x" in V:
            V, T = V.split("x")
            self.anyd = T.endswith("a"); self.XT = int(T.rstrip("a"))
        self.cont, self.xmoves = {}, 0
        super().__init__(*a, V=V, **kw)

    def observe(self, x, y):
        if self.XT is None or not self.mu:
            return super().observe(x, y)
        s, j, d = self.nearest(x)
        t0 = self.cont.get((s, y))
        if s != y and t0 is not None and self.t - t0 <= self.XT and len(self.mu[s]) > 1:
            r = self.rad[s][j]
            ok = self.anyd or (r is not None and d <= r)
            if ok and not self.rival(x, y, d) and not PB.absent_rel(self, s):
                self.log.append((self.t, s, y)); self.move(s, j, y); self.xmoves += 1
                return
        n0 = len(self.log)
        super().observe(x, y)
        for (tt, a, b) in self.log[n0:]:
            self.cont[(a, b)] = tt


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **G.FRESH[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: XStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in NV}
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
                errs[v][st.who(x[0]) if st.P is not None else ("unpaired" if pred != -1 else "other")] += 1
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
        post_mv = [e for e in st.log if sw <= e[0] < sw + 300] if st.P is None else []
        stats[v] = dict(commit=len(comm), first_commit=comm[:4], ans_lat=al,
                        swap300=float((p300[:, 3] == p300[:, 2]).mean() * 100),
                        swap1000=float((post[post[:, 0] < sw + 1000][:, 3] == post[post[:, 0] < sw + 1000][:, 2]).mean() * 100),
                        pre_acc=float((pre[:, 3] == pre[:, 2]).mean() * 100), errs=dict(errs[v]),
                        moves300=len(post_mv), xmoves=getattr(st, "xmoves", 0))
    print(f"{wk} seed={seed} {json.dumps({v: (s['commit'], s['ans_lat'], round(s['swap300'],1)) for v, s in stats.items()})}", flush=True)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    seeds = list(range(12, 30)) + list(range(120, 138))
    jobs = [(wk, s) for wk in G.FRESH for s in seeds]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 120)
    M = lambda cells, v, f: float(np.mean([fr[c][v][f] for c in cells]))
    S = lambda cells: [c for c in cells if c[0] == "s49f"]
    sel = min(CAND, key=lambda v: (-M(dev, v, "swap300"), M(dev, v, "ans_lat")))
    md = ["# Class contagion (`contagion.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29", dev), ("Held-out fresh seeds 120-137", test)):
        md += [f"\n## {name} (18 streams, 36 runs)\n",
               "| Store | Committed false remaps (sum) | Streams with any | Max per stream | ans_lat s49f | ans_lat bg002f | swap300 s49f | swap300 bg002f | swap1000 | pre_acc | moves in first 300 | contagion moves |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in NV:
            cm = [fr[c][v]["commit"] for c in S(cells)]
            sub = lambda wk: [c for c in cells if c[0] == wk]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {sum(cm)} | {sum(x > 0 for x in cm)} | {max(cm)} | "
                      f"{M(sub('s49f'), v, 'ans_lat'):.1f} | {M(sub('bg002f'), v, 'ans_lat'):.1f} | {M(sub('s49f'), v, 'swap300'):.1f} | "
                      f"{M(sub('bg002f'), v, 'swap300'):.1f} | {M(cells, v, 'swap1000'):.1f} | {M(cells, v, 'pre_acc'):.2f} | "
                      f"{M(cells, v, 'moves300'):.1f} | {M(cells, v, 'xmoves'):.1f} |")
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
    tc = sum(fr[c][sel]["commit"] for c in S(test)); bc = sum(fr[c]["m0.3n4"]["commit"] for c in S(test))
    verdict = dict(commit_total=int(tc), commit_base=int(bc),
                   swap300=M(test, sel, "swap300") - M(test, "m0.3n4", "swap300"),
                   ans_lat=M(test, sel, "ans_lat") - M(test, "m0.3n4", "ans_lat"),
                   pre_acc=M(test, sel, "pre_acc") - M(test, "z5", "pre_acc"))
    passed = (verdict["swap300"] >= 2 and tc <= 2 * bc and tc <= bc + 4 and verdict["ans_lat"] <= 2
              and verdict["pre_acc"] >= -0.3)
    md.append(f"\n## Verdict H14 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
