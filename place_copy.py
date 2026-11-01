"""Content-placed copies: write the new memory where its evidence is (loop 08:00 UTC+8 1 Nov, pre-registered before running).

Context: seed_copy.py (07:00) fixed copy-store latency with copy-first retrieval (m0.3c600s, ans_lat -16.7 vs z5, 0 false
   remaps) but swap first 300 stays 6.4 points behind the move store m0.3n4. Error attribution: the gap is ~+4.7
   unpaired old-label prototypes, +3.6 originals winning back, +4.9 copies answering wrong. Every copy is made at
   exactly its original's location, so the twins differ only by label: the copy covers the same region as the old
   label, not the region where the contradicting items actually fell.
   Bittner et al. 2017 (Science 357:1033): one plateau both creates a CA1 place field and places it at the animal's
   position over the preceding seconds (behavioural time scale plasticity), i.e. a new field is written at its
   evidence. Gu & Dao 2023 (Mamba): an LTI write (fixed B) puts the same thing into state whatever arrives; a selective
   write B(x) puts the input's content there. The copy rule is the LTI write.
H13: each prototype keeps the inputs of its last 10 in-radius contradictions. When a copy is made under label y it is
   placed at 'p' = the mean of the contradicting inputs with label y, or 'h' = halfway between the original and that
   mean. Everything else is m0.3c600s (copy-first, last-win retrieval, prune after 600 ticks without a win).
   Variants m0.3c600sp, m0.3c600sh; controls z5, m0.3n4, m0.3c600s.
Selection on dev fresh seeds 12-29: among p/h, highest mean swap300, ties lower ans_lat.
Prediction on held-out fresh seeds 102-119 (18 streams, never used): selected variant committed pre-swap remaps <= 4
   with no stream > 2; ans_lat <= z5 - 8; swap300 >= m0.3n4 - 2; pre_acc >= z5 - 0.3; dev seed 26 committed <= 4.
   Kill: any bar missed. (Null = placement does not matter: the copy-store gap is in prototypes that never get a copy,
   and 'unpaired' errors stay ~87/run while 'copy' errors do not fall.)
Usage: OMP_NUM_THREADS=1 python3 place_copy.py
"""
import json, os
from collections import Counter, deque
from multiprocessing import Pool
import numpy as np
import life as LF
import guard_remap as G
import copy_prune as CP
import seed_copy as SC
from data import load_mnist

OD = "results/place_copy"
NV = ["z5", "m0.3n4", "m0.3c600s", "m0.3c600sp", "m0.3c600sh"]
CAND = ["m0.3c600sp", "m0.3c600sh"]


class PStore(SC.SStore):
    def __init__(self, *a, V="z5", **kw):
        self.place = V[-1] if V[-1] in "ph" else None
        self.buf = {}
        super().__init__(*a, V=V[:-1] if self.place else V, **kw)

    def copy(self, s, j, y):
        h = self.hist[s][j]
        xs = [x for (yy, x) in self.buf.get(id(h), ()) if yy == y]
        super().copy(s, j, y)
        if self.place and xs:
            m = np.mean(xs, axis=0)
            if self.place == "h":
                m = 0.5 * (m + self.mu[s][self.idx(s, h)])
            self.mu[y][-1] = m
        self.buf.pop(id(h), None)

    def observe(self, x, y):
        if self.P is None or not self.mu or not self.place:
            return super().observe(x, y)
        self.prune()
        s, j, d = self.nearest(x)
        p = self.pair_of(s, self.hist[s][j])
        if p is not None:
            lab = {p["o"][0]: "o", p["c"][0]: "c"}
            s, dq = p[lab[y]] if y in lab else p[self.winner(p)]
        else:
            dq = self.hist[s][j]
        key = None
        if s != y:
            key = id(dq); b = self.buf.setdefault(key, deque(maxlen=10)); b.append((y, x[0].copy()))
        n0 = len(self.cev)
        super().observe(x, y)
        grew = len(self.cev) > n0 or (self.cev and self.cev[-1][0] == self.t)
        if key is not None and not grew and key in self.buf and self.buf[key] and self.buf[key][-1][0] == y:
            self.buf[key].pop()


def run(job):
    wk, seed = job
    w = dict(LF.WORLD, **G.FRESH[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = 16 * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    sts = {v: PStore(9, np.random.default_rng(seed + 1000), noise=0.0, V=v) for v in NV}
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
        stats[v] = dict(commit=len(comm), first_commit=comm[:4], ans_lat=al,
                        swap300=float((p300[:, 3] == p300[:, 2]).mean() * 100),
                        swap1000=float((post[post[:, 0] < sw + 1000][:, 3] == post[post[:, 0] < sw + 1000][:, 2]).mean() * 100),
                        pre_acc=float((pre[:, 3] == pre[:, 2]).mean() * 100), errs=dict(errs[v]))
    print(f"{wk} seed={seed} {json.dumps({v: (s['commit'], s['ans_lat'], round(s['swap300'],1)) for v, s in stats.items()})}", flush=True)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    seeds = list(range(12, 30)) + list(range(102, 120))
    jobs = [(wk, s) for wk in G.FRESH for s in seeds]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 102)
    M = lambda cells, v, f: float(np.mean([fr[c][v][f] for c in cells]))
    S = lambda cells: [c for c in cells if c[0] == "s49f"]
    sel = min(CAND, key=lambda v: (-M(dev, v, "swap300"), M(dev, v, "ans_lat")))
    md = ["# Content-placed copies (`place_copy.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29", dev), ("Held-out fresh seeds 102-119", test)):
        md += [f"\n## {name} (18 streams, 36 runs)\n",
               "| Store | Committed false remaps (sum) | Streams with any | Max per stream | ans_lat s49f | ans_lat bg002f | swap300 s49f | swap300 bg002f | swap1000 | pre_acc |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for v in NV:
            cm = [fr[c][v]["commit"] for c in S(cells)]
            sub = lambda wk: [c for c in cells if c[0] == wk]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {sum(cm)} | {sum(x > 0 for x in cm)} | {max(cm)} | "
                      f"{M(sub('s49f'), v, 'ans_lat'):.1f} | {M(sub('bg002f'), v, 'ans_lat'):.1f} | {M(sub('s49f'), v, 'swap300'):.1f} | "
                      f"{M(sub('bg002f'), v, 'swap300'):.1f} | {M(cells, v, 'swap1000'):.1f} | {M(cells, v, 'pre_acc'):.2f} |")
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
                   swap300_vs_s=M(test, sel, "swap300") - M(test, "m0.3c600s", "swap300"),
                   pre_acc=M(test, sel, "pre_acc") - M(test, "z5", "pre_acc"),
                   seed26=int(fr[("s49f", 26)][sel]["commit"]))
    passed = (verdict["commit_total"] <= 4 and verdict["commit_max"] <= 2 and verdict["ans_lat"] <= -8
              and verdict["swap300"] >= -2 and verdict["pre_acc"] >= -0.3 and verdict["seed26"] <= 4)
    md.append(f"\n## Verdict H13 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
