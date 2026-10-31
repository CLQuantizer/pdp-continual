"""Majority-vote twin retrieval on the copy-then-prune store (loop 06:00 UTC+8 1 Nov, pre-registered before running).

Context: copy_prune.py (05:00) removed false remaps (dev seed 26 68 -> 1, 0 on held-out seeds 48-65) but was killed on
   latency: twin retrieval answered with the twin that won most recently, and after a swap the originals still win
   10-16% of twin hits (border 9s now labelled 4 landing on old-4 prototypes), so the answer flips back after each.
   Gold & Shadlen 2007: a choice that follows only the last sample is the worst-case integrator.
H11: 'v{K}': same storage and pruning as c{P} (copy on remap, delete a twin after P ticks without a win), but retrieval
   answers with the twin holding the majority of the last K labelled twin hits (tie, incl. no hits yet: original).
   Variants: m0.3n4c600v10, m0.3n4c600v5, m0.3n4c300v10; controls z5, m0.3n4, m0.3n4c600 (last-win).
Metrics as copy_prune.py: committed pre-swap remaps (s49f streams), ans_lat, swap300, pre_acc.
Selection on dev fresh seeds 12-29: among v variants, fewest committed pre-swap remaps, ties by lower mean ans_lat.
Prediction on held-out fresh seeds 66-83 (18 streams, never used): selected variant total committed pre-swap remaps
   <= 4 with no stream > 2; mean ans_lat <= z5 - 8 ticks; mean swap300 (both worlds) >= m0.3n4 - 2 points;
   pre_acc >= z5 - 0.3; dev seed 26 committed <= 4. Kill: any bar missed. (Null = the 10-16% original wins are not
   the cause; the copy is simply made too late / too weak, so integrating the vote cannot beat a move store.)
Usage: OMP_NUM_THREADS=1 python3 majority_twin.py
"""
import json, os
from collections import deque
from multiprocessing import Pool
import numpy as np
import copy_prune as CP
import guard_remap as G

OD = "results/majority_twin"
NV = ["z5", "m0.3n4", "m0.3n4c600", "m0.3n4c600v10", "m0.3n4c600v5", "m0.3n4c300v10"]
CAND = [v for v in NV if "v" in v]


class MVStore(CP.CPStore):
    def __init__(self, *a, V="z5", **kw):
        self.K_vote = None
        if "v" in V:
            V, K = V.split("v"); self.K_vote = int(K)
        super().__init__(*a, V=V, **kw)

    def copy(self, s, j, y):
        super().copy(s, j, y)
        self.pairs[-1]["h"] = deque(maxlen=self.K_vote or 1)

    def winner(self, p):
        if self.K_vote is None:
            return super().winner(p)
        h = p["h"]
        return "c" if sum(e == "c" for e in h) > sum(e == "o" for e in h) else "o"

    def observe(self, x, y):
        if self.K_vote is not None and self.mu:
            s, j, _ = self.nearest(x)
            p = self.pair_of(s, self.hist[s][j])
            if p is not None:
                lab = {p["o"][0]: "o", p["c"][0]: "c"}
                if y in lab:
                    p["h"].append(lab[y])
        return super().observe(x, y)


CP.CPStore_orig = CP.CPStore
CP.CPStore = MVStore
CP.NV = NV


def main():
    os.makedirs(OD, exist_ok=True)
    seeds = list(range(12, 30)) + list(range(66, 84))
    jobs = [(wk, s) for wk in G.FRESH for s in seeds]
    with Pool(8) as p:
        res = p.map(CP.run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    dev = sorted(c for c in fr if c[1] < 30); test = sorted(c for c in fr if c[1] >= 66)
    M = lambda cells, v, f: float(np.mean([fr[c][v][f] for c in cells]))
    S = lambda cells: [c for c in cells if c[0] == "s49f"]
    sel = min(CAND, key=lambda v: (M(S(dev), v, "commit"), M(dev, v, "ans_lat")))
    md = ["# Majority-vote twin retrieval (`majority_twin.py`)\n", __doc__, f"\nSelected on dev seeds 12-29: **{sel}**\n"]
    for name, cells in (("Dev fresh seeds 12-29", dev), ("Held-out fresh seeds 66-83", test)):
        md += [f"\n## {name} (18 streams, 36 runs)\n",
               "| Store | Committed false remaps (sum) | Streams with any | Max per stream | Copies pre-swap / stream | ans_lat s49f | ans_lat bg002f | swap300 s49f | swap300 bg002f | pre_acc |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for v in NV:
            cm = [fr[c][v]["commit"] for c in S(cells)]
            sub = lambda wk: [c for c in cells if c[0] == wk]
            md.append(f"| {v}{' (sel)' if v == sel else ''} | {sum(cm)} | {sum(x > 0 for x in cm)} | {max(cm)} | {M(S(cells), v, 'copies'):.2f} | "
                      f"{M(sub('s49f'), v, 'ans_lat'):.1f} | {M(sub('bg002f'), v, 'ans_lat'):.1f} | {M(sub('s49f'), v, 'swap300'):.1f} | "
                      f"{M(sub('bg002f'), v, 'swap300'):.1f} | {M(cells, v, 'pre_acc'):.2f} |")
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
    md.append(f"\n## Verdict H11 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}},
              open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
