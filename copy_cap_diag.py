"""Diagnostic (loop 06:00 UTC+8 1 Nov, dev seeds 12-29 only, not a pre-registered test): is the copy store slow
because CPStore.copy enforces the per-class cap (len > 2k) by deleting the least-used FREE prototype of the
receiving class? Under a 4<->9 swap that deletes not-yet-copied originals, which the move store would have relabelled.
'x' = same as m0.3n4c600v10 but no cap deletion on copy. Also counts cap deletions per run.
Usage: OMP_NUM_THREADS=1 python3 copy_cap_diag.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import copy_prune as CP
import majority_twin as MT
import guard_remap as G

OD = "results/copy_cap_diag"
NV = ["z5", "m0.3n4", "m0.3n4c600v10", "m0.3n4c600v10x"]


class XStore(MT.MVStore):
    def __init__(self, *a, V="z5", **kw):
        self.nocap = V.endswith("x"); self.capdel = 0
        super().__init__(*a, V=V.rstrip("x"), **kw)

    def copy(self, s, j, y):
        k0 = self.k
        before = len(self.mu.get(y, [])) + 1
        if self.nocap:
            self.k = 10 ** 6
        super().copy(s, j, y)
        self.k = k0
        self.capdel += before - len(self.mu[y])


CP.CPStore = XStore
CP.NV = NV
_run = CP.run


def run(job):
    wk, seed, stats = _run(job)
    return wk, seed, stats


def main():
    os.makedirs(OD, exist_ok=True)
    jobs = [(wk, s) for wk in G.FRESH for s in range(12, 30)]
    with Pool(8) as p:
        res = p.map(run, jobs)
    fr = {(wk, s): st for wk, s, st in res}
    out = {}
    for v in NV:
        out[v] = {f: float(np.mean([fr[c][v][f] for c in fr if c[0] == wk]))
                  for f in ("ans_lat", "swap300") for wk in [None]} if False else {
            f"{f}_{wk}": float(np.mean([fr[c][v][f] for c in fr if c[0] == wk])) for f in ("ans_lat", "swap300") for wk in G.FRESH}
        out[v]["commit_s49f"] = int(sum(fr[c][v]["commit"] for c in fr if c[0] == "s49f"))
    json.dump({"summary": out, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}}, open(f"{OD}/results.json", "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
