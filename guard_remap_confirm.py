"""Fresh-seed confirmation of the n4 floor (loop 03:00 UTC+8 1 Nov, pre-registered before running).

guard_remap.py (H8) selected m0.3n3 by its rule and it failed on fresh seeds 12-17 (0.50 pre-swap remaps per run).
Post hoc, m0.3n4 met every H8 bar there (0 false remaps, first remap 60 vs 72 ticks; cached swap +0.9 to +3.4,
returns -0.3 to +0.5). Because it was chosen after seeing seeds 12-17, it is confirmed here on store-only seeds
18-29 in s49 (4<->9) and bg002 (2<->3), 24 runs never used.
H8c: m0.3n4 vs z5: pre-swap remaps per run <= z5 + 0.2 and mean first post-swap remap <= z5 - 8 ticks. Kill: either missed.
Usage: OMP_NUM_THREADS=1 python3 guard_remap_confirm.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import guard_remap as G

OD = "results/guard_remap_confirm"
G.NV = ["z5", "m0.3n4", "m0.35n4"]


def main():
    os.makedirs(OD, exist_ok=True)
    jobs = [(wk, s, False) for wk in G.FRESH for s in range(18, 30)]
    with Pool(8) as p:
        res = p.map(G.run, jobs)
    fr = {(wk, s): st for wk, s, _, st in res}
    L = lambda ls: [l if l is not None else 2000 for l in ls]
    def M(cells, v, f):
        xs = []
        for c in cells:
            xs += L(fr[c][v][f]) if f == "lat" else [fr[c][v][f]]
        return float(np.mean(xs))
    fc = sorted(fr)
    md = ["# Fresh-seed confirmation of the n4 floor (`guard_remap_confirm.py`)\n", __doc__,
          "\n| Store | Pre-swap remaps / run | First remap (ticks) | s49f | bg002f |", "|---|---|---|---|---|"]
    for v in G.NV:
        md.append(f"| {v} | {M(fc,v,'pre_swap'):.2f} | {M(fc,v,'lat'):.1f} | {M([c for c in fc if c[0]=='s49f'],v,'lat'):.1f} | "
                  f"{M([c for c in fc if c[0]=='bg002f'],v,'lat'):.1f} |")
    md.append("\n## Per seed\n")
    for c in fc:
        md.append(f"- {c[0]} seed {c[1]}: " + "; ".join(f"{v} {fr[c][v]['pre_swap']} {fr[c][v]['lat']}" for v in G.NV))
    verdict = {v: {f: M(fc, v, f) - M(fc, "z5", f) for f in ("pre_swap", "lat")} for v in G.NV[1:]}
    passed = verdict["m0.3n4"]["pre_swap"] <= 0.2 and verdict["m0.3n4"]["lat"] <= -8
    md.append(f"\n## Verdict H8c (vs z5; m0.35n4 shown, not tested)\n\n{json.dumps(verdict, indent=1)}\n\n"
              f"Prediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    json.dump({"passed": passed, "verdict": verdict, "fresh": {f"{a}:{b}": v for (a, b), v in fr.items()}}, open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
