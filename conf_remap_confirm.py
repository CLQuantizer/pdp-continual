"""Fresh-seed confirmation of confidence-weighted remap (loop 02:00 UTC+8 1 Nov, pre-registered before running).

conf_remap.py (H7, m0.3 selected) failed only on pre-swap false remaps (one transient 5->8->5 pair on s49 seed 4);
the post-hoc neighbour sweep (results/conf_remap_sweep) showed m0.35 meeting every H7 bar, but it was picked after
seeing the held-out groups. This runs m0.3 and m0.35 on bg 0.02 (swap 2<->3) seeds 3-5 and 9-11, whose FF logs
were never used to choose q (gate_probe_s345, gate_probe_s91011).
H7c prediction (m0.35 and, separately, m0.3) vs the z5 store, both routed by j+0.0b: first post-swap remap <= -10
ticks, swap first 300 >= +0.5, returns >= -0.5, lifetime >= -0.02, pre-swap false remaps per run <= z5 + 0.2.
Kill: any bar missed.
Usage: OMP_NUM_THREADS=1 python3 conf_remap_confirm.py
"""
import json, os
from multiprocessing import Pool
import numpy as np
import confirm_recency as CR
import conf_remap as CW

CR.WORLDS = {"bg002a": (dict(bg=0.02), "results/gate_probe_s345", [3, 4, 5]),
             "bg002b": (dict(bg=0.02), "results/gate_probe_s91011", [9, 10, 11])}
CW.OD = OD = "results/conf_remap_confirm"
CW.VARS = VARS = ["z5", "m0.3", "m0.35"]


def main():
    os.makedirs(OD, exist_ok=True)
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    jobs = sorted(logs)
    with Pool(6) as p:
        res = p.map(CW.run, jobs)
    stats = {(wk, s): st for wk, s, st in res}
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per = {k: {v: dict(CW.score(k[0], np.load(f"{OD}/Z_{k[0]}_{k[1]}.npz")[v], logs[k], fe, se), **stats[k][v]) for v in VARS} for k in jobs}
    def M(v, f):
        xs = []
        for c in jobs:
            val = per[c][v][f]
            xs += ([l if l is not None else 2000 for l in val] if f == "lat" else [val])
        return float(np.nanmean(xs))
    F = ("lifetime", "returns", "swap", "fired", "lat", "pre_swap", "dbl")
    md = ["# Fresh-seed confirmation of confidence-weighted remap (`conf_remap_confirm.py`)\n", __doc__,
          "\n## bg 0.02 swap 2<->3, seeds 3-5 and 9-11, mean (route = j+0.0b on that store)\n",
          "| Store | Lifetime % | Returns % | Swap first 300 % | Store answers % | First post-swap remap (ticks) | Pre-swap remaps / run | Doubled / run |",
          "|---|---|---|---|---|---|---|---|"]
    for v in VARS:
        md.append(f"| {v} | {M(v,'lifetime')*100:.2f} | {M(v,'returns')*100:.1f} | {M(v,'swap')*100:.1f} | {M(v,'fired')*100:.2f} | "
                  f"{M(v,'lat'):.0f} | {M(v,'pre_swap'):.2f} | {M(v,'dbl'):.0f} |")
    verdict = {}
    for v in VARS[1:]:
        d = {f: M(v, f) - M("z5", f) for f in ("lat", "swap", "returns", "lifetime", "pre_swap")}
        d["passed"] = d["lat"] <= -10 and d["swap"] >= 0.005 and d["returns"] >= -0.005 and d["lifetime"] >= -0.0002 and d["pre_swap"] <= 0.2
        verdict[v] = d
    md.append("\n## Per seed (swap first 300 %, latency per swapped class, pre-swap remaps)\n")
    for k in jobs:
        md.append(f"- {k[0]} seed {k[1]}: " + "; ".join(f"{v} {per[k][v]['swap']*100:.1f} {per[k][v]['lat']} {per[k][v]['pre_swap']}" for v in VARS))
    md.append(f"\n## Verdict H7c (vs z5)\n\n{json.dumps(verdict, indent=1)}\n")
    json.dump({"verdict": verdict, "raw": {f"{a}:{b}": v for (a, b), v in per.items()}}, open(f"{OD}/results.json", "w"))
    open(f"{OD}/RESULTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
