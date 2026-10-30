"""Fresh-seed confirmation of onset-aware presence s100 (pre-registered 05:05 UTC+8 31 Oct).

Same code as onset_presence.py (store a25w250h2k2sS replayed against gate_probe FF logs), on fresh seeds 6-8 at bg 0.02
(FF logs results/gate_probe_s678, never used for any choice). Confirm s100 (adopt as default store) if, under mq30_s2:
swap gain over s0 >= 1.0, pre-swap remaps <= 5 per run, returns drop <= 1.0. Otherwise default stays s0.
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
import onset_presence as OP
from onset_presence import run, score, LF, load_mnist


def main():
    Ks = ["a25w250h2k2s0", "a25w250h2k2s100"]
    out_dir = "results/onset_s678"; os.makedirs(out_dir, exist_ok=True)
    logs = {(0.02, r["seed"]): r for r in json.load(open("results/gate_probe_s678/results.json"))["raw"]}
    load_mnist("mnist")
    with Pool(min(8, os.cpu_count())) as p:
        res = p.map(run, [(s, 16, dict(bg=bg), Ks) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"]); w = dict(LF.WORLD, bg=bg)
        assert np.array_equal(out[Ks[0]][:, 1], R[:, 2])
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    keys = sorted(allres)
    md = ["# Onset presence s100, fresh seeds 6-8 (`onset_s678.py`)\n", __doc__,
          f"\n## bg 0.02, seeds {keys}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) | First post-swap remap, mean ticks |", "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for K in Ks:
        lat = [v for s in keys for v in allres[s][K]["post_lat"].values() if v is not None]
        for g in ("store", "mq30_s2"):
            m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
            md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.1f}) | {np.mean(lat):.0f} |")
        summ[K] = {k: float(np.nanmean([allres[s][K]['routes']['mq30_s2'][k] for s in keys])) for k in ("lifetime", "returns", "swap")}
        summ[K]["pre_swap"] = float(np.mean([allres[s][K]['pre_swap'] for s in keys]))
    a, b = summ[Ks[0]], summ[Ks[1]]
    ok = (b["swap"] - a["swap"]) * 100 >= 1.0 and b["pre_swap"] <= 5 and (a["returns"] - b["returns"]) * 100 <= 1.0
    md.append(f"\nVerdict: swap gain {100*(b['swap']-a['swap']):+.1f}, returns {100*(b['returns']-a['returns']):+.1f}, pre-swap remaps {b['pre_swap']:.1f} -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    json.dump(dict(runs=allres, summary=summ, confirmed=bool(ok)), open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
