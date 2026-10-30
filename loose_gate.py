"""Loosen the margin gate with the rate-presence remap store (loop 23:05 hyp 1).

Pre-registered (00:10 HKT 31 Oct): replay stores none and r3 (remap_probe.py) joined with gate_probe FF logs, bg 0.02.
Route grid: FF-margin rolling quantile q in {10, 20, 30, 50} x store-margin threshold s in {0.1, 0.2}, plus
(prot = FF class count >= 1000 kept as in mq10_s2). Selection is on seeds 0-2, test on held-out 3-5.
Prediction: some setting reaches held-out returns >= 68 at lifetime >= 96.0.
Kill: no setting beats mq10_s2+r3 held-out returns (65.8) by >= 2 points at lifetime >= 96.0.
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
from gate_probe import rolling_quantile
import remap_probe as RP

QS, SS = (0.1, 0.2, 0.3, 0.5), (0.1, 0.2)


def score(R, S, SM, w, swap_at, novel_at):
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    routes = dict(ff=np.zeros_like(dis), store=dis)
    for q in QS:
        lo = FFM < rolling_quantile(FFM, q)
        for s in SS:
            routes[f"mq{int(q*100)}_s{int(s*10)}"] = dis & prot & lo & (SM >= s)
    res = {}
    for g, use in routes.items():
        P = np.where(use, S, F); ok = P == Y
        first = lambda dg, at: float(ok[np.flatnonzero(np.isin(C, dg) & (T >= at))[:300]].mean())
        res[g] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()), off=float(ok[INS == 0].mean()),
                      swap=first(list(w["swap"]), swap_at), novel=first(list(w["novel"]), novel_at), fired=float(use.mean()))
    return res


def main():
    out_dir = "results/loose_gate"; os.makedirs(out_dir, exist_ok=True)
    w = dict(LF.WORLD, bg=0.02); logs = {}
    for path in ("results/gate_probe", "results/gate_probe_s345"):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:3]:
            logs[r["seed"]] = r
    load_mnist("mnist")
    with Pool(len(logs)) as p:
        res = p.map(RP.run, [(s, 16, dict(bg=0.02), [None, "r3"]) for s in sorted(logs)], chunksize=1)
    allres = {}
    for seed, out, _ in res:
        r = logs[seed]; R = np.array(r["rows"])
        assert np.array_equal(out["None"][:, 2], R[:, 6])
        allres[seed] = {K: score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]) for K, v in out.items()}
    md = ["# Loosened margin gate with r3 store (`loose_gate.py`)\n", __doc__]
    for name, seeds in (("seeds 0-2 (selection)", [0, 1, 2]), ("held-out seeds 3-5", [3, 4, 5])):
        md += [f"\n## {name}, bg 0.02, mean ± std %\n", "| Store | Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers |", "|---|---|---|---|---|---|---|---|"]
        for K in ("None", "r3"):
            for g in allres[0][K]:
                m = lambda k: f"{np.mean([allres[s][K][g][k] for s in seeds])*100:.1f} ± {np.std([allres[s][K][g][k] for s in seeds])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('off')} | {m('swap')} | {m('novel')} | {m('fired')} |")
    json.dump({str(s): v for s, v in allres.items()}, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
