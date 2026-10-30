"""Change-point (NE-style) veto on the a25 store + loosened gate (pre-registered 01:15 HKT 31 Oct).

Yu & Dayan 2005 (Neuron 46:681): ACh tracks *expected* uncertainty (a cue's known unreliability), NE tracks
*unexpected* uncertainty (a change of context), and the two call for different responses. veto_probe.py (21:10)
vetoed a store class on an absolute error count (3 of its last 50 confident claims), an expected-uncertainty
test, so it fired on the store's ordinary error rate and cost returns. Here the veto is a change-point test:
per store class k, over its confident claims (store margin >= 0.2), veto k when the error rate in the last 10
claims is >= 0.5 AND >= 4x k's long-run confident error rate (EMA, alpha 0.02, init 0.05, frozen while vetoed);
clear after 10 consecutive agreeing confident claims. Causal: state updates after the answer. Compared with the
old absolute veto (3/50, clear 20) on the same store/route.
Store a25w250, routes mq30_s2 and mq50_s2. bg 0.02 seeds 3-5 held out (0-2 shown), bg 0.2 seeds 0-5 held out.
Prediction: cp veto lifts held-out bg 0.02 swap to >= 74 with returns >= 67.5 and lifetime >= 96.0, and bg 0.2
lifetime >= 94.3.  Kill: returns < 67, or swap gain < 2 over the unvetoed route.
"""
import json, os
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import a25_gate as AG
from gate_probe import rolling_quantile


def veto_abs(Y, S, SM, win=50, bad=3, clear=20, thr=0.2):
    hist = {k: [] for k in range(10)}; veto = np.zeros(10, bool); streak = np.zeros(10, int); out = np.zeros(len(S), bool)
    for i in range(len(S)):
        s = int(S[i])
        if s < 0: continue
        out[i] = veto[s]
        if SM[i] >= thr:
            okc = s == Y[i]; hist[s] = (hist[s] + [not okc])[-win:]
            if veto[s]:
                streak[s] = streak[s] + 1 if okc else 0
                if streak[s] >= clear: veto[s] = False; streak[s] = 0; hist[s] = []
            elif sum(hist[s]) >= bad: veto[s] = True; streak[s] = 0
    return out


def veto_cp(Y, S, SM, short=10, hi=0.5, ratio=4.0, alpha=0.02, clear=10, thr=0.2):
    hist = {k: [] for k in range(10)}; base = np.full(10, 0.05); veto = np.zeros(10, bool); streak = np.zeros(10, int)
    out = np.zeros(len(S), bool)
    for i in range(len(S)):
        s = int(S[i])
        if s < 0: continue
        out[i] = veto[s]
        if SM[i] >= thr:
            err = float(s != Y[i]); hist[s] = (hist[s] + [err])[-short:]
            if veto[s]:
                streak[s] = streak[s] + 1 if not err else 0
                if streak[s] >= clear: veto[s] = False; streak[s] = 0; hist[s] = []
            else:
                r = np.mean(hist[s])
                if len(hist[s]) >= short and r >= hi and r >= ratio * base[s]:
                    veto[s] = True; streak[s] = 0
                else:
                    base[s] += alpha * (err - base[s])
    return out


def score(R, S, SM, w, swap_at, novel_at):
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    vetos = dict(none=np.zeros_like(dis), abs=veto_abs(Y, S, SM), cp=veto_cp(Y, S, SM))
    res = {}
    for q in (0.3, 0.5):
        base = dis & prot & (FFM < rolling_quantile(FFM, q)) & (SM >= 0.2)
        for vn, v in vetos.items():
            use = base & ~v; P = np.where(use, S, F); ok = P == Y
            first = lambda dg, at: float(ok[np.flatnonzero(np.isin(C, dg) & (T >= at))[:300]].mean())
            res[f"mq{int(q*100)}_s2+{vn}"] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else np.nan,
                swap=first(list(w["swap"]), swap_at), fired=float(use.mean()), vetoed=float(v.mean()),
                veto_pre=float(v[T < swap_at].mean()))
    return res


def main():
    out_dir = "results/cp_veto"; os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    with Pool(8) as p:
        res = p.map(AG.run, [(s, 16, dict(bg=bg), ["a25w250"]) for bg, s in sorted(logs)], chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"]); v = out["a25w250"]
        allres[f"{bg}_{seed}"] = score(R, v[:, 2], v[:, 3], dict(LF.WORLD, bg=bg), r["swap_at"], r["novel_at"])
    md = ["# Change-point veto on a25w250 store (`cp_veto.py`)\n", __doc__]
    for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5, held out")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Route + veto | Lifetime | Returns | Swap first 300 | Store answers | Vetoed (pre-swap) |", "|---|---|---|---|---|---|"]
        for g in allres[keys[0]]:
            m = lambda k: f"{np.nanmean([allres[s][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][g][k] for s in keys])*100:.1f}"
            md.append(f"| {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {m('vetoed')} ({m('veto_pre')}) |")
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
