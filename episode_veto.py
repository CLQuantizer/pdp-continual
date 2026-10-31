"""Episode-scoped surprise veto for the store-cortex arbiter (loop 00:00 UTC+8 1 Nov, pre-registered before running).

Context: 23:00 found the class error-streak veto v2b gives the highest lifetime yet (+0.15) and +1 to +2.3 swap, but
   costs 3.4-4.0 returns, because 74-84% of disagreements it blocks on returns are ones the store had right: the
   streak was built before the class left. Gershman, Blei & Niv 2010: a context break (here, a class's absence and
   return) should start a new latent cause, so evidence from the old episode should not carry over.
H5: as v{m}b (streak over store-predicted class c of disagreements where store wrong and FF right; veto hands class c
   back to the FF while streak >= m; released by a store-right disagreement), plus an episode reset:
   'e': before answering, if the label c = store's answer has not been observed for >= RET_GAP ticks, streak[c] = 0
        (causal: uses only labels already seen);
   'a': also release streak[c] whenever the store is right on an agreement of class c;
   'ea': both. m in 2, 3, 5. OR'd with mq30_s2 ('b').
H5b (added after H5 ran and failed, pre-registered before running, same bars, selected separately among 'p' tags):
   'p'/'pe': the streak is keyed by the (store answer, FF answer) pair, so only a run of disagreements where the FF
   keeps naming the SAME rival label d vetoes c->store for items where the FF says d. A label swap produces
   consistent pairs (store 4, FF 9); return-period errors are scattered. 'pe' adds the episode reset of 'e'.
Selection on s49 seeds 0-2: best swap first-300 among variants with lifetime >= j+0.0b - 0.02 and returns >=
   j+0.0b - 0.5. Prediction on held-out s49 3-5 and bg002 6-8: returns >= j+0.0b - 0.5, swap >= j+0.0b + 0.5,
   lifetime >= j+0.0b - 0.02; bg 0.2 lifetime >= j+0.0b - 0.02. Kill: any bar missed.
Mechanism prior: 'e' removes stale streaks at return onset, so returns should recover toward 77-78; 'a' releases
   the veto on stable classes, so it should also cut the swap gain (the veto then fires less inside the swap window,
   where the store agrees on unswapped items only of other classes, so maybe not).
Pure re-scoring of cached confirm_recency Z and FF logs (no new training).
Usage: OMP_NUM_THREADS=1 python3 episode_veto.py
"""
import json, os
import numpy as np
import life as LF
import confirm_recency as CR
from gate_probe import rolling_quantile
from arbiter_decay import arb as arb_decay
from change_arbiter import veto

OD = "results/episode_veto"
MS = [2, 3, 5]


def veto_ep(dis, T, S, Y, s_ok, f_ok, m, ep, agree, F=None):
    pair = F is not None
    blk = np.zeros(len(dis), bool); streak = np.zeros((10, 10) if pair else 10); last = {}; cleared = 0
    for i in range(len(T)):
        c = int(S[i])
        if c >= 0:
            if ep and c in last and T[i] - last[c] >= LF.RET_GAP and np.any(streak[c] > 0):
                streak[c] = 0; cleared += 1
            if dis[i]:
                k = (c, int(F[i])) if pair else c
                blk[i] = streak[k] >= m
                if s_ok[i]:
                    streak[c] = 0
                elif f_ok[i]:
                    streak[k] += 1
            elif agree and s_ok[i]:
                streak[c] = 0
        last[int(Y[i])] = T[i]
    return blk, cleared


def score(wk, seed, Z, log, fe, se):
    w = dict(LF.WORLD, **CR.WORLDS[wk][0])
    R = np.array(log["rows"])
    T, C, Y, INS, RET, F, _, FFM, _, AS, AF, CF = R.T
    S, SM = Z[:, 2], Z[:, 3]
    dis = (S >= 0) & (S != F); prot = CF >= 1000
    base = dis & prot & (FFM < rolling_quantile(FFM, 0.3)) & (SM >= 0.2)
    fb, sb = np.digitize(FFM, fe), np.digitize(SM, se)
    s_ok, f_ok = (S == Y).astype(float), (F == Y).astype(float)
    cls = np.clip(S, 0, 9).astype(int); nb = len(fe) + 1
    j = arb_decay(dis, T, fb, sb, cls, s_ok, f_ok, nb, 0, False)
    routes = {"ff": np.zeros(len(T), bool), "j+0.0b": j | base}
    diag = {}
    for m in MS:
        routes[f"v{m}b"] = (j & ~veto(dis, T, S, s_ok, f_ok, np.zeros((0, 3)), m, False)) | base
        for tag, ep, ag, pf in (("e", True, False, None), ("a", False, True, None), ("ea", True, True, None),
                                ("p", False, False, F), ("pe", True, False, F)):
            blk, cl = veto_ep(dis, T, S, Y, s_ok, f_ok, m, ep, ag, pf)
            routes[f"{tag}{m}b"] = (j & ~blk) | base
            diag[f"{tag}{m}b"] = dict(blocked=int((j & blk).sum()), cleared=cl,
                                      ret_blocked_store_right=float(s_ok[(j & blk) & (RET == 1)].mean()) if ((j & blk) & (RET == 1)).any() else float("nan"))
    sw = log["swap_at"]; g = {}
    for k, use in routes.items():
        ok = np.where(use, S, F) == Y
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw))[:300] if w.get("swap") else np.array([], int)
        g[k] = dict(lifetime=float(ok.mean()), returns=float(ok[RET == 1].mean()) if (RET == 1).any() else float("nan"),
                    swap=float(ok[idx].mean()) if len(idx) else float("nan"), fired=float(use.mean()))
    return g, diag


def main():
    os.makedirs(OD, exist_ok=True)
    jobs = [(wk, s) for wk, (_, _, ss) in CR.WORLDS.items() for s in ss]
    logs = {}
    for wk, (_, d, seeds) in CR.WORLDS.items():
        for r in json.load(open(os.path.join(d, "results.json")))["raw"]:
            if r["seed"] in seeds:
                logs[(wk, r["seed"])] = r
    fe, se = map(np.array, json.load(open("results/arbiter/results.json"))["edges"])
    per, dg = {}, {}
    for k in jobs:
        per[k], dg[k] = score(k[0], k[1], np.load(f"results/confirm_recency/Z_{k[0]}_{k[1]}.npy"), logs[k], fe, se)
    M = lambda cells, k, f: float(np.nanmean([per[c][k][f] for c in cells]))
    sel_cells = [("s49", s) for s in (0, 1, 2)]
    ref = "j+0.0b"
    cands = [k for k in per[sel_cells[0]] if k[0] in "ea"]
    cands_p = [k for k in per[sel_cells[0]] if k[0] == "p"]
    ok = lambda cells, k: (M(cells, k, "lifetime") >= M(cells, ref, "lifetime") - 0.0002 and
                           M(cells, k, "returns") >= M(cells, ref, "returns") - 0.005)
    pick = lambda cs: max(cs, key=lambda k: M(sel_cells, k, "swap") if ok(sel_cells, k) else -1)
    sel, sel_p = pick(cands), pick(cands_p)
    groups = {"s49 selection 0-2": sel_cells, "s49 held-out 3-5": [("s49", s) for s in (3, 4, 5)],
              "bg002 (swap 2<->3) seeds 6-8": [("bg002", s) for s in (6, 7, 8)], "bg 0.2 seeds 0-5": [("bg02", s) for s in range(6)]}
    md = ["# Episode-scoped surprise veto (`episode_veto.py`)\n", __doc__,
          f"\nSelected on s49 seeds 0-2: **{sel}** (passes selection bars: {ok(sel_cells, sel)}); "
          f"H5b pair veto: **{sel_p}** (passes: {ok(sel_cells, sel_p)})\n"]
    summary, verdict = {}, {}
    for name, cells in groups.items():
        md += [f"\n## {name}, mean %\n",
               "| Route | Lifetime | Returns | Swap first 300 | Store answers | Swap vs j+0.0b | Returns vs j+0.0b | blocked | stale-cleared | blocked-on-return store right |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for k in per[cells[0]]:
            d = [dg[c].get(k) for c in cells]
            dtxt = (f"{np.mean([x['blocked'] for x in d]):.0f} | {np.mean([x['cleared'] for x in d]):.0f} | "
                    f"{np.nanmean([x['ret_blocked_store_right'] for x in d])*100:.0f}%") if d[0] else "| |"
            md.append(f"| {k}{' (sel)' if k == sel else ''} | {M(cells,k,'lifetime')*100:.2f} | {M(cells,k,'returns')*100:.1f} | "
                      f"{M(cells,k,'swap')*100:.1f} | {M(cells,k,'fired')*100:.2f} | {(M(cells,k,'swap')-M(cells,ref,'swap'))*100:+.1f} | "
                      f"{(M(cells,k,'returns')-M(cells,ref,'returns'))*100:+.1f} | {dtxt} |")
        summary[name] = {k: {f: M(cells, k, f) for f in ("lifetime", "returns", "swap", "fired")} for k in per[cells[0]]}
    def judge(sel):
        verdict = {}
        for name in ("s49 held-out 3-5", "bg002 (swap 2<->3) seeds 6-8"):
            cs = groups[name]
            verdict[name] = dict(swap_gain=M(cs, sel, "swap") - M(cs, ref, "swap"),
                                 life=M(cs, sel, "lifetime") - M(cs, ref, "lifetime"), ret=M(cs, sel, "returns") - M(cs, ref, "returns"))
        cs = groups["bg 0.2 seeds 0-5"]; verdict["bg02_life"] = M(cs, sel, "lifetime") - M(cs, ref, "lifetime")
        passed = all(v["swap_gain"] >= 0.005 and v["life"] >= -0.0002 and v["ret"] >= -0.005 for n, v in verdict.items() if n != "bg02_life") \
            and verdict["bg02_life"] >= -0.0002
        return verdict, passed
    verdict, passed = judge(sel); verdict_p, passed_p = judge(sel_p)
    md.append(f"\n## Verdict H5 ({sel})\n\n{json.dumps(verdict, indent=1)}\n\nPrediction {'PASSED' if passed else 'FAILED (killed)'}.\n")
    md.append(f"\n## Verdict H5b ({sel_p})\n\n{json.dumps(verdict_p, indent=1)}\n\nPrediction {'PASSED' if passed_p else 'FAILED (killed)'}.\n")
    json.dump({"sel": sel, "passed": passed, "verdict": verdict, "sel_p": sel_p, "passed_p": passed_p, "verdict_p": verdict_p, "summary": summary,
               "raw": {f"{a}:{b}": v for (a, b), v in per.items()}, "diag": {f"{a}:{b}": v for (a, b), v in dg.items()}},
              open(os.path.join(OD, "results.json"), "w"))
    open(os.path.join(OD, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
