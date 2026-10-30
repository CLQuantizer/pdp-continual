import json, numpy as np, life as LF, gate_probe as GP, veto_probe as VP
w = dict(LF.WORLD, bg=0.02)
for src in ("results/gate_probe", "results/gate_probe_s345"):
    res = json.load(open(src + "/results.json"))["raw"][:3]
    for bad in (3, 5, 8, 10, 15, 20):
        acc = []
        for r in res:
            R = np.array(r["rows"]); T, C, Y, INS, RET, F, S, FFM, SM, AS, AF, CF = R.T
            conf = (S >= 0) & (SM >= 0.2)
            base = float((S[conf & (T < r["swap_at"])] != Y[conf & (T < r["swap_at"])]).mean())
            dis = (S >= 0) & (S != F); prot = CF >= 1000; low10 = FFM < GP.rolling_quantile(FFM, 0.1)
            g = dis & prot & low10 & (SM >= 0.2) & ~VP.veto_mask(C, Y, S, SM, bad=bad)
            ok = np.where(g, S, F) == Y
            sw = float(ok[np.flatnonzero(np.isin(C, [2, 3]) & (T >= r["swap_at"]))[:300]].mean())
            acc.append((ok.mean(), ok[RET == 1].mean(), sw, base))
        a = np.mean(acc, 0) * 100
        print(src[-4:], bad, " ".join(f"{v:.1f}" for v in a))
