"""Cortex-vetted confirmation (loop 12:05 UTC+8 31 Oct, pre-registered before running).

post_conf (11:05) proved impostor confirmations are real (54% of post-swap hits on un-remapped 4/9 prototypes) but
invisible to store geometry (relative weight 0.579 vs 0.589). CLS (McClelland, McNaughton & O'Reilly 1995): right
after a relabelling the slow cortex still calls a true 9 a 9. Rule cv inside z5: a same-label hit on prototype s
counts as a confirmation (history 0) only if the FF cortex's prediction for that arrival (gate_probe log column f,
made before learning on it) is s; otherwise it is logged as a neutral non-confirmation (3). The radius update is kept.
  H1 Prediction: cv gives s49 swap (mq30_s2) >= z5 + 0.8 and <= 1 pre-swap false remap per run on all 24 runs, and
  swap within 1.0 of z5 on the 18 old runs. Kill: either of the first two lines broken.
Diagnostic: vetoed confirmations pre-swap (cortex errors on real members) vs post-swap on 4/9 prototypes.
Usage: python3 cortex_conf.py A|B
"""
import sys, json, os
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import rival_remap as RR
import rival_s49 as RS


class VStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.cv = K.endswith("_cv")
        if self.cv:
            K = K[:-3]
        super().__init__(*a, K=K, **kw)
        self.f = None; self.sw = None; self.swp = set(); self.vd = dict(pre=0, pre_n=0, post=0, post_n=0)

    def observe(self, x, y):
        if not self.mu:
            return super().observe(x, y)
        s, j, d = self.nearest(x)
        super().observe(x, y)
        if s == y:
            post = self.sw is not None and self.t >= self.sw
            k = ("post" if post else "pre")
            if not post or s in self.swp:
                self.vd[k + "_n"] += 1
            if self.f != s:
                if not post or s in self.swp:
                    self.vd[k] += 1
                if self.cv:
                    self.hist[s][j][-1] = 3


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    F = FCOL[(w["bg"], seed)]
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: VStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    for st in stores.values():
        st.sw, st.swp = world.swap_at, set(w["swap"])
    out = {K: [] for K in Ks}; i = 0
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.f = int(F[i]); st.observe(x, y); st.add(x, y)
        i += 1
    sw = world.swap_at
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), rivals=s.rivals, veto=s.vd, post_lat={str(c): next((t - sw for t, s_, _ in s.log if t >= sw and s_ == c), None) for c in w["swap"]}) for K, s in stores.items()}


FCOL = {}
if __name__ == "__main__":
    part = sys.argv[1]
    RR.run = run
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_cv"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/cortex_conf_s49"
        RR.TITLE = "# Cortex-vetted confirmation, confusable swap (`cortex_conf.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/cortex_conf"
        RR.TITLE = "# Cortex-vetted confirmation, 18 old runs (`cortex_conf.py B`)\n"
    for path, bg, n in RR.LOGS:
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:n]:
            FCOL[(bg, r["seed"])] = np.array(r["rows"])[:, 5]
    RR.main()
