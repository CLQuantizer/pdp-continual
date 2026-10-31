"""Confusable swap (4<->9) and the extinction override (loop 08:05 UTC+8 31 Oct, pre-registered before running).

The default store a25w250h2k2_c1.5_uinf vetoes conflict evidence labelled y when a y prototype is a close rival
(squared distance <= 1.5 x that to the s prototype). In every world so far the swapped pair was not confusable, so a
relabelled digit sat far from every real y prototype. World bg002s49 swaps 4<->9: a 4 now labelled 9 lands near 4
prototypes AND near 9 prototypes, so the rival test may veto the true remap.
Part A (decisive, s49 FF logs seeds 0-5, made after the default was chosen):
  H1 Prediction (rival test is a general rule): c1.5_uinf swap (mq30_s2) within 1.0 of base, pre-swap remaps <= 1/run.
  Kill: swap loss > 2.0 on s49.
Part B (extinction override, latent-cause inference, Gershman, Blei & Niv 2010): a lure arrives while its prototype
keeps being confirmed by its own label; a real relabelling stops all confirmations. Variant z5: the rival veto is
lifted when the s prototype's last 5 history entries hold no confirming (same-label) hit.
  H2 Prediction: z5 keeps pre-swap false remaps <= 1/run on all 18 old runs and on s49, and on s49 recovers swap to
  within 1.0 of base. Kill: any run > 5 pre-swap remaps, or s49 swap loss > 1.0 vs base.
Usage: python3 rival_s49.py A|B
"""
import sys
import rival_remap as RR
import rival_robust as RB


class ZStore(RB.RStore):
    def __init__(self, *a, K=None, **kw):
        self.z = None
        if K.endswith("_z5"):
            self.z = 5; K = K[:-3]
        super().__init__(*a, K=K, **kw)

    def nearest(self, x):
        b = super().nearest(x); self._j = b[1]; return b

    def rival(self, x, y, d):
        r = super().rival(x, y, d)
        if r and self.z:
            h = list(self.hist[self._s][self._j])[-self.z:]
            if len(h) >= self.z and 0 not in h:
                return False
        return r


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = ZStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf", "a25w250h2k2_c1.5_uinf_z5"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/rival_s49"
        RR.TITLE = "# Confusable swap 4<->9 vs rival test (`rival_s49.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/rival_z5"
        RR.TITLE = "# Extinction override on the 18 old runs (`rival_s49.py B`)\n"
    RR.main()
