"""Retroactive credit on rival lift (loop 14:00 UTC+8 31 Oct, pre-registered before running, after cusum_lift A).

run_lift and cusum_lift A: lifting the c1.5 rival veto earlier (streak or CUSUM) leaves first post-swap remap at 73-77
ticks (z5 77, base 59). Suspected cause: vetoed conflicts are written as non-evidence (2), so when the veto lifts the
H2K2 remap test starts counting from zero. Rule cuP_hH_rc: as cusum_lift, but vetoed conflicts are kept as tagged
latent evidence (5, y); when the CUSUM lift fires on a prototype of s, all of s's (5, y) entries become (1, y).
(Reconsolidation-style: a memory that was protected from updating becomes labile once a mismatch is confirmed;
Nader, Schafe & LeDoux 2000; Sevenster, Beckers & Kindt 2013 show a prediction error is required.)
  H: cu0.5_h3_rc gives s49 first post-swap remap <= 65 ticks and swap (mq30_s2) >= z5 + 0.8 with <= 1 pre-swap false
  remap per run on all 24 runs. Kill: false remaps > 1 on any run, or swap gain < 0.8.
Usage: python3 retro_lift.py A|B
"""
import sys
import rival_remap as RR
import cusum_lift as CL


class RCStore(CL.CuStore):
    def __init__(self, *a, K=None, **kw):
        self.rc = K.endswith("_rc"); self._vet = None
        super().__init__(*a, K=K[:-3] if self.rc else K, **kw)

    def observe(self, x, y):
        self._vet = None
        if self.rc and self.mu:
            s, j, d = self.nearest(x)
        super().observe(x, y)
        if self._vet is not None and self.hist[s][j] and self.hist[s][j][-1] == 2:
            self.hist[s][j][-1] = (5, self._vet)

    def rival(self, x, y, d):
        r = CL.RS.ZStore.rival(self, x, y, d)
        if self.P is None:
            return r
        key = (self._s, self._j); ly, v = self.S.get(key, (y, 0.0))
        v = v + 1 if ly == y else 1.0
        self.S[key] = (y, v)
        if r and v >= self.Hc:
            if self.rc:
                for h in self.hist[self._s]:
                    for i, e in enumerate(h):
                        if e == (5, y):
                            h[i] = (1, y)
            return False
        if r and self.rc:
            self._vet = y
        return r


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = RCStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_cu0.5_h3",
                      "a25w250h2k2_c1.5_uinf_z5_cu0.5_h3_rc", "a25w250h2k2_c1.5_uinf_z5_cu0.5_h5_rc"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/retro_lift_s49"
        RR.TITLE = "# Retroactive credit on rival lift, confusable swap (`retro_lift.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/retro_lift"
        RR.TITLE = "# Retroactive credit on rival lift, 18 old runs (`retro_lift.py B`)\n"
    RR.main()
