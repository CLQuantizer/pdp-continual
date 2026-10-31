"""CUSUM rival lift (loop 14:00 UTC+8 31 Oct, pre-registered before running, after run_lift A).

run_lift A: streak-based lift r2/r3/r5 gives s49 swap 68.5/68.2/68.1 vs z5 68.1 and latency 73/77/77: after a 4<->9
swap a 4 prototype receives true 4s now labelled 9 (conflicts) interleaved with true 9s now labelled 4 (impostor
confirmations, ~54% of hits, post_conf), so streaks of >= 3 conflicts are rare. An ideal change detector (Page 1954
CUSUM; Gallistel, Mark, King & Latham 2001 show rats approximate one for reward rates) does not need a streak: it
integrates conflict minus weighted confirmation and fires on net drift. Rule cuP_hH inside z5: per prototype,
S <- max(0, S + 1) on an in-radius conflict with label y (S reset if y changes), S <- max(0, S - P) on a same-label
confirmation; lift the rival veto when S >= H. Pre-swap conflict share per prototype is small, so drift is negative;
post-swap ~46% conflicts vs ~54% impostor confirmations gives drift ~ +0.19 per hit at P = 0.5.
  H: cu0.5_h3 gives s49 swap (mq30_s2) >= z5 + 0.8 and first post-swap remap <= 65 ticks with <= 1 pre-swap false
  remap per run on all 24 runs. Kill: false remaps > 1 on any run, or swap gain < 0.8.
Variants cu0.5_h3, cu0.5_h5, cu0.25_h3. Usage: python3 cusum_lift.py A|B
"""
import sys, re
import rival_remap as RR
import rival_s49 as RS


class CuStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.P = None; self.S = {}
        m = re.search(r"_cu([\d.]+)_h(\d+)$", K)
        if m:
            self.P = float(m.group(1)); self.Hc = int(m.group(2)); K = K[:m.start()]
        super().__init__(*a, K=K, **kw)

    def observe(self, x, y):
        if self.P is not None and self.mu:
            s, j, d = self.nearest(x)
            if s == y and (s, j) in self.S:
                ly, v = self.S[(s, j)]; self.S[(s, j)] = (ly, max(0.0, v - self.P))
        return super().observe(x, y)

    def rival(self, x, y, d):
        r = super().rival(x, y, d)
        if self.P is not None:
            key = (self._s, self._j); ly, v = self.S.get(key, (y, 0.0))
            v = v + 1 if ly == y else 1.0
            self.S[key] = (y, v)
            if r and v >= self.Hc:
                return False
        return r


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = CuStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_cu0.5_h3",
                      "a25w250h2k2_c1.5_uinf_z5_cu0.5_h5", "a25w250h2k2_c1.5_uinf_z5_cu0.25_h3"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/cusum_lift_s49"
        RR.TITLE = "# CUSUM rival lift, confusable swap (`cusum_lift.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/cusum_lift"
        RR.TITLE = "# CUSUM rival lift, 18 old runs (`cusum_lift.py B`)\n"
    RR.main()
