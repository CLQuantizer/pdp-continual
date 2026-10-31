"""Confirmation-only remap evidence (loop 09:05 UTC+8 31 Oct, pre-registered before running).

rival_s49 (08:05) showed the fixed-ratio rival test fails on a confusable swap (4<->9), and that the extinction
override z5 (lift the rival veto once the prototype has gone 5 entries unconfirmed) is safe on all 24 runs but still
costs 1.6 swap on s49, because the geometric rival test vetoes the first true hits. Gershman, Blei & Niv (2010):
infer a new latent cause when the old cause stops being confirmed, not when one observation is surprising.
Rule nN (no distance ratio at all): an in-radius conflicting hit counts as remap evidence only if the prototype's last
N history entries hold no same-label confirmation (0). Variants n3, n5, n8 on the plain base store.
  H1 Prediction: base_n5 keeps pre-swap false remaps <= 1/run on all 24 runs, and on s49 swap (mq30_s2) is within 1.0
  of base (beats z5's -1.6). Kill: any run > 5 pre-swap remaps, or s49 swap loss > 1.0.
  H3 (window sensitivity): if n3 and n8 also pass, N is a free choice; if both break a line, the rule is a tuned point.
Usage: python3 conf_trace.py A|B
"""
import sys
import rival_remap as RR
import rival_s49 as RS


class NStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.nconf = None
        head, tail = K.rsplit("_", 1)
        if tail.startswith("n") and tail[1:].isdigit():
            self.nconf = int(tail[1:]); K = head
        super().__init__(*a, K=K, **kw)

    def rival(self, x, y, d):
        r = super().rival(x, y, d)
        if self.nconf:
            return r or 0 in list(self.hist[self._s][self._j])[-self.nconf:]
        return r


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = NStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_base_n3", "a25w250h2k2_base_n5",
                      "a25w250h2k2_base_n8"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/conf_trace_s49"
        RR.TITLE = "# Confirmation-only remap evidence, confusable swap (`conf_trace.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/conf_trace"
        RR.TITLE = "# Confirmation-only remap evidence, 18 old runs (`conf_trace.py B`)\n"
    RR.main()
