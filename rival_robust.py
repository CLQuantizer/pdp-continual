"""Robustness of the rival test with directional unlock (loop 06:10 UTC+8 31 Oct, pre-registered before running).

rival_unlock.py chose C = 1.5 and U = 2000 after seeing seed 6, so c1.5_u may be a tuned point rather than a rule.
Sweep the neighbours on the same 15 runs: C in {1.2, 2.0, 3.0} at U = 2000, and U in {500, 8000, inf} at C = 1.5.
Prediction (rule is robust): every neighbour keeps pre-swap false remaps <= 1 per run (seed 6 <= 5) and held-out
bg 0.02 swap (mq30_s2) within 1.0 of base.  Kill (fragile, tuned point): two or more neighbours break either line.
Side question: if U = inf (a pair stays unlocked for life once it has remapped) is as good as U = 2000, drop U.
"""
import rival_remap as RR
import rival_unlock as RU


class RStore(RU.UStore):
    def __init__(self, *a, K=None, **kw):
        base, rule = K.rsplit("_", 1) if "_u" in K else (K, None)
        if rule and rule.startswith("u") and len(rule) > 1:
            u = rule[1:]; self.U = float("inf") if u == "inf" else int(u); K = base + "_u"
        super().__init__(*a, K=K, **kw)


if __name__ == "__main__":
    RR.STORE = RStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_u", "a25w250h2k2_c1.2_u", "a25w250h2k2_c2.0_u",
                      "a25w250h2k2_c3.0_u", "a25w250h2k2_c1.5_u500", "a25w250h2k2_c1.5_u8000", "a25w250h2k2_c1.5_uinf"]
    RR.OUT_OVERRIDE = "results/rival_robust"
    RR.TITLE = "# Robustness of rival test + directional unlock (`rival_robust.py`)\n"
    RR.DOC = __doc__
    RR.main()
