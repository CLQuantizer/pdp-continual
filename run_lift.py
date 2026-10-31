"""Run-length rival lift (loop 14:00 UTC+8 31 Oct, pre-registered before running).

z_window (13:00) showed the z5 extinction window is only ~2 of the 18-tick swap latency cost (z5 77 vs base 59 ticks);
the c1.5 rival veto is the rest: right after a swap the true 9 prototypes sit close to the relabelled 4s, so early
conflicts on 4 prototypes are vetoed. Rule rL inside z5: count, per prototype, consecutive in-radius conflicting hits
that all carry the same new label y (reset by a same-label confirmation of that prototype or by an in-radius conflict
with another label). When the run reaches L, lift the rival veto for that hit. Rationale: a borderline 9 is an isolated
event inside a 4 prototype; a relabelling produces an unbroken streak (Gershman, Blei & Niv 2010: a run of surprising
outcomes, not one, signals a new latent cause).
  H: r3 gives s49 first post-swap remap <= 65 ticks, s49 swap (mq30_s2) >= z5 + 0.8 (z5 68.1 on mq30_s2 in z_window_s49),
  and <= 1 pre-swap false remap per run on all 24 runs (s49 seeds 0-5 + 18 old runs).
  Kill: false remaps return on any run (> 1), or swap gain < 0.8.
Variants r2, r3, r5. Usage: python3 run_lift.py A|B
"""
import sys, re
import rival_remap as RR
import rival_s49 as RS


class LStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.L = None; self.runs = {}; self.lifts = 0
        m = re.search(r"_r(\d+)$", K)
        if m:
            self.L = int(m.group(1)); K = K[:m.start()]
        super().__init__(*a, K=K, **kw)

    def observe(self, x, y):
        if self.L and self.mu:
            s, j, d = self.nearest(x)
            if s == y:
                self.runs.pop((s, j), None)
        return super().observe(x, y)

    def rival(self, x, y, d):
        r = super().rival(x, y, d)
        if self.L:
            key = (self._s, self._j); ly, n = self.runs.get(key, (y, 0))
            n = n + 1 if ly == y else 1
            self.runs[key] = (y, n)
            if r and n >= self.L:
                self.lifts += 1; return False
        return r


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = LStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_r2",
                      "a25w250h2k2_c1.5_uinf_z5_r3", "a25w250h2k2_c1.5_uinf_z5_r5"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/run_lift_s49"
        RR.TITLE = "# Run-length rival lift, confusable swap (`run_lift.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/run_lift"
        RR.TITLE = "# Run-length rival lift, 18 old runs (`run_lift.py B`)\n"
    RR.main()
