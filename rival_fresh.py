"""Fresh-seed test of the rival test, step vs graded (loop 07:05 UTC+8 31 Oct, pre-registered before running).

rival_robust.py: permanent directional unlock (c1.5_uinf) gave 0 pre-swap false remaps on seeds 0-8 / bg 0.2 at no
swap cost, but C = 1.5 was picked after seeing seed 6 and its neighbours fail (1.2 leaks, 2.0 costs 9 swap).
Part A (confirmation): FF logs for bg 0.02 seeds 9-11 were made after C was chosen.
  Prediction: c1.5_uinf keeps pre-swap false remaps <= 1 per run and swap (mq30_s2) within 1.0 of base.
  Kill: any run > 5 pre-swap remaps, or swap loss > 1.0.
Part B (graded, CA1-like comparator; Lisman & Grace 2005, Duncan et al. 2012 graded novelty): instead of a step at C, a
conflict hit counts with weight w = clip((q - a) / (b - a), 0, 1), q = d_y / d_s; remap when summed weight >= H = 2.
Variants g1-2 and g1.2-3, both with permanent unlock. Run on all 18 runs (seeds 0-11 bg 0.02, 0-5 bg 0.2).
  Prediction (graded beats step): at least one g variant has <= 1 pre-swap remap per run everywhere and swap within
  0.5 of base; and it tolerates a 2x change of its scale (both variants pass).  Kill: both g variants fail either line.
"""
import numpy as np
import rival_remap as RR
import rival_robust as RB


class GStore(RB.RStore):
    def __init__(self, *a, K=None, **kw):
        self.g = None
        parts = K.split("_")
        if len(parts) > 1 and parts[1].startswith("g"):
            lo, hi = parts[1][1:].split("-"); self.g = (float(lo), float(hi)); parts[1] = "c1.5"
            K = "_".join(parts)
        super().__init__(*a, K=K, **kw)

    def weight(self, x, y, d):
        if self.u:
            s = self._s
            if (s, y) in self.unlock and self.t - self.unlock[(s, y)] <= self.U:
                return 1.0
        if y not in self.mu or not len(self.mu[y]):
            return 1.0
        q = ((self.mu[y] - x[0]) ** 2).sum(1).min() / max(d, 1e-12)
        a, b = self.g
        return float(np.clip((q - a) / (b - a), 0, 1))

    def observe(self, x, y):
        if self.g is None:
            return super().observe(x, y)
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        if not self.mu:
            return
        s, j, d = self.nearest(x)
        r = self.rad[s][j]
        if s == y:
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
        inr = r is not None and d <= r
        wt = self.weight(x, y, d) if inr else 0.0
        if inr and wt < 1:
            self.rivals += 1
        if not (inr and wt > 0):
            self.hist[s][j].append(2); return
        self.hist[s][j].append((1, y, wt))
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e[2] for e in h if isinstance(e, tuple) and e[1] == y) >= self.H - 1e-9]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if RR.PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


if __name__ == "__main__":
    RR.STORE = GStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf", "a25w250h2k2_g1-2_uinf", "a25w250h2k2_g1.2-3_uinf"]
    RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
               ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
    RR.GROUPS = ((0.02, [9, 10, 11], "FRESH seeds 9-11"), (0.02, [3, 4, 5, 6, 7, 8], "held-out seeds 3-8"),
                 (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
    RR.OUT_OVERRIDE = "results/rival_fresh"
    RR.TITLE = "# Fresh-seed and graded rival test (`rival_fresh.py`)\n"
    RR.DOC = __doc__
    RR.main()
