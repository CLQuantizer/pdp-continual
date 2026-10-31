"""Deferred confirmation gate (loop 09:05 UTC+8 31 Oct, pre-registered after conf_trace, before running).

conf_trace killed the confirmation-only veto nN: 0 pre-swap false remaps on all 24 runs, but swap (mq30_s2) falls
4-12 points, monotone in N, and first post-swap remap latency rises 88-539 ticks. Mechanism: a vetoed hit is logged
as out-of-radius (2), so the evidence that arrives while the prototype still carries pre-swap confirmations is thrown
away and must be collected again. Gershman, Blei & Niv (2010) infer a new cause from the whole run of unconfirmed
observations, including those seen before the inference is made. Rule dN: every in-radius conflicting hit is logged as
evidence (1, y) as in base; a prototype may only be remapped (counted as hot) when its last N history entries hold no
same-label confirmation. Evidence is deferred, not discarded.
  Prediction: d5 keeps pre-swap false remaps <= 1/run on all 24 runs and swap within 1.0 of base on every group,
  including s49. Kill: any run > 5 pre-swap remaps, or swap loss > 1.0 on any group.
Usage: python3 conf_defer.py A|B
"""
import sys
import rival_remap as RR
import presence_bg as PB
import conf_trace as CT


class DStore(CT.NStore):
    def __init__(self, *a, K=None, **kw):
        self.dconf = None
        head, tail = K.rsplit("_", 1)
        if tail.startswith("d") and tail[1:].isdigit():
            self.dconf = int(tail[1:]); K = head
        super().__init__(*a, K=K, **kw)

    def observe(self, x, y):
        if not self.dconf:
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
        self.hist[s][j].append((1, y) if inr else 2)
        if not inr:
            return
        N = self.dconf
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e == (1, y) for e in h) >= self.H]
        stale = [i for i in hot if 0 not in list(self.hist[s][i])[-N:]]
        if len(stale) < len(hot):
            self.rivals += 1
        if len(stale) >= self.K and len(self.mu[s]) > len(stale):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(stale, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = DStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_base_d3", "a25w250h2k2_base_d5"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/conf_defer_s49"
        RR.TITLE = "# Deferred confirmation gate, confusable swap (`conf_defer.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/conf_defer"
        RR.TITLE = "# Deferred confirmation gate, 18 old runs (`conf_defer.py B`)\n"
    RR.main()
