"""Posterior-weighted confirmation (REMERGE) + origin tracking (loop 11:05 UTC+8 31 Oct, pre-registered before running).

core_conf (10:05) showed a confirmation's absolute distance cannot tell a lure from a member: both sit at the mean
radius. REMERGE (Kumaran & McClelland 2012) says support for a stored item is relative: a probe nearly as close to a
rival item gives about half the support. Rule pP inside z5: each same-label hit on prototype (s, j) carries weight
w = d_r / (d_s + d_r), d_r = squared distance to the nearest prototype of any other label; every other history entry
carries 0. The rival veto is lifted when the summed weight of the last 5 entries is < P (z5 is P -> 1 with w = 1).
  H1 Prediction: some P in {1.0, 1.5} gives s49 swap (mq30_s2) >= z5 + 0.8 and 0 runs with > 5 pre-swap false remaps
  on all 24 runs, swap within 1.0 of z5 on the 18 old runs. Kill: no P meets both lines.
Diagnostic O (origin tracking, settles the 09:05 mechanism): each prototype carries the label it was created with.
After the swap, for prototypes of the swapped pair that still carry their original label, count history entries:
same-label confirmations (all by construction impostors, i.e. true partner digits) vs in-radius conflicts.
  O Prediction (mechanism real): impostor confirmations >= 20% of (confirmations + in-radius conflicts) on those
  prototypes, and their mean w is lower than pre-swap confirmations' mean w by >= 0.05. Kill: share < 10%.
Usage: python3 post_conf.py A|B
"""
import sys
from collections import deque
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import rival_remap as RR
import rival_s49 as RS


class PStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.P = None
        head, tail = K.rsplit("_", 1)
        if tail.startswith("p"):
            self.P = float(tail[1:]); K = head
        super().__init__(*a, K=K, **kw)
        self.meta = {}  # id(deque) -> [deque, origin label, weights deque]
        self.sw = None; self.diag = dict(pre_w=[], post_w=[], post_conf=0, post_conf_inr=0, post_out=0)

    def _m(self, s, j):
        h = self.hist[s][j]; m = self.meta.get(id(h))
        if m is None or m[0] is not h:
            m = self.meta[id(h)] = [h, s, deque(maxlen=10)]
        return m

    def move(self, s, j, y):
        org = self._m(s, j)[1]
        super().move(s, j, y)
        for k in range(len(self.hist[y])):
            h = self.hist[y][k]
            if len(h) == 0 and id(h) not in self.meta:
                self.meta[id(h)] = [h, org, deque(maxlen=10)]

    def rival(self, x, y, d):
        r = super(RS.ZStore, self).rival(x, y, d)
        if r and self.z:
            if self.P is None:
                h = list(self.hist[self._s][self._j])[-self.z:]
                if len(h) >= self.z and 0 not in h:
                    return False
            else:
                wv = list(self._m(self._s, self._j)[2])[-self.z:]
                if len(wv) >= self.z and sum(wv) < self.P:
                    return False
        return r

    def observe(self, x, y):
        if not self.mu:
            return super().observe(x, y)
        s, j, d = self.nearest(x)
        m = self._m(s, j); r = self.rad[s][j]
        post = self.sw is not None and self.t >= self.sw
        if s == y:
            dr = min((((self.mu[c] - x[0]) ** 2).sum(1).min() for c in self.mu if c != s and len(self.mu[c])), default=np.inf)
            w = 1.0 if not np.isfinite(dr) else dr / (d + dr)
            m[2].append(w)
            if s in self.swp:
                if not post: self.diag["pre_w"].append(w)
                elif m[1] == s: self.diag["post_w"].append(w); self.diag["post_conf"] += 1
        else:
            m[2].append(0.0)
            if post and s in self.swp and m[1] == s:
                if r is not None and d <= r: self.diag["post_conf_inr"] += 1
                else: self.diag["post_out"] += 1
        return super().observe(x, y)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: PStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    for st in stores.values():
        st.sw, st.swp = world.swap_at, set(w["swap"])
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st.observe(x, y); st.add(x, y)
    sw = world.swap_at
    def dg(st):
        D = st.diag; n = D["post_conf"] + D["post_conf_inr"]
        return dict(post_conf=D["post_conf"], post_conflict_inr=D["post_conf_inr"], post_out=D["post_out"],
                    share=D["post_conf"] / max(1, n), pre_w=float(np.mean(D["pre_w"])) if D["pre_w"] else None,
                    post_w=float(np.mean(D["post_w"])) if D["post_w"] else None)
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), rivals=s.rivals, diag=dg(s), post_lat={str(c): next((t - sw for t, s_, _ in s.log if t >= sw and s_ == c), None) for c in w["swap"]}) for K, s in stores.items()}


if __name__ == "__main__":
    part = sys.argv[1]
    RR.run = run
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_p1.0", "a25w250h2k2_c1.5_uinf_z5_p1.5"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/post_conf_s49"
        RR.TITLE = "# Posterior-weighted confirmation, confusable swap (`post_conf.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/post_conf"
        RR.TITLE = "# Posterior-weighted confirmation, 18 old runs (`post_conf.py B`)\n"
    RR.main()
