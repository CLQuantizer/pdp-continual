"""Core-only confirmation inside the extinction override (loop 10:05 UTC+8 31 Oct, pre-registered before running).

z5 (rival_s49) lifts the rival veto once a prototype's last 5 history entries hold no same-label confirmation (0).
On the confusable swap 4<->9 it still costs 1.6 swap vs base (68.1 vs 69.7), and the window-only confirmation rules
of 09:05 cost more. Suspected mechanism (09:05): after the swap, real 9s arrive labelled 4; being confusable, some
land inside 4 prototypes and are logged as same-label confirmations, so the stale cause keeps being "confirmed".
Mechanism check (M): count post-swap same-label hits on prototypes of the swapped pair, split by whether the
arrival's true class is the partner (impostor confirmation) or the owner, and their d/r (squared distances).
  M Prediction: impostor confirmations are a material share (>= 20%) of post-swap confirmations on swapped
  prototypes, and sit at larger d/r than genuine ones (median impostor d/r > genuine + 0.2). Kill of the mechanism:
  share < 10% or medians within 0.1.
Rule qQ (Yu & Dayan 2005: rim hits on a confusable border are known-unreliable): a same-label hit still updates the
radius, but is logged as a confirmation (0) only if d <= Q x r; a rim hit is logged as neutral (3). Inside z5.
  H1 Prediction: z5_q0.5 on s49 swap (mq30_s2) >= z5 + 0.8, and 0 runs with > 5 pre-swap false remaps on all 24 runs,
  and swap within 1.0 of z5 on the 18 old runs. Kill: s49 gain < +0.8 or any run > 5 false remaps.
Usage: python3 core_conf.py A|B
"""
import sys
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import rival_remap as RR
import rival_s49 as RS


class QStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.q = None
        head, tail = K.rsplit("_", 1)
        if tail.startswith("q"):
            self.q = float(tail[1:]); K = head
        super().__init__(*a, K=K, **kw)
        self.conf = []  # (t, s, true class, d/r) for same-label hits

    def observe(self, x, y):
        if self.mu and y in self.mu and len(self.mu[y]):
            s, j, d = self.nearest(x)
            r = self.rad[s][j]
            if s == y and r is not None:
                self.conf.append((self.t, s, self._c, d / r if r > 0 else 0.0))
                if self.q is not None and d > self.q * r:
                    self.win.append((self.t, y))
                    while self.win and self.win[0][0] < self.t - self.W:
                        self.win.popleft()
                    self.rad[s][j] = r + 0.1 * (d - r); self.hist[s][j].append(3); return
        return super().observe(x, y)


def run(args):
    seed, days, w, Ks = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: QStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    out = {K: [] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t; st._c = int(c); st.observe(x, y); st.add(x, y)
    sw = world.swap_at; pair = set(w["swap"])
    def mech(st):
        post = [(c != s, q) for t, s, c, q in st.conf if t >= sw and s in pair]
        imp = [q for i, q in post if i]; gen = [q for i, q in post if not i]
        return dict(n=len(post), imp=len(imp), share=len(imp) / max(1, len(post)),
                    med_imp=float(np.median(imp)) if imp else None, med_gen=float(np.median(gen)) if gen else None,
                    imp_core=sum(q <= 0.5 for q in imp), gen_core=sum(q <= 0.5 for q in gen))
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), rivals=s.rivals, mech=mech(s), post_lat={str(c): next((t - sw for t, s_, _ in s.log if t >= sw and s_ == c), None) for c in w["swap"]}) for K, s in stores.items()}


if __name__ == "__main__":
    part = sys.argv[1]
    RR.run = run
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z5", "a25w250h2k2_c1.5_uinf_z5_q0.5", "a25w250h2k2_c1.5_uinf_z5_q0.25"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/core_conf_s49"
        RR.TITLE = "# Core-only confirmation, confusable swap (`core_conf.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/core_conf"
        RR.TITLE = "# Core-only confirmation, 18 old runs (`core_conf.py B`)\n"
    RR.main()
