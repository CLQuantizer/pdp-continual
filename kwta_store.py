"""Pattern-separated store code (loop 15:00 UTC+8 31 Oct, pre-registered before running; 14:00 hypothesis 1).

Lorden 1971: at a fixed false-alarm rate, remap latency ~ log(gamma)/I, so only raising per-hit evidence I can close the
s49 gap (z5 first post-swap remap 77 ticks vs base 59). In pixel space a 4 and a 9 overlap, so a 4 prototype's
post-swap hits mix true 4s (conflicts) with impostor 9s (confirmations, ~54%, post_conf). A dentate-gyrus-like code
(fixed random expansion to N units, k-winners-take-all, binary; Marr 1971, Treves & Rolls 1994, Babadi & Sompolinsky
2014) decorrelates similar inputs, so impostors should fall outside a 4 prototype's radius more often than true 4s.
Store K suffix _xN_kK: prototypes, radii, rival test and answers all live in the code space h = top_k(W (x - xbar)).
Control _xN_lin: same random projection, no k-WTA (distances ~ preserved, isolates the sparsification).
  H: _x2000_k100 inside z5 gives s49 swap (mq30_s2) >= z5 + 0.8, first post-swap remap <= 65 ticks, 0 pre-swap false
  remaps on s49 (and later on the 18 old runs). Kill: latency > 70 or swap gain < 0.8 or store lifetime loss > 1.0.
Usage: python3 kwta_store.py A (s49 seeds 0-5) | B (18 old runs)
"""
import sys, re
import numpy as np
import rival_remap as RR
import rival_s49 as RS
import remap_probe as RP
from data import load_mnist

_W = {}
_XBAR = None


def phi(x, N, k):
    global _XBAR
    if _XBAR is None:
        _XBAR = load_mnist("mnist")[0].mean(0).astype(np.float32)
    if N not in _W:
        _W[N] = np.random.default_rng(12345).standard_normal((784, N)).astype(np.float32) / np.sqrt(784)
    h = (x.reshape(1, -1) - _XBAR) @ _W[N]
    if k is None:
        return h.astype(np.float32)
    out = np.zeros_like(h); idx = np.argpartition(-h[0], k)[:k]; out[0, idx] = 1.0
    return out


class KStore(RS.ZStore):
    def __init__(self, *a, K=None, **kw):
        self.N = None
        m = re.search(r"_x(\d+)_(k(\d+)|lin)$", K)
        if m:
            self.N = int(m.group(1)); self.kk = int(m.group(3)) if m.group(3) else None; K = K[:m.start()]
            kw["dim"] = self.N; kw["clip"] = (None, None)
        super().__init__(*a, K=K, **kw)
        self._cx = (None, None)

    def code(self, x):
        if self.N is None:
            return x
        if self._cx[0] is not x:
            self._cx = (x, phi(x, self.N, self.kk))
        return self._cx[1]

    def observe(self, x, y):
        return super().observe(self.code(x), y)

    def add(self, x, y, surprise=False):
        return super().add(self.code(x), y, surprise)


_answer = RP.answer
RP.answer = lambda st, x: _answer(st, st.code(x) if hasattr(st, "code") else x)

if __name__ == "__main__":
    part = sys.argv[1]
    RR.STORE = KStore
    z = "a25w250h2k2_c1.5_uinf_z5"
    RR.KS_OVERRIDE = ["a25w250h2k2_base", z, z + "_x2000_k100", z + "_x2000_k40", z + "_x2000_lin"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/kwta_store_s49"
        RR.TITLE = "# Pattern-separated store code, confusable swap (`kwta_store.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/kwta_store"
        RR.TITLE = "# Pattern-separated store code, 18 old runs (`kwta_store.py B`)\n"
    RR.main()
