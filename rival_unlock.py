"""Rival-aware remap with directional unlock (loop 05:00 hyp 1b, pre-registered 05:20 UTC+8 31 Oct).

rival_remap.py: vetoing conflict evidence when a prototype of the new label y is a close rival removes the
ping-pong false remaps (c1.5: 0 pre-swap remaps on every run, seed 6 from 34) but costs 3.4 swap points, because
after the first true remap s->y the remapped prototype is itself a y rival sitting among the s prototypes and vetoes
the rest of the remap. Fix: once a remap s->y has fired, the pair (s, y) is unlocked (no rival check) for U = 2000 ticks.
The reverse pair (y, s), the ping-pong direction, stays checked. Variants rad_u, c1.5_u.
Test: c1.5_u.  Prediction: pre-swap remaps <= 1 per run everywhere (seed 6 <= 5); swap (mq30_s2) within 0.5 of base on
bg 0.02 seeds 3-8 and bg 0.2; returns within 1.0.  Kill: swap loss > 1.0 or seed-6 pre-swap remaps > 10.
"""
import rival_remap as RR


class UStore(RR.CStore):
    U = 2000

    def __init__(self, *a, K=None, **kw):
        self.unlock = {}
        u = K.endswith("_u"); super().__init__(*a, K=K[:-2] if u else K, **kw); self.u = u
        _move = self.move

        def move(s, j, y):
            self.unlock[(s, y)] = self.t; _move(s, j, y)
        self.move = move

    def rival(self, x, y, d):
        if self.u:
            s = self._s
            if (s, y) in self.unlock and self.t - self.unlock[(s, y)] <= self.U:
                return False
        return super().rival(x, y, d)

    def nearest(self, x):
        b = super().nearest(x); self._s = b[0]; return b


if __name__ == "__main__":
    RR.STORE = UStore
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5", "a25w250h2k2_rad_u", "a25w250h2k2_c1.5_u"]
    RR.OUT_OVERRIDE = "results/rival_unlock"
    RR.TITLE = "# Rival-aware remap with directional unlock (`rival_unlock.py`)\n"
    RR.DOC = __doc__
    RR.main()
