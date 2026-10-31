"""Extinction-window length (loop 13:00 UTC+8 31 Oct, pre-registered before running).

cortex_conf C (slow cortex, wake LR x0.1) vetoed 13.5% of post-swap 4/9 confirmations yet moved first post-swap remap
latency by 1 tick (76 vs 77; base 59). So impostor confirmations are not what delays remapping; the z5 window itself
(5 hist entries without a same-label confirmation before the rival veto lifts) is the likely 18-tick cost.
  H: z3 gives s49 swap (mq30_s2) >= z5 + 0.8 with <= 1 pre-swap false remap per run on all 24 runs and swap within
  1.0 of z5 on the 18 old runs -> new default. Kill: false remaps return on any run, or swap gain < 0.8.
Usage: python3 z_window.py A|B
"""
import sys, json, os
import numpy as np
import rival_remap as RR
import cortex_conf as CC

if __name__ == "__main__":
    part = sys.argv[1]
    RR.run = CC.run
    RR.KS_OVERRIDE = ["a25w250h2k2_base", "a25w250h2k2_c1.5_uinf_z2", "a25w250h2k2_c1.5_uinf_z3", "a25w250h2k2_c1.5_uinf_z5"]
    RR.DOC = __doc__
    if part == "A":
        RR.WEXTRA = dict(swap=(4, 9))
        RR.LOGS = (("results/gate_probe_s49", 0.02, 6),)
        RR.GROUPS = ((0.02, range(6), "CONFUSABLE swap 4<->9 (bg002s49), seeds 0-5"),)
        RR.OUT_OVERRIDE = "results/z_window_s49"
        RR.TITLE = "# Extinction-window length, confusable swap (`z_window.py A`)\n"
    else:
        RR.LOGS = (("results/gate_probe", 0.02, 3), ("results/gate_probe_s345", 0.02, 3), ("results/gate_probe_s678", 0.02, 3),
                   ("results/gate_probe_s91011", 0.02, 3), ("results/gate_probe_bg02", 0.2, 6))
        RR.GROUPS = ((0.02, [3, 4, 5, 6, 7, 8, 9, 10, 11], "held-out seeds 3-11"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5"))
        RR.OUT_OVERRIDE = "results/z_window"
        RR.TITLE = "# Extinction-window length, 18 old runs (`z_window.py B`)\n"
    for path, bg, n in RR.LOGS:
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:n]:
            CC.FCOL[(bg, r["seed"])] = np.array(r["rows"])[:, 5]
    RR.main()
