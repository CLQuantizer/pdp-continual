# ACh-style mismatch veto (`veto_probe.py`)

ACh-style mismatch veto on the margin gate (loop 19:40 hyp 2 / 20:25 hyp 1), scored offline.

Hasselmo 2006: acetylcholine signals mismatch between hippocampal recall and input; high ACh suppresses
recall-driven output and favours encoding. Here: per store class k, look at the store's *confident* claims
(store says k with relative margin >= 0.2). After 3 wrong of class k's last 50 confident claims, veto the
store's answers for class k until 20 consecutive confident claims of k agree with the label. State updates
after the answer (causal). Applied on top of mq10_s1 / mq10_s2 from gate_probe.py, on gate_probe's logged
arrivals (seeds 0-2 and held-out 3-5, bg 0.02). Only the veto half of the hypothesis is testable offline;
the prototype-lr half needs a store re-run and is tested only if the veto passes.
Pre-registered: pass = swap first 300 >= 76 with returns >= 64 and lifetime >= 96.0; kill = swap < 74.


## results/gate_probe seeds [0, 1, 2], mean ± std %

| Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers % |
|---|---|---|---|---|---|---|
| mq10_s1 | 95.6 ± 0.1 | 71.8 ± 2.7 | 61.2 ± 1.5 | 64.0 ± 4.7 | 76.0 ± 2.8 | 1.8 ± 0.1 |
| mq10_s1_veto | 96.1 ± 0.2 | 57.0 ± 0.4 | 47.9 ± 2.1 | 76.7 ± 4.1 | 76.0 ± 2.8 | 0.7 ± 0.0 |
| mq10_s2 | 96.0 ± 0.2 | 68.5 ± 3.3 | 57.5 ± 2.0 | 67.0 ± 5.2 | 75.2 ± 3.0 | 0.8 ± 0.1 |
| mq10_s2_veto | 96.1 ± 0.2 | 56.5 ± 0.9 | 46.8 ± 1.9 | 76.8 ± 4.0 | 75.2 ± 3.0 | 0.3 ± 0.0 |
| ff | 95.9 ± 0.2 | 53.5 ± 1.2 | 42.8 ± 2.1 | 76.8 ± 4.0 | 74.2 ± 3.0 | 0.0 ± 0.0 |

Veto active on 15.5% of arrivals; 6.4% before the swap; 81.9% of post-swap arrivals where the store says 2 or 3.

## results/gate_probe_s345 seeds [3, 4, 5], mean ± std %

| Route | Lifetime | Returns | Off season | Swap first 300 | Novel first 300 | Store answers % |
|---|---|---|---|---|---|---|
| mq10_s1 | 95.7 ± 0.1 | 67.9 ± 3.1 | 61.2 ± 1.8 | 67.1 ± 2.5 | 78.7 ± 4.1 | 1.7 ± 0.1 |
| mq10_s1_veto | 96.2 ± 0.0 | 56.5 ± 2.0 | 48.7 ± 1.8 | 77.8 ± 1.3 | 78.7 ± 4.1 | 0.7 ± 0.0 |
| mq10_s2 | 96.1 ± 0.1 | 65.1 ± 2.8 | 57.9 ± 1.7 | 70.3 ± 1.9 | 77.6 ± 4.2 | 0.8 ± 0.0 |
| mq10_s2_veto | 96.2 ± 0.0 | 56.1 ± 2.0 | 47.7 ± 1.9 | 78.4 ± 1.0 | 77.6 ± 4.2 | 0.3 ± 0.0 |
| ff | 96.0 ± 0.0 | 53.6 ± 2.0 | 43.9 ± 1.0 | 78.4 ± 1.0 | 76.3 ± 3.2 | 0.0 ± 0.0 |

Veto active on 15.8% of arrivals; 5.8% before the swap; 80.4% of post-swap arrivals where the store says 2 or 3.

## Sensitivity sweep of the wrong-claim count (`veto_sweep.py`, mq10_s2 + veto, window 50, clear 20)

Pre-swap confident-claim error rate of the store is only 1.4% (0-2) / 1.3% (3-5), so a 3/50 trigger is not
firing on base-rate noise alone. Columns: lifetime, returns, swap first 300, base error (all %).

| bad (of 50) | seeds 0-2 | held-out 3-5 |
|---|---|---|
| 3 (pre-reg) | 96.1 / 56.5 / 76.8 | 96.2 / 56.1 / 78.4 |
| 5 | 96.1 / 57.8 / 76.8 | 96.2 / 57.6 / 78.3 |
| 8 | 96.1 / 60.6 / 76.6 | 96.2 / 58.9 / 78.3 |
| 10 | 96.1 / 62.1 / 76.4 | 96.2 / 59.3 / 78.1 |
| 15 | 96.1 / 65.3 / 75.4 | 96.2 / 62.0 / 77.8 |
| 20 | 96.1 / 66.0 / 74.0 | 96.2 / 62.4 / 76.6 |

## Verdict

Hypothesis fails as pre-registered: swap recovers fully (76.8 / 78.4, equal to FF) but returns fall from
68.5 / 65.1 to 56.5 / 56.1, below the 64 floor. No trigger count passes both on held-out seeds: the sweep is a
monotone trade-off (every point of swap bought back costs returns). A per-claimed-class mismatch counter
cannot separate "this class's label changed" from "the store's confident claims for this class are where the
returns gain comes from". The veto half of the ACh hypothesis is closed; the prototype-lr half is not tested.
