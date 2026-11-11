I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is W1c: measure allocation headroom on one shared cortex. I'll read W1b's results, code, and Kaggle setup before copying them into the new experiment.W1b measured separate learners. W1c uses one shared cortex and the same seven streams. I'll add the scheduled reader, smoke-test it on CPU, then run the three private GPU kernels.The full schedules land on the quotas and the three checkpoints. Next is the CPU smoke test.U7 is training on a T4 and the streams match W1b. N0 is in the other GPU slot. I'll queue OR when a slot frees and poll until all three finish.**headroom_shared = -1.891746 bits/byte** at 3,500,000 bytes. The minimum is OR. **M7(U7) − M7(N0) = -1.990203**.

| Arm | M7 | learnable5 |
|---|---:|---:|
| U7 | 2.782672 | 2.218766 |
| N0 | 4.772876 | 2.367554 |
| OR | 4.674418 | 2.303579 |

One shared H29c cortex per arm (lr 3e-4, replay batch 8, seed 0). Bytes read matched the quotas, including 0 on noise for N0 and OR. Stream hashes match W1-0 and W1b. The CPU smoke passed: U7 read 512 bytes from each channel, N0 read 0 noise bytes, bits were finite, and every replay window stayed inside bytes already read.

| Kernel | Status | Elapsed seconds | GPU hours |
|---|---|---:|---:|
| ezzzzzx/w1c-u7 | COMPLETE v1 | 1428.597 | 0.3968 |
| ezzzzzx/w1c-n0 | COMPLETE v1 | 1464.949 | 0.4069 |
| ezzzzzx/w1c-or | COMPLETE v1 | 1447.924 | 0.4022 |

Total GPU hours: **1.2060** on Tesla T4. Numbers are in `results/w1c_shared/RESULTS.md` and `results/w1c_shared/results.json`. No verdict. Nothing committed.
