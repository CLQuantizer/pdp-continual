I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a Kaggle GPU experiment on content-keyed replay. I'll read the cited docs and the H31 code before writing anything.H31 is the baseline. Next I’ll read the sleep script and the Kaggle kernel layout so the cued-replay copy stays equivalent.E3M matches the registered hash, and 17.19 GPU hours remain, which is above the 11.5-hour floor. I'll add content-keyed replay on top of H31's interleaved path, keeping the uniform arm on the same sampler.The 512-byte uniform arm matches H31 exactly. Next I’ll run the required CPU smokes and add the Kaggle kernel files.H26 is in `results/h26/RESULTS.md` and `results/h26/results.json`. All three arms finished on a Tesla T4, 23,438 replay steps each. Total GPU time is 0.9694 hours. No kernel was rerun. No verdict is in the files.

| Arm | Bits/byte | Gap vs uniform |
|---|---:|---:|
| uniform | 1.958153 | |
| cued8 | 2.020962 | +0.062809 |
| mix44 | 1.973144 | +0.014991 |

The uniform slices match the H31 interleaved run. The bars were left unscored.

Content-keyed windows are older and cover less of the log than uniform draws at the same steps. Mean age is 6086.6 chunks for cued8 and 6035.9 for mix44, against 5875.1 realized and 5863.7 expected under uniform sampling. The share of cued windows in the most recent 10% of the eligible log is 0.0967 (cued8) and 0.0958 (mix44), against 0.1001 for uniform. Coverage of stored chunks is 0.835 (cued8) and 0.878 (mix44), against 0.886 for uniform.

CPU smoke: `h26_cued.py --arm uniform` and `h31_sleep.py --sleep-block 1` on 8,192 bytes both scored 42063.298164869797 bits. On 65,536 bytes, the five logged cued8 retrievals all had chunk index ≤ cue chunk − 16. All three arms were finite on 8,192 bytes.

Kernels are `ezzzzzx/h26-uniform`, `ezzzzzx/h26-cued8`, and `ezzzzzx/h26-mix44`, each version 1. Code dataset is `ezzzzzx/pdp-code-h26`. Nothing was committed.
