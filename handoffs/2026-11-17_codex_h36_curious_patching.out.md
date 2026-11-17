Implemented code, 24 kernel folders, dispatcher, and summarizer. **Smoke stopped at the entropy budget failure**, as requested.

| Rule | Achieved ρ | Junk share | Estimated 400-doc wall |
|---|---:|---:|---:|
| fixed | 0.250000 | 0.500000 | 43.7 min |
| entropy | **0.339119** | 0.941843 | 53.5 min |

Allowed ρ: `[0.23, 0.27]`. Threshold ties use seeded 25% sampling.

All four prefix checks passed exactly. Dense GRU bits fell from `5.257840` to `3.755196`.

Gain/progress allocation checks remain unrun. H35 was missing when kernels were prepared; it now exists, but the code snapshot remains unrefreshed. No network calls, kernel launches, or commits.

[Partial results and smoke output](/root/pdp/results/h36/RESULTS.md)