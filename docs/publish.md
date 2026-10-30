# Publishing: learning.mywords.io

The public research log at <https://learning.mywords.io> is generated from this repo:
`docs/loop.md` (newest first), `docs/goal.md`, `docs/notes.md`, `audit.md`,
`docs/cls_route.md`, `README.md`, and every `results/*/RESULTS.md`.

## Hook for the hourly loop

After each material loop commit (on the server, from the repo root):

```sh
tools/publish.sh "$(git rev-parse HEAD)"   # or just tools/publish.sh
```

It rebuilds the static site in ~10 s and swaps it in atomically. No restart, no deploy step.
If it fails, the previous site stays up; the loop should log the error and carry on.

## Public-content rules (enforced by `tools/publish_site.py` scrub, but write clean anyway)

- Byline is **Ez** only.
- Times are **UTC+8**; never write HKT or a city.
- No hostnames, IPs, paths under home directories, emails, or tokens in docs.
- `results/` data stays private (gitignored); only `RESULTS.md` write-ups are published.
