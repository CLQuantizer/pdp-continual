# Handoff: papers library → Cloudflare R2, indexed in D1 (worker: grok)

Read /root/pdp/CLAUDE.md first. Use the `cf` CLI (already authenticated). Discover commands with
`cf cli search "<task>"` and `<command> --help`. Keep those searches anonymous: no names, IDs, or tokens in the query.

## Hard rules
- Create and touch ONLY these two resources: R2 bucket **`pdp-papers`** and D1 database **`pdp-papers-index`**. Never
  modify, write to or delete any other bucket or database in the account (others exist; leave them alone).
- No deletes at all, not even in the new resources, unless an upload you made in this run is corrupt (then re-upload).
- Do not enable public access, custom domains or CORS on the bucket.
- Never print, log or write tokens / account IDs into files. Do not git commit. Do not edit docs/loop.md or audit.md.

## Task
1. Create the bucket and the database if they do not exist (idempotent).
2. D1 schema (`CREATE TABLE IF NOT EXISTS`):
   `documents(id INTEGER PRIMARY KEY, r2_key TEXT UNIQUE NOT NULL, filename TEXT NOT NULL, content_type TEXT, bytes INTEGER,
   sha256 TEXT NOT NULL, title TEXT, authors TEXT, year INTEGER, venue TEXT, arxiv_id TEXT, doi TEXT, status TEXT,
   local_path TEXT, added_at TEXT, updated_at TEXT, notes TEXT)`, plus an index on sha256 and on year.
3. Documents: every file in `/root/pdp/docs/papers/` with extension pdf, html, htm, docx, epub, ps, djvu, txt or md,
   **excluding** names starting with `._` and `README.md`. R2 key = `papers/<filename>`; set the right content type.
4. Metadata (title, authors, year, venue, arxiv_id, doi, status) from `/root/pdp/docs/papers/README.md`. Status uses that
   file's terms: read / checked / memory. Leave unknown fields NULL; do not guess.
5. Write a repeatable sync script **`/root/pdp/tools/sync_papers.py`** (python3 stdlib; shells out to `cf`) that
   uploads only new or changed files (by sha256), upserts their D1 rows (`INSERT ... ON CONFLICT(r2_key) DO UPDATE`), and
   prints one line per file: uploaded / unchanged. It must be safe to run daily. A `--dry` flag lists actions without
   doing them. Run it once for real to do steps 3–4.
6. Verify, and paste the results into your final message:
   - the bucket's object list (key, size);
   - `SELECT r2_key, bytes, sha256, title, year, status FROM documents`;
   - download one object back to /tmp and check its sha256 matches the local file;
   - a second run of the sync script reports every file as unchanged.

## Output
Write `/root/pdp/handoffs/2026-11-09_grok_r2_papers.out.md` with: resources created (names only), the verification
output, the exact `python3 tools/sync_papers.py` usage, and anything that failed.
