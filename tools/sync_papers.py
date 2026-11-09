#!/usr/bin/env python3
"""Sync docs/papers to R2 bucket pdp-papers and D1 database pdp-papers-index.

Uploads a file only when its sha256 differs from the indexed object. Upserts
one documents row per file. Safe to re-run. --dry prints actions and writes
nothing.

Usage (from the repo root):
  python3 tools/sync_papers.py
  python3 tools/sync_papers.py --dry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "pdp-papers"
DB_NAME = "pdp-papers-index"
EXTENSIONS = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".epub": "application/epub+zip",
    ".ps": "application/postscript",
    ".djvu": "image/vnd.djvu",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "docs" / "papers"
README = PAPERS / "README.md"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY,
  r2_key TEXT UNIQUE NOT NULL,
  filename TEXT NOT NULL,
  content_type TEXT,
  bytes INTEGER,
  sha256 TEXT NOT NULL,
  title TEXT,
  authors TEXT,
  year INTEGER,
  venue TEXT,
  arxiv_id TEXT,
  doi TEXT,
  status TEXT,
  local_path TEXT,
  added_at TEXT,
  updated_at TEXT,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
"""

UPSERT_SQL = """
INSERT INTO documents (
  r2_key, filename, content_type, bytes, sha256, title, authors, year, venue,
  arxiv_id, doi, status, local_path, added_at, updated_at, notes
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
ON CONFLICT(r2_key) DO UPDATE SET
  filename = excluded.filename,
  content_type = excluded.content_type,
  bytes = excluded.bytes,
  sha256 = excluded.sha256,
  title = excluded.title,
  authors = excluded.authors,
  year = excluded.year,
  venue = excluded.venue,
  arxiv_id = excluded.arxiv_id,
  doi = excluded.doi,
  status = excluded.status,
  local_path = excluded.local_path,
  updated_at = excluded.updated_at
"""

_ARXIV_RE = re.compile(
    r"arXiv\s*:?\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-zA-Z.\-]+/\d{7})",
    re.I,
)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s)>\]]+)", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*/\s*\d{2,4}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_VENUE_REJECT = re.compile(r"^(see|and|or|the|a|an)$", re.I)


class CfError(RuntimeError):
    pass


def redact(text: str) -> str:
    text = re.sub(r"/accounts/[0-9a-fA-F]{32}", "/accounts/<redacted>", text)
    text = _UUID_RE.sub("<redacted>", text)
    return text


def run_cf(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    return subprocess.run(
        ["cf", *args, "-q"],
        capture_output=True,
        env=env,
    )


def parse_json_out(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        raise CfError(redact(text.strip() or "empty response from cf")[:800])
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise CfError(redact(f"cf returned non-JSON: {text[:800]}")) from exc


def cf_ok(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    proc = run_cf(args)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        out = proc.stdout.decode("utf-8", errors="replace")
        raise CfError(redact((err or out).strip())[:800])
    return proc


def d1_query(database_id: str, sql: str, params: list | None = None):
    body: dict = {"sql": sql}
    if params is not None:
        body["params"] = params
    proc = cf_ok(["d1", "query", database_id, "--body", json.dumps(body)])
    data = parse_json_out(proc.stdout)
    if isinstance(data, dict):
        data = [data]
    for stmt in data:
        if not stmt.get("success", False):
            raise CfError(redact(json.dumps(stmt)[:800]))
    return data


def ensure_bucket() -> None:
    proc = run_cf(["r2", "buckets", "get", BUCKET])
    if proc.returncode == 0:
        return
    err = (
        proc.stderr.decode("utf-8", errors="replace")
        + proc.stdout.decode("utf-8", errors="replace")
    )
    if "does not exist" not in err and "10006" not in err:
        raise CfError(redact(err.strip())[:800])
    cf_ok(["r2", "buckets", "create-by-name", BUCKET])


def lookup_database() -> str | None:
    proc = cf_ok(["d1", "list", "--name", DB_NAME])
    data = parse_json_out(proc.stdout)
    if not isinstance(data, list):
        raise CfError("unexpected database list payload")
    for row in data:
        if row.get("name") == DB_NAME and row.get("uuid"):
            return row["uuid"]
    return None


def ensure_database() -> str:
    found = lookup_database()
    if found:
        return found
    proc = cf_ok(["d1", "create", "--name", DB_NAME])
    created = parse_json_out(proc.stdout)
    if created.get("name") != DB_NAME or not created.get("uuid"):
        raise CfError("database create returned an unexpected payload")
    return created["uuid"]


def ensure_schema(database_id: str) -> None:
    d1_query(database_id, SCHEMA_SQL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_documents() -> list[Path]:
    found = []
    for path in PAPERS.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("._") or name == "README.md":
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        found.append(path)
    return sorted(found, key=lambda p: p.name)


def filenames_from_cell(cell: str) -> list[str]:
    names: list[str] = []
    stem: str | None = None
    for tick in re.findall(r"`([^`]+)`", cell):
        token = tick.strip()
        if "/" in token or "\\" in token:
            continue
        ext = Path(token).suffix.lower()
        if ext in EXTENSIONS:
            names.append(Path(token).name)
            stem = Path(token).stem
            continue
        if stem and re.fullmatch(r"\.[A-Za-z0-9]+", token) and token.lower() in EXTENSIONS:
            names.append(stem + token.lower())
    return names


def normalize_status(cell: str) -> str | None:
    text = re.sub(r"[*`]", "", cell).strip().lower()
    if re.match(r"memory\b", text):
        return "memory"
    if re.match(r"checked\b", text):
        return "checked"
    if re.match(r"read\b", text):
        return "read"
    return None


def parse_authors(before: str) -> str | None:
    text = re.sub(r"\([^)]*\)", " ", before)
    text = re.sub(r"\s+", " ", text).strip(" .;,\t")
    if not text:
        return None
    ranged = _YEAR_RANGE_RE.search(text)
    if ranged and ranged.end() == len(text.rstrip()):
        text = text[: ranged.start()].strip(" .;,")
    else:
        year_tail = re.match(r"^(.*?)[,\s]+((?:19|20)\d{2})$", text)
        if year_tail:
            text = year_tail.group(1).strip(" .;,")
    return text or None


def venue_from_after(after: str) -> tuple[str | None, int | None]:
    text = re.sub(r"\*\*[^*]+\*\*", " ", after)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"arXiv\s*:?\s*\S+", " ", text, flags=re.I)
    text = re.sub(r"\bsee\b.*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" .;,\t")
    if not text or _VENUE_REJECT.match(text):
        return None, None
    year = None
    year_tail = re.match(r"^(.*?)(?:\s+((?:19|20)\d{2}))$", text)
    if year_tail:
        text = year_tail.group(1).strip(" .;,")
        year = int(year_tail.group(2))
    if not text or _VENUE_REJECT.match(text):
        return None, year
    return text, year


def parse_paper(cell: str) -> dict:
    """Pull only fields the README states. Year ranges and gaps stay null."""
    nobold = re.sub(r"\*\*([^*]+)\*\*", " ", cell)
    title = None
    before = nobold
    after = ""
    italic = re.search(r"\*([^*]+)\*", nobold)
    if italic:
        title = re.sub(r"\s+", " ", italic.group(1)).strip().rstrip(" .")
        before = nobold[: italic.start()]
        after = nobold[italic.end() :]
    arxiv = _ARXIV_RE.search(cell)
    doi_match = _DOI_RE.search(cell)
    doi = doi_match.group(1).rstrip(".,;") if doi_match else None
    range_years = {int(match.group(1)) for match in _YEAR_RANGE_RE.finditer(nobold)}
    years = [int(match.group(1)) for match in _YEAR_RE.finditer(nobold)]
    clean_years = [year for year in years if year not in range_years]
    venue, venue_year = venue_from_after(after)
    unique_years = set(clean_years)
    if venue_year is not None and venue_year not in range_years:
        year = venue_year
    elif len(unique_years) == 1:
        year = unique_years.pop()
    else:
        year = None
    return {
        "title": title,
        "authors": parse_authors(before),
        "year": year,
        "venue": venue,
        "arxiv_id": arxiv.group(1) if arxiv else None,
        "doi": doi,
    }


def metadata_by_filename(readme_text: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    lines = readme_text.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not header.startswith("|"):
            i += 1
            continue
        cells = [part.strip().strip("`") for part in header.strip().strip("|").split("|")]
        headers = [cell.lower() for cell in cells]
        if "file" not in headers or "status" not in headers:
            i += 1
            continue
        i += 1
        if i < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i].strip()):
            i += 1
        while i < len(lines) and lines[i].strip().startswith("|"):
            row_cells = [part.strip() for part in lines[i].strip().strip("|").split("|")]
            row = dict(zip(headers, row_cells))
            info = parse_paper(row.get("paper", ""))
            info["status"] = normalize_status(row.get("status", ""))
            for name in filenames_from_cell(row.get("file", "")):
                meta[name] = dict(info)
            i += 1
    return meta


def indexed_hashes(database_id: str) -> dict[str, str]:
    data = d1_query(database_id, "SELECT r2_key, sha256 FROM documents")
    rows = data[0].get("results") or []
    return {row["r2_key"]: row["sha256"] for row in rows}


def upload_object(key: str, path: Path, content_type: str) -> None:
    cf_ok(
        [
            "r2",
            "objects",
            "put",
            key,
            "--bucket-name",
            BUCKET,
            "--file",
            str(path),
            "--content-type",
            content_type,
            "--content-length",
            str(path.stat().st_size),
        ]
    )


def upsert_row(database_id: str, params: list) -> None:
    d1_query(database_id, UPSERT_SQL, params)


def sync(dry: bool) -> int:
    if not README.is_file():
        print(f"missing {README}", file=sys.stderr)
        return 1
    catalogue = metadata_by_filename(README.read_text(encoding="utf-8"))
    files = iter_documents()
    database_id = None
    known: dict[str, str] = {}
    if dry:
        database_id = lookup_database()
        if database_id:
            known = indexed_hashes(database_id)
    else:
        ensure_bucket()
        database_id = ensure_database()
        ensure_schema(database_id)
        known = indexed_hashes(database_id)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for path in files:
        name = path.name
        key = f"papers/{name}"
        digest = sha256_file(path)
        changed = known.get(key) != digest
        if dry:
            print(("would upload " if changed else "unchanged ") + name)
            continue
        content_type = EXTENSIONS[path.suffix.lower()]
        if changed:
            upload_object(key, path, content_type)
        info = catalogue.get(name)
        if info is None:
            print(f"no README metadata for {name}; indexed fields left null", file=sys.stderr)
            info = {
                "title": None,
                "authors": None,
                "year": None,
                "venue": None,
                "arxiv_id": None,
                "doi": None,
                "status": None,
            }
        upsert_row(
            database_id,
            [
                key,
                name,
                content_type,
                path.stat().st_size,
                digest,
                info.get("title"),
                info.get("authors"),
                info.get("year"),
                info.get("venue"),
                info.get("arxiv_id"),
                info.get("doi"),
                info.get("status"),
                f"docs/papers/{name}",
                now,
                now,
                None,
            ],
        )
        print(("uploaded " if changed else "unchanged ") + name)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sync docs/papers to R2 and D1.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="list upload / unchanged actions without writing",
    )
    args = parser.parse_args(argv)
    try:
        return sync(args.dry)
    except CfError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
