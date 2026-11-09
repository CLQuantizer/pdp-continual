#!/usr/bin/env python3
"""Sync the papers library to R2 bucket pdp-papers and D1 database pdp-papers-index.

One D1 row per paper (table papers). A local file in docs/papers/ is uploaded
only when its sha256 differs from the indexed object. --fetch downloads rows
marked to_fetch (arXiv, or an open-access PDF already recorded for the row).
--wanted prints the paywalled list. --dry writes nothing.

Usage (from the repo root):
  python3 tools/sync_papers.py
  python3 tools/sync_papers.py --dry
  python3 tools/sync_papers.py --fetch
  python3 tools/sync_papers.py --wanted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.request
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
PRIMARY_RANK = {".pdf": 0, ".html": 1, ".htm": 1, ".epub": 2, ".docx": 3, ".ps": 4, ".djvu": 5, ".txt": 6, ".md": 7}

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "docs" / "papers"
README = PAPERS / "README.md"
UA = "pdp-papers-registry/1.0 (mailto:agent@learning.mywords.io)"
SHADOW = (
    "sci-hub",
    "scihub",
    "libgen",
    "library.lol",
    "annas-archive",
    "z-lib",
    "zlib",
    "booksc",
    "nexus",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers(
  id INTEGER PRIMARY KEY,
  cite_key TEXT UNIQUE NOT NULL,
  title TEXT,
  authors TEXT,
  year INTEGER,
  venue TEXT,
  arxiv_id TEXT,
  doi TEXT,
  url TEXT,
  status TEXT CHECK(status IN ('read','checked','memory')),
  file_status TEXT NOT NULL CHECK(file_status IN ('stored','to_fetch','wanted','unavailable')),
  r2_key TEXT UNIQUE,
  filename TEXT,
  content_type TEXT,
  bytes INTEGER,
  sha256 TEXT,
  cited_in TEXT,
  notes TEXT,
  added_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_file_status ON papers(file_status);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
"""

UPSERT_SQL = """
INSERT INTO papers (
  cite_key, title, authors, year, venue, arxiv_id, doi, url, status, file_status,
  r2_key, filename, content_type, bytes, sha256, cited_in, notes, added_at, updated_at
) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)
ON CONFLICT(cite_key) DO UPDATE SET
  title = COALESCE(excluded.title, papers.title),
  authors = COALESCE(excluded.authors, papers.authors),
  year = COALESCE(excluded.year, papers.year),
  venue = COALESCE(excluded.venue, papers.venue),
  arxiv_id = COALESCE(excluded.arxiv_id, papers.arxiv_id),
  doi = COALESCE(excluded.doi, papers.doi),
  url = COALESCE(excluded.url, papers.url),
  status = COALESCE(excluded.status, papers.status),
  file_status = excluded.file_status,
  r2_key = COALESCE(excluded.r2_key, papers.r2_key),
  filename = COALESCE(excluded.filename, papers.filename),
  content_type = COALESCE(excluded.content_type, papers.content_type),
  bytes = COALESCE(excluded.bytes, papers.bytes),
  sha256 = COALESCE(excluded.sha256, papers.sha256),
  cited_in = COALESCE(papers.cited_in, excluded.cited_in),
  notes = COALESCE(excluded.notes, papers.notes),
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
_OA_PDF_RE = re.compile(r"oa_pdf=(\S+)")
_COMPANION_RE = re.compile(r"companion (\S+) sha256=([0-9a-f]{64}) bytes=(\d+)")
_VENUE_REJECT = re.compile(r"^(see|and|or|the|a|an)$", re.I)


class CfError(RuntimeError):
    pass


def redact(text: str) -> str:
    text = re.sub(r"/accounts/[0-9a-fA-F]{32}", "/accounts/<redacted>", text)
    text = _UUID_RE.sub("<redacted>", text)
    return text


def fold_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z]", "", text).lower()


def cite_key_from(authors: str | None, year: int | None) -> str | None:
    if not authors or year is None:
        return None
    first = re.split(r",|&| and ", authors, maxsplit=1)[0].strip()
    surname = first.split()[-1] if first else ""
    key = fold_key(surname)
    if not key:
        return None
    return f"{key}{year}"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cf(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    return subprocess.run(["cf", *args, "-q"], capture_output=True, env=env)


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
    err = proc.stderr.decode("utf-8", errors="replace") + proc.stdout.decode("utf-8", errors="replace")
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
    # D1 runs one statement per call. Split on semicolons that end a statement.
    for statement in SCHEMA_SQL.split(";"):
        statement = statement.strip()
        if statement:
            d1_query(database_id, statement)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            info["cite_key"] = cite_key_from(info.get("authors"), info.get("year"))
            for name in filenames_from_cell(row.get("file", "")):
                meta[name] = dict(info)
            i += 1
    return meta


def load_rows(database_id: str) -> list[dict]:
    data = d1_query(
        database_id,
        "SELECT cite_key, filename, r2_key, sha256, notes, file_status, arxiv_id, doi, url, "
        "title, authors, year, venue, status, content_type, bytes, cited_in, added_at "
        "FROM papers",
    )
    return data[0].get("results") or []


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


def upload_bytes(key: str, data: bytes, content_type: str, path: Path) -> None:
    path.write_bytes(data)
    upload_object(key, path, content_type)


def upsert_row(database_id: str, params: list) -> None:
    d1_query(database_id, UPSERT_SQL, params)


def row_params(row: dict, stamp: str) -> list:
    return [
        row.get("cite_key"),
        row.get("title"),
        row.get("authors"),
        row.get("year"),
        row.get("venue"),
        row.get("arxiv_id"),
        row.get("doi"),
        row.get("url"),
        row.get("status"),
        row.get("file_status"),
        row.get("r2_key"),
        row.get("filename"),
        row.get("content_type"),
        row.get("bytes"),
        row.get("sha256"),
        row.get("cited_in"),
        row.get("notes"),
        row.get("added_at") or stamp,
        stamp,
    ]


def legal_url(url: str) -> bool:
    host = urllib.parse_host(url)
    return not any(bad in host for bad in SHADOW)


def urllib_parse_host(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower()


# keep a local alias used above
urllib.parse_host = urllib_parse_host  # type: ignore[attr-defined]


def download_pdf(url: str) -> bytes:
    if not legal_url(url):
        raise CfError("refusing a non-open-access host")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise CfError("download was not a PDF")
    if len(data) < 1000:
        raise CfError("download was too small to be a paper")
    return data


def arxiv_pdf_url(arxiv_id: str) -> str:
    base = re.sub(r"v\d+$", "", arxiv_id or "")
    return f"https://arxiv.org/pdf/{base}.pdf"


def unpaywall_pdfs(doi: str) -> list[str]:
    url = f"https://api.unpaywall.org/v2/{doi}?email=agent@learning.mywords.io"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    found: list[str] = []
    locations = []
    if payload.get("best_oa_location"):
        locations.append(payload["best_oa_location"])
    locations.extend(payload.get("oa_locations") or [])
    for loc in locations:
        pdf = loc.get("url_for_pdf") or ""
        if pdf and legal_url(pdf) and pdf not in found:
            found.append(pdf)
    return found


def sync_local(database_id: str | None, dry: bool) -> int:
    if not README.is_file():
        print(f"missing {README}", file=sys.stderr)
        return 1
    catalogue = metadata_by_filename(README.read_text(encoding="utf-8"))
    files = iter_documents()
    rows = load_rows(database_id) if database_id else []
    by_name = {row["filename"]: row for row in rows if row.get("filename")}
    by_key = {row["cite_key"]: row for row in rows if row.get("cite_key")}
    groups: dict[str, list[Path]] = {}
    loose: list[Path] = []
    for path in files:
        info = catalogue.get(path.name) or {}
        key = info.get("cite_key")
        if key:
            groups.setdefault(key, []).append(path)
        else:
            loose.append(path)

    stamp = now_stamp()
    for key, paths in groups.items():
        paths.sort(key=lambda p: (PRIMARY_RANK.get(p.suffix.lower(), 9), p.name))
        primary = paths[0]
        companions = paths[1:]
        info = catalogue.get(primary.name) or {}
        existing = by_name.get(primary.name) or by_key.get(key)
        digest = sha256_file(primary)
        changed = not existing or existing.get("sha256") != digest
        if dry:
            print(("would upload " if changed else "unchanged ") + primary.name)
        else:
            assert database_id is not None
            content_type = EXTENSIONS[primary.suffix.lower()]
            r2_key = (existing or {}).get("r2_key") or f"papers/{primary.name}"
            if changed:
                upload_object(r2_key, primary, content_type)
            notes = (existing or {}).get("notes")
            row = {
                "cite_key": key,
                "title": info.get("title") or (existing or {}).get("title"),
                "authors": info.get("authors") or (existing or {}).get("authors"),
                "year": info.get("year") if info.get("year") is not None else (existing or {}).get("year"),
                "venue": info.get("venue") or (existing or {}).get("venue"),
                "arxiv_id": info.get("arxiv_id") or (existing or {}).get("arxiv_id"),
                "doi": info.get("doi") or (existing or {}).get("doi"),
                "url": (existing or {}).get("url"),
                "status": info.get("status") or (existing or {}).get("status"),
                "file_status": "stored",
                "r2_key": r2_key,
                "filename": primary.name,
                "content_type": content_type,
                "bytes": primary.stat().st_size,
                "sha256": digest,
                "cited_in": (existing or {}).get("cited_in") or "docs/papers/README.md",
                "notes": notes,
                "added_at": (existing or {}).get("added_at") or stamp,
            }
            upsert_row(database_id, row_params(row, stamp))
            print(("uploaded " if changed else "unchanged ") + primary.name)
            existing = row
        for extra in companions:
            extra_digest = sha256_file(extra)
            note_src = (existing or {}).get("notes") or ""
            match = _COMPANION_RE.search(note_src)
            same = match and match.group(2) == extra_digest
            label = extra.name
            if dry:
                print(("would upload companion " if not same else "unchanged companion ") + label)
                continue
            assert database_id is not None
            if not same:
                companion_key = f"papers/{extra.name}"
                upload_object(companion_key, extra, EXTENSIONS[extra.suffix.lower()])
                note = f"companion {companion_key} sha256={extra_digest} bytes={extra.stat().st_size}"
                # keep any non-companion note text
                rest = _COMPANION_RE.sub("", note_src).strip()
                notes = (note + (" " + rest if rest else "")).strip()
                existing["notes"] = notes
                upsert_row(database_id, row_params(existing, stamp))
                print("uploaded companion " + label)
            else:
                print("unchanged companion " + label)

    for path in loose:
        digest = sha256_file(path)
        existing = by_name.get(path.name)
        changed = not existing or existing.get("sha256") != digest
        if dry:
            print(("would upload " if changed else "unchanged ") + path.name)
            continue
        assert database_id is not None
        key = (existing or {}).get("cite_key") or fold_key(path.stem) or path.stem.lower()
        content_type = EXTENSIONS[path.suffix.lower()]
        r2_key = (existing or {}).get("r2_key") or f"papers/{path.name}"
        if changed:
            upload_object(r2_key, path, content_type)
        row = existing or {
            "cite_key": key,
            "title": None,
            "authors": None,
            "year": None,
            "venue": None,
            "arxiv_id": None,
            "doi": None,
            "url": None,
            "status": None,
            "cited_in": "docs/papers/README.md",
            "notes": None,
            "added_at": stamp,
        }
        row.update(
            {
                "file_status": "stored",
                "r2_key": r2_key,
                "filename": path.name,
                "content_type": content_type,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        upsert_row(database_id, row_params(row, stamp))
        print(("uploaded " if changed else "unchanged ") + path.name)
    return 0


def fetch_row(row: dict) -> bytes:
    if row.get("arxiv_id"):
        return download_pdf(arxiv_pdf_url(row["arxiv_id"]))
    notes = row.get("notes") or ""
    oa = _OA_PDF_RE.search(notes)
    if oa:
        return download_pdf(oa.group(1))
    if row.get("doi"):
        for url in unpaywall_pdfs(row["doi"]):
            try:
                return download_pdf(url)
            except Exception:
                continue
    url = row.get("url") or ""
    if url.lower().endswith(".pdf"):
        return download_pdf(url)
    raise CfError("no open-access PDF for " + str(row.get("cite_key")))


def sync_fetch(database_id: str, dry: bool) -> int:
    rows = [
        row
        for row in load_rows(database_id)
        if row.get("file_status") == "to_fetch"
    ]
    if not rows:
        print("no to_fetch rows")
        return 0
    stamp = now_stamp()
    tmp = Path("/tmp/pdp-paper-fetch.pdf")
    for row in rows:
        key = row["cite_key"]
        if dry:
            print("would fetch " + key)
            continue
        try:
            data = fetch_row(row)
        except Exception as exc:
            print(f"fetch failed {key}: {redact(str(exc))[:200]}", file=sys.stderr)
            continue
        digest = sha256_bytes(data)
        filename = f"{key}.pdf"
        r2_key = f"papers/{filename}"
        upload_bytes(r2_key, data, "application/pdf", tmp)
        row.update(
            {
                "file_status": "stored",
                "r2_key": r2_key,
                "filename": filename,
                "content_type": "application/pdf",
                "bytes": len(data),
                "sha256": digest,
            }
        )
        upsert_row(database_id, row_params(row, stamp))
        print("fetched " + key)
    return 0


def print_wanted(database_id: str) -> int:
    data = d1_query(
        database_id,
        "SELECT cite_key, title, authors, year, url FROM papers "
        "WHERE file_status = 'wanted' ORDER BY year, cite_key",
    )
    for row in data[0].get("results") or []:
        year = "" if row.get("year") is None else str(row["year"])
        print(
            "\t".join(
                [
                    row.get("cite_key") or "",
                    (row.get("title") or "").replace("\t", " "),
                    (row.get("authors") or "").replace("\t", " "),
                    year,
                    row.get("url") or "",
                ]
            )
        )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sync the papers library to R2 and D1.")
    parser.add_argument("--dry", action="store_true", help="list actions without writing")
    parser.add_argument("--fetch", action="store_true", help="download to_fetch rows and store them")
    parser.add_argument("--wanted", action="store_true", help="print wanted rows for Ez")
    args = parser.parse_args(argv)
    try:
        if args.dry:
            database_id = lookup_database()
        else:
            ensure_bucket()
            database_id = ensure_database()
            ensure_schema(database_id)
        if database_id is None and (args.fetch or args.wanted):
            print("database not found", file=sys.stderr)
            return 1
        code = 0
        if args.wanted:
            assert database_id is not None
            code = print_wanted(database_id)
        elif args.fetch:
            assert database_id is not None
            code = sync_fetch(database_id, args.dry)
        else:
            code = sync_local(database_id, args.dry)
        return code
    except CfError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
