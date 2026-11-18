"""Build the H35c pool and evaluation sets. Streaming, one document at a time.

Pool: first documents of C4 en.noclean (Hugging Face allenai/c4), in shard and
line order. Drop utf-8 under 200 bytes, truncate to 4,096 bytes on a character
boundary. Record the shard file and the 0-based line id inside that shard.

Clean eval, never mixed into the pool:
- C4 en validation, documents in order, until the stored bytes reach 2_000_000.
- WikiText-103 raw test (Salesforce/wikitext, wikitext-103-raw-v1).

Reference text: the first 5_000_000 bytes of local enwik8 (English Wikipedia).
"""
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path('/root/pdp')
OUT = ROOT / 'data' / 'h35c'
ENWIK8 = ROOT / 'data' / 'enwik8' / 'enwik8'
POOL_N = 60_000
MIN_BYTES = 200
MAX_BYTES = 4_096
EVAL_BYTES = 2_000_000
REF_BYTES = 5_000_000
UA = 'pdp-h35c-research'
C4 = 'https://huggingface.co/datasets/allenai/c4/resolve/main/'
WIKI = (
    'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/'
    'wikitext-103-raw-v1/test-00000-of-00001.parquet'
)


def hf_stream(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    resp = urllib.request.urlopen(req, timeout=120)
    return gzip.GzipFile(fileobj=resp), resp


def truncate(text):
    raw = text.encode('utf-8')
    if len(raw) < MIN_BYTES:
        return None
    if len(raw) > MAX_BYTES:
        raw = raw[:MAX_BYTES]
        text = raw.decode('utf-8', errors='ignore')
        raw = text.encode('utf-8')
        if len(raw) < MIN_BYTES:
            return None
    return raw


def build_pool():
    OUT.mkdir(parents=True, exist_ok=True)
    bin_path = OUT / 'pool.bin'
    meta_path = OUT / 'pool_meta.jsonl'
    offsets = [0]
    kept = 0
    seen = 0
    dropped_short = 0
    shards = []
    sha = hashlib.sha256()
    shard_i = 0
    with bin_path.open('wb') as bout, meta_path.open('w') as mout:
        while kept < POOL_N:
            name = f'en.noclean/c4-train.{shard_i:05d}-of-07168.json.gz'
            url = C4 + name
            print(f'shard {name}', flush=True)
            try:
                gz, resp = hf_stream(url)
            except Exception as exc:
                raise SystemExit(f'cannot stream {name}: {exc}') from exc
            shards.append(name)
            line_i = 0
            try:
                for line in gz:
                    seen += 1
                    line_i += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw = truncate(obj.get('text') or '')
                    if raw is None:
                        dropped_short += 1
                    else:
                        bout.write(raw)
                        sha.update(raw)
                        offsets.append(offsets[-1] + len(raw))
                        rec = {
                            'pool_id': kept,
                            'shard': name,
                            'line': line_i - 1,
                            'url': (obj.get('url') or '')[:500],
                            'n_bytes': len(raw),
                        }
                        mout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        kept += 1
                        if kept >= POOL_N:
                            break
                    if seen % 20000 == 0:
                        print(f'  seen={seen} kept={kept} short={dropped_short}', flush=True)
            finally:
                gz.close()
                resp.close()
            shard_i += 1
            if shard_i > 8 and kept < POOL_N:
                raise SystemExit(f'only kept {kept} documents in {shard_i} shards')
    offsets = np.asarray(offsets, dtype=np.int64)
    np.save(OUT / 'pool_offsets.npy', offsets)
    manifest = {
        'source': 'allenai/c4 en.noclean train, documents in shard/line order',
        'shards': shards,
        'n_docs': kept,
        'n_seen': seen,
        'n_dropped_short': dropped_short,
        'n_bytes': int(offsets[-1]),
        'min_bytes': MIN_BYTES,
        'max_bytes': MAX_BYTES,
        'sha256': sha.hexdigest(),
    }
    (OUT / 'pool_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    return manifest


def build_c4_val():
    name = 'en/c4-validation.00000-of-00008.json.gz'
    gz, resp = hf_stream(C4 + name)
    docs = []
    total = 0
    line_i = 0
    try:
        for line in gz:
            line_i += 1
            obj = json.loads(line)
            raw = (obj.get('text') or '').encode('utf-8')
            if len(raw) < MIN_BYTES:
                continue
            if total >= EVAL_BYTES and docs:
                break
            if len(raw) > EVAL_BYTES and not docs:
                raw = raw[:EVAL_BYTES]
                raw = raw.decode('utf-8', errors='ignore').encode('utf-8')
            docs.append(raw)
            total += len(raw)
            if total >= EVAL_BYTES:
                break
    finally:
        gz.close()
        resp.close()
    path = OUT / 'c4_val.bin'
    offsets = [0]
    sha = hashlib.sha256()
    with path.open('wb') as f:
        for raw in docs:
            f.write(raw)
            sha.update(raw)
            offsets.append(offsets[-1] + len(raw))
    np.save(OUT / 'c4_val_offsets.npy', np.asarray(offsets, dtype=np.int64))
    meta = {
        'source': 'allenai/c4 en validation ' + name,
        'n_docs': len(docs),
        'n_bytes': total,
        'target_bytes': EVAL_BYTES,
        'lines_scanned': line_i,
        'sha256': sha.hexdigest(),
    }
    (OUT / 'c4_val_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


def build_wiki():
    import pyarrow.parquet as pq
    dest = OUT / 'wikitext103_test.parquet'
    if not dest.exists() or dest.stat().st_size < 1000:
        req = urllib.request.Request(WIKI, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open('wb') as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    table = pq.read_table(dest, columns=['text'])
    lines = table.column('text').to_pylist()
    # The parquet stores the raw file one line per row.
    raw = ('\n'.join(lines) + '\n').encode('utf-8')
    (OUT / 'wikitext103_test.bin').write_bytes(raw)
    meta = {
        'source': 'Salesforce/wikitext wikitext-103-raw-v1 test',
        'n_lines': len(lines),
        'n_bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }
    (OUT / 'wikitext_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


def build_ref():
    blob = ENWIK8.read_bytes()
    digest = hashlib.sha256(blob[:10_000_000]).hexdigest()
    expect = '5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97'
    if digest != expect:
        raise SystemExit(f'enwik8 sha mismatch {digest}')
    raw = blob[:REF_BYTES]
    (OUT / 'wiki_ref.bin').write_bytes(raw)
    meta = {
        'source': 'enwik8 first 5_000_000 bytes (English Wikipedia)',
        'n_bytes': len(raw),
        'enwik8_e10m_sha256': digest,
        'sha256': hashlib.sha256(raw).hexdigest(),
    }
    (OUT / 'wiki_ref_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


def main():
    which = sys.argv[1:] or ['all']
    if 'all' in which or 'pool' in which:
        build_pool()
    if 'all' in which or 'eval' in which:
        build_c4_val()
        build_wiki()
        build_ref()


if __name__ == '__main__':
    main()
