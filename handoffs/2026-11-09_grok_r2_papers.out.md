# Papers library → R2 + D1

## Resources created

- R2 bucket `pdp-papers` (private: no public access, custom domain, or CORS)
- D1 database `pdp-papers-index`
- Table `documents` in that database, with indexes `idx_documents_sha256` and `idx_documents_year`

No other bucket or database was modified. Nothing was deleted.

## Usage

From the repo root:

```
python3 tools/sync_papers.py
python3 tools/sync_papers.py --dry
```

`--dry` lists `would upload` or `unchanged` and writes nothing. A real run uploads only when the local sha256 differs, then upserts the D1 row (`INSERT ... ON CONFLICT(r2_key) DO UPDATE`). README metadata is refreshed even when the file is unchanged. Unknown fields stay NULL.

## Verification

### Bucket object list (key, size)

```
papers/journe2022_hebbian_deep_learning_without_feedback.pdf    27441097
papers/schmidhuber2009_driven_by_compression_progress.html       317444
papers/schmidhuber2009_driven_by_compression_progress.pdf        537077
```

Content types on those objects: `application/pdf`, `text/html`, `application/pdf`.

### `SELECT r2_key, bytes, sha256, title, year, status FROM documents`

```
r2_key                                                                  bytes     sha256                                                            title                                                                                                                                                                      year  status
papers/journe2022_hebbian_deep_learning_without_feedback.pdf            27441097  00432ed0d6edc6c357b9a8ad160b36223088b4f8108ecc3fce397f5a01172f39  Hebbian Deep Learning Without Feedback                                                                                                                                         2023  read
papers/schmidhuber2009_driven_by_compression_progress.html               317444  2e2315b104ddc53b9ea51d99dffe263f69bcfd089cf54725ef07b876c2806584  Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes  2009  read
papers/schmidhuber2009_driven_by_compression_progress.pdf                537077  e971d466392c52f0a5212fcacc022c1cffab16fcf120ff25106bd71ed685557d  Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes  2009  read
```

Three rows. Journé's year is 2023 and venue is ICLR, from the README line "ICLR 2023". Schmidhuber's venue is NULL (the README names arXiv, which is stored in `arxiv_id` as `0812.4360v2`). DOI is NULL on all three. Status is `read`.

### Round-trip sha256

Downloaded `papers/schmidhuber2009_driven_by_compression_progress.pdf` to `/tmp` (537077 bytes). sha256 `e971d466392c52f0a5212fcacc022c1cffab16fcf120ff25106bd71ed685557d`, identical to the local file.

### Second sync

```
unchanged journe2022_hebbian_deep_learning_without_feedback.pdf
unchanged schmidhuber2009_driven_by_compression_progress.html
unchanged schmidhuber2009_driven_by_compression_progress.pdf
```

The first real run printed `uploaded` for the same three files.

## Failures

None.
