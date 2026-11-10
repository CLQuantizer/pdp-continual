I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a lookup: find small, research-licensed Kaggle streams that fit the compression world, then write a ranked table. I'll read the target-machine section and start the dataset search.Search results are in. I'll pull the truncated sections and inspect the strongest candidates' files and licences.The eight probes are in. DNA bases are the most compressible, the synthetic transaction file is the least, and the licence-clean trio is Bible, chess moves, and SQuAD. I'll write that up as the table.The ranked table is in `handoffs/2026-11-10_grok_kaggle_worlds.table.md`. Samples are under `data/kaggle_samples/` (gitignored). Nothing was committed.

Probe on every file: first 1,000,000 bytes, `xz -9 -T1`, bits/byte = 8 × compressed size / 1e6. The second number is the same probe on the symbol stream (moves, bases, or question text).

| Rank | Ref | What | Bytes | Licence | Needs | xz file | xz stream |
|---|---|---|---|---|---|---|---|
| 1 | `datasnaek/chess` | 20,058 Lichess games, SAN moves | 7,672,655 | CC0-1.0 | hard-learnable | 2.2760 | moves 2.3217, alphabet 29 |
| 2 | `stanfordu/stanford-question-answering-dataset` `dev-v1.1.json` | SQuAD v1.1 dev: 48 articles, 10,570 questions | 4,854,279 | CC-BY-SA-4.0 | episodic | 1.3794 | plain text 2.1877 |
| 3 | `tentotheminus9/religious-and-philosophical-texts` `pg10.txt` | King James Bible, Gutenberg #10 | 4,452,070 | card unknown; text is public domain | easy | 1.8457 | same |
| 4 | `martinsn/high-frequency-crypto-limit-order-book-data` `BTC_5min.csv` | Coinbase BTC book, 3,367 rows, 7–19 Apr 2021 | 6,606,836 | CC0-1.0 | medium numeric | 1.9153 | same, alphabet 35 |
| 5 | `dnkumars/cryptocurrency-transaction-analytics-btc-and-eth` | 50,000 synthetic tx rows | 10,394,272 | CC0-1.0 | highest entropy here | 3.5467 | same |
| 6 | `nageshsingh/dna-sequence-dataset` `human.txt` | 4,380 labeled fragments, median 865 bp | 5,547,716 | unknown | easy, alphabet ACGTN | 0.6963 | bases 0.6953 |
| 7 | `tunguz/200000-jeopardy-questions` | 216,930 clues; 29,573 answers repeat | 34,854,057 | other | episodic, hard language | 2.1871 | Q/A text 2.6844 |
| 8 | `raj5287/abc-notation-of-tunes` | 4,694 ABC tunes | 1,751,713 | unknown | easy music | 1.5425 | same |

SQuAD: on all 48 articles, at least one answer string occurs at least twice in that article. Jeopardy repeats are stronger (China 216, Australia 215, Japan 196). The Jeopardy card licence is `other`, source a reddit dump of the show's clues.

Published cousins, full files, not these slices. Mahoney's Large Text Compression Benchmark (http://mattmahoney.net/dc/text.html, page date 14 Oct 2026), enwik8 = 1e8 bytes: gzip 1.3.5 -9 is 2.9156 bits/byte, xz 5.2.1 preset 9e is 1.9763, ppmd J1 is 1.7111, cmix v21 -t is 1.1699. Hutter enwik8 phda9 total (4 Nov 2017, decompressor included) is 15,284,944 bytes, 1.2228 bits/byte (http://prize.hutter1.net/). Canterbury `bible.txt` gzip -9 is 2.326 bits/byte (Balkenhol); that file is 4,047,392 bytes, and `pg10.txt` is 4,452,070 with a Gutenberg wrapper. enwik9 (1e9 bytes) and text8 (1e8 bytes) were over the 50 MB cap and were not downloaded.

The transaction file is synthetic: ids are 14 characters, every sender and receiver is unique, and BTC rows use `0x` addresses. The real 5-minute BTC book compresses to 1.9153, below chess moves. Tick files small enough to take were not in the search (`SP.csv` is 481 MB; the Bitcoin minute csv is 397 MB).

**Three to combine**

1. `pg10.txt` — easy. Public-domain prose at 1.8457 bits/byte, with the Canterbury Bible as the published cousin.
2. `datasnaek/chess` moves column — hard-learnable. SAN over 20,058 games, 2.3217 bits/byte, alphabet 29. Drop the player-id columns.
3. SQuAD dev — episodic. The answer string comes back inside the article, then the article changes. Plain text 2.1877 bits/byte. CC-BY-SA-4.0.

The noise seat is empty. The only probe above 3 bits/byte is the synthetic transaction table.
