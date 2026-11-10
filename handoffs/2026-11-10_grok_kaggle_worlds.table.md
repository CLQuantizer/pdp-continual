# Kaggle channel lookup (2026-11-10)

Lookup only. No experiment code. Samples are under `data/kaggle_samples/` (gitignored). Nothing committed.

Probe, same for every file: first 1,000,000 bytes, `xz -9 -T1`, bits/byte = 8 × compressed bytes / 1,000,000. All eight files are longer than 1 MB, so every probe is a 1 MB slice. A second number, where given, is the same probe on a derived symbol stream (moves only, bases only, or question text only). That second number is extra; the file number is the spec probe.

Searches were `kaggle datasets list -s "<term>" --sort-by votes`. Terms included enwik8, text8, Hutter prize, Wikipedia, chess PGN, Lichess, MIDI, ABC, source code, DNA, protein, Jeopardy, trivia, weather, crypto prices, bitcoin minute, forex, sensor, event log, game replay, Gutenberg, codeforces, and leetcode. Files and licences come from `kaggle datasets files` and `kaggle datasets metadata`.

## Ranked by fit

Fit means: a real stream, a research-usable licence, a 1–20 MB slice, and a job in the world (easy, hard-learnable, episodic, or noise). Needs are the four in the handoff: (1) learnable structure, (2) recurring facts, (3) near-irreducible, (4) a published compression reference.

| Rank | Ref | File probed | Bytes | Licence on the Kaggle card | Needs | xz bits/byte, file | xz bits/byte, symbol stream | Byte alphabet |
|---|---|---|---|---|---|---|---|---|
| 1 | `datasnaek/chess` | `games.csv` | 7,672,655 | CC0-1.0 | 1 hard | 2.2760 (284,496 B) | moves only 2.3217 (290,208 B), alphabet 29 | 77 |
| 2 | `stanfordu/stanford-question-answering-dataset` | `dev-v1.1.json` | 4,854,279 | CC-BY-SA-4.0 | 1 medium, 2 | 1.3794 (172,420 B) | plain text 2.1877 (273,460 B), alphabet 158 | 88 |
| 3 | `tentotheminus9/religious-and-philosophical-texts` | `pg10.txt` | 4,452,070 | unknown (file is Project Gutenberg #10) | 1 easy, 4 cousin | 1.8457 (230,708 B) | same file | 80 |
| 4 | `martinsn/high-frequency-crypto-limit-order-book-data` | `BTC_5min.csv` | 6,606,836 | CC0-1.0 | 1 medium numeric | 1.9153 (239,416 B) | same file | 35 |
| 5 | `dnkumars/cryptocurrency-transaction-analytics-btc-and-eth` | `Cryptocurrency Transaction Data.csv` | 10,394,272 | CC0-1.0 | 3 relative | 3.5467 (443,336 B) | same file | 70 |
| 6 | `nageshsingh/dna-sequence-dataset` | `human.txt` | 5,547,716 | unknown | 1 easy, small alphabet | 0.6963 (87,032 B) | bases only 0.6953 (86,916 B), alphabet ACGTN | 22 |
| 7 | `tunguz/200000-jeopardy-questions` | `JEOPARDY_CSV.csv` | 34,854,057 | other | 1 hard language, 2 | 2.1871 (273,388 B) | question/answer text 2.6844 (335,544 B), alphabet 120 | 117 |
| 8 | `raj5287/abc-notation-of-tunes` | `input.txt` | 1,751,713 | unknown | 1 easy music | 1.5425 (192,808 B) | same file | 96 |

Usable slice for a channel is the whole probed file, except chess (the moves column is 5,027,572 bytes) and Jeopardy (34.9 MB, above the 20 MB preference; a 20 MB prefix is inside the file).

### 1. Lichess games — hard-learnable

`datasnaek/chess`, 1,530 votes. 20,058 games. Columns include `moves` (SAN), `opening_eco`, `opening_name`, and `white_id` / `black_id`. Move tokens in the whole file: 1,212,827. 365 ECO codes; the most common is A00 (1,007 games). SAN alphabet on the move stream: space, newline, `#+-12345678=BKNOQRabcdefghx`.

Excerpt (moves only):

```
d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4
d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6 Qe5+ Nxe5 c4 Bb4+
e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc6 bxc6 Ra6 Nc4 a4
```

Published compression reference: none found for this file.

Risk: player ids are in the csv. The move column is the channel. CC0-1.0 is on the Kaggle card; this is a 20k-game Lichess sample, not the full Lichess dump.

### 2. SQuAD v1.1 dev — episodic

`stanfordu/stanford-question-answering-dataset`, 244 votes. Dev file: 48 Wikipedia articles, 2,067 paragraphs, 10,570 questions. Every article has more than one paragraph. On every article, at least one answer string occurs at least twice in that article's text. `train-v1.1.json` is 30,288,272 bytes (same schema, not downloaded).

The raw JSON probe is 1.3794 bits/byte because keys repeat. The plain-text probe (title, context, question, answer) is 2.1877.

Excerpt:

```
Super_Bowl_50
Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
Which NFL team represented the AFC at Super Bowl 50? => Denver Broncos
```

Published reference: this file has none. The calibration corpus for Wikipedia text is enwik8/enwik9, below. SQuAD is a few MB of article extracts in JSON, not that dump.

Risk: CC-BY-SA-4.0 (attribution and share-alike). Passages are Wikipedia.

### 3. King James Bible — easy prose

`tentotheminus9/religious-and-philosophical-texts`, 82 votes. `pg10.txt` is Project Gutenberg EBook #10, "The King James Bible", release date in the file March 2, 2011. Genesis 1:1 starts at byte offset 699. A Gutenberg footer is present. Card licence is `unknown`. The file's own Gutenberg header says the eBook may be copied and reused under the Project Gutenberg License. The King James text is public domain.

Excerpt:

```
Title: The King James Bible
1:1 In the beginning God created the heavens and the earth.
1:1 Now these are the names of the children of Israel, which came
```

Published cousin, not this file: Large Canterbury `bible.txt` is 4,047,392 bytes (this PG file is 4,452,070). Bernhard Balkenhol's table (https://www.balkenhol.net/comp/canterbury.html): gzip -9 compresses `bible.txt` to 1,176,645 bytes, 2.326 bits/byte; bbz -x to 638,401 bytes, 1.262 bits/byte. data-compression.info lists GZIP93 on `bible.txt` at 2.330 bits/byte (https://www.data-compression.info/Comparisons/index.html). Those figures are the full Canterbury file with gzip, not this 1 MB xz probe.

Risk: strip the Gutenberg header and footer before use. The Kaggle card does not name a licence.

### 4. BTC 5-minute order book — medium numeric, real market

`martinsn/high-frequency-crypto-limit-order-book-data`, 44 votes. Card text: Coinbase Pro websocket order books, aggregated to 1-second, 1-minute, and 5-minute snapshots. Probed file: `BTC_5min.csv`, 3,367 rows, 156 columns, 2021-04-07 11:37:41 UTC through 2021-04-19 09:50:00 UTC. First midpoint 55899.995, spread 0.01. Last midpoint 56878.09, spread 6.54.

Same dataset, not downloaded: `BTC_1min.csv` 33,161,767 bytes; `BTC_1sec.csv` 1,755,134,589 bytes (over the 50 MB cap).

Excerpt (first fields only):

```
system_time, midpoint, spread, buys, sells
2021-04-07 11:37:41.122161+00:00, 55899.995, 0.01, 11993042.137718448, 13486155.647609372
2021-04-19 09:50:00.386544+00:00, 56878.09, 6.54
```

Published reference: none found.

Risk: 12 days is a short life. At 1.9153 bits/byte the 5-minute book sits with the Bible, below chess moves (2.32). It is a structured price channel. It is a weak noisy-TV channel. CC0-1.0.

### 5. Transaction table — highest entropy in the set, synthetic

`dnkumars/cryptocurrency-transaction-analytics-btc-and-eth`, 70 votes. 50,000 rows. Columns: Transaction_ID, Sender_Address, Receiver_Address, Amount, Transaction_Fee, Timestamp, Block_ID, Mining_Pool, Currency, Transaction_Type, Transaction_Status, Gas_Price_Gwei.

Every transaction id is unique (length 14). All 50,000 senders are unique. All 50,000 receivers are unique. Currency: ETH 25,012, BTC 24,988. Every type is Transfer. Every status is Confirmed. Mining pools: Ethermine 30,005, Poolin 5,039, Antpool 5,016, Slush Pool 5,004, BTC.TOP 4,936. Addresses are 42 characters and `0x`-prefixed on both the ETH and the BTC rows. Real Bitcoin txids are 64 hex characters. This table is synthetic.

Excerpt (addresses and ids omitted):

```
Amount, Transaction_Fee, Currency, Transaction_Type, Transaction_Status
11.39618058, 6.226e-05, ETH, Transfer, Confirmed
9.50400363, 4.342e-05
```

Published reference: none found.

Risk: the entropy is in made-up ids, not in a market. No party recurs, so a memory store has nothing to reuse. CC0-1.0 on the card. 3.5467 bits/byte is the highest probe here, and it is still far from 8 because of the repeated CSV words (Confirmed, Transfer, Ethermine, decimal amounts).

### 6. Human DNA fragments — easiest probe, 4-letter alphabet

`nageshsingh/dna-sequence-dataset`, 60 votes. `human.txt`: 4,380 sequences plus an integer class. Lengths: min 5, median 865, max 18,921. Class counts: 0:531, 1:534, 2:349, 3:672, 4:711, 5:240, 6:1,343. No codebook in the file. Base stream alphabet: A 248,749, C 267,257, G 267,515, T 216,383, N 96, on the first 1,000,000 bases. Bases-only xz is 0.6953 bits/byte. A uniform 4-symbol code stored as ASCII would be about 2 bits/byte, so xz is using sequence redundancy beyond the alphabet.

The other files in the dataset are `chimpanzee.txt` (3,258,260), `dog.txt` (1,667,375), `example_dna.fa` (3,099). Not downloaded.

Excerpt:

```
sequence	class
ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATTCCTCATCACCCAACTAAAAATATTAAACACAAACTACCACCTACCTCCCTCACCAAAGCCCATAAAAATAAAAAATTATAACAAACCCTGAGAACCAAAATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAG	4
ATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAGGCCTACCCGCCGCAGTACTGATCATTCTATTTCCCCCTCTATTGATCCCCACCTCCAAATATCTCATCAACAACCGACTAATCACCACCCAACAATGACTAATCAAACTAACCTCAAAACAAATGATAACCATACACAACACTAAAGGACGAACCTGATCTC
```

Published cousin, different file: Large Canterbury `E.coli`, 4,638,690 bytes, gzip -9 to 1,299,066 bytes, 2.240 bits/byte; bbz -x 1.880 bits/byte (same Balkenhol table). That is one contiguous genome. This Kaggle file is 4,380 labeled fragments.

Risk: card licence `unknown`. These are short labeled fragments, not one chromosome, and not a personal genome (a personal genome is not a 5.5 MB file of class-labeled pieces). The class integer is unexplained.

### 7. Jeopardy — strongest fact recurrence, licence blocks it

`tunguz/200000-jeopardy-questions`, 96 votes. 216,930 rows. Columns: Show Number, Air Date, Round, Category, Value, Question, Answer. 88,253 distinct answers. 29,573 answers appear more than once. Most repeated answers: China 216, Australia 215, Japan 196. 27,995 categories. Most common categories: BEFORE & AFTER 547, SCIENCE 519, LITERATURE 496, AMERICAN HISTORY 418, POTPOURRI 401.

Question/answer text is the hardest natural-language probe in the set: 2.6844 bits/byte.

Excerpt:

```
HISTORY | For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory | Copernicus
HISTORY | Built in 312 B.C. to link Rome & the South of Italy, it's still in use today | the Appian Way
HISTORY | In 1000 Rajaraja I of the Cholas battled to take this Indian Ocean island now known for its tea | Ceylon (or Sri Lanka)
```

Published reference: none found.

Risk: card licence is `other`. Card source is https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/ . The clues are Jeopardy copyright. This is the episodic channel on the numbers, and it is the one I would not ship.

### 8. ABC tunes — easy music, emails in the file

`raj5287/abc-notation-of-tunes`, 7 votes. 4,694 tunes (`T:` lines). 2,831 lines contain `@`. Headers cite O'Neill's. The notation itself repeats within a tune (the first tune restates its opening bars).

Excerpt (notation only; transcriber lines omitted):

```
T: The Enchanted Valley
M: 2/4
K:Gm
G3-A (Bcd=e) | f4 (g2dB) | ({d}c3-B) G2-E2 | F4 (D2=E^F) |
```

Published reference: none found.

Risk: card licence `unknown`. The file contains transcriber email addresses. Strip every `Z:` / email line before any use. Some tunes are old traditional repertoire; the file does not say which encodings are copyrighted.

## Published numbers for the files that did not fit the cap

`jamesmcguigan/hutter-prize` has one file, `enwik9`, 1,000,000,000 bytes. Card licence CC-BY-SA-3.0. Card source http://mattmahoney.net/dc/enwik9.zip . Over 50 MB, not downloaded. `text8` on `yorkyong/text8-zip` and `tonystark897/text8-txt` is one file of 100,000,000 bytes. Over 50 MB, not downloaded. No separate text8 bits/byte figure was retrieved.

Matt Mahoney, Large Text Compression Benchmark, http://mattmahoney.net/dc/text.html (page last update on that page: Oct 14, 2026). The number below is the compressed size of the data file. Decompressor size is extra, except the Hutter total.

enwik8 is the first 100,000,000 bytes. enwik9 is the first 1,000,000,000.

| Program | enwik8 bytes | enwik8 bits/byte | enwik9 bytes | enwik9 bits/byte |
|---|---|---|---|---|
| gzip 1.3.5 -9 | 36,445,248 | 2.9156 | 322,591,995 | 2.5807 |
| bzip2 1.0.2 -9 | 29,008,736 | 2.3207 | 253,977,839 | 2.0318 |
| xz 5.2.1 `--lzma2=preset=9e,dict=1GiB,lc=4,pb=0` | 24,703,772 | 1.9763 | 197,331,816 | 1.5787 |
| ppmd J1 `-m256 -o10 -r1` | 21,388,296 | 1.7111 | 183,964,915 | 1.4717 |
| cmix v21 `-t` | 14,623,723 | 1.1699 | 107,963,380 | 0.8637 |

Hutter prize, enwik8, http://prize.hutter1.net/ : Alexander Rhatushnyak, phda9, 4 Nov 2017, total size 15,284,944 bytes including a 42,448 byte decompressor, 1.2228 bits/byte on 1e8. The prize table prints bits/char 1.225 for that row.

These are full-file records with large dictionaries. The 1 MB `xz -9` probe above is a different measurement.

## Inspected, not downloaded

Over the 50 MB file cap: `mczielinski/bitcoin-historical-data` `btcusd_1-min_data.csv` 396,771,887 bytes; `finnhub/sp-500-futures-tick-data-sp` `SP.csv` 481,181,612 bytes; `nphantawee/pump-sensor-data` `sensor.csv` 124,058,460 bytes; `tgtanalytics/nq-futures-1min-bar-2022-2025` 72,522,264 bytes; `jrobischon/wikipedia-movie-plots` 81,193,310 bytes; `mylesoneill/drosophila-melanogaster-genome` `genome.fa` 146,638,899 bytes (the peptide table `ensembl-pep.csv` in that dataset is 20,473,906 bytes and was not downloaded).

Under the cap, left for a later pass: `tianmin/alphazero-vs-stockfish-110-chess-games` `alphazero_stockfish_moves_df.csv` 26,011,225 bytes, CC0-1.0, source the DeepMind AlphaZero resources, 110 games. `lichess/chess-openings` `openings.csv` 772,922 bytes, CC0-1.0. `selfishgene/historical-hourly-weather-data` `temperature.csv` 13,971,171 bytes, DbCL-1.0. `soumikrakshit/classical-music-midi`, dataset 2,445,834 bytes, many MIDI files under 80 KB each, card licence unknown, described as scraped from piano-midi.de. `erichartford/leetcode-solutions` `leetcode-solutions.jsonl` 10,732,730 bytes, card licence "GNU Lesser General Public License 3.0"; that tag is on a scraped solution dump, so the file was left alone. `yeoyunsianggeremie/codeforces-code-dataset` is 112 MB of separate `.java` files, a few KB each. Personal genomes (`zusmani/mygenome`, `zusmani/family-genome-dataset`) were skipped.

## Three to combine

1. `pg10.txt` — easy. Public-domain prose, xz 1.8457 bits/byte on the file, with the Canterbury Bible as the published cousin.
2. `datasnaek/chess` moves column — hard-learnable. SAN over 20,058 games, xz 2.3217 bits/byte, alphabet 29. CC0 if the id columns are dropped.
3. SQuAD `dev-v1.1.json` — episodic. The same answer string comes back inside the article, then the next article starts. Plain text xz 2.1877 bits/byte. CC-BY-SA-4.0.

The noise seat is empty. The transaction file is the only probe above 3 bits/byte (3.5467), and the ids are synthetic, with no repeated party. The real 5-minute BTC book is 1.9153 bits/byte. A real tick file small enough to download was not in the search (`SP.csv` is 481 MB; the Bitcoin minute file is 397 MB).

Jeopardy is the stronger episodic count (29,573 answers repeat) and the harder language probe (2.6844 on the question text). The card licence is `other`.
