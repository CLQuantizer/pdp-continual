# Papers registry, 2026-11-09

One row per cited paper in D1 database `pdp-papers-index`, table `papers`. Files are in private R2 bucket `pdp-papers`. No public access and no custom domain were turned on. Nothing was committed.

`tools/sync_papers.py` now uses this schema. From the repo root:

```
python3 tools/sync_papers.py
python3 tools/sync_papers.py --dry
python3 tools/sync_papers.py --fetch
python3 tools/sync_papers.py --wanted
```

The default command uploads files in `docs/papers/` when the sha256 differs. `--fetch` downloads rows marked `to_fetch` (arXiv, an `oa_pdf=` URL in notes, or an Unpaywall PDF). `--wanted` prints the list below. `--dry` writes nothing. Re-running does not replace a filled field with NULL. `status` is how well we know the paper (`read`, `checked`, `memory`). `file_status` is where the file is. Downloading a PDF does not change `status`.

## Counts

| file_status | rows |
| --- | ---: |
| stored | 59 |
| wanted | 50 |
| to_fetch | 0 |
| unavailable | 0 |
| **total** | **109** |

`status`: read 2, checked 84, memory 23. Every row has a title and a `cited_in` path.

`sqlite_master` tables: `_cf_KV`, `papers`.

## Migration from `documents`

`documents` held 3 file rows and 2 papers. Both PDFs are stored on their original keys. The HTML companion is still in the bucket and is recorded on the Schmidhuber row, not as a second paper. Each of the three objects was downloaded again and its sha256 and size matched before `DROP TABLE documents`.

| object | bytes | sha256 |
| --- | ---: | --- |
| `papers/schmidhuber2009_driven_by_compression_progress.pdf` | 537077 | `e971d466392c52f0a5212fcacc022c1cffab16fcf120ff25106bd71ed685557d` |
| `papers/schmidhuber2009_driven_by_compression_progress.html` | 317444 | `2e2315b104ddc53b9ea51d99dffe263f69bcfd089cf54725ef07b876c2806584` |
| `papers/journe2022_hebbian_deep_learning_without_feedback.pdf` | 27441097 | `00432ed0d6edc6c357b9a8ad160b36223088b4f8108ecc3fce397f5a01172f39` |

New open-access files use `papers/<cite_key>.pdf`. The two files above kept the keys they already had.

## Wanted (50)

`python3 tools/sync_papers.py --wanted` printed these 50 rows (`cite_key`, title, authors, year, url):

```
lorden1971	Procedures for Reacting to a Change in Distribution	G. Lorden	1971	https://doi.org/10.1214/aoms/1177693055
teyler1986	The hippocampal memory indexing theory.	Timothy J. Teyler, Pascal DiScenna	1986	https://doi.org/10.1037/0735-7044.100.2.147
desieno1988	Adding a conscience to competitive learning	DeSieno	1988	https://doi.org/10.1109/icnn.1988.23839
anderson1991	Reflections of the Environment in Memory	John R. Anderson, Lael J. Schooler	1991	https://doi.org/10.1111/j.1467-9280.1991.tb00174.x
french1991	Using Semi-Distributed Representations to Overcome Catastrophic Forgetting in Connectionist Networks	Robert M. French	1991	https://escholarship.org/uc/item/7xt8t468
schmidhuber1991a	A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers	Juergen Schmidhuber	1991	https://doi.org/10.5555/116517.116542
schmidhuber1991b	Curious model-building control systems	J. Schmidhuber	1991	https://doi.org/10.1109/ijcnn.1991.170605
schmidhuber1992	Learning Complex, Extended Sequences Using the Principle of History Compression	Jürgen Schmidhuber	1992	https://doi.org/10.1162/neco.1992.4.2.234
oreilly1994	Hippocampal conjunctive encoding, storage, and recall: Avoiding a trade‐off	Randall C. O'Reilly, James L. McClelland	1994	https://doi.org/10.1002/hipo.450040605
wilson1994	Reactivation of Hippocampal Ensemble Memories During Sleep	Matthew A. Wilson, Bruce L. McNaughton	1994	https://doi.org/10.1126/science.8036517
hinton1995	The "Wake-Sleep" Algorithm for Unsupervised Neural Networks	Geoffrey E. Hinton, Peter Dayan, Brendan J. Frey, Radford M. Neal	1995	https://doi.org/10.1126/science.7761831
krogh1995	Neural Network Ensembles, Cross Validation, and Active Learning	Anders Krogh, Jesper Vedelsby	1995	https://proceedings.neurips.cc/paper_files/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html
mcclelland1995	Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory.	James L. McClelland, Bruce L. McNaughton, Randall C. O'Reilly	1995	https://doi.org/10.1037/0033-295x.102.3.419
gallistel2000	Time, rate, and conditioning.	C. R. Gallistel, John Gibbon	2000	https://doi.org/10.1037/0033-295x.107.2.289
howard2002	A Distributed Representation of Temporal Context	Marc W. Howard, Michael J. Kahana	2002	https://doi.org/10.1006/jmps.2001.1388
norman2003	Modeling hippocampal and neocortical contributions to recognition memory: A complementary-learning-systems approach.	Kenneth A. Norman, Randall C. O'Reilly	2003	https://doi.org/10.1037/0033-295x.110.4.611
poirazi2003	Pyramidal Neuron as Two-Layer Neural Network	Panayiota Poirazi, Terrence Brannon, Bartlett W. Mel	2003	https://doi.org/10.1016/s0896-6273(03)00149-1
daw2005	Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control	Nathaniel D Daw, Yael Niv, Peter Dayan	2005	https://doi.org/10.1038/nn1560
leutgeb2005a	Independent Codes for Spatial and Episodic Memory in Hippocampal Neuronal Ensembles	Stefan Leutgeb, Jill K. Leutgeb, Carol A. Barnes, Edvard I. Moser, Bruce L. McNaughton, May-Britt Moser	2005	https://doi.org/10.1126/science.1114037
leutgeb2005b	Progressive Transformation of Hippocampal Neuronal Representations in “Morphed” Environments	Jill K. Leutgeb, Stefan Leutgeb, Alessandro Treves, Retsina Meyer, Carol A. Barnes, Bruce L. McNaughton, May-Britt Moser, Edvard I. Moser	2005	https://doi.org/10.1016/j.neuron.2005.09.007
lisman2005	The Hippocampal-VTA Loop: Controlling the Entry of Information into Long-Term Memory	John E. Lisman, Anthony A. Grace	2005	https://doi.org/10.1016/j.neuron.2005.05.002
wills2005	Attractor Dynamics in the Hippocampal Representation of the Local Environment	Tom J. Wills, Colin Lever, Francesca Cacucci, Neil Burgess, John O'Keefe	2005	https://doi.org/10.1126/science.1108905
yu2005	Uncertainty, Neuromodulation, and Attention	Angela J. Yu, Peter Dayan	2005	https://doi.org/10.1016/j.neuron.2005.04.026
courville2006	Bayesian theories of conditioning in a changing world	Aaron C. Courville, Nathaniel D. Daw, David S. Touretzky	2006	https://doi.org/10.1016/j.tics.2006.05.004
hasselmo2006	The role of acetylcholine in learning and memory	Michael E Hasselmo	2006	https://doi.org/10.1016/j.conb.2006.09.002
huang2006	Extreme learning machine: Theory and applications	Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew	2006	https://doi.org/10.1016/j.neucom.2005.12.126
behrens2007	Learning the value of information in an uncertain world	Timothy E J Behrens, Mark W Woolrich, Mark E Walton, Matthew F S Rushworth	2007	https://doi.org/10.1038/nn1954
gold2007	The Neural Basis of Decision Making	Joshua I. Gold, Michael N. Shadlen	2007	https://doi.org/10.1146/annurev.neuro.29.051605.113038
kumaran2007	Match–Mismatch Processes Underlie Human Hippocampal Responses to Associative Novelty	Dharshan Kumaran, Eleanor A. Maguire	2007	https://doi.org/10.1523/jneurosci.1677-07.2007
lengyel2007	Hippocampal Contributions to Control: The Third Way	Máté Lengyel, Peter Dayan	2007	https://papers.nips.cc/paper/3311-hippocampal-contributions-to-control-the-third-way
oudeyer2007	Intrinsic Motivation Systems for Autonomous Mental Development	Pierre-Yves Oudeyer, Frdric Kaplan, Verena V. Hafner	2007	https://doi.org/10.1109/tevc.2006.890271
karlsson2009	Awake replay of remote experiences in the hippocampus	Mattias P Karlsson, Loren M Frank	2009	https://doi.org/10.1038/nn.2344
diekelmann2010	The memory function of sleep	Susanne Diekelmann, Jan Born	2010	https://doi.org/10.1038/nrn2762
gershman2010	Context, learning, and extinction.	Samuel J. Gershman, David M. Blei, Yael Niv	2010	https://doi.org/10.1037/a0017808
nassar2010	An Approximately Bayesian Delta-Rule Model Explains the Dynamics of Belief Updating in a Changing Environment	Matthew R. Nassar, Robert C. Wilson, Benjamin Heasly, Joshua I. Gold	2010	https://doi.org/10.1523/jneurosci.0822-10.2010
schmidhuber2010	Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990–2010)	Jürgen Schmidhuber	2010	https://doi.org/10.1109/tamd.2010.2056368
jezek2011	Theta-paced flickering between place-cell maps in the hippocampus	Karel Jezek, Espen J. Henriksen, Alessandro Treves, Edvard I. Moser, May-Britt Moser	2011	https://doi.org/10.1038/nature10439
lacy2011	Distinct pattern separation related transfer functions in human CA3/dentate and CA1 revealed using high-resolution fMRI and variable mnemonic similarity	Joyce W. Lacy, Michael A. Yassa, Shauna M. Stark, L. Tugan Muftuler, Craig E.L. Stark	2011	https://doi.org/10.1101/lm.1971111
yassa2011	Pattern separation in the hippocampus	Michael A. Yassa, Craig E.L. Stark	2011	https://doi.org/10.1016/j.tins.2011.06.006
kumaran2012	Generalization through the recurrent interaction of episodic memories: A model of the hippocampal system.	Dharshan Kumaran, James L. McClelland	2012	https://doi.org/10.1037/a0028681
babadi2014	Sparseness and Expansion in Sensory Representations	Baktash Babadi, Haim Sompolinsky	2014	https://doi.org/10.1016/j.neuron.2014.07.035
ambrose2016	Reverse Replay of Hippocampal Place Cells Is Uniquely Modulated by Changing Reward	R. Ellen Ambrose, Brad E. Pfeiffer, David J. Foster	2016	https://doi.org/10.1016/j.neuron.2016.07.047
kumaran2016	What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated	Dharshan Kumaran, Demis Hassabis, James L. McClelland	2016	https://doi.org/10.1016/j.tics.2016.05.004
bittner2017	Behavioral time scale synaptic plasticity underlies CA1 place fields	Katie C. Bittner, Aaron D. Milstein, Christine Grienberger, Sandro Romani, Jeffrey C. Magee	2017	https://doi.org/10.1126/science.aan3846
richards2017	The Persistence and Transience of Memory	Blake A. Richards, Paul W. Frankland	2017	https://doi.org/10.1016/j.neuron.2017.04.037
rule2019	Causes and consequences of representational drift	Michael E Rule, Timothy O’Leary, Christopher D Harvey	2019	https://doi.org/10.1016/j.conb.2019.08.005
gidon2020	Dendritic action potentials and computation in human layer 2/3 cortical neurons	Albert Gidon, Timothy Adam Zolnik, Pawel Fidzinski, Felix Bolduan, Athanasia Papoutsi, Panayiota Poirazi, Martin Holtkamp, Imre Vida, Matthew Evan Larkum	2020	https://doi.org/10.1126/science.aax6239
lillicrap2020	Backpropagation and the brain	Timothy P. Lillicrap, Adam Santoro, Luke Marris, Colin J. Akerman, Geoffrey Hinton	2020	https://doi.org/10.1038/s41583-020-0277-3
doro2023	Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier	Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G. Bellemare, Aaron Courville	2023	https://openreview.net/forum?id=OpC-9aBBVJe
sun2025	DeeperForward: Enhanced Forward-Forward Training for Deeper and Better Performance	Liang Sun, Yang Zhang, Weizhao He, Jiajun Wen, Linlin Shen, Weicheng Xie	2025	https://proceedings.iclr.cc/paper_files/paper/2025/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html
```

Seven of these have a free copy that this host could not download. They are not paywall requests:

- `french1991`: eScholarship returned 403.
- `krogh1995` and `lengyel2007`: the NeurIPS proceedings PDFs timed out.
- `lorden1971`: Project Euclid returned HTML, not a PDF.
- `sun2025`: the ICLR 2025 proceedings PDF timed out. OpenReview also returned 403.
- `doro2023`: the OpenReview PDF returned 403.
- `jezek2011`: the SISSA repository PDF returned 403.

The other wanted rows are publisher or DOI pages. Unpaywall either had no legal PDF, or the publisher URL returned 403. No shadow library was used. Crossref spells Oudeyer's second author as "Frdric Kaplan"; that string is stored as the landing page returned it.

## Stored-object checks

Three stored rows, drawn with `random.Random(20261109)` from the 59 stored rows. Each object was downloaded from R2 and hashed again.

| cite_key | r2 key | bytes | sha256 | match |
| --- | --- | ---: | --- | --- |
| scodellaro2025 | `papers/scodellaro2025.pdf` | 3089658 | `156af99da2c76890e0b2a6856b50bb92206ce830ce66fa81df533c6f74301fd5` | yes |
| lee2023 | `papers/lee2023.pdf` | 1589919 | `70d90bb8e9389aab86db56f5fe474ddd2d245ec3e77228340b7030dddd86cef3` | yes |
| dohare2024 | `papers/dohare2024.pdf` | 9163939 | `273b70522c1c428744b1e9abf2f7523c08845a2d5ecb3ee0959c825eefdadccd` | yes |

## Second sync

After the fetch, a second `python3 tools/sync_papers.py` printed:

```
unchanged journe2022_hebbian_deep_learning_without_feedback.pdf
unchanged schmidhuber2009_driven_by_compression_progress.pdf
unchanged companion schmidhuber2009_driven_by_compression_progress.html
```

Exit code 0. A `--dry` run before the first real sync printed the same three lines and wrote nothing.

## Left unregistered

Tononi and Cirelli are named in `docs/notes.md` and `audit.md` with no year and no title. Both the 2006 and the 2014 synaptic-homeostasis papers match that note, so no row was created.

"Jacobi preconditioning" and "Page's CUSUM" are named as methods, without a year, and are not separate papers.

The loop's "Schmidhuber 1991" citation is recorded on both `schmidhuber1991a` (SAB curiosity and boredom) and `schmidhuber1991b` (IJCNN curious model-building). The history compressor is only `schmidhuber1992`.

## Landing-page decisions

- `schmidhuber2009` keeps arXiv `0812.4360v2`, a null DOI, and a null venue. The LNCS DOI `10.1007/978-3-642-02565-5_4` is a different short version.
- `pathak2017` is the ICML paper, arXiv `1705.05363`. The CVPR-workshop DOI is a different record.
- `chaudhry2024` is "Long sequence Hopfield memory" (Chaudhry, Zavatone-Veth, Krotov, Pehlevan, J. Stat. Mech. 2024, arXiv `2306.04532`). The README also says "Krotov, Hoover et al., ICML 2025", which is a different tutorial. This row is also distinct from `chaudhry2019`.
- `dohare2024` is the Nature 2024 article, DOI `10.1038/s41586-024-07711-7`, and the PDF is stored. arXiv `2306.13812` is titled "Maintaining Plasticity in Deep Continual Learning" and is not this paper's id.
- `hooker2020` uses the repo year 2020. The Communications of the ACM record is 2021. The arXiv PDF is stored.
- `lacy2011` uses the January 2011 print issue (volume 18). Crossref's issued year is 2010.
- `fedus2022`: the repo gives no year. jmlr.org lists 23(120):1–39, 2022. The arXiv PDF is stored.
- `journe2023` uses the ICLR 2023 venue year. The local filename still says 2022.
- `kafraj2026` folds the last token of "Mohadeseh Shafiei Kafraj". `deepseekai2024` folds "DeepSeek-AI". `dai2024` is DeepSeekMoE (the repo gives no year). `scodellaro2025`, `spyra2025`, `casnici2026`, and `millidge2025` use the landing-page first author. `leutgeb2005a` is Science and `leutgeb2005b` is Neuron. `burda2019a` is RND and `burda2019b` is the large-scale study.
- Where the stored file is a bioRxiv preprint of a later journal article, the row notes say so. The cite year stays the journal year.
