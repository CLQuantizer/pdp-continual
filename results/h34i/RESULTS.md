| c4 seed | Own tail gap | Wait full gap | See tail gap | Bars meeting | Bars evaluated |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.135338867 | -0.690705002 | -0.047952230 | 3/3 | 3/3 |
| 1 | -0.158462290 | -0.681923366 | -0.053400390 | 3/3 | 3/3 |

| bible seed | Own tail gap | Wait full gap | See tail gap | Bars meeting | Bars evaluated |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.163756350 | -0.630239125 | -0.055805445 | 3/3 | 3/3 |
| 1 | -0.171888147 | -0.611781684 | -0.062806377 | 3/3 | 3/3 |

| mnist seed | Own tail gap | Wait full gap | See tail gap | Bars meeting | Bars evaluated |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.014606481 | -0.705753311 | -0.005065966 | 1/3 | 3/3 |
| 1 | -0.001006766 | -0.736298887 | -0.005694421 | 1/3 | 3/3 |

| Trick | Bar | c4 seeds meeting | bible seeds meeting | mnist seeds meeting | Total seeds meeting | Seeds evaluated | Streams with 2/2 | enwik8 s0 gap | enwik8 s1 gap | enwik8 seeds meeting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Own | -0.05 | 2/2 | 2/2 | 0/2 | 4/6 | 6/6 | 2/3 | -0.184277648 | -0.203066980 | 2/2 |
| Wait | -0.10 | 2/2 | 2/2 | 2/2 | 6/6 | 6/6 | 3/3 | -0.732934247 | -0.763597990 | 2/2 |
| See | -0.02 | 2/2 | 2/2 | 0/2 | 4/6 | 6/6 | 2/3 | -0.062003120 | -0.061783905 | 2/2 |

| Stream | Arm | Seed | Tail bits | Full bits | First 40 KB bits | enwik8 tail | enwik8 full | enwik8 first 40 KB | Reference source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| c4 | plain_1 | 0 | 3.781851999 | 3.890304666 | 4.269888999 | 3.264204845 | 3.533424381 | 4.475692753 | h34f |
| c4 | plain_1 | 1 | 3.807236726 | 3.923473965 | 4.330304304 | 3.285754554 | 3.565700932 | 4.545513255 | h34f |
| c4 | plain_6 | 0 | 3.646513132 | 4.463139657 | 7.321332493 | 3.079927197 | 4.107748392 | 7.705122572 | h34f |
| c4 | plain_6 | 1 | 3.648774436 | 4.469113566 | 7.340300518 | 3.082687574 | 4.150529688 | 7.887977086 | h34f |
| c4 | zgate_6 | 0 | 3.644675909 | 3.772434655 | 4.219590264 | 3.078244907 | 3.374814145 | 4.412806476 | h34f |
| c4 | zgate_6 | 1 | 3.648583752 | 3.787190200 | 4.272312765 | 3.077944571 | 3.386931698 | 4.468386643 | h34f |
| c4 | dense_6 | 0 | 3.596723679 | 3.730098566 | 4.196910671 | 3.016241787 | 3.320349905 | 4.384728317 | h34h |
| c4 | dense_6 | 1 | 3.595183362 | 3.740222437 | 4.247859198 | 3.016160666 | 3.330644108 | 4.431336156 | h34h |
| c4 | dense_6_ones | 0 | 3.592710819 | 4.371955399 | 7.099311426 | 3.010042782 | 3.997741283 | 7.454686036 | h34h |
| c4 | dense_6_ones | 1 | 3.590002656 | 4.386957224 | 7.176298209 | 3.011322334 | 4.019653114 | 7.548810842 | h34h |
| bible | plain_1 | 0 | 2.840087524 | 3.007913286 | 3.595303450 | 3.264204845 | 3.533424381 | 4.475692753 | h34f |
| bible | plain_1 | 1 | 2.849604090 | 3.028185646 | 3.653221093 | 3.285754554 | 3.565700932 | 4.545513255 | h34f |
| bible | plain_6 | 0 | 2.676331175 | 3.487899634 | 6.328389240 | 3.079927197 | 4.107748392 | 7.705122572 | h34f |
| bible | plain_6 | 1 | 2.677715943 | 3.482870060 | 6.300909473 | 3.082687574 | 4.150529688 | 7.887977086 | h34f |
| bible | zgate_6 | 0 | 2.672156881 | 2.857660508 | 3.506923204 | 3.078244907 | 3.374814145 | 4.412806476 | h34f |
| bible | zgate_6 | 1 | 2.674950602 | 2.871088377 | 3.557570588 | 3.077944571 | 3.386931698 | 4.468386643 | h34f |
| bible | dense_6 | 0 | 2.616351436 | 2.807459109 | 3.476335962 | 3.016241787 | 3.320349905 | 4.384728317 | h34h |
| bible | dense_6 | 1 | 2.612144225 | 2.812675983 | 3.514537138 | 3.016160666 | 3.330644108 | 4.431336156 | h34h |
| bible | dense_6_ones | 0 | 2.616024242 | 3.387056726 | 6.085670419 | 3.010042782 | 3.997741283 | 7.454686036 | h34h |
| bible | dense_6_ones | 1 | 2.611488120 | 3.376385112 | 6.053524583 | 3.011322334 | 4.019653114 | 7.548810842 | h34h |
| mnist | plain_1 | 0 | 1.342671048 | 1.378627418 | 1.504474713 | 3.264204845 | 3.533424381 | 4.475692753 | h34f |
| mnist | plain_1 | 1 | 1.337111367 | 1.378863434 | 1.524995669 | 3.285754554 | 3.565700932 | 4.545513255 | h34f |
| mnist | plain_6 | 0 | 1.328064567 | 2.035347671 | 4.510838536 | 3.079927197 | 4.107748392 | 7.705122572 | h34f |
| mnist | plain_6 | 1 | 1.336104601 | 2.065782353 | 4.619654483 | 3.082687574 | 4.150529688 | 7.887977086 | h34f |
| mnist | zgate_6 | 0 | 1.280784712 | 1.329594359 | 1.500428126 | 3.078244907 | 3.374814145 | 4.412806476 | h34f |
| mnist | zgate_6 | 1 | 1.276275221 | 1.329483466 | 1.515712323 | 3.077944571 | 3.386931698 | 4.468386643 | h34f |
| mnist | dense_6 | 0 | 1.275718746 | 1.325673507 | 1.500515170 | 3.016241787 | 3.320349905 | 4.384728317 | h34h |
| mnist | dense_6 | 1 | 1.270580800 | 1.325406053 | 1.517294439 | 3.016160666 | 3.330644108 | 4.431336156 | h34h |
| mnist | dense_6_ones | 0 | 1.298699676 | 1.923817328 | 4.111729112 | 3.010042782 | 3.997741283 | 7.454686036 | h34h |
| mnist | dense_6_ones | 1 | 1.309747174 | 1.975279826 | 4.304644110 | 3.011322334 | 4.019653114 | 7.548810842 | h34h |

| Stream | Seed | dense_6_ones − dense_6 tail | dense_6_ones − dense_6 full |
|---|---:|---:|---:|
| c4 | 0 | -0.004012859 | 0.641856833 |
| c4 | 1 | -0.005180706 | 0.646734787 |
| bible | 0 | -0.000327195 | 0.579597617 |
| bible | 1 | -0.000656104 | 0.563709129 |
| mnist | 0 | 0.022980930 | 0.598143821 |
| mnist | 1 | 0.039166374 | 0.649873773 |
| enwik8 | 0 | -0.006199005 | 0.677391378 |
| enwik8 | 1 | -0.004838332 | 0.689009006 |

| Runs scored | Runs expected | Smoke | Tail start | Tail end |
|---:|---:|---|---:|---:|
| 30 | 30 | True | 40000 | 180000 |

| Stream | Arm | Seed | Segment wall seconds | Run wall seconds | Kernel |
|---|---|---:|---:|---:|---|
| c4 | plain_1 | 0 | 340.353675196 | 340.353675196 | ezzzzzx/h34i-c4-plain-1-s0 |
| c4 | plain_1 | 1 | 298.668563655 | 298.668563655 | ezzzzzx/h34i-c4-plain-1-s1 |
| c4 | plain_6 | 0 | 2910.478550601 | 2910.478550601 | ezzzzzx/h34i-c4-plain-6-s0 |
| c4 | plain_6 | 1 | 3385.973907068 | 3385.973907068 | ezzzzzx/h34i-c4-plain-6-s1 |
| c4 | zgate_6 | 0 | 3022.010191311 | 3022.010191311 | ezzzzzx/h34i-c4-zgate-6-s0 |
| c4 | zgate_6 | 1 | 2906.987336203 | 2906.987336203 | ezzzzzx/h34i-c4-zgate-6-s1 |
| c4 | dense_6 | 0 | 2966.760789834 | 2966.760789834 | ezzzzzx/h34i-c4-dense-6-s0 |
| c4 | dense_6 | 1 | 1312.047737595 | 1312.047737595 | ezzzzzx/h34i-c4-dense-6-s1 |
| c4 | dense_6_ones | 0 | 3012.870137777 | 3012.870137777 | ezzzzzx/h34i-c4-dense-6-ones-s0 |
| c4 | dense_6_ones | 1 | 3319.040423859 | 3319.040423859 | ezzzzzx/h34i-c4-dense-6-ones-s1 |
| bible | plain_1 | 0 | 110.189889299 | 110.189889299 | ezzzzzx/h34i-bible-plain-1-s0 |
| bible | plain_1 | 1 | 241.523985139 | 241.523985139 | ezzzzzx/h34i-bible-plain-1-s1 |
| bible | plain_6 | 0 | 2963.290307108 | 2963.290307108 | ezzzzzx/h34i-bible-plain-6-s0 |
| bible | plain_6 | 1 | 2918.341249932 | 2918.341249932 | ezzzzzx/h34i-bible-plain-6-s1 |
| bible | zgate_6 | 0 | 2670.221545616 | 2670.221545616 | ezzzzzx/h34i-bible-zgate-6-s0 |
| bible | zgate_6 | 1 | 2670.678097085 | 2670.678097085 | ezzzzzx/h34i-bible-zgate-6-s1 |
| bible | dense_6 | 0 | 1382.987894963 | 1382.987894963 | ezzzzzx/h34i-bible-dense-6-s0 |
| bible | dense_6 | 1 | 2613.445160347 | 2613.445160347 | ezzzzzx/h34i-bible-dense-6-s1 |
| bible | dense_6_ones | 0 | 2968.964817834 | 2968.964817834 | ezzzzzx/h34i-bible-dense-6-ones-s0 |
| bible | dense_6_ones | 1 | 2907.837148621 | 2907.837148621 | ezzzzzx/h34i-bible-dense-6-ones-s1 |
| mnist | plain_1 | 0 | 396.930305004 | 396.930305004 | ezzzzzx/h34i-mnist-plain-1-s0 |
| mnist | plain_1 | 1 | 108.507621100 | 108.507621100 | ezzzzzx/h34i-mnist-plain-1-s1 |
| mnist | plain_6 | 0 | 3481.103587945 | 3481.103587945 | ezzzzzx/h34i-mnist-plain-6-s0 |
| mnist | plain_6 | 1 | 3880.400566611 | 3880.400566611 | ezzzzzx/h34i-mnist-plain-6-s1 |
| mnist | zgate_6 | 0 | 3474.117022345 | 3474.117022345 | ezzzzzx/h34i-mnist-zgate-6-s0 |
| mnist | zgate_6 | 1 | 3816.079802921 | 3816.079802921 | ezzzzzx/h34i-mnist-zgate-6-s1 |
| mnist | dense_6 | 0 | 4088.568673320 | 4088.568673320 | ezzzzzx/h34i-mnist-dense-6-s0 |
| mnist | dense_6 | 1 | 3783.894987976 | 3783.894987976 | ezzzzzx/h34i-mnist-dense-6-s1 |
| mnist | dense_6_ones | 0 | 4483.530238332 | 4483.530238332 | ezzzzzx/h34i-mnist-dense-6-ones-s0 |
| mnist | dense_6_ones | 1 | 3624.622962190 | 3624.622962190 | ezzzzzx/h34i-mnist-dense-6-ones-s1 |
| enwik8 | plain_1 | 0 | 312.435891968 | 312.435891968 | ezzzzzx/h34f-plain-1-s0 |
| enwik8 | plain_1 | 1 | 346.055265508 | 346.055265508 | ezzzzzx/h34f-plain-1-s1 |
| enwik8 | plain_6 | 0 | 3357.314462746 | 3357.314462746 | ezzzzzx/h34f-plain-6-s0 |
| enwik8 | plain_6 | 1 | 3599.511882545 | 3599.511882545 | ezzzzzx/h34f-plain-6-s1 |
| enwik8 | zgate_6 | 0 | 3255.959925905 | 3255.959925905 | ezzzzzx/h34f-zgate-6-s0 |
| enwik8 | zgate_6 | 1 | 3382.259732914 | 3382.259732914 | ezzzzzx/h34f-zgate-6-s1 |
| enwik8 | dense_6 | 0 | 3419.851387238 | 3419.851387238 | ezzzzzx/h34h-dense-6-s0 |
| enwik8 | dense_6 | 1 | 2889.970922083 | 2889.970922083 | ezzzzzx/h34h-dense-6-s1 |
| enwik8 | dense_6_ones | 0 | 3233.928259879 | 3233.928259879 | ezzzzzx/h34h-dense-6-ones-s0 |
| enwik8 | dense_6_ones | 1 | 3389.781466745 | 3389.781466745 | ezzzzzx/h34h-dense-6-ones-s1 |

| Stream | Bytes | SHA256 | Source SHA256 |
|---|---:|---|---|
| c4 | 180000 | ce571b05827c84849622acedfb7a7d3d6c6a46faccf5015d7e1865d60befbc65 | ca3b5fd714e95e42cffe2ddb93c44e233cfb05540f087f5e9b209da9b40d9c5e |
| bible | 180000 | 4d0bf3117914cb7269e88ac3d891a95742c732574676d34aecc51344d1a5cf4b | ef4a0a6f3bf7254c2304a0d7b577200f2b13c5c37c9bb5c28875e4017bbf14e9 |
| mnist | 180000 | 16cfada011d644ca6fb5bd1717760085c8d96874c7f82346ddeb544fb946276d | 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609 |

| Smoke check | Measurements |
|---|---|
| enwik8_equivalence | {"arm": "plain_1", "bits_per_byte": 5.142095267713781, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34b_plain_depth", "reference_bits": 5.142095267713781, "seed": 0, "wall_seconds": 17.425855874083936} |
| enwik8_equivalence | {"arm": "plain_1", "bits_per_byte": 5.142095267713781, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 5.142095267713781, "seed": 0, "wall_seconds": 17.425855874083936} |
| enwik8_equivalence | {"arm": "plain_6", "bits_per_byte": 11.727748600175858, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34b_plain_depth", "reference_bits": 11.727748600175858, "seed": 0, "wall_seconds": 186.5542665347457} |
| enwik8_equivalence | {"arm": "plain_6", "bits_per_byte": 11.727748600175858, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 11.727748600175858, "seed": 0, "wall_seconds": 186.5542665347457} |
| enwik8_equivalence | {"arm": "zgate_6", "bits_per_byte": 5.136672077864404, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 5.136672077864404, "seed": 0, "wall_seconds": 185.7900735391304} |
| enwik8_equivalence | {"arm": "zgate_6", "bits_per_byte": 5.136672077864404, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34g_depth", "reference_bits": 5.136672077864404, "seed": 0, "wall_seconds": 185.7900735391304} |
| enwik8_equivalence | {"arm": "dense_6", "bits_per_byte": 5.1429546675619315, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34g_depth", "reference_bits": 5.1429546675619315, "seed": 0, "wall_seconds": 190.61359696369618} |
| enwik8_equivalence | {"arm": "dense_6", "bits_per_byte": 5.1429546675619315, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34h_dense", "reference_bits": 5.1429546675619315, "seed": 0, "wall_seconds": 190.61359696369618} |
| enwik8_equivalence | {"arm": "dense_6_ones", "bits_per_byte": 11.23465416704458, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34h_dense", "reference_bits": 11.23465416704458, "seed": 0, "wall_seconds": 186.96048393752426} |
| enwik8_equivalence | {"arm": "plain_1", "bits_per_byte": 5.295348220502141, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34b_plain_depth", "reference_bits": 5.295348220502141, "seed": 1, "wall_seconds": 11.708576166070998} |
| enwik8_equivalence | {"arm": "plain_1", "bits_per_byte": 5.295348220502141, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 5.295348220502141, "seed": 1, "wall_seconds": 11.708576166070998} |
| enwik8_equivalence | {"arm": "plain_6", "bits_per_byte": 12.072928327396916, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34b_plain_depth", "reference_bits": 12.072928327396916, "seed": 1, "wall_seconds": 189.39115606620908} |
| enwik8_equivalence | {"arm": "plain_6", "bits_per_byte": 12.072928327396916, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 12.072928327396916, "seed": 1, "wall_seconds": 189.39115606620908} |
| enwik8_equivalence | {"arm": "zgate_6", "bits_per_byte": 5.276438589593237, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34f_zero_gate", "reference_bits": 5.276438589593237, "seed": 1, "wall_seconds": 189.88369769509882} |
| enwik8_equivalence | {"arm": "zgate_6", "bits_per_byte": 5.276438589593237, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34g_depth", "reference_bits": 5.276438589593237, "seed": 1, "wall_seconds": 189.88369769509882} |
| enwik8_equivalence | {"arm": "dense_6", "bits_per_byte": 5.288052342338434, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34g_depth", "reference_bits": 5.288052342338434, "seed": 1, "wall_seconds": 191.12518419604748} |
| enwik8_equivalence | {"arm": "dense_6", "bits_per_byte": 5.288052342338434, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34h_dense", "reference_bits": 5.288052342338434, "seed": 1, "wall_seconds": 191.12518419604748} |
| enwik8_equivalence | {"arm": "dense_6_ones", "bits_per_byte": 11.439297013204204, "max_bits_abs": 0.0, "n_bytes": 8192, "reference": "h34h_dense", "reference_bits": 11.439297013204204, "seed": 1, "wall_seconds": 190.9884139597416} |
| stream_identity | {"end": 180000, "file": "c4.bin", "n_bytes": 180000, "preview": "The woman who died after falling from a bridge over the A21 has been identified as a Sevenoaks mum..Marta Kendle, 37, fell from the Gracious Lane bridge on the morning of February 19..Police were call", "sha256": "ce571b05827c84849622acedfb7a7d3d6c6a46faccf5015d7e1865d60befbc65", "source": "data/h35c/c4_val.bin", "source_manifest": {"lines_scanned": 907, "n_bytes": 2001861, "n_docs": 868, "sha256": "ca3b5fd714e95e42cffe2ddb93c44e233cfb05540f087f5e9b209da9b40d9c5e", "source": "allenai/c4 en validation en/c4-validation.00000-of-00008.json.gz", "target_bytes": 2000000}, "source_sha256": "ca3b5fd714e95e42cffe2ddb93c44e233cfb05540f087f5e9b209da9b40d9c5e", "start": 0, "stream": "c4"} |
| new_stream_plain_1 | {"bits_per_byte": 4.896332294693652, "n_bytes": 8192, "seed": 0, "stream": "c4", "wall_seconds": 14.542214838787913} |
| new_stream_plain_1 | {"bits_per_byte": 5.089318372607185, "n_bytes": 8192, "seed": 1, "stream": "c4", "wall_seconds": 14.863923863507807} |
| stream_identity | {"end": 180000, "file": "bible.bin", "n_bytes": 180000, "preview": "...The Project Gutenberg EBook of The King James Bible....This eBook is for the use of anyone anywhere at no cost and with..almost no restrictions whatsoever.  You may copy it, give it away or..re-use", "sha256": "4d0bf3117914cb7269e88ac3d891a95742c732574676d34aecc51344d1a5cf4b", "source": "data/pg10.txt", "source_sha256": "ef4a0a6f3bf7254c2304a0d7b577200f2b13c5c37c9bb5c28875e4017bbf14e9", "start": 0, "stream": "bible"} |
| new_stream_plain_1 | {"bits_per_byte": 4.217409151621755, "n_bytes": 8192, "seed": 0, "stream": "bible", "wall_seconds": 15.902113638818264} |
| new_stream_plain_1 | {"bits_per_byte": 4.423972411810908, "n_bytes": 8192, "seed": 1, "stream": "bible", "wall_seconds": 16.387336568906903} |
| stream_identity | {"complete_images": 229, "end": 180000, "file": "mnist.bin", "final_image_bytes": 464, "idx_header": [2051, 60000, 28, 28], "n_bytes": 180000, "order": "image index, row, column", "preview": ["00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 03 12 12 12 7e 88 af 1a a6 ff f7 7f 00 00 00 00", "00 00 00 00 00 00 00 00 1e 24 5e 9a aa fd fd fd fd fd e1 ac fd f2 c3 40 00 00 00 00", "00 00 00 00 00 00 00 31 ee fd fd fd fd fd fd fd fd fb 5d 52 52 38 27 00 00 00 00 00", "00 00 00 00 00 00 00 12 db fd fd fd fd fd c6 b6 f7 f1 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 50 9c 6b fd fd cd 0b 00 2b 9a 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 0e 01 9a fd 5a 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 8b fd be 02 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 0b be fd 46 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 23 f1 e1 a0 6c 01 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 51 f0 fd fd 77 19 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 2d ba fd fd 96 1b 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 5d fc fd bb 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f9 fd f9 40 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 2e 82 b7 fd fd cf 02 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 27 94 e5 fd fd fd fa b6 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 18 72 dd fd fd fd fd c9 4e 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 17 42 d5 fd fd fd fd c6 51 02 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 12 ab db fd fd fd fd c3 50 09 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 37 ac e2 fd fd fd fd f4 85 0b 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 88 fd fd fd d4 87 84 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00", "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"], "sha256": "16cfada011d644ca6fb5bd1717760085c8d96874c7f82346ddeb544fb946276d", "skip_header_bytes": 16, "source": "data/train-images-idx3-ubyte.gz", "source_sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", "start": 0, "stream": "mnist"} |
| new_stream_plain_1 | {"bits_per_byte": 1.8917444507060877, "n_bytes": 8192, "seed": 0, "stream": "mnist", "wall_seconds": 18.73302111402154} |
| new_stream_plain_1 | {"bits_per_byte": 1.9239684716857088, "n_bytes": 8192, "seed": 1, "stream": "mnist", "wall_seconds": 23.118545881472528} |
| prequential_prefix | {"arm": "plain_1", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0, "stream": "c4"} |
| prequential_prefix | {"arm": "plain_1", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1, "stream": "c4"} |
| prequential_prefix | {"arm": "plain_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0, "stream": "c4"} |
| prequential_prefix | {"arm": "plain_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1, "stream": "c4"} |
| prequential_prefix | {"arm": "zgate_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0, "stream": "c4"} |
| prequential_prefix | {"arm": "zgate_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1, "stream": "c4"} |
| prequential_prefix | {"arm": "dense_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0, "stream": "c4"} |
| prequential_prefix | {"arm": "dense_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1, "stream": "c4"} |
| prequential_prefix | {"arm": "dense_6_ones", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0, "stream": "c4"} |
| prequential_prefix | {"arm": "dense_6_ones", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_1", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 0, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_1", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 0, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_1", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 1, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_1", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 1, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 0, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 0, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 1, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "plain_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 1, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "zgate_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 0, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "zgate_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 0, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "zgate_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 1, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "zgate_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 1, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 0, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 0, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 1, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 1, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6_ones", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 0, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6_ones", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 0, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6_ones", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 0, "seed": 1, "split_at": 1024, "state_exact": true, "stream": "c4"} |
| checkpoint_equivalence | {"arm": "dense_6_ones", "committed_bytes": 1024, "max_bits_abs": 0.0, "n_bytes": 2048, "pending_bytes": 16, "seed": 1, "split_at": 1040, "state_exact": true, "stream": "c4"} |
| wall_estimates | {"estimated_kernel_seconds": {"dense_6": 4199.527973057684, "dense_6_ones": 4196.522767670103, "plain_1": 507.9758616534491, "plain_6": 4161.426769032914, "zgate_6": 4172.249216933324}, "estimated_segment_seconds": {"dense_6": 4199.527973057684, "dense_6_ones": 4196.522767670103, "plain_1": 507.9758616534491, "plain_6": 4161.426769032914, "zgate_6": 4172.249216933324}, "limit": 5100, "n_bytes": 180000, "split_arms": []} |
