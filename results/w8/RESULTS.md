n_images 500 per_class 50 pool_per_class 50 opacity 768

L=16

| policy | accuracy | n_correct | logloss_nats | coverage |
|---|---:|---:|---:|---:|
| full | 0.978000 | 489 | 0.056925 | 1.000000 |
| none | 0.100000 | 50 | 3.384273 | 0.000000 |
| still | 0.228000 | 114 | 3.235962 | 0.250000 |
| random | 0.672000 | 336 | 1.028108 | 0.609795 |
| sweep | 0.984000 | 492 | 0.053447 | 0.997484 |
| curious_prog | 0.918000 | 459 | 0.242795 | 0.805088 |
| curious_surp | 0.918000 | 459 | 0.242795 | 0.805088 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |
|---|---:|---:|---:|---:|---:|---:|
| random-still | 0.444000 | 0.398000 | 0.492000 | -2.207854 | -2.394003 | -2.024988 |
| curious_prog-random | 0.246000 | 0.204000 | 0.288000 | -0.785313 | -0.907813 | -0.665173 |

L=16 full_minus_still_accuracy 0.750000

L=4

| policy | accuracy | n_correct | logloss_nats | coverage |
|---|---:|---:|---:|---:|
| full | 0.978000 | 489 | 0.056925 | 1.000000 |
| none | 0.100000 | 50 | 3.384273 | 0.000000 |
| still | 0.228000 | 114 | 3.235962 | 0.250000 |
| random | 0.372000 | 186 | 2.405636 | 0.396160 |
| sweep | 0.736000 | 368 | 0.822767 | 0.704885 |
| curious_prog | 0.466000 | 233 | 1.860370 | 0.427174 |
| curious_surp | 0.466000 | 233 | 1.860370 | 0.427174 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |
|---|---:|---:|---:|---:|---:|---:|
| random-still | 0.144000 | 0.104000 | 0.184000 | -0.830327 | -0.966102 | -0.695459 |
| curious_prog-random | 0.094000 | 0.054000 | 0.134000 | -0.545266 | -0.665618 | -0.426552 |

L=4 full_minus_still_accuracy 0.750000

L=8

| policy | accuracy | n_correct | logloss_nats | coverage |
|---|---:|---:|---:|---:|
| full | 0.978000 | 489 | 0.056925 | 1.000000 |
| none | 0.100000 | 50 | 3.384273 | 0.000000 |
| still | 0.228000 | 114 | 3.235962 | 0.250000 |
| random | 0.498000 | 249 | 1.794911 | 0.490186 |
| sweep | 0.954000 | 477 | 0.144614 | 0.919025 |
| curious_prog | 0.730000 | 365 | 0.815452 | 0.594838 |
| curious_surp | 0.730000 | 365 | 0.815452 | 0.594838 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |
|---|---:|---:|---:|---:|---:|---:|
| random-still | 0.270000 | 0.226000 | 0.316000 | -1.441051 | -1.604217 | -1.281810 |
| curious_prog-random | 0.232000 | 0.188000 | 0.276000 | -0.979458 | -1.111445 | -0.849683 |

L=8 full_minus_still_accuracy 0.750000

L=32

| policy | accuracy | n_correct | logloss_nats | coverage |
|---|---:|---:|---:|---:|
| full | 0.978000 | 489 | 0.056925 | 1.000000 |
| none | 0.100000 | 50 | 3.384273 | 0.000000 |
| still | 0.228000 | 114 | 3.235962 | 0.250000 |
| random | 0.842000 | 421 | 0.485816 | 0.737035 |
| sweep | 0.978000 | 489 | 0.056925 | 0.999996 |
| curious_prog | 0.968000 | 484 | 0.097046 | 0.954023 |
| curious_surp | 0.968000 | 484 | 0.097046 | 0.954023 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |
|---|---:|---:|---:|---:|---:|---:|
| random-still | 0.614000 | 0.570000 | 0.658000 | -2.750146 | -2.954800 | -2.554540 |
| curious_prog-random | 0.126000 | 0.094000 | 0.158000 | -0.388769 | -0.479575 | -0.303288 |

L=32 full_minus_still_accuracy 0.750000

L=64

| policy | accuracy | n_correct | logloss_nats | coverage |
|---|---:|---:|---:|---:|
| full | 0.978000 | 489 | 0.056925 | 1.000000 |
| none | 0.100000 | 50 | 3.384273 | 0.000000 |
| still | 0.228000 | 114 | 3.235962 | 0.250000 |
| random | 0.934000 | 467 | 0.188464 | 0.867824 |
| sweep | 0.978000 | 489 | 0.056925 | 0.999996 |
| curious_prog | 0.978000 | 489 | 0.056712 | 0.998170 |
| curious_surp | 0.978000 | 489 | 0.056712 | 0.998170 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |
|---|---:|---:|---:|---:|---:|---:|
| random-still | 0.706000 | 0.664000 | 0.746000 | -3.047499 | -3.245469 | -2.851268 |
| curious_prog-random | 0.044000 | 0.022000 | 0.066000 | -0.131752 | -0.182047 | -0.083936 |

L=64 full_minus_still_accuracy 0.750000

synsets

| order | wnid | imagenet_index | category |
|---:|---|---:|---|
| 0 | n01440764 | 0 | tench |
| 1 | n02102040 | 217 | English springer |
| 2 | n02979186 | 482 | cassette player |
| 3 | n03000684 | 491 | chain saw |
| 4 | n03028079 | 497 | church |
| 5 | n03394916 | 566 | French horn |
| 6 | n03417042 | 569 | garbage truck |
| 7 | n03425413 | 571 | gas pump |
| 8 | n03445777 | 574 | golf ball |
| 9 | n03888257 | 701 | parachute |

source

dataset imagenette2-160
split val
order sorted filename within wnid, wnid order is the synset table
pool_per_class 50
scored_per_class 50
seed pool index, 50 per class, first scored_per_class of each class
resize short side 224 bilinear, centre crop 224, offset (size-224)//2
weight_preset_resize_256 not used
exif_transpose no
exif_orientation_not_1 0
pool_counts 387,395,357,386,409,394,389,419,399,390
grey integer BT.601 (299R+587G+114B)/1000, mean of 7x7, floor(mean*8/256)
opacity 768
scene_shift 1
dot_shift 2
eye torchvision resnet18 IMAGENET1K_V1 frozen eval
softmax 10 logits, synset table order
logloss nats
normalize_mean 0.485 0.456 0.406
normalize_std 0.229 0.224 0.225
unseen imagenet mean colour
bootstrap 10000 paired percentile, numpy quantile linear, default_rng(0) per contrast
runner vps
device cpu
torch_threads 2
kernel_id none
torch 2.15.0+cpu
torchvision 0.29.2+cpu
numpy 2.5.4
files_sha256 4a7d039aff8fae3a936ad78729fb4cd7bcc3faf78a2b5257da8fcb7740fa649f
smoke_full_accuracy_10way 0.978000
smoke_full_accuracy_1000way 0.666000
load_average_start 0.36

walls

w7_s 2.705
load_s 1.228
model_s 0.837
full_eye_s 33.547
image0_s 0.070
time20_s 48.742
per_image_s 2.437
estimate_500_s 1264.461
estimate_300_s 777.038
rest_s 1145.271
demo_s 0.350
wall_s 1244.014

w7 seed 10

max_abs_probe 0.000e+00
max_abs_view 0.000e+00
max_abs_logo 0.000e+00
max_abs_cover 0.000e+00
rows 60
wall_s 2.705e+00

image 0 grey

2 2 5 6 4 3 3 5 6 6 3 2 1 3 6 6 6 6 5 4 4 2 2 3 4 4 4 4 4 4 4 4
1 2 3 4 4 3 5 6 6 6 6 6 5 5 2 2 4 6 6 5 5 3 2 3 4 5 5 4 5 4 4 4
2 2 2 3 3 5 6 6 6 6 7 6 7 6 4 5 3 2 1 1 5 7 5 5 4 6 6 5 4 4 4 4
4 1 1 3 4 6 6 6 5 6 5 5 7 6 4 5 3 4 3 1 1 2 2 3 6 6 5 5 4 4 4 4
4 2 1 3 5 5 6 6 6 6 5 4 5 5 4 4 4 4 4 4 2 3 1 1 2 5 6 6 5 5 5 4
4 2 2 4 5 5 5 6 6 5 4 4 4 4 4 3 4 5 4 5 3 5 3 5 5 3 2 3 5 6 6 5
2 2 3 3 3 3 4 4 5 5 5 3 3 3 4 3 3 2 5 6 2 3 3 5 7 3 4 3 2 2 3 5
3 2 4 5 4 4 3 3 3 3 4 3 3 3 2 2 1 2 2 3 2 2 4 5 4 4 4 5 5 3 2 2
3 5 5 5 5 5 5 5 4 3 3 2 2 1 0 1 0 2 3 2 1 2 5 5 3 4 5 5 4 3 4 5
2 4 4 4 5 5 5 5 5 5 4 4 3 2 0 0 0 0 0 0 1 2 2 2 2 3 4 2 1 1 2 2
2 3 4 3 4 5 5 5 4 4 4 4 2 2 0 1 0 0 0 0 0 1 2 2 3 3 2 2 2 2 1 2
2 2 4 4 3 4 5 5 5 4 4 3 2 2 1 0 0 0 0 0 0 0 0 1 3 3 4 4 5 6 5 3
2 1 2 3 4 3 4 5 4 4 4 3 2 3 2 0 0 0 0 0 0 0 0 1 5 6 6 6 5 4 4 4
5 3 1 2 3 4 4 4 5 4 3 3 3 4 2 1 1 0 0 1 1 1 1 3 4 5 4 3 3 4 5 5
5 4 4 3 2 2 2 4 5 4 4 2 3 4 2 1 1 1 1 1 1 2 3 3 4 3 3 4 4 5 5 5
5 4 4 5 5 2 1 2 3 3 3 2 3 3 3 2 2 3 2 2 3 2 3 5 4 3 4 4 4 4 5 5
5 4 4 4 5 5 3 3 4 4 5 4 4 4 3 3 3 3 3 4 3 1 3 3 3 3 3 3 4 4 5 4
4 3 3 4 4 4 5 5 5 5 5 5 4 4 3 3 3 3 2 3 3 3 4 3 2 3 4 4 4 5 4 3
4 3 3 4 5 6 7 6 6 5 5 5 5 4 4 4 4 3 3 4 4 4 4 4 4 4 4 4 3 3 3 2
4 5 5 4 4 6 6 6 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3
4 4 5 4 4 4 4 4 4 4 4 4 3 3 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
4 4 4 3 5 4 4 4 5 4 4 4 3 3 3 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 2 2
4 4 3 3 4 3 4 4 4 4 4 3 3 3 3 2 2 1 2 2 2 2 1 1 1 1 1 2 2 2 2 2
2 2 2 3 3 2 3 4 4 4 3 3 3 2 2 2 1 1 2 2 1 1 1 1 0 0 0 0 0 1 1 1
2 1 1 1 2 1 0 1 1 1 1 2 2 2 1 1 1 1 1 1 0 0 2 2 1 0 0 1 2 2 3 2
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 3 3 3 2 3 2 1 1 1 1 2 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 5 5 5 5 4 4 4 4 4 3 2 2 6
1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 2 3 4 5 5 5 6 5 5 5 5 4 5 4 6 3
5 3 3 3 2 2 2 2 2 3 3 3 4 5 5 5 5 4 5 4 5 4 4 4 5 5 5 4 5 6 4 3
5 5 5 4 4 4 4 4 4 5 4 5 5 5 5 5 5 4 4 4 4 4 4 4 5 5 5 4 6 5 5 3
5 5 5 5 5 5 5 5 5 5 4 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 4 4 2
5 5 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 4 4 4 1

image 0 curious_prog first 10 viewpoints (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0) (9,0) (10,0)

demos

img000_still_L1_garbage_truck_0.3063_L4_garbage_truck_0.3063_L16_garbage_truck_0.3063_L64_garbage_truck_0.3063.png
img000_random_L1_English_springer_0.3364_L4_garbage_truck_0.4751_L16_garbage_truck_0.7427_L64_tench_0.9986.png
img000_curious_prog_L1_English_springer_0.3364_L4_garbage_truck_0.7170_L16_tench_0.6780_L64_tench_0.9942.png
img001_still_L1_parachute_0.7420_L4_parachute_0.7420_L16_parachute_0.7420_L64_parachute_0.7420.png
img001_random_L1_parachute_0.7683_L4_tench_0.7070_L16_tench_0.9999_L64_tench_1.0000.png
img001_curious_prog_L1_tench_0.7160_L4_tench_0.9962_L16_tench_0.9994_L64_tench_1.0000.png
img002_still_L1_church_0.9490_L4_church_0.9490_L16_church_0.9490_L64_church_0.9490.png
img002_random_L1_church_0.9294_L4_church_0.9051_L16_church_0.9890_L64_tench_0.9986.png
img002_curious_prog_L1_church_0.9294_L4_church_0.8129_L16_tench_1.0000_L64_tench_1.0000.png
