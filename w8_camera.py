"""W8-0 curious camera.

The policies in w7_run run unchanged on a grey proxy of a photograph.
The frozen ResNet reads the RGB scene: a cell seen at least once keeps its
true block, and an unseen cell is the ImageNet mean.
"""
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_HOME"] = os.path.join(ROOT, "data", "w8", "torch")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

import w7_run as R
import w7_world as W

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_grad_enabled(False)

CELL = 7
SIDE = 224
POOL = 50
OPACITY = 768
LS = (4, 8, 16, 32, 64)
DEMO_LS = (1, 4, 16, 64)
DEMO_INDEX = (0, 1, 2)
DEMO_POLICIES = ("still", "random", "curious_prog")
POLICIES = R.POLICIES
N_BOOT = 10_000
BUDGET_S = 90 * 60
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Integer BT.601: (299R + 587G + 114B) / 1000. The weights sum to 1000, so a
# flat grey of value v stays v. Bins are floor(mean * 8 / 256), i.e. [0, 32) ... [224, 256).

# Imagenette wnid order is the sorted validation folder order.
# Logit index is the ImageNet-1K class of that wnid.
SYNSETS = (
    ("n01440764", "tench", 0),
    ("n02102040", "English springer", 217),
    ("n02979186", "cassette player", 482),
    ("n03000684", "chain saw", 491),
    ("n03028079", "church", 497),
    ("n03394916", "French horn", 566),
    ("n03417042", "garbage truck", 569),
    ("n03425413", "gas pump", 571),
    ("n03445777", "golf ball", 574),
    ("n03888257", "parachute", 701),
)
SYN_INDEX = torch.tensor([row[2] for row in SYNSETS], dtype=torch.long)


def log(msg):
    print(msg, flush=True)


def load_average():
    with open("/proc/loadavg", encoding="utf-8") as f:
        return float(f.read().split()[0])


def f6(x):
    return f"{float(x):.6f}"


def slug(name):
    return name.replace(" ", "_")


def grey_proxy(rgb_u8):
    """32×32 greys 0..7. Mean BT.601 luminance of each 7×7 cell, then equal bins."""
    if rgb_u8.shape != (SIDE, SIDE, 3):
        raise RuntimeError(f"rgb shape {rgb_u8.shape}")
    rgb = rgb_u8.astype(np.int64)
    acc = 299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2]
    # Sum of 1000*luminance over the 49 pixels. Divide by 49*1000*256/8.
    total = acc.reshape(W.N, CELL, W.N, CELL).sum(axis=(1, 3))
    q = ((total * 8) // (CELL * CELL * 1000 * 256)).astype(np.int16)
    if int(q.min()) < 0 or int(q.max()) > 7:
        raise RuntimeError("grey bin outside 0..7")
    return q


def reconstruct(rgb_f, seen):
    """Scene-coordinate RGB in 0..1. Unseen cells are the ImageNet mean."""
    out = np.empty((SIDE, SIDE, 3), dtype=np.float32)
    out[...] = IMAGENET_MEAN
    mask = np.repeat(np.repeat(seen, CELL, axis=0), CELL, axis=1)
    out[mask] = rgb_f[mask]
    return out


def to_u8(recon):
    return np.clip(np.rint(recon * 255.0), 0, 255).astype(np.uint8)


def seen_of(grey, occ, path):
    counts, _visits = W.accumulate(grey, occ, path, W.SCENE_SHIFT, W.DOT_SHIFT)
    return counts.sum(axis=2) > 0


def resize_crop(im, side=SIDE):
    im = im.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        raise RuntimeError("empty image")
    if w <= h:
        nw = side
        nh = int(round(h * side / w))
    else:
        nh = side
        nw = int(round(w * side / h))
    nw = max(nw, side)
    nh = max(nh, side)
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    left = (nw - side) // 2
    top = (nh - side) // 2
    im = im.crop((left, top, left + side, top + side))
    arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (side, side, 3):
        raise RuntimeError(f"crop shape {arr.shape}")
    return arr


def check_geometry():
    if (W.N, W.P, W.G, W.SCENE_SHIFT, W.DOT_SHIFT) != (32, 32, 8, 1, 2):
        raise RuntimeError("w7 geometry does not match the W8 spec")
    if CELL * W.N != SIDE:
        raise RuntimeError("cell grid does not fill 224")
    for value, expect in ((0, 0), (31, 0), (32, 1), (223, 6), (224, 7), (255, 7)):
        rgb = np.full((SIDE, SIDE, 3), value, dtype=np.uint8)
        got = grey_proxy(rgb)
        if int(got[0, 0]) != expect or not np.all(got == expect):
            raise RuntimeError(f"grey bin {value} -> {int(got[0, 0])}")
    rgb = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    rgb[7:14, 14:21] = 255
    placed = grey_proxy(rgb)
    if int(placed[1, 2]) != 7 or int(placed[0, 0]) != 0 or int(placed.sum()) != 7:
        raise RuntimeError("grey cell is not scene-aligned")
    rgb_f = np.zeros((SIDE, SIDE, 3), dtype=np.float32)
    for sy in range(W.N):
        for sx in range(W.N):
            block = rgb_f[sy * CELL:(sy + 1) * CELL, sx * CELL:(sx + 1) * CELL]
            block[..., 0] = (sy + 1) / 255.0
            block[..., 1] = (sx + 1) / 255.0
    seen = np.zeros((W.N, W.N), dtype=bool)
    seen[0, 0] = True
    seen[1, 2] = True
    out = reconstruct(rgb_f, seen)
    if not np.allclose(out[0:7, 0:7, 0], 1.0 / 255.0):
        raise RuntimeError("reconstruct cell 0,0")
    if not np.allclose(out[7:14, 14:21, 1], 3.0 / 255.0):
        raise RuntimeError("reconstruct cell 1,2")
    if not np.allclose(out[0, 7], IMAGENET_MEAN):
        raise RuntimeError("reconstruct unseen")
    log("smoke geometry ok")


def reproduce_w7_seed10():
    path = os.path.join(ROOT, "results", "w7", "results.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    want = {}
    for row in payload["rows"]:
        if int(row["seed"]) != 10:
            continue
        key = (int(row["opacity"]), int(row["life"]), row["shift"], row["policy"])
        want[key] = row
    scene = W.make_scene()
    shifts = {
        "parallax": (W.SCENE_SHIFT, W.DOT_SHIFT),
        "glued": (1, 1),
    }
    probes = W.probe_points(10)
    maxes = {field: 0.0 for field in ("probe", "view", "logo", "cover")}
    n = 0
    t0 = time.perf_counter()
    for n_opaque in (512, 768, 922):
        occ = W.occluder(W.make_field(10), n_opaque)
        for shift_name, (ss, sd) in shifts.items():
            paths = {
                policy: R.make_path(policy, 64, 10, scene, occ, ss, sd)
                for policy in POLICIES
            }
            for life in (25, 64):
                for policy in POLICIES:
                    got = W.evaluate(scene, occ, paths[policy][:life], probes, ss, sd)
                    ref = want[(n_opaque, life, shift_name, policy)]
                    for field in maxes:
                        err = abs(float(got[field]) - float(ref[field]))
                        if err > maxes[field]:
                            maxes[field] = err
                    if int(got["n_pix"]) != int(ref["n_pix"]):
                        raise RuntimeError(f"n_pix mismatch {(n_opaque, life, shift_name, policy)}")
                    n += 1
    wall = time.perf_counter() - t0
    worst = max(maxes.values())
    log(f"smoke w7 seed 10 rows {n} max_abs {worst:.3e} wall_s {wall:.3f}")
    for field, err in maxes.items():
        log(f"smoke w7 seed 10 max_abs_{field} {err:.3e}")
    if n != 60:
        raise RuntimeError(f"w7 seed 10 row count {n}")
    if worst > 1e-9:
        raise RuntimeError(f"w7 seed 10 drifted {worst:.3e}")
    return maxes, wall


def ensure_imagenette():
    root = os.path.join(ROOT, "data", "w8")
    dest = os.path.join(root, "imagenette2-160")
    val = os.path.join(dest, "val")
    os.makedirs(root, exist_ok=True)
    if os.path.isdir(val):
        return dest
    tar_path = os.path.join(root, "imagenette2-160.tgz")
    if not os.path.isfile(tar_path):
        log("download imagenette2-160")
        urllib.request.urlretrieve(IMAGENETTE_URL, tar_path)
    log("extract imagenette2-160")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root, filter="data")
    if not os.path.isdir(val):
        raise RuntimeError("imagenette val missing after extract")
    return dest


def load_pool(dest):
    val = os.path.join(dest, "val")
    wnids = [row[0] for row in SYNSETS]
    found = sorted(d for d in os.listdir(val) if os.path.isdir(os.path.join(val, d)))
    if found != wnids:
        raise RuntimeError(f"val folders {found}")
    images = np.zeros((len(wnids) * POOL, SIDE, SIDE, 3), dtype=np.uint8)
    files = []
    counts = []
    exif_n = 0
    t0 = time.perf_counter()
    for ci, wnid in enumerate(wnids):
        folder = os.path.join(val, wnid)
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".jpeg", ".jpg")))
        counts.append(len(names))
        if len(names) < POOL:
            raise RuntimeError(f"{wnid} has {len(names)} images")
        for j, name in enumerate(names[:POOL]):
            path = os.path.join(folder, name)
            with Image.open(path) as im:
                orient = 1
                if hasattr(im, "getexif"):
                    orient = im.getexif().get(274, 1)
                if orient not in (None, 0, 1):
                    exif_n += 1
                images[ci * POOL + j] = resize_crop(im)
            files.append(f"{wnid}/{name}")
    wall = time.perf_counter() - t0
    if len(files) != len(wnids) * POOL:
        raise RuntimeError("pool size")
    log(f"smoke pool {len(files)} per_class_available {counts} exif_orientation {exif_n} load_s {wall:.3f}")
    log("smoke files " + " ".join(files[:3]))
    return images, files, counts, exif_n, wall


def load_eye():
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = list(weights.meta["categories"])
    if len(categories) != 1000:
        raise RuntimeError("category count")
    for _wnid, name, index in SYNSETS:
        if categories[index] != name:
            raise RuntimeError(f"synset {index} is {categories[index]!r}")
    t0 = time.perf_counter()
    model = resnet18(weights=weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    n_param = sum(int(param.numel()) for param in model.parameters())
    wall = time.perf_counter() - t0
    if n_param != 11_689_512:
        raise RuntimeError(f"resnet18 params {n_param}")
    if model.training:
        raise RuntimeError("eye left train mode")
    log(f"smoke eye resnet18 params {n_param} load_s {wall:.3f}")
    return model, wall


def forward_array(model, hwc):
    arr = (hwc - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (0, 3, 1, 2))
    tensor = torch.from_numpy(np.ascontiguousarray(chw))
    with torch.inference_mode():
        return model(tensor)


def scores_from_logits(logits, labels):
    labels = torch.as_tensor(labels, dtype=torch.long)
    logits10 = logits.index_select(1, SYN_INDEX)
    logp = torch.log_softmax(logits10, dim=1)
    pred = torch.argmax(logits10, dim=1)
    nll = -logp.gather(1, labels.view(-1, 1)).squeeze(1)
    prob = torch.softmax(logits10, dim=1)
    top_p, top_i = torch.max(prob, dim=1)
    return {
        "correct": pred.eq(labels).to(torch.int32).cpu().numpy(),
        "logloss": nll.detach().cpu().numpy().astype(np.float64),
        "top": top_i.cpu().numpy().astype(np.int32),
        "top_p": top_p.detach().cpu().numpy().astype(np.float64),
        "pred1000": torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32),
    }


def score_full(model, images, batch=16):
    n = int(images.shape[0])
    correct = np.zeros(n, dtype=np.int32)
    logloss = np.zeros(n, dtype=np.float64)
    pred1000 = np.zeros(n, dtype=np.int32)
    labels = np.arange(n) // POOL
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        hwc = images[start:stop].astype(np.float32) * (1.0 / 255.0)
        got = scores_from_logits(forward_array(model, hwc), labels[start:stop])
        correct[start:stop] = got["correct"]
        logloss[start:stop] = got["logloss"]
        pred1000[start:stop] = got["pred1000"]
    wall = time.perf_counter() - t0
    true1000 = np.array([SYNSETS[int(y)][2] for y in labels], dtype=np.int32)
    acc10 = float(correct.mean())
    acc1000 = float((pred1000 == true1000).mean())
    log(f"smoke_full_accuracy_10way {acc10:.6f}")
    log(f"smoke_full_accuracy_1000way {acc1000:.6f}")
    log(f"smoke_full_logloss_nats {float(logloss.mean()):.6f}")
    log(f"smoke_full_eye_s {wall:.3f}")
    return {
        "correct": correct,
        "logloss": logloss,
        "acc10": acc10,
        "acc1000": acc1000,
        "wall_s": wall,
    }


def score_none(model, n):
    hwc = np.empty((1, SIDE, SIDE, 3), dtype=np.float32)
    hwc[...] = IMAGENET_MEAN
    logits = forward_array(model, hwc)
    labels = np.arange(n) // POOL
    # One constant logit vector. Repeat it across the pool, then score each label.
    tiled = logits.repeat(n, 1)
    got = scores_from_logits(tiled, labels)
    top = int(got["top"][0])
    if not np.all(got["top"] == top):
        raise RuntimeError("none image was not constant")
    return {
        "correct": got["correct"].astype(np.int32),
        "logloss": got["logloss"].astype(np.float64),
        "top": top,
        "top_p": float(got["top_p"][0]),
        "logits10": torch.log_softmax(logits.index_select(1, SYN_INDEX), dim=1).exp().view(-1).detach().cpu().numpy().astype(np.float64),
    }


def score_image(model, rgb_u8, label, seed, demo):
    grey = grey_proxy(rgb_u8)
    occ = W.occluder(W.make_field(seed), OPACITY)
    t0 = time.perf_counter()
    paths = {
        policy: R.make_path(policy, 64, seed, grey, occ, W.SCENE_SHIFT, W.DOT_SHIFT)
        for policy in POLICIES
    }
    policy_s = time.perf_counter() - t0
    for policy, path in paths.items():
        if len(path) != 64:
            raise RuntimeError(f"{policy} path length {len(path)}")
    rgb_f = rgb_u8.astype(np.float32) * (1.0 / 255.0)
    order = []
    covers = {}
    recons = []
    for policy in POLICIES:
        levels = DEMO_LS[:1] + LS if demo and policy in DEMO_POLICIES else LS
        # DEMO_LS[0] is 1. The scored lives stay LS. L=1 is only for the strip.
        if levels[0] == 1 and 1 in LS:
            raise RuntimeError("L=1 collided with a scored life")
        for life in levels:
            seen = seen_of(grey, occ, paths[policy][:life])
            cover = float(seen.mean())
            if policy == "still" and abs(cover - (W.N * W.N - OPACITY) / float(W.N * W.N)) > 1e-12:
                raise RuntimeError(f"still coverage {cover}")
            covers[(policy, life)] = cover
            recons.append(reconstruct(rgb_f, seen))
            order.append((policy, life))
    t1 = time.perf_counter()
    got = scores_from_logits(forward_array(model, np.stack(recons, axis=0)), np.full(len(order), int(label)))
    eye_s = time.perf_counter() - t1
    scored = {}
    demo_rows = {}
    for i, (policy, life) in enumerate(order):
        row = {
            "correct": int(got["correct"][i]),
            "logloss": float(got["logloss"][i]),
            "coverage": covers[(policy, life)],
            "top": int(got["top"][i]),
            "top_p": float(got["top_p"][i]),
        }
        if life in LS:
            scored[(policy, life)] = row
        if demo and policy in DEMO_POLICIES and life in DEMO_LS:
            demo_rows[(policy, life)] = dict(row, u8=to_u8(recons[i]))
    return {
        "scored": scored,
        "demo": demo_rows,
        "grey": grey,
        "path0": paths["curious_prog"][:10],
        "policy_s": policy_s,
        "eye_s": eye_s,
    }


def load_font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.isfile(path):
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_strip(path, panels, captions):
    gap = 6
    bar = 36
    n = len(panels)
    canvas = Image.new("RGB", (n * SIDE + (n - 1) * gap, SIDE + bar), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    for i, (panel, cap) in enumerate(zip(panels, captions)):
        x = i * (SIDE + gap)
        canvas.paste(Image.fromarray(panel), (x, 0))
        draw.text((x + 4, SIDE + 8), cap, fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def write_demos(packet_by_index):
    out_dir = os.path.join(ROOT, "internal", "w8_demo")
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for index in DEMO_INDEX:
        packet = packet_by_index[index]
        for policy in DEMO_POLICIES:
            bits = []
            panels = []
            captions = []
            for life in DEMO_LS:
                row = packet["demo"][(policy, life)]
                name = slug(SYNSETS[row["top"]][1])
                bits.append(f"L{life}_{name}_{row['top_p']:.4f}")
                panels.append(row["u8"])
                captions.append(f"L{life} {SYNSETS[row['top']][1]} {row['top_p']:.2f}")
            fname = f"img{index:03d}_{policy}_" + "_".join(bits) + ".png"
            save_strip(os.path.join(out_dir, fname), panels, captions)
            records.append({
                "index": int(index),
                "policy": policy,
                "file": fname,
                "L": [
                    {
                        "L": int(life),
                        "top": int(packet["demo"][(policy, life)]["top"]),
                        "top_name": SYNSETS[packet["demo"][(policy, life)]["top"]][1],
                        "top_p": float(packet["demo"][(policy, life)]["top_p"]),
                    }
                    for life in DEMO_LS
                ],
            })
    return records


def selected_indices(per_class):
    idx = []
    for c in range(len(SYNSETS)):
        base = c * POOL
        idx.extend(range(base, base + int(per_class)))
    return idx


def paired_ci(diff):
    diff = np.asarray(diff, dtype=np.float64)
    n = int(diff.shape[0])
    rng = np.random.default_rng(0)
    draw = rng.integers(0, n, size=(N_BOOT, n))
    means = diff[draw].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return float(diff.mean()), float(lo), float(hi)


def summarise(indices, full, none, store):
    labels = [i // POOL for i in indices]
    summary = {}
    contrasts = {}
    correct = {policy: {} for policy in POLICIES}
    logloss = {policy: {} for policy in POLICIES}
    coverage = {policy: {} for policy in POLICIES}
    for life in LS:
        block = {}
        for policy in POLICIES:
            c = np.array([store[i]["scored"][(policy, life)]["correct"] for i in indices], dtype=np.float64)
            ll = np.array([store[i]["scored"][(policy, life)]["logloss"] for i in indices], dtype=np.float64)
            cov = np.array([store[i]["scored"][(policy, life)]["coverage"] for i in indices], dtype=np.float64)
            correct[policy][str(life)] = c.astype(np.int32).tolist()
            logloss[policy][str(life)] = ll.tolist()
            coverage[policy][str(life)] = cov.tolist()
            block[policy] = {
                "accuracy": float(c.mean()),
                "n_correct": int(c.sum()),
                "logloss_nats": float(ll.mean()),
                "coverage": float(cov.mean()),
            }
        full_c = full["correct"][indices].astype(np.float64)
        full_ll = full["logloss"][indices].astype(np.float64)
        none_c = none["correct"][indices].astype(np.float64)
        none_ll = none["logloss"][indices].astype(np.float64)
        block["full"] = {
            "accuracy": float(full_c.mean()),
            "n_correct": int(full_c.sum()),
            "logloss_nats": float(full_ll.mean()),
            "coverage": 1.0,
        }
        block["none"] = {
            "accuracy": float(none_c.mean()),
            "n_correct": int(none_c.sum()),
            "logloss_nats": float(none_ll.mean()),
            "coverage": 0.0,
        }
        summary[str(life)] = block
        contrast = {}
        pairs = (
            ("random-still", "random", "still"),
            ("curious_prog-random", "curious_prog", "random"),
        )
        for name, left, right in pairs:
            acc = np.array([
                store[i]["scored"][(left, life)]["correct"] - store[i]["scored"][(right, life)]["correct"]
                for i in indices
            ], dtype=np.float64)
            ll = np.array([
                store[i]["scored"][(left, life)]["logloss"] - store[i]["scored"][(right, life)]["logloss"]
                for i in indices
            ], dtype=np.float64)
            acc_mean, acc_lo, acc_hi = paired_ci(acc)
            ll_mean, ll_lo, ll_hi = paired_ci(ll)
            contrast[name] = {
                "accuracy": acc_mean,
                "accuracy_ci95_lo": acc_lo,
                "accuracy_ci95_hi": acc_hi,
                "logloss_nats": ll_mean,
                "logloss_ci95_lo": ll_lo,
                "logloss_ci95_hi": ll_hi,
            }
        contrasts[str(life)] = contrast
    return {
        "labels": labels,
        "summary": summary,
        "contrasts": contrasts,
        "correct": correct,
        "logloss": logloss,
        "coverage": coverage,
        "full_correct": full["correct"][indices].astype(int).tolist(),
        "full_logloss": full["logloss"][indices].astype(float).tolist(),
        "none_correct": none["correct"][indices].astype(int).tolist(),
        "none_logloss": none["logloss"][indices].astype(float).tolist(),
    }


def render(payload):
    lines = []
    lines.append(
        f"n_images {payload['n_images']} per_class {payload['per_class']} "
        f"pool_per_class {POOL} opacity {OPACITY}"
    )
    lines.append("")
    order = ("full", "none") + tuple(POLICIES)
    for life in (16, 4, 8, 32, 64):
        block = payload["summary"][str(life)]
        lines.append(f"L={life}")
        lines.append("")
        lines.append("| policy | accuracy | n_correct | logloss_nats | coverage |")
        lines.append("|---|---:|---:|---:|---:|")
        for policy in order:
            row = block[policy]
            lines.append(
                f"| {policy} | {f6(row['accuracy'])} | {row['n_correct']} | "
                f"{f6(row['logloss_nats'])} | {f6(row['coverage'])} |"
            )
        lines.append("")
        lines.append("| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name in ("random-still", "curious_prog-random"):
            row = payload["contrasts"][str(life)][name]
            lines.append(
                f"| {name} | {f6(row['accuracy'])} | {f6(row['accuracy_ci95_lo'])} | "
                f"{f6(row['accuracy_ci95_hi'])} | {f6(row['logloss_nats'])} | "
                f"{f6(row['logloss_ci95_lo'])} | {f6(row['logloss_ci95_hi'])} |"
            )
        lines.append("")
        lines.append(f"L={life} full_minus_still_accuracy {f6(block['full']['accuracy'] - block['still']['accuracy'])}")
        lines.append("")
    lines.append("synsets")
    lines.append("")
    lines.append("| order | wnid | imagenet_index | category |")
    lines.append("|---:|---|---:|---|")
    for i, (wnid, name, index) in enumerate(SYNSETS):
        lines.append(f"| {i} | {wnid} | {index} | {name} |")
    lines.append("")
    lines.append("source")
    lines.append("")
    for key, value in payload["source"].items():
        lines.append(f"{key} {value}")
    lines.append("")
    lines.append("walls")
    lines.append("")
    for key, value in payload["walls"].items():
        if isinstance(value, float):
            lines.append(f"{key} {value:.3f}")
        else:
            lines.append(f"{key} {value}")
    lines.append("")
    lines.append("w7 seed 10")
    lines.append("")
    for key, value in payload["w7_seed10"].items():
        if isinstance(value, float):
            lines.append(f"{key} {value:.3e}")
        else:
            lines.append(f"{key} {value}")
    lines.append("")
    lines.append("image 0 grey")
    lines.append("")
    for row in payload["image0"]["grey"]:
        lines.append(" ".join(str(int(v)) for v in row))
    lines.append("")
    vps = " ".join(f"({x},{y})" for x, y in payload["image0"]["curious_prog_first10"])
    lines.append(f"image 0 curious_prog first 10 viewpoints {vps}")
    lines.append("")
    lines.append("demos")
    lines.append("")
    for row in payload["demos"]:
        lines.append(row["file"])
    lines.append("")
    return "\n".join(lines)


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def store_keep(packet):
    """Keep scalars, and the three demo strips' panels. Drop the grey raster."""
    return {
        "scored": packet["scored"],
        "demo": packet["demo"],
        "policy_s": packet["policy_s"],
        "eye_s": packet["eye_s"],
    }


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("all", "repro"):
        raise SystemExit("usage: w8_camera.py [all|repro]")
    t0 = time.perf_counter()
    load = load_average()
    log(f"load_average {load:.2f}")
    if load >= 1.5:
        raise SystemExit(f"load {load:.2f} >= 1.5; this run belongs on a Kaggle kernel")
    check_geometry()
    maxes, w7_s = reproduce_w7_seed10()
    if stage == "repro":
        log(f"repro wall_s {time.perf_counter() - t0:.3f}")
        return
    dest = ensure_imagenette()
    images, files, counts, exif_n, load_s = load_pool(dest)
    model, model_s = load_eye()
    full = score_full(model, images)
    if full["acc10"] < 0.80:
        raise SystemExit("stop accuracy below 0.80")
    none = score_none(model, images.shape[0])
    grey0 = grey_proxy(images[0])
    occ0 = W.occluder(W.make_field(0), OPACITY)
    t_path = time.perf_counter()
    path0 = R.make_path("curious_prog", 64, 0, grey0, occ0, W.SCENE_SHIFT, W.DOT_SHIFT)[:10]
    image0_s = time.perf_counter() - t_path
    log("smoke image 0 grey")
    for row in grey0:
        log(" ".join(str(int(v)) for v in row))
    log("smoke image 0 curious_prog first 10 " + " ".join(f"({x},{y})" for x, y in path0))
    store = {}
    t20 = time.perf_counter()
    for index in range(20):
        packet = score_image(model, images[index], index // POOL, index, index in DEMO_INDEX)
        store[index] = store_keep(packet)
        log(
            f"smoke time image {index} policy_s {packet['policy_s']:.3f} "
            f"eye_s {packet['eye_s']:.3f}"
        )
    time20 = time.perf_counter() - t20
    per = time20 / 20.0
    elapsed = time.perf_counter() - t0
    est_500 = elapsed + per * (500 - 20)
    est_300 = elapsed + per * (300 - 20)
    log(f"smoke time20_s {time20:.3f} per_image_s {per:.3f}")
    log(f"smoke estimate_500_s {est_500:.3f} estimate_300_s {est_300:.3f} elapsed_s {elapsed:.3f}")
    per_class = 50
    if est_500 > BUDGET_S:
        per_class = 30
        log("smoke scored set first 30 per class")
    else:
        log("smoke scored set first 50 per class")
    indices = selected_indices(per_class)
    t_rest = time.perf_counter()
    todo = [i for i in indices if i not in store]
    for n_done, index in enumerate(todo, start=1):
        packet = score_image(model, images[index], index // POOL, index, index in DEMO_INDEX)
        store[index] = store_keep(packet)
        if n_done == 1 or n_done % 10 == 0 or n_done == len(todo):
            spent = time.perf_counter() - t_rest
            rate = spent / n_done
            eta = rate * (len(todo) - n_done)
            log(f"image {index} done {n_done}/{len(todo)} elapsed_s {spent:.1f} eta_s {eta:.1f}")
    rest_s = time.perf_counter() - t_rest
    if any(i not in store for i in indices):
        raise RuntimeError("missing image score")
    labels = [i // POOL for i in indices]
    if sorted(set(np.bincount(labels, minlength=10))) != [per_class]:
        raise RuntimeError("class counts")
    t_demo = time.perf_counter()
    demos = write_demos(store)
    demo_s = time.perf_counter() - t_demo
    packed = summarise(indices, full, none, store)
    wall = time.perf_counter() - t0
    file_text = "\n".join(files[i] for i in indices)
    source = {
        "dataset": "imagenette2-160",
        "split": "val",
        "order": "sorted filename within wnid, wnid order is the synset table",
        "pool_per_class": POOL,
        "scored_per_class": per_class,
        "seed": "pool index, 50 per class, first scored_per_class of each class",
        "resize": "short side 224 bilinear, centre crop 224, offset (size-224)//2",
        "weight_preset_resize_256": "not used",
        "exif_transpose": "no",
        "exif_orientation_not_1": exif_n,
        "pool_counts": ",".join(str(c) for c in counts),
        "grey": "integer BT.601 (299R+587G+114B)/1000, mean of 7x7, floor(mean*8/256)",
        "opacity": OPACITY,
        "scene_shift": W.SCENE_SHIFT,
        "dot_shift": W.DOT_SHIFT,
        "eye": "torchvision resnet18 IMAGENET1K_V1 frozen eval",
        "softmax": "10 logits, synset table order",
        "logloss": "nats",
        "normalize_mean": "0.485 0.456 0.406",
        "normalize_std": "0.229 0.224 0.225",
        "unseen": "imagenet mean colour",
        "bootstrap": "10000 paired percentile, numpy quantile linear, default_rng(0) per contrast",
        "runner": "vps",
        "device": "cpu",
        "torch_threads": 2,
        "kernel_id": "none",
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "files_sha256": hashlib.sha256(file_text.encode("utf-8")).hexdigest(),
        "smoke_full_accuracy_10way": f6(full["acc10"]),
        "smoke_full_accuracy_1000way": f6(full["acc1000"]),
        "load_average_start": f"{load:.2f}",
    }
    walls = {
        "w7_s": w7_s,
        "load_s": load_s,
        "model_s": model_s,
        "full_eye_s": full["wall_s"],
        "image0_s": image0_s,
        "time20_s": time20,
        "per_image_s": per,
        "estimate_500_s": est_500,
        "estimate_300_s": est_300,
        "rest_s": rest_s,
        "demo_s": demo_s,
        "wall_s": wall,
    }
    w7_out = {f"max_abs_{k}": float(v) for k, v in maxes.items()}
    w7_out["rows"] = 60
    w7_out["wall_s"] = w7_s
    image0 = {
        "grey": grey0.astype(int).tolist(),
        "curious_prog_first10": [[int(x), int(y)] for x, y in path0],
    }
    payload = {
        "n_images": len(indices),
        "per_class": per_class,
        "pool_per_class": POOL,
        "indices": indices,
        "files": [files[i] for i in indices],
        "pool_files": files,
        "L": list(LS),
        "policies": list(POLICIES),
        "summary": packed["summary"],
        "contrasts": packed["contrasts"],
        "correct": packed["correct"],
        "logloss": packed["logloss"],
        "coverage": packed["coverage"],
        "full_correct": packed["full_correct"],
        "full_logloss": packed["full_logloss"],
        "none_correct": packed["none_correct"],
        "none_logloss": packed["none_logloss"],
        "none_top": int(none["top"]),
        "none_top_name": SYNSETS[none["top"]][1],
        "none_top_p": float(none["top_p"]),
        "none_prob10": none["logits10"].tolist(),
        "labels": packed["labels"],
        "synsets": [
            {"order": i, "wnid": wnid, "imagenet_index": index, "category": name}
            for i, (wnid, name, index) in enumerate(SYNSETS)
        ],
        "source": source,
        "walls": walls,
        "w7_seed10": w7_out,
        "image0": image0,
        "demos": demos,
    }
    text = render(payload)
    out_dir = os.path.join(ROOT, "results", "w8")
    os.makedirs(out_dir, exist_ok=True)
    atomic_write(os.path.join(out_dir, "RESULTS.md"), text)
    atomic_write(
        os.path.join(out_dir, "results.json"),
        json.dumps(payload, indent=2) + "\n",
    )
    b = payload["summary"]["16"]
    c = payload["contrasts"]["16"]
    log(
        "L16 acc "
        + " ".join(f"{name} {f6(b[name]['accuracy'])}" for name in ("full", "none") + tuple(POLICIES))
    )
    log(
        "L16 random-still acc "
        f"{f6(c['random-still']['accuracy'])} "
        f"[{f6(c['random-still']['accuracy_ci95_lo'])}, {f6(c['random-still']['accuracy_ci95_hi'])}]"
    )
    log(
        "L16 curious_prog-random acc "
        f"{f6(c['curious_prog-random']['accuracy'])} "
        f"[{f6(c['curious_prog-random']['accuracy_ci95_lo'])}, {f6(c['curious_prog-random']['accuracy_ci95_hi'])}] "
        "logloss "
        f"{f6(c['curious_prog-random']['logloss_nats'])} "
        f"[{f6(c['curious_prog-random']['logloss_ci95_lo'])}, {f6(c['curious_prog-random']['logloss_ci95_hi'])}]"
    )
    log(f"wall_s {wall:.3f} n {len(indices)} per_class {per_class}")


if __name__ == "__main__":
    main()
