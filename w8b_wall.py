"""W8-1: a curious camera in front of a wall of four pictures.

Imports w8_camera, w7_world and w7_run. Does not modify them.

Clock: decision t = 0..1023. Before the glance, a fast panel replaces its image
when t > 0 and t % 32 == 0, and a slow panel when t % 128 == 0. The glance
then uses that image. Scores land after t+1 in {16, 32, ..., 1024}, on the
image the glance just used.

Curious policies are the W7 region bandit on 4 panels x 16 regions. The sort
key is W7's EMA (beta = w7_world.BETA). curious_prog and curious_surp rank by
the aged key v + (v_max - v) (1 - exp(-age / 256)); the stored EMA is not
aged. Unvisited progress arms stay at +inf. Unvisited surprise arms stay at
log2(8). Neither prior is aged. v_max is the largest per-glance reward so far.
Age is completed decisions minus the decision that arm was last observed.
No epsilon. Walk is w7_run.step_toward until the viewpoint is inside the
region; the entering step is travel. The next 8 decisions are lawn_step
inside the region. Round-robin takes W7's sweep lattice, 8 glances per panel,
cursor kept per panel.
"""
import hashlib
import json
import math
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np

import w7_run as R
import w7_world as W
import w8_camera as cam

ROOT = cam.ROOT
OUT_DIR = os.path.join(ROOT, "results", "w8b")
DEMO_DIR = os.path.join(ROOT, "internal", "w8_demo", "wall")
PARTIAL = os.path.join(OUT_DIR, "partial.json")

# Pre-registered bars. Do not change.
BAR_ACC = 0.03
BAR_TV_PROG = 0.15
BAR_TV_SURP = 0.35
N_LIVES_FULL = 20
N_LIVES_SHORT = 12
BUDGET_S = 90 * 60

N_DECISIONS = 1024
N_PANELS = 4
N_REGIONS = 16
N_ARMS = N_PANELS * N_REGIONS
DWELL = 8
TAU = 256
CHECK_EVERY = 16
DEMO_EVERY = 128
FAST_EVERY = 32
SLOW_EVERY = 128
N_BOOT = cam.N_BOOT

KINDS = ("static", "fast", "slow", "tv")
SCORED = ("static", "fast", "slow")
POLICIES = (
    "random",
    "round_robin",
    "curious_prog",
    "curious_surp",
    "curious_prog_noage",
)
DEMO_POLICIES = ("curious_prog", "round_robin")
PERIOD = {"static": 0, "fast": FAST_EVERY, "slow": SLOW_EVERY, "tv": 0}

CURIOUS = {
    "curious_prog": ("prog", True),
    "curious_surp": ("surp", True),
    "curious_prog_noage": ("prog", False),
}


def log(msg):
    print(msg, flush=True)


def f6(x):
    return f"{float(x):.6f}"


def code_sha():
    path = os.path.abspath(__file__)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_average():
    with open("/proc/loadavg", encoding="utf-8") as f:
        return float(f.read().split()[0])


def aged_value(v, age, v_max, tau=TAU):
    """v + (v_max - v) * (1 - exp(-age / tau)). Age 0 returns v."""
    gap = float(v_max) - float(v)
    return float(v) + gap * (1.0 - math.exp(-float(age) / float(tau)))


def make_world(life, n_images=500):
    """Panel kinds at default_rng(9800+life). Image order at default_rng(9900+life)."""
    kind_rng = np.random.default_rng(9800 + int(life))
    kinds = tuple(KINDS[int(i)] for i in kind_rng.permutation(len(KINDS)))
    if sorted(kinds) != sorted(KINDS):
        raise RuntimeError("panel kinds are not one of each")
    img_rng = np.random.default_rng(9900 + int(life))
    pool = [int(i) for i in img_rng.permutation(int(n_images))]
    schedule = np.empty((N_PANELS, N_DECISIONS), dtype=np.int32)
    cursor = 0
    current = [0] * N_PANELS
    for panel in range(N_PANELS):
        current[panel] = pool[cursor]
        cursor += 1
    for t in range(N_DECISIONS):
        if t > 0:
            for panel in range(N_PANELS):
                per = PERIOD[kinds[panel]]
                if per and t % per == 0:
                    current[panel] = pool[cursor]
                    cursor += 1
        schedule[:, t] = current
    if cursor != 42:
        raise RuntimeError(f"life {life} used {cursor} images")
    if len(np.unique(schedule)) != 42:
        raise RuntimeError("image ids repeated across the wall")
    return kinds, schedule


def centers():
    out = []
    side = W.P // W.REG
    for region in range(N_REGIONS):
        ry, rx = divmod(region, side)
        out.append((rx * W.REG + W.REG // 2, ry * W.REG + W.REG // 2))
    return out


CENTERS = centers()
DY = np.arange(cam.CELL, dtype=np.int32)


def build_maps():
    sy = np.empty((W.P, W.P, W.N, W.N), dtype=np.int16)
    sx = np.empty_like(sy)
    oy = np.empty_like(sy)
    ox = np.empty_like(sy)
    for py in range(W.P):
        for px in range(W.P):
            sx_i, sy_i, ox_i, oy_i = W.maps(px, py, W.SCENE_SHIFT, W.DOT_SHIFT)
            sy[py, px] = sy_i
            sx[py, px] = sx_i
            oy[py, px] = oy_i
            ox[py, px] = ox_i
    return sy, sx, oy, ox


def glance(maps, scene, occ, px, py):
    sy_m, sx_m, oy_m, ox_m = maps
    sy = sy_m[py, px]
    sx = sx_m[py, px]
    oy = oy_m[py, px]
    ox = ox_m[py, px]
    vis = ~occ[oy, ox]
    grey = scene[sy, sx]
    return grey, vis, sx, sy


def paint_image(canvas, seen, image, ys, xs):
    if ys.size == 0:
        return
    y = ys.astype(np.int32)[:, None] * cam.CELL + DY
    x = xs.astype(np.int32)[:, None] * cam.CELL + DY
    canvas[y[:, :, None], x[:, None, :], :] = image[y[:, :, None], x[:, None, :], :]
    seen[ys, xs] = True


def paint_colors(canvas, seen, colors, ys, xs):
    if ys.size == 0:
        return
    y = ys.astype(np.int32)[:, None] * cam.CELL + DY
    x = xs.astype(np.int32)[:, None] * cam.CELL + DY
    canvas[y[:, :, None], x[:, None, :], :] = colors[:, None, None, :]
    seen[ys, xs] = True


def tv_grey(colors):
    lum = (
        299 * colors[:, 0].astype(np.int32)
        + 587 * colors[:, 1].astype(np.int32)
        + 114 * colors[:, 2].astype(np.int32)
    )
    return (lum * 8) // 256000


def display_u8(canvas, seen):
    rgb_f = canvas.astype(np.float32) * (1.0 / 255.0)
    return cam.to_u8(cam.reconstruct(rgb_f, seen))


def recon_f(canvas, seen):
    rgb_f = canvas.astype(np.float32) * (1.0 / 255.0)
    return cam.reconstruct(rgb_f, seen)


def step_action(px, py, action):
    if action == 0:
        return (px + 1) % W.P, py
    if action == 1:
        return (px - 1) % W.P, py
    if action == 2:
        return px, (py + 1) % W.P
    if action == 3:
        return px, (py - 1) % W.P
    raise RuntimeError("action")


def sweep_viewpoints(n_panels, budget):
    """Round-robin chunks of W7's sweep. One panel keeps the continuing lattice."""
    sweep = R.path_sweep(int(budget))
    cursor = [0] * int(n_panels)
    panel = 0
    left = DWELL
    out = []
    for _ in range(int(budget)):
        if left == 0:
            panel = (panel + 1) % int(n_panels)
            left = DWELL
        x, y = sweep[cursor[panel]]
        cursor[panel] += 1
        left -= 1
        out.append((panel, int(x), int(y)))
    return out


def choose_arm(ema, visits, last_end, v_max, t, use_age):
    score = np.array(ema, dtype=np.float64, copy=True)
    if use_age and v_max is not None:
        known = (last_end >= 0) & np.isfinite(score)
        if np.any(known):
            age = (int(t) - last_end[known]).astype(np.float64)
            raw = score[known]
            score[known] = raw + (float(v_max) - raw) * (1.0 - np.exp(-age / float(TAU)))
    order = np.lexsort((np.arange(score.size), visits, -score))
    return int(order[0])


def update_ema(mode, ema, visits, arm, reward):
    if mode == "prog" and int(visits[arm]) == 0:
        ema[arm] = float(reward)
    else:
        ema[arm] = (1.0 - W.BETA) * float(ema[arm]) + W.BETA * float(reward)
    visits[arm] += 1


class LifeRun:
    def __init__(self, life, kinds, schedule, pack, policy, demo):
        self.life = int(life)
        self.kinds = kinds
        self.schedule = schedule
        self.pack = pack
        self.policy = policy
        self.demo = bool(demo)
        self.panel_of = {kind: kinds.index(kind) for kind in KINDS}
        self.canvas = np.zeros((N_PANELS, cam.SIDE, cam.SIDE, 3), dtype=np.uint8)
        self.seen = np.zeros((N_PANELS, W.N, W.N), dtype=bool)
        self.px = [0] * N_PANELS
        self.py = [0] * N_PANELS
        self.tv_rng = np.random.default_rng(9700 + int(life))
        self.counts = None
        self.reservoirs = None
        self.mode = None
        self.use_age = False
        self.ema = None
        self.visits = None
        self.last_end = None
        self.v_max = None
        self.share_n = {kind: 0 for kind in KINDS}
        self.recons = []
        self.labels = []
        self.demo_snaps = []
        self.panel_seq = []
        self.n_choices = 0
        self.n_walk = 0
        self.n_dwell = 0
        self.first_pass_t = None
        if policy in CURIOUS:
            mode, use_age = CURIOUS[policy]
            self.mode = mode
            self.use_age = use_age
            self.counts = np.zeros((N_PANELS, W.N, W.N, W.G), dtype=np.int32)
            if mode == "prog":
                self.reservoirs = [W.GlanceReservoir(int(life)) for _ in range(N_PANELS)]
                self.ema = np.full(N_ARMS, np.inf, dtype=np.float64)
            else:
                self.ema = np.full(N_ARMS, np.log2(W.G), dtype=np.float64)
            self.visits = np.zeros(N_ARMS, dtype=np.int32)
            self.last_end = np.full(N_ARMS, -1, dtype=np.int32)

    def observe(self, panel, t):
        kind = self.kinds[panel]
        index = int(self.schedule[panel, t])
        image = self.pack["images"][index]
        scene = self.pack["greys"][index]
        occ = self.pack["occs"][index]
        px, py = int(self.px[panel]), int(self.py[panel])
        grey, vis, sx, sy = glance(self.pack["maps"], scene, occ, px, py)
        ys = sy[vis]
        xs = sx[vis]
        if kind == "tv":
            n = int(ys.size)
            if n:
                colors = self.tv_rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
                paint_colors(self.canvas[panel], self.seen[panel], colors, ys, xs)
                grey = np.array(grey, copy=True)
                grey[vis] = tv_grey(colors).astype(np.int16)
        else:
            paint_image(self.canvas[panel], self.seen[panel], image, ys, xs)
        reward = 0.0
        if self.mode is not None:
            reservoir = None if self.reservoirs is None else self.reservoirs[panel]
            reward = float(R._reward(
                self.mode, self.counts[panel], reservoir, grey, vis, sx, sy
            ))
            if not math.isfinite(reward):
                raise RuntimeError("nonfinite reward")
            landed = panel * N_REGIONS + R.region_id(px, py)
            if self.v_max is None or reward > self.v_max:
                self.v_max = float(reward)
            update_ema(self.mode, self.ema, self.visits, landed, reward)
            self.last_end[landed] = int(t) + 1
            if self.first_pass_t is None and int(self.visits.min()) >= 1:
                self.first_pass_t = int(t) + 1
        self.share_n[kind] += 1
        self.panel_seq.append(int(panel))
        completed = int(t) + 1
        if completed % CHECK_EVERY == 0:
            self.checkpoint(completed)
        return reward

    def checkpoint(self, completed):
        base = len(self.recons)
        for kind in SCORED:
            panel = self.panel_of[kind]
            index = int(self.schedule[panel, completed - 1])
            self.recons.append(recon_f(self.canvas[panel], self.seen[panel]))
            self.labels.append(int(index) // cam.POOL)
        if self.demo and self.policy in DEMO_POLICIES and completed % DEMO_EVERY == 0:
            self.demo_snaps.append({
                "t": int(completed),
                "images": [display_u8(self.canvas[p], self.seen[p]) for p in range(N_PANELS)],
                "score_at": {kind: base + i for i, kind in enumerate(SCORED)},
                "labels": {
                    kind: int(self.schedule[self.panel_of[kind], completed - 1]) // cam.POOL
                    for kind in SCORED
                },
            })

    def finish(self, model):
        n_checks = N_DECISIONS // CHECK_EVERY
        if len(self.recons) != n_checks * len(SCORED):
            raise RuntimeError(f"checks {len(self.recons)}")
        if len(self.panel_seq) != N_DECISIONS:
            raise RuntimeError("panel sequence length")
        if sum(self.share_n.values()) != N_DECISIONS:
            raise RuntimeError("share count")
        if self.mode == "prog":
            for panel in range(N_PANELS):
                glances = int(self.share_n[self.kinds[panel]])
                reservoir = self.reservoirs[panel]
                if reservoir.n_seen != glances:
                    raise RuntimeError("reservoir missed a glance")
                if len(reservoir.slots) != min(glances, reservoir.cap):
                    raise RuntimeError("reservoir size")
        got = score_recons(model, self.recons, self.labels)
        correct = got["correct"].reshape(n_checks, len(SCORED))
        logloss = got["logloss"].reshape(n_checks, len(SCORED))
        top = got["top"].reshape(n_checks, len(SCORED))
        top_p = got["top_p"].reshape(n_checks, len(SCORED))
        demo_files = []
        if self.demo_snaps:
            demo_files = save_demos(
                self.policy, self.kinds, self.demo_snaps, top, top_p
            )
        del self.recons
        del self.canvas
        return {
            "accuracy": float(correct.mean()),
            "logloss": float(logloss.mean()),
            "panel_accuracy": {
                kind: float(correct[:, i].mean()) for i, kind in enumerate(SCORED)
            },
            "panel_logloss": {
                kind: float(logloss[:, i].mean()) for i, kind in enumerate(SCORED)
            },
            "checks": {
                kind: {
                    "correct": correct[:, i].astype(int).tolist(),
                    "logloss": logloss[:, i].astype(float).tolist(),
                }
                for i, kind in enumerate(SCORED)
            },
            "share": {kind: self.share_n[kind] / float(N_DECISIONS) for kind in KINDS},
            "share_n": dict(self.share_n),
            "n_choices": int(self.n_choices),
            "n_walk": int(self.n_walk),
            "n_dwell": int(self.n_dwell),
            "first_pass_t": self.first_pass_t,
            "demo_files": demo_files,
            "panel_seq": self.panel_seq,
        }


def run_fixed(run):
    if run.policy == "round_robin":
        steps = sweep_viewpoints(N_PANELS, N_DECISIONS)
        for t, (panel, x, y) in enumerate(steps):
            run.px[panel], run.py[panel] = int(x), int(y)
            run.observe(panel, t)
        return
    if run.policy != "random":
        raise RuntimeError(run.policy)
    rng = np.random.default_rng(8000 + run.life)
    for t in range(N_DECISIONS):
        panel = int(rng.integers(0, N_PANELS))
        action = int(rng.integers(0, 4))
        run.px[panel], run.py[panel] = step_action(run.px[panel], run.py[panel], action)
        run.observe(panel, t)


def run_curious(run):
    phase = None
    arm = None
    dwell_left = 0
    walk_steps = 0
    region = None
    for t in range(N_DECISIONS):
        if phase is None:
            arm = choose_arm(
                run.ema, run.visits, run.last_end, run.v_max, t, run.use_age
            )
            run.n_choices += 1
            phase = "walk"
            dwell_left = DWELL
            walk_steps = 0
            region = arm % N_REGIONS
        panel = arm // N_REGIONS
        if phase == "walk" and R.region_id(run.px[panel], run.py[panel]) == region:
            phase = "dwell"
        if phase == "walk":
            cx, cy = CENTERS[region]
            run.px[panel], run.py[panel] = R.step_toward(
                run.px[panel], run.py[panel], cx, cy
            )
            walk_steps += 1
            run.n_walk += 1
            if walk_steps > 48:
                raise RuntimeError("walk did not reach the region")
        else:
            run.px[panel], run.py[panel] = R.lawn_step(run.px[panel], run.py[panel])
            if R.region_id(run.px[panel], run.py[panel]) != region:
                raise RuntimeError("dwell left the region")
            dwell_left -= 1
            run.n_dwell += 1
            if dwell_left == 0:
                phase = None
        run.observe(panel, t)
    if run.n_walk + run.n_dwell != N_DECISIONS:
        raise RuntimeError("walk and dwell do not cover the life")


def score_recons(model, recons, labels, batch=16):
    n = len(recons)
    correct = np.zeros(n, dtype=np.int32)
    logloss = np.zeros(n, dtype=np.float64)
    top = np.zeros(n, dtype=np.int32)
    top_p = np.zeros(n, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        hwc = np.stack(recons[start:stop], axis=0)
        got = cam.scores_from_logits(cam.forward_array(model, hwc), labels[start:stop])
        correct[start:stop] = got["correct"]
        logloss[start:stop] = got["logloss"]
        top[start:stop] = got["top"]
        top_p[start:stop] = got["top_p"]
    return {"correct": correct, "logloss": logloss, "top": top, "top_p": top_p}


def save_demos(policy, kinds, snaps, top, top_p):
    os.makedirs(DEMO_DIR, exist_ok=True)
    names = []
    flat_top = top.reshape(-1)
    flat_p = top_p.reshape(-1)
    for snap in snaps:
        captions = []
        for panel, kind in enumerate(kinds):
            if kind == "tv":
                captions.append("tv")
                continue
            true_i = snap["labels"][kind]
            score_i = snap["score_at"][kind]
            pred_i = int(flat_top[score_i])
            prob = float(flat_p[score_i])
            captions.append(
                f"{kind} {cam.SYNSETS[true_i][1]}\n{cam.SYNSETS[pred_i][1]} {prob:.2f}"
            )
        name = f"life0_{policy}_t{snap['t']:04d}.png"
        path = os.path.join(DEMO_DIR, name)
        save_strip(path, snap["images"], captions)
        names.append(name)
    return names


def save_strip(path, panels, captions):
    from PIL import Image, ImageDraw
    gap = 6
    bar = 56
    n = len(panels)
    canvas = Image.new("RGB", (n * cam.SIDE + (n - 1) * gap, cam.SIDE + bar), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = cam.load_font(13)
    for i, (panel, cap) in enumerate(zip(panels, captions)):
        x = i * (cam.SIDE + gap)
        canvas.paste(Image.fromarray(panel), (x, 0))
        draw.multiline_text((x + 4, cam.SIDE + 4), cap, fill=(0, 0, 0), font=font, spacing=2)
    canvas.save(path)


def run_policy(life, kinds, schedule, pack, policy, model, demo):
    run = LifeRun(life, kinds, schedule, pack, policy, demo)
    t0 = time.perf_counter()
    if policy in CURIOUS:
        run_curious(run)
    else:
        run_fixed(run)
    row = run.finish(model)
    row["wall_s"] = time.perf_counter() - t0
    if life == 0 and policy == "curious_prog":
        log(
            "life 0 panel kinds "
            + " ".join(f"{i}={kind}" for i, kind in enumerate(kinds))
        )
        log(
            "life 0 curious_prog panels "
            + " ".join(str(p) for p in row["panel_seq"][:64])
        )
    row["panel_seq_head"] = row["panel_seq"][:64]
    return row


def prepare_pack():
    t0 = time.perf_counter()
    dest = cam.ensure_imagenette()
    images, files, counts, exif_n, load_s = cam.load_pool(dest)
    greys = np.empty((images.shape[0], W.N, W.N), dtype=np.int16)
    occs = np.empty((images.shape[0], W.N, W.N), dtype=bool)
    t1 = time.perf_counter()
    for i in range(images.shape[0]):
        greys[i] = cam.grey_proxy(images[i])
        occs[i] = W.occluder(W.make_field(i), cam.OPACITY)
    prep_s = time.perf_counter() - t1
    maps = build_maps()
    log(f"pack images {images.shape[0]} prep_s {prep_s:.3f} load_s {load_s:.3f}")
    return {
        "images": images,
        "files": files,
        "counts": counts,
        "exif_n": exif_n,
        "greys": greys,
        "occs": occs,
        "maps": maps,
        "load_s": load_s,
        "prep_s": prep_s,
        "pack_s": time.perf_counter() - t0,
    }


def check_maps(pack):
    sy_m, sx_m, oy_m, ox_m = pack["maps"]
    for py in range(0, W.P, 3):
        for px in range(0, W.P, 5):
            g1, v1, sx1, sy1 = W.frame(
                pack["greys"][0], pack["occs"][0], px, py, W.SCENE_SHIFT, W.DOT_SHIFT
            )
            g2, v2, sx2, sy2 = glance(pack["maps"], pack["greys"][0], pack["occs"][0], px, py)
            if not (
                np.array_equal(g1, g2)
                and np.array_equal(v1, v2)
                and np.array_equal(sx1, sx2)
                and np.array_equal(sy1, sy2)
            ):
                raise RuntimeError(f"glance map mismatch at {(px, py)}")
            if not (
                np.array_equal(sy_m[py, px], sy1)
                and np.array_equal(sx_m[py, px], sx1)
            ):
                raise RuntimeError("map tables drifted")
    log("smoke maps ok")


def check_formulas():
    if aged_value(1.25, 0, 9.0) != 1.25:
        raise RuntimeError("age 0 did not return v")
    got = aged_value(0.0, TAU, 1.0)
    expect = 1.0 - math.exp(-1.0)
    if abs(got - expect) > 1e-15:
        raise RuntimeError("age tau formula")
    color = np.array([10, 20, 30], dtype=np.uint8)
    image = np.empty((cam.SIDE, cam.SIDE, 3), dtype=np.uint8)
    image[...] = color
    q = int(cam.grey_proxy(image)[0, 0])
    q2 = int(tv_grey(color.reshape(1, 3))[0])
    if q != q2:
        raise RuntimeError(f"tv grey {q2} != grey_proxy {q}")
    ema = np.full(N_ARMS, np.inf)
    visits = np.zeros(N_ARMS, dtype=np.int32)
    last = np.full(N_ARMS, -1, dtype=np.int32)
    if choose_arm(ema, visits, last, None, 0, True) != 0:
        raise RuntimeError("first arm is not 0")
    log("smoke formulas ok")


def check_worlds():
    digests = []
    for life in range(N_LIVES_FULL):
        a_kinds, a_sched = make_world(life)
        b_kinds, b_sched = make_world(life)
        if a_kinds != b_kinds or not np.array_equal(a_sched, b_sched):
            raise RuntimeError(f"world {life} is not repeatable")
        fast = a_kinds.index("fast")
        slow = a_kinds.index("slow")
        static = a_kinds.index("static")
        tv = a_kinds.index("tv")
        if len(np.unique(a_sched[fast])) != 32:
            raise RuntimeError("fast image count")
        if len(np.unique(a_sched[slow])) != 8:
            raise RuntimeError("slow image count")
        if len(np.unique(a_sched[static])) != 1 or len(np.unique(a_sched[tv])) != 1:
            raise RuntimeError("static or tv image count")
        if not np.array_equal(a_sched[fast, 0:32], np.full(32, a_sched[fast, 0])):
            raise RuntimeError("fast window 0")
        if int(a_sched[fast, 32]) == int(a_sched[fast, 31]):
            raise RuntimeError("fast did not replace at t=32")
        if not np.array_equal(a_sched[slow, 0:128], np.full(128, a_sched[slow, 0])):
            raise RuntimeError("slow window 0")
        if int(a_sched[slow, 128]) == int(a_sched[slow, 127]):
            raise RuntimeError("slow did not replace at t=128")
        digests.append(hashlib.sha256(a_sched.tobytes()).hexdigest()[:12])
    kinds, schedule = make_world(0)
    log("smoke worlds ok life0 " + " ".join(f"{i}={k}" for i, k in enumerate(kinds)))
    return kinds, schedule, digests


def check_sweep_coverage(pack):
    with open(os.path.join(ROOT, "results", "w8", "results.json"), encoding="utf-8") as f:
        stored = json.load(f)["coverage"]["sweep"]
    rows = []
    for index in (0, 1, 2):
        for life in (4, 8, 16, 32, 64):
            steps = sweep_viewpoints(1, life)
            path = [(x, y) for _panel, x, y in steps]
            sweep = [(int(x), int(y)) for x, y in R.path_sweep(life)]
            if path != sweep:
                raise RuntimeError(f"single-panel sweep diverged at L={life}")
            seen = cam.seen_of(pack["greys"][index], pack["occs"][index], path)
            got = float(seen.mean())
            exp = float(stored[str(life)][index])
            if got != exp:
                raise RuntimeError(f"coverage image {index} L={life} {got} != {exp}")
            if life in (16, 64):
                canvas = np.zeros((cam.SIDE, cam.SIDE, 3), dtype=np.uint8)
                seen_paint = np.zeros((W.N, W.N), dtype=bool)
                for x, y in path:
                    grey, vis, sx, sy = glance(
                        pack["maps"], pack["greys"][index], pack["occs"][index], x, y
                    )
                    paint_image(canvas, seen_paint, pack["images"][index], sy[vis], sx[vis])
                if not np.array_equal(seen_paint, seen):
                    raise RuntimeError(f"paint seen mismatch image {index} L={life}")
                rgb_f = pack["images"][index].astype(np.float32) * (1.0 / 255.0)
                want = cam.reconstruct(rgb_f, seen)
                got_r = recon_f(canvas, seen_paint)
                if not np.array_equal(got_r, want):
                    raise RuntimeError(f"paint recon mismatch image {index} L={life}")
            rows.append({"image": index, "L": life, "coverage": got})
    long = sweep_viewpoints(1, N_DECISIONS)
    if [(x, y) for _p, x, y in long[:64]] != [(int(x), int(y)) for x, y in R.path_sweep(64)]:
        raise RuntimeError("long sweep prefix")
    four = sweep_viewpoints(N_PANELS, N_DECISIONS)
    counts = [0, 0, 0, 0]
    for t, (panel, _x, _y) in enumerate(four):
        counts[panel] += 1
        if panel != (t // DWELL) % N_PANELS:
            raise RuntimeError("round-robin panel order")
    if counts != [256, 256, 256, 256]:
        raise RuntimeError(f"round-robin counts {counts}")
    log("smoke coverage exact images 0-2 L=4,8,16,32,64")
    return rows


def check_trace_shape(seq):
    if len(seq) < 64:
        raise RuntimeError("trace shorter than 64")
    # Arm 0 is panel 0. The first commitment is 8 dwells before any other panel,
    # unless panel 0 region 0 is left, which the lawnmower does not do.
    if any(p != seq[0] for p in seq[:8]):
        raise RuntimeError("first dwell was not 8 decisions on one panel")
    log("smoke first dwell 8 on panel " + str(seq[0]))


def paired_ci(diff):
    diff = np.asarray(diff, dtype=np.float64)
    n = int(diff.shape[0])
    rng = np.random.default_rng(0)
    draw = rng.integers(0, n, size=(N_BOOT, n))
    means = diff[draw].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return float(diff.mean()), float(lo), float(hi)


def summarise(store, n_lives):
    lives = list(range(n_lives))
    accuracy = {policy: [] for policy in POLICIES}
    logloss = {policy: [] for policy in POLICIES}
    share = {policy: {kind: [] for kind in KINDS} for policy in POLICIES}
    panel_acc = {policy: {kind: [] for kind in SCORED} for policy in POLICIES}
    panel_ll = {policy: {kind: [] for kind in SCORED} for policy in POLICIES}
    for life in lives:
        block = store[str(life)]["policies"]
        for policy in POLICIES:
            row = block[policy]
            accuracy[policy].append(float(row["accuracy"]))
            logloss[policy].append(float(row["logloss"]))
            for kind in KINDS:
                share[policy][kind].append(float(row["share"][kind]))
            for kind in SCORED:
                panel_acc[policy][kind].append(float(row["panel_accuracy"][kind]))
                panel_ll[policy][kind].append(float(row["panel_logloss"][kind]))
    contrasts = {}
    for name, left, right in (
        ("curious_prog-round_robin", "curious_prog", "round_robin"),
        ("curious_prog-curious_surp", "curious_prog", "curious_surp"),
    ):
        acc = np.array(accuracy[left]) - np.array(accuracy[right])
        ll = np.array(logloss[left]) - np.array(logloss[right])
        acc_m, acc_lo, acc_hi = paired_ci(acc)
        ll_m, ll_lo, ll_hi = paired_ci(ll)
        contrasts[name] = {
            "accuracy": acc_m,
            "accuracy_ci95_lo": acc_lo,
            "accuracy_ci95_hi": acc_hi,
            "logloss_nats": ll_m,
            "logloss_ci95_lo": ll_lo,
            "logloss_ci95_hi": ll_hi,
            "n_accuracy_gt_0": int(np.sum(acc > 0)),
            "n_accuracy_lt_0": int(np.sum(acc < 0)),
            "n_accuracy_eq_0": int(np.sum(acc == 0)),
        }
    mean_share = {
        policy: {kind: float(np.mean(share[policy][kind])) for kind in KINDS}
        for policy in POLICIES
    }
    prog = mean_share["curious_prog"]
    surp = mean_share["curious_surp"]
    gap = contrasts["curious_prog-round_robin"]
    ci_excludes_0 = gap["accuracy_ci95_lo"] > 0 or gap["accuracy_ci95_hi"] < 0
    bars = {
        "n_lives": int(n_lives),
        "curiosity_beats_round_robin": bool(
            gap["accuracy"] >= BAR_ACC and ci_excludes_0
        ),
        "ignores_tv": bool(prog["tv"] <= BAR_TV_PROG and surp["tv"] >= BAR_TV_SURP),
        "watches_changes": bool(prog["fast"] > prog["slow"] > prog["static"]),
        "accuracy_gap": gap["accuracy"],
        "accuracy_ci_excludes_0": bool(ci_excludes_0),
        "prog_tv": prog["tv"],
        "surp_tv": surp["tv"],
        "prog_fast": prog["fast"],
        "prog_slow": prog["slow"],
        "prog_static": prog["static"],
    }
    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "share": share,
        "mean_share": mean_share,
        "panel_accuracy": panel_acc,
        "panel_logloss": panel_ll,
        "mean_panel_accuracy": {
            policy: {kind: float(np.mean(panel_acc[policy][kind])) for kind in SCORED}
            for policy in POLICIES
        },
        "mean_panel_logloss": {
            policy: {kind: float(np.mean(panel_ll[policy][kind])) for kind in SCORED}
            for policy in POLICIES
        },
        "contrasts": contrasts,
        "bars": bars,
    }


def render(payload):
    lines = []
    n = payload["n_lives"]
    lines.append(
        f"n_lives {n} decisions {N_DECISIONS} checks_every {CHECK_EVERY} "
        f"panels 4 opacity {cam.OPACITY} tau {TAU} dwell {DWELL}"
    )
    lines.append("")
    header = "| life | " + " | ".join(POLICIES) + " | prog−robin | prog−surp |"
    sep = "|---:|" + "|".join(["---:"] * (len(POLICIES) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    acc = payload["summary"]["accuracy"]
    for life in range(n):
        cells = [f6(acc[policy][life]) for policy in POLICIES]
        gap_r = acc["curious_prog"][life] - acc["round_robin"][life]
        gap_s = acc["curious_prog"][life] - acc["curious_surp"][life]
        lines.append(
            f"| {life} | " + " | ".join(cells) + f" | {f6(gap_r)} | {f6(gap_s)} |"
        )
    lines.append("")
    lines.append("| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi | n_acc_gt_0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("curious_prog-round_robin", "curious_prog-curious_surp"):
        row = payload["summary"]["contrasts"][name]
        lines.append(
            f"| {name} | {f6(row['accuracy'])} | {f6(row['accuracy_ci95_lo'])} | "
            f"{f6(row['accuracy_ci95_hi'])} | {f6(row['logloss_nats'])} | "
            f"{f6(row['logloss_ci95_lo'])} | {f6(row['logloss_ci95_hi'])} | "
            f"{row['n_accuracy_gt_0']}/{n} |"
        )
    lines.append("")
    lines.append("| policy | tv | static | fast | slow |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in POLICIES:
        share = payload["summary"]["mean_share"][policy]
        lines.append(
            "| " + policy + " | "
            + " | ".join(f6(share[kind]) for kind in ("tv", "static", "fast", "slow"))
            + " |"
        )
    lines.append("")
    bars = payload["summary"]["bars"]
    lines.append("| bar | observed | required | result |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| 1 curiosity beats round robin | {f6(bars['accuracy_gap'])} "
        f"ci_excludes_0 {str(bars['accuracy_ci_excludes_0']).lower()} | "
        f">= {BAR_ACC:.2f} and CI excludes 0 | "
        f"{'PASS' if bars['curiosity_beats_round_robin'] else 'FAIL'} |"
    )
    lines.append(
        f"| 2 ignores the tv | prog {f6(bars['prog_tv'])} surp {f6(bars['surp_tv'])} | "
        f"prog <= {BAR_TV_PROG:.2f} and surp >= {BAR_TV_SURP:.2f} | "
        f"{'PASS' if bars['ignores_tv'] else 'FAIL'} |"
    )
    lines.append(
        f"| 3 watches what changes | fast {f6(bars['prog_fast'])} slow {f6(bars['prog_slow'])} "
        f"static {f6(bars['prog_static'])} | fast > slow > static | "
        f"{'PASS' if bars['watches_changes'] else 'FAIL'} |"
    )
    lines.append("")
    lines.append("per-life share")
    lines.append("")
    lines.append("| life | policy | tv | static | fast | slow |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for life in range(n):
        for policy in POLICIES:
            share = payload["summary"]["share"][policy]
            lines.append(
                f"| {life} | {policy} | "
                + " | ".join(f6(share[kind][life]) for kind in ("tv", "static", "fast", "slow"))
                + " |"
            )
    lines.append("")
    lines.append("per-panel life accuracy, mean over lives")
    lines.append("")
    lines.append("| policy | static | fast | slow |")
    lines.append("|---|---:|---:|---:|")
    for policy in POLICIES:
        row = payload["summary"]["mean_panel_accuracy"][policy]
        lines.append(
            f"| {policy} | {f6(row['static'])} | {f6(row['fast'])} | {f6(row['slow'])} |"
        )
    lines.append("")
    lines.append("per-panel life logloss nats, mean over lives")
    lines.append("")
    lines.append("| policy | static | fast | slow | life |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in POLICIES:
        row = payload["summary"]["mean_panel_logloss"][policy]
        life_ll = float(np.mean(payload["summary"]["logloss"][policy]))
        lines.append(
            f"| {policy} | {f6(row['static'])} | {f6(row['fast'])} | {f6(row['slow'])} | {f6(life_ll)} |"
        )
    lines.append("")
    lines.append("per-life per-panel accuracy")
    lines.append("")
    lines.append("| life | policy | static | fast | slow | logloss |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for life in range(n):
        for policy in POLICIES:
            row = payload["summary"]["panel_accuracy"][policy]
            lines.append(
                f"| {life} | {policy} | {f6(row['static'][life])} | {f6(row['fast'][life])} | "
                f"{f6(row['slow'][life])} | {f6(payload['summary']['logloss'][policy][life])} |"
            )
    lines.append("")
    lines.append("life 0 curious_prog panels, first 64 decisions")
    lines.append("")
    lines.append(" ".join(str(p) for p in payload["life0_panels"]))
    lines.append("")
    lines.append("life 0 panel kinds " + " ".join(
        f"{i}={kind}" for i, kind in enumerate(payload["life0_kinds"])
    ))
    lines.append("")
    lines.append("mechanism")
    lines.append("")
    lines.append("| life | policy | n_choices | n_walk | n_dwell | first_pass_t | wall_s |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for life in range(n):
        for policy in POLICIES:
            row = payload["lives"][str(life)]["policies"][policy]
            passed = row["first_pass_t"]
            passed_s = "" if passed is None else str(int(passed))
            lines.append(
                f"| {life} | {policy} | {row['n_choices']} | {row['n_walk']} | "
                f"{row['n_dwell']} | {passed_s} | {f6(row['wall_s'])} |"
            )
    lines.append("")
    smoke = payload["smoke"]
    lines.append("smoke")
    lines.append("")
    lines.append(
        "single-panel round_robin coverage matches results/w8 sweep for images 0-2 "
        "at L=4,8,16,32,64, exact float"
    )
    for row in smoke["coverage"]:
        lines.append(f"image {row['image']} L {row['L']} coverage {f6(row['coverage'])}")
    lines.append("")
    lines.append("walls")
    lines.append("")
    for key, value in payload["walls"].items():
        lines.append(f"{key} {value}")
    lines.append("")
    lines.append("source")
    lines.append("")
    for key, value in payload["source"].items():
        lines.append(f"{key} {value}")
    lines.append("")
    if payload["demos"]:
        lines.append("demos")
        lines.append("")
        for name in payload["demos"]:
            lines.append(name)
        lines.append("")
    return "\n".join(lines)


def public_row(row):
    return {
        "accuracy": row["accuracy"],
        "logloss": row["logloss"],
        "panel_accuracy": row["panel_accuracy"],
        "panel_logloss": row["panel_logloss"],
        "checks": row["checks"],
        "share": row["share"],
        "share_n": row["share_n"],
        "n_choices": row["n_choices"],
        "n_walk": row["n_walk"],
        "n_dwell": row["n_dwell"],
        "first_pass_t": row["first_pass_t"],
        "wall_s": row["wall_s"],
        "demo_files": row["demo_files"],
    }


def load_partial():
    if not os.path.isfile(PARTIAL):
        return None
    with open(PARTIAL, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("sha") != code_sha():
        log("partial sha mismatch; ignoring")
        return None
    return payload


def save_partial(payload):
    atomic_write(PARTIAL, json.dumps(payload))


def empty_partial(pack, smoke, load):
    return {
        "sha": code_sha(),
        "target_lives": None,
        "smoke": smoke,
        "pack": {
            "load_s": pack["load_s"],
            "prep_s": pack["prep_s"],
            "exif_n": pack["exif_n"],
            "pool_counts": pack["counts"],
            "files_head": pack["files"][:3],
        },
        "load_average_start": load,
        "lives": {},
        "timing": {},
    }


def run_lives(pack, model, partial, lives):
    for life in lives:
        key = str(life)
        if key not in partial["lives"]:
            kinds, schedule = make_world(life)
            partial["lives"][key] = {"kinds": list(kinds), "policies": {}}
        else:
            kinds = tuple(partial["lives"][key]["kinds"])
            kinds2, schedule = make_world(life)
            if kinds2 != kinds:
                raise RuntimeError("resumed kinds do not match the world")
        for policy in POLICIES:
            if policy in partial["lives"][key]["policies"]:
                log(f"life {life} {policy} resume")
                continue
            demo = life == 0
            log(f"life {life} {policy} start")
            row = run_policy(life, kinds, schedule, pack, policy, model, demo)
            if life == 0 and policy == "curious_prog":
                check_trace_shape(row["panel_seq"])
                partial["life0_panels"] = row["panel_seq"][:64]
            seq = row.pop("panel_seq")
            head = row.pop("panel_seq_head")
            if policy == "curious_prog" and life == 0:
                partial["life0_panels"] = head
            partial["lives"][key]["policies"][policy] = public_row(row)
            partial["timing"].setdefault(key, {})[policy] = row["wall_s"]
            if policy == "curious_prog":
                partial["lives"][key]["prog_seq_digest"] = hashlib.sha256(
                    np.asarray(seq, dtype=np.int16).tobytes()
                ).hexdigest()[:12]
            if life == 0 and policy == "curious_prog_noage" and "life0_panels" in partial:
                pass
            save_partial(partial)
            log(
                f"life {life} {policy} acc {row['accuracy']:.4f} "
                f"tv {row['share']['tv']:.3f} fast {row['share']['fast']:.3f} "
                f"slow {row['share']['slow']:.3f} static {row['share']['static']:.3f} "
                f"wall_s {row['wall_s']:.1f}"
            )
        if life == 0 and partial["target_lives"] is None:
            spent = sum(partial["timing"]["0"].values())
            est20 = spent * N_LIVES_FULL
            target = N_LIVES_FULL if est20 <= BUDGET_S else N_LIVES_SHORT
            partial["target_lives"] = int(target)
            partial["estimate_20_s"] = est20
            partial["life0_s"] = spent
            save_partial(partial)
            log(f"life0_s {spent:.1f} estimate_20_s {est20:.1f} target_lives {target}")
    return partial


def assemble(partial, pack, load_start, t_wall):
    n_lives = int(partial["target_lives"])
    for life in range(n_lives):
        got = set(partial["lives"][str(life)]["policies"])
        if got != set(POLICIES):
            raise RuntimeError(f"life {life} missing policies {set(POLICIES) - got}")
    summary = summarise(partial["lives"], n_lives)
    demos = []
    for policy in DEMO_POLICIES:
        demos.extend(partial["lives"]["0"]["policies"][policy]["demo_files"])
    kinds0 = partial["lives"]["0"]["kinds"]
    walls = {
        "life0_s": f"{partial['life0_s']:.3f}",
        "estimate_20_s": f"{partial['estimate_20_s']:.3f}",
        "wall_s": f"{t_wall:.3f}",
        "pack_load_s": f"{pack['load_s']:.3f}",
        "pack_prep_s": f"{pack['prep_s']:.3f}",
        "load_average_start": f"{load_start:.2f}",
        "load_average_end": f"{load_average():.2f}",
        "target_lives": str(n_lives),
        "mean_first_pass_t_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["first_pass_t"] or N_DECISIONS
            for i in range(n_lives)
        ])),
        "mean_n_choices_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_choices"]
            for i in range(n_lives)
        ])),
        "mean_n_walk_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_walk"]
            for i in range(n_lives)
        ])),
    }
    source = {
        "dataset": "imagenette2-160",
        "split": "val",
        "pool_per_class": str(cam.POOL),
        "opacity": str(cam.OPACITY),
        "scene_shift": str(W.SCENE_SHIFT),
        "dot_shift": str(W.DOT_SHIFT),
        "eye": "torchvision resnet18 IMAGENET1K_V1 frozen eval",
        "softmax": "10 logits, synset table order",
        "logloss": "nats",
        "unseen": "imagenet mean colour",
        "stale": "cell keeps the rgb block from the generation that was seen",
        "tv": "uniform random rgb block per clear cell, default_rng(9700+life) per policy",
        "kinds_rng": "default_rng(9800+life).permutation",
        "images_rng": "default_rng(9900+life).permutation, consumed in panel order",
        "random_rng": "default_rng(8000+life), panel then action",
        "reservoir": "GlanceReservoir(life) per panel, progress policies only",
        "ema_beta": str(W.BETA),
        "tau": str(TAU),
        "dwell": str(DWELL),
        "age": "completed decisions minus last observation of that arm; sort key only",
        "v_max": "largest per-glance reward so far",
        "unvisited_prog": "+inf, not aged",
        "unvisited_surp": "log2(8), not aged",
        "epsilon": "none",
        "bootstrap": "10000 paired percentile, numpy quantile linear, default_rng(0) per contrast",
        "runner": "vps",
        "device": "cpu",
        "torch_threads": "2",
        "torch": torch_version(),
        "numpy": np.__version__,
        "code_sha256": code_sha(),
        "n_lives_registered": str(N_LIVES_FULL),
        "n_lives_run": str(n_lives),
        "short_run": "yes" if n_lives != N_LIVES_FULL else "no",
    }
    # Drop the bulky per-decision checks from the on-disk lives copy used by
    # markdown, but keep them in results.json.
    payload = {
        "n_lives": n_lives,
        "n_decisions": N_DECISIONS,
        "policies": list(POLICIES),
        "summary": summary,
        "lives": partial["lives"],
        "life0_kinds": kinds0,
        "life0_panels": partial["life0_panels"],
        "smoke": partial["smoke"],
        "walls": walls,
        "source": source,
        "demos": demos,
        "synsets": [
            {"order": i, "wnid": wnid, "imagenet_index": index, "category": name}
            for i, (wnid, name, index) in enumerate(cam.SYNSETS)
        ],
    }
    text = render(payload)
    atomic_write(os.path.join(OUT_DIR, "RESULTS.md"), text)
    atomic_write(os.path.join(OUT_DIR, "results.json"), json.dumps(payload))
    log("wrote results/w8b/RESULTS.md")
    bars = summary["bars"]
    log(
        "bars "
        f"1 {bars['curiosity_beats_round_robin']} "
        f"2 {bars['ignores_tv']} "
        f"3 {bars['watches_changes']} "
        f"gap {bars['accuracy_gap']:.4f} "
        f"tv_prog {bars['prog_tv']:.3f} tv_surp {bars['surp_tv']:.3f} "
        f"fast {bars['prog_fast']:.3f} slow {bars['prog_slow']:.3f} static {bars['prog_static']:.3f}"
    )
    return payload


def torch_version():
    import torch
    import torchvision
    return f"{torch.__version__} torchvision {torchvision.__version__}"


def fast_checks(pack):
    check_formulas()
    check_maps(pack)
    _kinds, _schedule, digests = check_worlds()
    coverage = check_sweep_coverage(pack)
    return {"coverage": coverage, "world_digest_head": digests[0]}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("smoke", "all", "render"):
        raise SystemExit("usage: w8b_wall.py [smoke|all|render]")
    load = load_average()
    log(f"load_average {load:.2f}")
    if stage == "all" and load >= 1.5:
        raise SystemExit(f"load_average {load:.2f} >= 1.5; not starting the VPS run")
    if stage == "render":
        partial = load_partial()
        if partial is None or partial.get("target_lives") is None:
            raise SystemExit("nothing to render")
        # render needs pack timings already stored; assemble reads partial only.
        raise SystemExit("render is done by all")
    pack = prepare_pack()
    smoke = fast_checks(pack)
    if stage == "smoke":
        model, eye_s = cam.load_eye()
        partial = empty_partial(pack, smoke, load)
        partial["eye_s"] = eye_s
        run_lives(pack, model, partial, [0])
        log(
            f"smoke done life0_s {partial['life0_s']:.1f} "
            f"estimate_20_s {partial['estimate_20_s']:.1f} "
            f"target_lives {partial['target_lives']}"
        )
        return
    model, eye_s = cam.load_eye()
    partial = load_partial()
    if partial is None:
        partial = empty_partial(pack, smoke, load)
        partial["eye_s"] = eye_s
    else:
        partial["smoke"] = smoke
        log(f"resuming lives {sorted(partial['lives'])}")
    t0 = time.perf_counter()
    # Life 0 decides the target. Run it first, then the rest.
    if "0" not in partial["lives"] or set(partial["lives"]["0"]["policies"]) != set(POLICIES):
        run_lives(pack, model, partial, [0])
    target = int(partial["target_lives"])
    run_lives(pack, model, partial, list(range(target)))
    assemble(partial, pack, partial.get("load_average_start", load), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
