"""W8-1b: curiosity chooses the panel. The sweep inside a panel is shared.

Imports w8b_wall, w8_camera, w7_world and w7_run. Does not modify them.

World, images, lives 0-19, L = 1024, the eye and the score are W8-1.
Every policy moves with W7's planned sweep. Each panel keeps its own cursor.
Replacing a panel's image does not move that cursor.

A visit is 8 decisions on one panel. The policies differ only in the next panel.
round_robin takes the panels in turn. random draws one uniform panel per visit.
curious_prog is W5/W6 prog_age_s over the 4 panels: optimistic first pass,
then epsilon 0.1, EMA beta 0.3, ageing tau 256 on the greedy step only.
The visit reward is the sum of W7's per-glance progress. curious_surp sums
W7's per-glance surprise. curious_prog_noage is curious_prog without ageing.
"""
import copy
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

import curio
import w8b_wall as wall
import w7_run as R

ROOT = wall.ROOT
OUT_DIR = os.path.join(ROOT, "results", "w8c")
DEMO_DIR = os.path.join(ROOT, "internal", "w8_demo", "wall_b")
PARTIAL = os.path.join(OUT_DIR, "partial.json")

N_DECISIONS = wall.N_DECISIONS
N_PANELS = wall.N_PANELS
DWELL = wall.DWELL
TAU = wall.TAU
CHECK_EVERY = wall.CHECK_EVERY
DEMO_EVERY = wall.DEMO_EVERY
N_LIVES = wall.N_LIVES_FULL
POLICIES = wall.POLICIES
KINDS = wall.KINDS
SCORED = wall.SCORED
DEMO_POLICIES = wall.DEMO_POLICIES
EPS = curio.EPS
BETA = curio.BETA

CURIOUS = {
    "curious_prog": ("prog", True),
    "curious_surp": ("surp", True),
    "curious_prog_noage": ("prog", False),
}

if EPS != 0.1 or BETA != 0.3 or TAU != 256:
    raise RuntimeError(f"bandit constants drifted: eps {EPS} beta {BETA} tau {TAU}")
if N_DECISIONS % DWELL != 0:
    raise RuntimeError("the life does not divide into visits")


def log(msg):
    print(msg, flush=True)


def f6(x):
    return wall.f6(x)


def code_sha():
    path = os.path.abspath(__file__)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def aged_value(v, age, v_max, tau=TAU):
    """v + (v_max - v) * (1 - exp(-age / tau)). Age 0 returns v."""
    gap = float(v_max) - float(v)
    return float(v) + gap * (1.0 - math.exp(-float(age) / float(tau)))


def nearest_unvisited(src, unvisited, n):
    """W5/W6 ring rule. Equal distances break toward +index."""
    best, best_key = None, None
    for dst in unvisited:
        right = (int(dst) - int(src)) % int(n)
        left = (int(src) - int(dst)) % int(n)
        dist = right if right <= left else left
        key = (dist, 0 if right <= left else 1, int(dst))
        if best_key is None or key < best_key:
            best, best_key = int(dst), key
    return best


def ema(values, counts, arm, reward):
    """W5/W6 update. The first visit writes the reward. Later visits use beta."""
    if counts[arm] == 0:
        values[arm] = float(reward)
    else:
        values[arm] = (1.0 - BETA) * float(values[arm]) + BETA * float(reward)
    counts[arm] += 1


def choose_panel(room, unvisited, values, counts, last_end, v_max, t, rng, use_age):
    """W5/W6 choose_room on 4 panels. The stored EMA is not written or aged."""
    n_arms = len(values)
    if int(room) < 0 or int(room) >= n_arms:
        raise RuntimeError("choice panel is off the wall")
    stored = [float(v) for v in values]
    stored_counts = list(counts)
    if unvisited:
        target = nearest_unvisited(int(room), unvisited, n_arms)
        unvisited.remove(target)
        mode = "first"
    elif float(rng.random()) < EPS:
        target = int(rng.integers(n_arms))
        mode = "eps"
    elif use_age:
        if v_max is None:
            raise RuntimeError("greedy age with no visit reward")
        if any(last_end[i] is None for i in range(n_arms)):
            raise RuntimeError("greedy age before the first pass finished")
        aged = []
        for i in range(n_arms):
            age = int(t) - int(last_end[i])
            aged.append(aged_value(values[i], age, v_max))
            if age == 0 and aged[-1] != float(values[i]):
                raise RuntimeError("age 0 changed the value")
        target = max(range(n_arms), key=lambda k: (aged[k], float(rng.random())))
        mode = "greedy"
    else:
        target = max(range(n_arms), key=lambda k: (float(values[k]), float(rng.random())))
        mode = "greedy"
    if [float(v) for v in values] != stored or list(counts) != stored_counts:
        raise RuntimeError("the choice changed the stored EMA")
    return int(target), mode


class PanelSweep:
    """One W7 lattice, one cursor per panel. Nothing resets a cursor."""

    def __init__(self, budget):
        self.points = [(int(x), int(y)) for x, y in R.path_sweep(int(budget))]
        if len(self.points) != int(budget):
            raise RuntimeError("sweep length")
        self.cursor = [0] * N_PANELS

    def step(self, panel):
        i = self.cursor[int(panel)]
        x, y = self.points[i]
        self.cursor[int(panel)] = i + 1
        return x, y


class PanelBandit:
    def __init__(self, life, use_age):
        self.use_age = bool(use_age)
        self.rng = np.random.default_rng(10_000 + int(life))
        self.values = [0.0] * N_PANELS
        self.counts = [0] * N_PANELS
        self.last_end = [None] * N_PANELS
        self.v_max = None
        self.unvisited = set(range(N_PANELS))
        self.room = 0
        self.n_choices = 0
        self.n_first = 0
        self.n_eps = 0
        self.n_greedy = 0
        self.first_pass_t = None
        self.choices = []

    def choose(self, t):
        target, mode = choose_panel(
            self.room, self.unvisited, self.values, self.counts, self.last_end,
            self.v_max, int(t), self.rng, self.use_age,
        )
        self.n_choices += 1
        if mode == "first":
            self.n_first += 1
        elif mode == "eps":
            self.n_eps += 1
        elif mode == "greedy":
            self.n_greedy += 1
        else:
            raise RuntimeError(f"choice mode {mode}")
        self.choices.append({"t": int(t), "panel": int(target), "mode": mode})
        return int(target), mode

    def finish(self, panel, reward, t_done):
        reward = float(reward)
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite visit reward")
        ema(self.values, self.counts, int(panel), reward)
        if self.v_max is None or reward > self.v_max:
            self.v_max = float(reward)
        self.last_end[int(panel)] = int(t_done)
        self.room = int(panel)
        if self.first_pass_t is None and min(self.counts) >= 1:
            self.first_pass_t = int(t_done)


class PanelLife:
    def __init__(self, life, kinds, schedule, pack, policy, demo):
        self.life = int(life)
        self.kinds = kinds
        self.schedule = schedule
        self.pack = pack
        self.policy = policy
        self.demo = bool(demo)
        self.panel_of = {kind: kinds.index(kind) for kind in KINDS}
        self.canvas = np.zeros((N_PANELS, wall.cam.SIDE, wall.cam.SIDE, 3), dtype=np.uint8)
        self.seen = np.zeros((N_PANELS, wall.W.N, wall.W.N), dtype=bool)
        self.px = [0] * N_PANELS
        self.py = [0] * N_PANELS
        self.tv_rng = np.random.default_rng(9700 + int(life))
        self.counts = None
        self.reservoirs = None
        self.mode = None
        self.share_n = {kind: 0 for kind in KINDS}
        self.recons = []
        self.labels = []
        self.demo_snaps = []
        self.panel_seq = []
        if policy in CURIOUS:
            mode, _use_age = CURIOUS[policy]
            self.mode = mode
            self.counts = np.zeros((N_PANELS, wall.W.N, wall.W.N, wall.W.G), dtype=np.int32)
            if mode == "prog":
                self.reservoirs = [
                    wall.W.GlanceReservoir(int(life)) for _ in range(N_PANELS)
                ]

    def observe(self, panel, t):
        kind = self.kinds[panel]
        index = int(self.schedule[panel, t])
        image = self.pack["images"][index]
        scene = self.pack["greys"][index]
        occ = self.pack["occs"][index]
        px, py = int(self.px[panel]), int(self.py[panel])
        grey, vis, sx, sy = wall.glance(self.pack["maps"], scene, occ, px, py)
        ys = sy[vis]
        xs = sx[vis]
        if kind == "tv":
            n = int(ys.size)
            if n:
                colors = self.tv_rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
                wall.paint_colors(self.canvas[panel], self.seen[panel], colors, ys, xs)
                grey = np.array(grey, copy=True)
                grey[vis] = wall.tv_grey(colors).astype(np.int16)
        else:
            wall.paint_image(self.canvas[panel], self.seen[panel], image, ys, xs)
        reward = 0.0
        if self.mode is not None:
            reservoir = None if self.reservoirs is None else self.reservoirs[panel]
            reward = float(R._reward(
                self.mode, self.counts[panel], reservoir, grey, vis, sx, sy
            ))
            if not math.isfinite(reward):
                raise RuntimeError("nonfinite glance reward")
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
            self.recons.append(wall.recon_f(self.canvas[panel], self.seen[panel]))
            self.labels.append(int(index) // wall.cam.POOL)
        if self.demo and self.policy in DEMO_POLICIES and completed % DEMO_EVERY == 0:
            self.demo_snaps.append({
                "t": int(completed),
                "images": [
                    wall.display_u8(self.canvas[p], self.seen[p]) for p in range(N_PANELS)
                ],
                "score_at": {kind: base + i for i, kind in enumerate(SCORED)},
                "labels": {
                    kind: int(self.schedule[self.panel_of[kind], completed - 1]) // wall.cam.POOL
                    for kind in SCORED
                },
            })

    def finish(self, model, bandit):
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
        got = wall.score_recons(model, self.recons, self.labels)
        correct = got["correct"].reshape(n_checks, len(SCORED))
        logloss = got["logloss"].reshape(n_checks, len(SCORED))
        top = got["top"].reshape(n_checks, len(SCORED))
        top_p = got["top_p"].reshape(n_checks, len(SCORED))
        demo_files = []
        if self.demo_snaps:
            demo_files = save_demos(self.policy, self.kinds, self.demo_snaps, top, top_p)
        del self.recons
        del self.canvas
        n_choices = 0 if bandit is None else int(bandit.n_choices)
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
            "n_choices": n_choices,
            "n_walk": 0,
            "n_dwell": int(N_DECISIONS),
            "n_first": 0 if bandit is None else int(bandit.n_first),
            "n_eps": 0 if bandit is None else int(bandit.n_eps),
            "n_greedy": 0 if bandit is None else int(bandit.n_greedy),
            "first_pass_t": None if bandit is None else bandit.first_pass_t,
            "demo_files": demo_files,
            "panel_seq": self.panel_seq,
            "choices": [] if bandit is None else bandit.choices,
        }


def save_demos(policy, kinds, snaps, top, top_p):
    os.makedirs(DEMO_DIR, exist_ok=True)
    names = []
    flat_top = top.reshape(-1)
    flat_p = top_p.reshape(-1)
    for snap in snaps:
        captions = []
        for _panel, kind in enumerate(kinds):
            if kind == "tv":
                captions.append("tv")
                continue
            true_i = snap["labels"][kind]
            score_i = snap["score_at"][kind]
            pred_i = int(flat_top[score_i])
            prob = float(flat_p[score_i])
            captions.append(
                f"{kind} {wall.cam.SYNSETS[true_i][1]}\n"
                f"{wall.cam.SYNSETS[pred_i][1]} {prob:.2f}"
            )
        name = f"life0_{policy}_t{snap['t']:04d}.png"
        wall.save_strip(os.path.join(DEMO_DIR, name), snap["images"], captions)
        names.append(name)
    return names


def run_policy(life, kinds, schedule, pack, policy, model, demo):
    run = PanelLife(life, kinds, schedule, pack, policy, demo)
    sweep = PanelSweep(N_DECISIONS)
    bandit = None
    if policy in CURIOUS:
        _mode, use_age = CURIOUS[policy]
        bandit = PanelBandit(life, use_age)
    rng = np.random.default_rng(8000 + int(life))
    panel = 0
    visit_sum = 0.0
    t0 = time.perf_counter()
    for t in range(N_DECISIONS):
        if t % DWELL == 0:
            if policy == "round_robin":
                panel = (t // DWELL) % N_PANELS
            elif policy == "random":
                panel = int(rng.integers(0, N_PANELS))
            elif bandit is not None:
                panel, _mode = bandit.choose(t)
            else:
                raise RuntimeError(policy)
            visit_sum = 0.0
        x, y = sweep.step(panel)
        run.px[panel], run.py[panel] = x, y
        visit_sum += run.observe(panel, t)
        if bandit is not None and (t + 1) % DWELL == 0:
            bandit.finish(panel, visit_sum, t + 1)
    for p in range(N_PANELS):
        if sweep.cursor[p] != int(run.share_n[kinds[p]]):
            raise RuntimeError("sweep cursor is not the number of glances")
    if sum(sweep.cursor) != N_DECISIONS:
        raise RuntimeError("sweep cursors do not cover the life")
    row = run.finish(model, bandit)
    row["wall_s"] = time.perf_counter() - t0
    if bandit is not None:
        if bandit.n_choices != N_DECISIONS // DWELL:
            raise RuntimeError("choice count")
        if bandit.n_first != N_PANELS or bandit.first_pass_t != N_PANELS * DWELL:
            raise RuntimeError(
                f"first pass n {bandit.n_first} t {bandit.first_pass_t}"
            )
        first = [c["panel"] for c in bandit.choices[:N_PANELS]]
        if first != list(range(N_PANELS)):
            raise RuntimeError(f"first pass order {first}")
        if any(c["mode"] != "first" for c in bandit.choices[:N_PANELS]):
            raise RuntimeError("first pass mode")
    seq = row["panel_seq"]
    for start in range(0, N_DECISIONS, DWELL):
        if any(p != seq[start] for p in seq[start:start + DWELL]):
            raise RuntimeError("a visit left its panel")
    return row


def check_bandit():
    if aged_value(1.25, 0, 9.0) != 1.25:
        raise RuntimeError("age 0 did not return v")
    got = aged_value(0.0, TAU, 1.0)
    expect = 1.0 - math.exp(-1.0)
    if abs(got - expect) > 1e-15:
        raise RuntimeError("age tau formula")
    values = [0.0]
    counts = [0]
    ema(values, counts, 0, 10.0)
    if values[0] != 10.0 or counts[0] != 1:
        raise RuntimeError("first ema")
    ema(values, counts, 0, 0.0)
    if abs(values[0] - 7.0) > 1e-15 or counts[0] != 2:
        raise RuntimeError("second ema")
    band = PanelBandit(0, True)
    clone = np.random.default_rng()
    clone.bit_generator.state = copy.deepcopy(band.rng.bit_generator.state)
    order = []
    for i in range(N_PANELS):
        panel, mode = band.choose(i * DWELL)
        order.append((panel, mode))
        band.finish(panel, 0.0, (i + 1) * DWELL)
    if order != [(i, "first") for i in range(N_PANELS)]:
        raise RuntimeError(f"optimistic order {order}")
    if float(band.rng.random()) != float(clone.random()):
        raise RuntimeError("the first pass consumed the choice rng")
    if band.first_pass_t != N_PANELS * DWELL:
        raise RuntimeError("first pass time")
    values = [0.0, 0.2, 0.2, 0.2]
    counts = [1, 1, 1, 1]
    last_end = [0, TAU, TAU, TAU]
    rng = np.random.default_rng(1)
    aged_pick, aged_mode = choose_panel(
        0, set(), values, counts, last_end, 1.0, TAU, rng, True
    )
    raw_pick, raw_mode = choose_panel(
        0, set(), values, counts, last_end, 1.0, TAU, rng, False
    )
    if aged_mode != "greedy" or aged_pick != 0:
        raise RuntimeError(f"ageing picked {aged_pick}")
    if raw_mode != "greedy" or raw_pick == 0:
        raise RuntimeError(f"no-age picked {raw_pick}")
    if list(values) != [0.0, 0.2, 0.2, 0.2]:
        raise RuntimeError("choice wrote the EMA")
    log("smoke bandit ok")


def check_mover():
    sweep = PanelSweep(N_DECISIONS)
    got = []
    panel = 0
    left = DWELL
    for _ in range(N_DECISIONS):
        if left == 0:
            panel = (panel + 1) % N_PANELS
            left = DWELL
        x, y = sweep.step(panel)
        got.append((panel, x, y))
        left -= 1
    exp = wall.sweep_viewpoints(N_PANELS, N_DECISIONS)
    if got != exp:
        raise RuntimeError("round-robin sweep diverged from W8-1")
    div = [(t // DWELL) % N_PANELS for t in range(N_DECISIONS)]
    if [p for p, _x, _y in got] != div:
        raise RuntimeError("round-robin panel index diverged")
    one = PanelSweep(64)
    path = [one.step(0) for _ in range(64)]
    lattice = [(int(x), int(y)) for x, y in R.path_sweep(64)]
    if path != lattice:
        raise RuntimeError("single-panel sweep diverged")
    if sweep.cursor != [N_DECISIONS // N_PANELS] * N_PANELS:
        raise RuntimeError(f"round-robin cursors {sweep.cursor}")
    log("smoke mover ok")


def check_trace(seq, choices, kinds):
    if len(choices) < 32:
        raise RuntimeError("fewer than 32 panel choices")
    if any(p != seq[0] for p in seq[:DWELL]):
        raise RuntimeError("first visit was not 8 decisions")
    head = choices[:32]
    rendered = " ".join(
        f"{c['panel']}={kinds[c['panel']]}" for c in head
    )
    log("life 0 curious_prog choices " + rendered)
    log(
        "life 0 curious_prog modes "
        + " ".join(c["mode"] for c in head)
    )
    return head


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
        "n_first": row["n_first"],
        "n_eps": row["n_eps"],
        "n_greedy": row["n_greedy"],
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
    wall.atomic_write(PARTIAL, json.dumps(payload))


def empty_partial(pack, smoke, load):
    return {
        "sha": code_sha(),
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
            kinds, schedule = wall.make_world(life)
            partial["lives"][key] = {"kinds": list(kinds), "policies": {}}
        else:
            kinds = tuple(partial["lives"][key]["kinds"])
            kinds2, schedule = wall.make_world(life)
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
                log(
                    "life 0 panel kinds "
                    + " ".join(f"{i}={kind}" for i, kind in enumerate(kinds))
                )
                head = check_trace(row["panel_seq"], row["choices"], kinds)
                partial["life0_choices"] = [
                    {
                        "t": c["t"],
                        "panel": c["panel"],
                        "kind": kinds[c["panel"]],
                        "mode": c["mode"],
                    }
                    for c in head
                ]
                partial["life0_kinds"] = list(kinds)
            seq = row.pop("panel_seq")
            row.pop("choices")
            partial["lives"][key]["policies"][policy] = public_row(row)
            partial["timing"].setdefault(key, {})[policy] = row["wall_s"]
            if policy == "curious_prog":
                partial["lives"][key]["prog_seq_digest"] = hashlib.sha256(
                    np.asarray(seq, dtype=np.int16).tobytes()
                ).hexdigest()[:12]
            save_partial(partial)
            log(
                f"life {life} {policy} acc {row['accuracy']:.4f} "
                f"tv {row['share']['tv']:.3f} fast {row['share']['fast']:.3f} "
                f"slow {row['share']['slow']:.3f} static {row['share']['static']:.3f} "
                f"wall_s {row['wall_s']:.1f}"
            )
    return partial


def render(payload):
    lines = []
    n = payload["n_lives"]
    lines.append(
        f"n_lives {n} decisions {N_DECISIONS} checks_every {CHECK_EVERY} "
        f"panels 4 opacity {wall.cam.OPACITY} tau {TAU} dwell {DWELL} "
        f"eps {EPS} beta {BETA}"
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
    lines.append(
        "| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi | n_acc_gt_0 |"
    )
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
        f">= {wall.BAR_ACC:.2f} and CI excludes 0 | "
        f"{'PASS' if bars['curiosity_beats_round_robin'] else 'FAIL'} |"
    )
    lines.append(
        f"| 2 ignores the tv | prog {f6(bars['prog_tv'])} surp {f6(bars['surp_tv'])} | "
        f"prog <= {wall.BAR_TV_PROG:.2f} and surp >= {wall.BAR_TV_SURP:.2f} | "
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
            f"| {policy} | {f6(row['static'])} | {f6(row['fast'])} | "
            f"{f6(row['slow'])} | {f6(life_ll)} |"
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
    lines.append("life 0 curious_prog panel choices, first 32")
    lines.append("")
    lines.append("| choice | t | panel | kind | mode |")
    lines.append("|---:|---:|---:|---|---|")
    for i, choice in enumerate(payload["life0_choices"]):
        lines.append(
            f"| {i} | {choice['t']} | {choice['panel']} | {choice['kind']} | {choice['mode']} |"
        )
    lines.append("")
    lines.append(
        " ".join(f"{c['panel']}={c['kind']}" for c in payload["life0_choices"])
    )
    lines.append("")
    lines.append("life 0 panel kinds " + " ".join(
        f"{i}={kind}" for i, kind in enumerate(payload["life0_kinds"])
    ))
    lines.append("")
    lines.append("mechanism")
    lines.append("")
    lines.append(
        "| life | policy | n_choices | n_walk | n_dwell | n_first | n_eps | n_greedy | first_pass_t | wall_s |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for life in range(n):
        for policy in POLICIES:
            row = payload["lives"][str(life)]["policies"][policy]
            passed = row["first_pass_t"]
            passed_s = "" if passed is None else str(int(passed))
            lines.append(
                f"| {life} | {policy} | {row['n_choices']} | {row['n_walk']} | "
                f"{row['n_dwell']} | {row['n_first']} | {row['n_eps']} | {row['n_greedy']} | "
                f"{passed_s} | {f6(row['wall_s'])} |"
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


def torch_version():
    import torch
    import torchvision
    return f"{torch.__version__} torchvision {torchvision.__version__}"


def assemble(partial, pack, load_start, t_wall):
    n_lives = N_LIVES
    for life in range(n_lives):
        got = set(partial["lives"][str(life)]["policies"])
        if got != set(POLICIES):
            raise RuntimeError(f"life {life} missing policies {set(POLICIES) - got}")
    summary = wall.summarise(partial["lives"], n_lives)
    demos = []
    for policy in DEMO_POLICIES:
        demos.extend(partial["lives"]["0"]["policies"][policy]["demo_files"])
    first_pass = [
        partial["lives"][str(i)]["policies"]["curious_prog"]["first_pass_t"]
        for i in range(n_lives)
    ]
    walls = {
        "life0_s": f"{sum(partial['timing']['0'].values()):.3f}",
        "estimate_20_s": f"{sum(partial['timing']['0'].values()) * N_LIVES:.3f}",
        "wall_s": f"{t_wall:.3f}",
        "pack_load_s": f"{pack['load_s']:.3f}",
        "pack_prep_s": f"{pack['prep_s']:.3f}",
        "load_average_start": f"{load_start:.2f}",
        "load_average_end": f"{wall.load_average():.2f}",
        "target_lives": str(n_lives),
        "first_pass_complete_curious_prog": str(sum(t is not None and t < N_DECISIONS for t in first_pass)),
        "mean_first_pass_t_curious_prog": f6(np.mean([
            N_DECISIONS if t is None else t for t in first_pass
        ])),
        "mean_n_choices_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_choices"]
            for i in range(n_lives)
        ])),
        "mean_n_walk_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_walk"]
            for i in range(n_lives)
        ])),
        "mean_n_first_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_first"]
            for i in range(n_lives)
        ])),
        "mean_n_eps_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_eps"]
            for i in range(n_lives)
        ])),
        "mean_n_greedy_curious_prog": f6(np.mean([
            partial["lives"][str(i)]["policies"]["curious_prog"]["n_greedy"]
            for i in range(n_lives)
        ])),
    }
    source = {
        "dataset": "imagenette2-160",
        "split": "val",
        "pool_per_class": str(wall.cam.POOL),
        "opacity": str(wall.cam.OPACITY),
        "scene_shift": str(wall.W.SCENE_SHIFT),
        "dot_shift": str(wall.W.DOT_SHIFT),
        "eye": "torchvision resnet18 IMAGENET1K_V1 frozen eval",
        "softmax": "10 logits, synset table order",
        "logloss": "nats",
        "unseen": "imagenet mean colour",
        "stale": "cell keeps the rgb block from the generation that was seen",
        "tv": "uniform random rgb block per clear cell, default_rng(9700+life) per policy",
        "kinds_rng": "default_rng(9800+life).permutation",
        "images_rng": "default_rng(9900+life).permutation, consumed in panel order",
        "random_rng": "default_rng(8000+life), one uniform panel per visit",
        "choice_rng": "default_rng(10000+life), prog_age_s",
        "reservoir": "GlanceReservoir(life) per panel, progress policies only",
        "mover": "W7 path_sweep, one cursor per panel, not reset when the image changes",
        "visit": "8 decisions, every policy",
        "bandit": "4 panels, W5/W6 prog_age_s",
        "ema_beta": str(BETA),
        "epsilon": str(EPS),
        "tau": str(TAU),
        "dwell": str(DWELL),
        "optimistic": "nearest unvisited on a 4-ring, start panel 0, values start at 0",
        "reward": "sum of per-glance W7 rewards over the visit; EMA once per visit",
        "age": "at the choice, completed decisions minus last_end; last_end is the completed count when the visit ended; greedy step only; stored EMA is not aged",
        "v_max": "largest visit reward so far",
        "noage": "curious_prog_noage ranks the raw EMA on the greedy step",
        "bootstrap": "10000 paired percentile, numpy quantile linear, default_rng(0) per contrast",
        "runner": "vps",
        "device": "cpu",
        "torch_threads": "2",
        "torch": torch_version(),
        "numpy": np.__version__,
        "code_sha256": code_sha(),
        "n_lives_registered": str(N_LIVES),
        "n_lives_run": str(n_lives),
        "short_run": "no",
    }
    payload = {
        "n_lives": n_lives,
        "n_decisions": N_DECISIONS,
        "policies": list(POLICIES),
        "summary": summary,
        "lives": partial["lives"],
        "life0_kinds": partial["life0_kinds"],
        "life0_choices": partial["life0_choices"],
        "smoke": partial["smoke"],
        "walls": walls,
        "source": source,
        "demos": demos,
        "synsets": [
            {"order": i, "wnid": wnid, "imagenet_index": index, "category": name}
            for i, (wnid, name, index) in enumerate(wall.cam.SYNSETS)
        ],
    }
    text = render(payload)
    os.makedirs(OUT_DIR, exist_ok=True)
    wall.atomic_write(os.path.join(OUT_DIR, "RESULTS.md"), text)
    wall.atomic_write(os.path.join(OUT_DIR, "results.json"), json.dumps(payload))
    log("wrote results/w8c/RESULTS.md")
    bars = summary["bars"]
    log(
        "bars "
        f"1 {bars['curiosity_beats_round_robin']} "
        f"2 {bars['ignores_tv']} "
        f"3 {bars['watches_changes']} "
        f"gap {bars['accuracy_gap']:.4f} "
        f"tv_prog {bars['prog_tv']:.3f} tv_surp {bars['surp_tv']:.3f} "
        f"fast {bars['prog_fast']:.3f} slow {bars['prog_slow']:.3f} "
        f"static {bars['prog_static']:.3f}"
    )
    return payload


def prepare():
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 2:
        raise RuntimeError(f"torch threads {torch.get_num_threads()}")
    load = wall.load_average()
    log(f"load_average {load:.2f} torch_threads {torch.get_num_threads()}")
    if load >= 1.5:
        raise SystemExit(f"load_average {load:.2f} >= 1.5; not starting the VPS run")
    check_bandit()
    check_mover()
    pack = wall.prepare_pack()
    smoke = wall.fast_checks(pack)
    return pack, smoke, load


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("checks", "smoke", "all"):
        raise SystemExit("usage: w8c_panels.py [checks|smoke|all]")
    pack, smoke, load = prepare()
    if stage == "checks":
        log("checks done")
        return
    import torch
    torch.set_grad_enabled(False)
    model, eye_s = wall.cam.load_eye()
    log(f"eye_s {eye_s:.3f}")
    partial = load_partial()
    if partial is None:
        partial = empty_partial(pack, smoke, load)
        partial["eye_s"] = eye_s
    else:
        partial["smoke"] = smoke
        log(f"resuming lives {sorted(int(k) for k in partial['lives'])}")
    t0 = time.perf_counter()
    if stage == "smoke":
        run_lives(pack, model, partial, [0])
        spent = sum(partial["timing"]["0"].values())
        log(
            f"smoke done life0_s {spent:.1f} estimate_20_s {spent * N_LIVES:.1f} "
            "one life per policy timed"
        )
        return
    run_lives(pack, model, partial, list(range(N_LIVES)))
    assemble(partial, pack, partial.get("load_average_start", load), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
