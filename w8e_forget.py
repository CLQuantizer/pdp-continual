"""W8-1d: forgetting counts and a recent reservoir on the watching wall.

curious_prog and curious_prog_noage score equation 5 on each panel's last
32 glances, under counts that multiply a seen pixel's 8 greys by gamma 0.8
before the new sight is added. The glance is scored before it is stored.
An empty reservoir scores 0. curious_surp_forget is the mean KT surprise
of the glance under those same counts, report only.

round_robin, random, curious_surp and curious_prog_lifetime are W8-1c.
curious_prog_lifetime is W8-1c's curious_prog. The eye still shows the last
block painted in a cell. Gamma and the reservoir length are fixed.
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

import w8b_wall as wall
import w8c_panels as panels
import w8d_progress as w8d

ROOT = wall.ROOT
OUT_DIR = os.path.join(ROOT, "results", "w8e")
DEMO_DIR = os.path.join(ROOT, "internal", "w8_demo", "wall_d")
PARTIAL = os.path.join(OUT_DIR, "partial.json")
W8D_JSON = os.path.join(ROOT, "results", "w8d", "results.json")
panels.DEMO_DIR = DEMO_DIR

N_DECISIONS = panels.N_DECISIONS
N_PANELS = panels.N_PANELS
DWELL = panels.DWELL
TAU = panels.TAU
CHECK_EVERY = panels.CHECK_EVERY
N_LIVES = panels.N_LIVES
EPS = panels.EPS
BETA = panels.BETA
KINDS = panels.KINDS
N = wall.W.N
G = wall.W.G
ALPHA = wall.W.ALPHA

# Fixed by the spec. Do not tune.
GAMMA = 0.8
RECENT = 32

POLICIES = (
    "random",
    "round_robin",
    "curious_prog",
    "curious_surp",
    "curious_prog_noage",
    "curious_surp_forget",
    "curious_prog_lifetime",
)
FORGET = {
    "curious_prog": ("prog", True),
    "curious_prog_noage": ("prog", False),
    "curious_surp_forget": ("surp", True),
}
UNCHANGED = (
    ("random", "random"),
    ("round_robin", "round_robin"),
    ("curious_surp", "curious_surp"),
    ("curious_prog_lifetime", "curious_prog"),
)
EXACT_FIELDS = w8d.EXACT_FIELDS
PARENT = {
    "curious_prog": "curious_prog",
    "curious_prog_noage": "curious_prog_noage",
    "curious_surp_forget": "curious_surp",
}


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


class FirstPassStop(Exception):
    """Life 0 curious_prog failed the TV-smallest gate."""

    def __init__(self, rewards):
        self.rewards = rewards


def sight(grey, vis, sx, sy):
    ys = sy[vis]
    xs = sx[vis]
    vs = grey[vis].astype(np.int64)
    return ys, xs, vs


class RecentReservoir:
    """Last `cap` glances, first in, first out. Each sight keeps its grey."""

    def __init__(self, cap=RECENT):
        self.cap = int(cap)
        self.n_seen = 0
        self.slots = []

    def include(self, ys, xs, vs):
        ys = np.asarray(ys, dtype=np.int64).copy()
        xs = np.asarray(xs, dtype=np.int64).copy()
        vs = np.asarray(vs, dtype=np.int64).copy()
        if vs.shape != ys.shape or xs.shape != ys.shape:
            raise RuntimeError("reservoir sight shape")
        self.n_seen += 1
        self.slots.append((ys, xs, vs))
        if len(self.slots) > self.cap:
            self.slots.pop(0)


def learn_forget(counts, ys, xs, vs, gamma=GAMMA):
    """Multiply each seen pixel's 8 grey counts by gamma, then add the sight."""
    if vs.size == 0:
        return
    if ys.size > 1:
        key = ys.astype(np.int64) * N + xs.astype(np.int64)
        if np.unique(key).size != key.size:
            raise RuntimeError("glance repeated a pixel")
    block = np.array(counts[ys, xs], dtype=np.float64, copy=True)
    block *= float(gamma)
    block[np.arange(vs.size), vs] += 1.0
    counts[ys, xs] = block


def progress_forget(counts, slots, ys, xs, vs, gamma=GAMMA):
    """Equation 5 on stored sights. The update is gamma, then +1 on the new grey.

    Sights already stored, and only pixels this glance shows. The current
    glance is not in `slots`.
    """
    if len(vs) == 0 or not slots:
        return 0.0
    parts_k = []
    parts_v = []
    for sy, sx, sv in slots:
        if sy.size == 0:
            continue
        parts_k.append(sy * N + sx)
        parts_v.append(sv)
    if not parts_k:
        return 0.0
    ak = np.concatenate(parts_k)
    av = np.concatenate(parts_v)
    key = ys.astype(np.int64) * N + xs.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    v_s = vs.astype(np.int64)[order]
    pos = np.searchsorted(key_s, ak)
    inb = np.where(pos < key_s.size, pos, 0)
    ok = (pos < key_s.size) & (key_s[inb] == ak)
    if not np.any(ok):
        return 0.0
    ak = ak[ok]
    av = av[ok]
    vn = v_s[inb[ok]]
    yy = ak // N
    xx = ak % N
    c_rows = counts[yy, xx]
    t = c_rows.sum(axis=1).astype(np.float64)
    c = c_rows[np.arange(av.size), av].astype(np.float64)
    match = (av == vn).astype(np.float64)
    c_after = c * float(gamma) + match
    t_after = t * float(gamma) + 1.0
    before = -np.log2((c + ALPHA) / (t + ALPHA * G))
    after = -np.log2((c_after + ALPHA) / (t_after + ALPHA * G))
    reward = float(np.sum(before - after))
    if not math.isfinite(reward):
        raise RuntimeError("nonfinite forget progress")
    return reward


def forget_reward(counts, reservoir, grey, vis, sx, sy):
    """Score stored sights, learn the glance, then store it."""
    ys, xs, vs = sight(grey, vis, sx, sy)
    reward = progress_forget(counts, reservoir.slots, ys, xs, vs)
    learn_forget(counts, ys, xs, vs)
    reservoir.include(ys, xs, vs)
    return float(reward)


def surp_forget_reward(counts, grey, vis, sx, sy):
    """Mean KT bits of this glance under the forgetting counts, then learn it."""
    ys, xs, vs = sight(grey, vis, sx, sy)
    if vs.size == 0:
        return 0.0
    bits = float(wall.W.kt_from_counts(counts[ys, xs], vs).mean())
    if not math.isfinite(bits):
        raise RuntimeError("nonfinite forget surprise")
    learn_forget(counts, ys, xs, vs)
    return bits


class ForgetLife(panels.PanelLife):
    """W8-1b's life. The visit reward uses forgetting counts."""

    def __init__(self, life, kinds, schedule, pack, policy, demo):
        if policy not in FORGET:
            raise RuntimeError(policy)
        if GAMMA != 0.8 or RECENT != 32:
            raise RuntimeError("gamma or reservoir length was tuned")
        super().__init__(life, kinds, schedule, pack, PARENT[policy], demo)
        self.policy = policy
        kind, _use_age = FORGET[policy]
        self.forget_kind = kind
        self.counts = np.zeros((N_PANELS, N, N, G), dtype=np.float64)
        if kind == "prog":
            if self.mode != "prog":
                raise RuntimeError("progress arm did not inherit prog")
            self.reservoirs = [RecentReservoir(RECENT) for _ in range(N_PANELS)]
            for reservoir in self.reservoirs:
                if reservoir.cap != RECENT:
                    raise RuntimeError("reservoir cap")
        elif kind == "surp":
            if self.mode != "surp":
                raise RuntimeError("surprise arm did not inherit surp")
            self.reservoirs = None
        else:
            raise RuntimeError(kind)

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
        if self.forget_kind == "prog":
            reward = forget_reward(
                self.counts[panel], self.reservoirs[panel], grey, vis, sx, sy
            )
        elif self.forget_kind == "surp":
            reward = surp_forget_reward(self.counts[panel], grey, vis, sx, sy)
        else:
            raise RuntimeError(self.forget_kind)
        self.share_n[kind] += 1
        self.panel_seq.append(int(panel))
        completed = int(t) + 1
        if completed % CHECK_EVERY == 0:
            self.checkpoint(completed)
        return float(reward)


def run_forget(life, kinds, schedule, pack, policy, model, demo):
    run = ForgetLife(life, kinds, schedule, pack, policy, demo)
    sweep = panels.PanelSweep(N_DECISIONS)
    _mode, use_age = FORGET[policy]
    bandit = panels.PanelBandit(life, use_age)
    visit_sum = 0.0
    first_pass = []
    gate_checked = False
    t0 = time.perf_counter()
    panel = 0
    for t in range(N_DECISIONS):
        if t % DWELL == 0:
            panel, _how = bandit.choose(t)
            visit_sum = 0.0
        x, y = sweep.step(panel)
        run.px[panel], run.py[panel] = x, y
        visit_sum += run.observe(panel, t)
        if (t + 1) % DWELL == 0:
            bandit.finish(panel, visit_sum, t + 1)
            if len(first_pass) < N_PANELS:
                first_pass.append({
                    "panel": int(panel),
                    "kind": kinds[panel],
                    "reward": float(visit_sum),
                })
            if (
                life == 0
                and policy == "curious_prog"
                and not gate_checked
                and len(first_pass) == N_PANELS
            ):
                gate_checked = True
                w8d.assert_first_pass(bandit)
                ok, _by = w8d.tv_smallest(first_pass)
                if not ok:
                    raise FirstPassStop(list(first_pass))
    for p in range(N_PANELS):
        if sweep.cursor[p] != int(run.share_n[kinds[p]]):
            raise RuntimeError("sweep cursor is not the number of glances")
    if sum(sweep.cursor) != N_DECISIONS:
        raise RuntimeError("sweep cursors do not cover the life")
    row = run.finish(model, bandit)
    row["wall_s"] = time.perf_counter() - t0
    row["first_pass"] = first_pass
    if bandit.n_choices != N_DECISIONS // DWELL:
        raise RuntimeError("choice count")
    w8d.assert_first_pass(bandit)
    if len(first_pass) != N_PANELS:
        raise RuntimeError("first-pass reward count")
    seq = row["panel_seq"]
    for start in range(0, N_DECISIONS, DWELL):
        if any(p != seq[start] for p in seq[start:start + DWELL]):
            raise RuntimeError("a visit left its panel")
    return row


def run_policy(life, kinds, schedule, pack, policy, model, demo):
    if policy in FORGET:
        return run_forget(life, kinds, schedule, pack, policy, model, demo)
    if policy == "curious_prog_lifetime":
        # W8-1c curious_prog, including its reservoir and its reward order.
        return w8d.run_policy(
            life, kinds, schedule, pack, "curious_prog", model, False
        )
    write_demo = bool(demo) and policy in panels.DEMO_POLICIES
    return w8d.run_policy(
        life, kinds, schedule, pack, policy, model, write_demo
    )


def _manual_bits(c, t):
    return -math.log2((float(c) + ALPHA) / (float(t) + ALPHA * G))


def check_formula():
    """Gamma 0.8 matches the hand update. Gamma 1 matches equation 5."""
    if GAMMA != 0.8 or RECENT != 32:
        raise RuntimeError("constants drifted")
    counts = np.zeros((N, N, G), dtype=np.float64)
    counts[4, 5, :] = np.arange(G, dtype=np.float64)
    ys = np.array([4, 1], dtype=np.int64)
    xs = np.array([5, 2], dtype=np.int64)
    vs = np.array([3, 0], dtype=np.int64)
    learn_forget(counts, ys, xs, vs)
    expect = np.arange(G, dtype=np.float64) * GAMMA
    expect[3] += 1.0
    if not np.allclose(counts[4, 5], expect):
        raise RuntimeError("forget update missed the seen grey")
    if counts[1, 2, 0] != 1.0 or not np.all(counts[1, 2, 1:] == 0.0):
        raise RuntimeError("forget update missed a fresh pixel")

    counts = np.zeros((N, N, G), dtype=np.float64)
    counts[4, 5, 3] = 1.0
    slots = [(
        np.array([4], dtype=np.int64),
        np.array([5], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )]
    same = progress_forget(
        counts, slots,
        np.array([4]), np.array([5]), np.array([3]),
    )
    c_after = 1.0 * GAMMA + 1.0
    t_after = 1.0 * GAMMA + 1.0
    want = _manual_bits(1.0, 1.0) - _manual_bits(c_after, t_after)
    if abs(same - want) > 1e-12:
        raise RuntimeError(f"same-grey progress {same} != {want}")
    other = progress_forget(
        counts, slots,
        np.array([4]), np.array([5]), np.array([1]),
    )
    want_other = _manual_bits(1.0, 1.0) - _manual_bits(GAMMA, t_after)
    if abs(other - want_other) > 1e-12:
        raise RuntimeError(f"other-grey progress {other} != {want_other}")
    if progress_forget(counts, [], np.array([4]), np.array([5]), np.array([3])) != 0.0:
        raise RuntimeError("empty reservoir did not score 0")

    rng = np.random.default_rng(0)
    mine_c = np.zeros((N, N, G), dtype=np.float64)
    obs_c = np.zeros((N, N, G), dtype=np.int32)
    recent = RecentReservoir(RECENT)
    observed = wall.W.ObservedReservoir(0, cap=256)
    max_abs = 0.0
    for _step in range(30):
        n = int(rng.integers(1, 50))
        ys = rng.integers(0, N, size=n)
        xs = rng.integers(0, N, size=n)
        _uniq, idx = np.unique(ys * N + xs, return_index=True)
        ys = ys[np.sort(idx)]
        xs = xs[np.sort(idx)]
        vs = rng.integers(0, G, size=ys.size)
        got = progress_forget(mine_c, recent.slots, ys, xs, vs, gamma=1.0)
        if len(observed.slots) == 0 or vs.size == 0:
            exp = 0.0
        else:
            exp = float(observed.progress(obs_c, ys, xs, vs))
        max_abs = max(max_abs, abs(got - exp))
        learn_forget(mine_c, ys, xs, vs, gamma=1.0)
        np.add.at(obs_c, (ys, xs, vs), 1)
        recent.include(ys, xs, vs)
        observed.include(ys, xs, vs)
        if not np.allclose(mine_c, obs_c):
            raise RuntimeError("gamma 1 counts diverged from equation 5")
    if max_abs > 1e-9:
        raise RuntimeError(f"gamma 1 progress disagreed {max_abs}")

    fifo = RecentReservoir(RECENT)
    fifo.include(np.array([7]), np.array([8]), np.array([1]))
    for i in range(RECENT):
        fifo.include(np.array([i]), np.array([0]), np.array([0]))
    if fifo.n_seen != RECENT + 1 or len(fifo.slots) != RECENT:
        raise RuntimeError("fifo size")
    if int(fifo.slots[0][0][0]) != 0 or int(fifo.slots[-1][0][0]) != RECENT - 1:
        raise RuntimeError("fifo did not drop the oldest glance")
    log(
        f"formula ok gamma {GAMMA} recent {RECENT} "
        f"same {same:.6f} other {other:.6f} eq5_max_abs {max_abs:.3e}"
    )
    log("W8E_FORMULA_OK")
    return {
        "gamma": GAMMA,
        "recent": RECENT,
        "same_grey": same,
        "other_grey": other,
        "eq5_max_abs": max_abs,
    }


def stare_visits(pack, schedule, panel, n_visits):
    """One panel, from t = 0, visit reward under curious_prog's learner."""
    sweep = panels.PanelSweep(N_DECISIONS)
    counts = np.zeros((N, N, G), dtype=np.float64)
    reservoir = RecentReservoir(RECENT)
    rows = []
    for visit in range(n_visits):
        total = 0.0
        t0 = sweep.cursor[panel]
        image = int(schedule[panel, t0])
        for _glance in range(DWELL):
            x, y = sweep.step(panel)
            t = sweep.cursor[panel] - 1
            index = int(schedule[panel, t])
            grey, vis, sx, sy = wall.glance(
                pack["maps"], pack["greys"][index], pack["occs"][index], x, y
            )
            total += forget_reward(counts, reservoir, grey, vis, sx, sy)
        t1 = sweep.cursor[panel] - 1
        rows.append({
            "visit": visit + 1,
            "t0": int(t0),
            "t1": int(t1),
            "image": image,
            "reward": float(total),
        })
    return rows


def check_static(pack, schedule, kinds):
    panel = kinds.index("static")
    if schedule[panel].min() != schedule[panel].max():
        raise RuntimeError("static panel changed image")
    rows = stare_visits(pack, schedule, panel, 6)
    first = rows[0]["reward"]
    sixth = rows[5]["reward"]
    if first == 0.0:
        ratio = None
        below = False
    else:
        ratio = sixth / first
        below = sixth < 0.1 * first
    log("smoke static life 0 curious_prog stare, 6 visits")
    for row in rows:
        log(
            f"static visit {row['visit']} t {row['t0']}-{row['t1']} "
            f"reward {f6(row['reward'])}"
        )
    ratio_s = "nan" if ratio is None else f6(ratio)
    log(
        f"static visit6 {f6(sixth)} visit1 {f6(first)} "
        f"ratio {ratio_s} below_10pct {str(below).lower()}"
    )
    return {
        "visits": rows,
        "visit1": first,
        "visit6": sixth,
        "ratio": ratio,
        "below_10pct": bool(below),
    }


def check_fast(pack, schedule, kinds):
    panel = kinds.index("fast")
    if int(schedule[panel, 32]) == int(schedule[panel, 31]):
        raise RuntimeError("fast did not replace at t=32")
    if int(schedule[panel, 64]) == int(schedule[panel, 63]):
        raise RuntimeError("fast did not replace at t=64")
    rows = stare_visits(pack, schedule, panel, 10)
    log("smoke fast life 0 curious_prog stare, visits around the first two replacements")
    for row in rows:
        log(
            f"fast visit {row['visit']} t {row['t0']}-{row['t1']} "
            f"image {row['image']} reward {f6(row['reward'])}"
        )
    # Visit 4 is t 24-31 (before t=32). Visit 5 is the first visit after it.
    # Visit 8 is t 56-63. Visit 9 is the first visit after t=64.
    pairs = []
    for before_i, after_i, at in ((4, 5, 32), (8, 9, 64)):
        before = rows[before_i - 1]
        after = rows[after_i - 1]
        if before["t1"] != at - 1 or after["t0"] != at:
            raise RuntimeError("replacement visit alignment")
        higher = after["reward"] > before["reward"]
        pairs.append({
            "t": at,
            "before_visit": before_i,
            "after_visit": after_i,
            "before": before["reward"],
            "after": after["reward"],
            "higher": bool(higher),
        })
        log(
            f"fast replacement t={at} before visit {before_i} {f6(before['reward'])} "
            f"after visit {after_i} {f6(after['reward'])} "
            f"higher {str(higher).lower()}"
        )
    return {"visits": rows, "replacements": pairs}


def load_w8d():
    with open(W8D_JSON, encoding="utf-8") as f:
        return json.load(f)


def check_exact(partial, previous, life):
    got_block = partial["lives"][str(life)]["policies"]
    exp_block = previous["lives"][str(life)]["policies"]
    for got_name, exp_name in UNCHANGED:
        got = got_block[got_name]
        exp = exp_block[exp_name]
        for field in EXACT_FIELDS:
            if field not in got or field not in exp:
                raise RuntimeError(f"life {life} {got_name} missing {field}")
            if not w8d._same(got[field], exp[field]):
                diff = w8d._max_abs(got[field], exp[field])
                raise RuntimeError(
                    f"life {life} {got_name} != w8d {exp_name} "
                    f"field {field} max_abs {diff}"
                )
    log(f"smoke reproduce life {life} exact")
    log(f"W8E_REPRO_OK life {life}")


def stored_from(row):
    seq = row.pop("panel_seq")
    choices = row.pop("choices")
    first_pass = row.pop("first_pass", [])
    row.pop("first_pass_by", None)
    row.pop("tv_smallest", None)
    stored = panels.public_row(row)
    if first_pass:
        stored["first_pass"] = first_pass
    return stored, seq, choices


def policy_order(life, partial):
    done = set(partial["lives"].get(str(life), {}).get("policies", {}))
    if life == 0 and "curious_prog" not in done:
        return ("curious_prog",) + tuple(p for p in POLICIES if p != "curious_prog")
    return POLICIES


def run_lives(pack, model, partial, lives, previous):
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
        for policy in policy_order(life, partial):
            if policy in partial["lives"][key]["policies"]:
                log(f"life {life} {policy} resume")
                continue
            demo = life == 0
            log(f"life {life} {policy} start")
            try:
                row = run_policy(life, kinds, schedule, pack, policy, model, demo)
            except FirstPassStop as stop:
                by = {item["kind"]: float(item["reward"]) for item in stop.rewards}
                partial["smoke"]["first_pass_curious_prog"] = by
                partial["smoke"]["tv_smallest"] = False
                partial["stopped"] = "tv_not_smallest"
                save_partial(partial)
                write_stop(partial, by)
                log("W8E_GATE_STOP")
                raise SystemExit(2)
            if life == 0 and policy == "curious_prog":
                log(
                    "life 0 panel kinds "
                    + " ".join(f"{i}={kind}" for i, kind in enumerate(kinds))
                )
                head = panels.check_trace(row["panel_seq"], row["choices"], kinds)
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
                by = {item["kind"]: float(item["reward"]) for item in row["first_pass"]}
                partial["smoke"]["first_pass_curious_prog"] = by
                partial["smoke"]["tv_smallest"] = True
                log("W8E_GATE_PASS")
            stored, seq, _choices = stored_from(row)
            partial["lives"][key]["policies"][policy] = stored
            partial["timing"].setdefault(key, {})[policy] = row["wall_s"]
            if policy in ("curious_prog", "curious_prog_lifetime"):
                digest = hashlib.sha256(
                    np.asarray(seq, dtype=np.int16).tobytes()
                ).hexdigest()[:12]
                partial["lives"][key][f"{policy}_seq_digest"] = digest
            save_partial(partial)
            log(
                f"life {life} {policy} acc {row['accuracy']:.4f} "
                f"tv {row['share']['tv']:.3f} fast {row['share']['fast']:.3f} "
                f"slow {row['share']['slow']:.3f} static {row['share']['static']:.3f} "
                f"wall_s {row['wall_s']:.1f}"
            )
        have = set(partial["lives"][key]["policies"])
        if set(POLICIES) <= have:
            check_exact(partial, previous, life)
            if life == 0:
                partial["smoke"]["reproduce_life0"] = True
                save_partial(partial)
                log("GATE life 0 reproduce passed")
        log(f"W8E_LIFE_DONE {life}")
    return partial


def save_partial(payload):
    wall.atomic_write(PARTIAL, json.dumps(payload))


def clear_demos():
    if not os.path.isdir(DEMO_DIR):
        return
    for name in os.listdir(DEMO_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(DEMO_DIR, name))


def load_partial():
    if not os.path.isfile(PARTIAL):
        return None
    with open(PARTIAL, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("sha") != code_sha():
        log("partial sha mismatch; ignoring")
        return None
    if payload.get("stopped") == "tv_not_smallest":
        log("partial is a tv stop; ignoring")
        return None
    return payload


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


def wait_for_load():
    while True:
        load = wall.load_average()
        log(f"load_average_1m {load:.2f}")
        if load < 1.0:
            return load
        log("W8E_WAITING")
        log("full run waiting 300s for load_average_1m < 1.0")
        time.sleep(300)


def summarise(store, n_lives):
    old = wall.POLICIES
    wall.POLICIES = POLICIES
    try:
        return wall.summarise(store, n_lives)
    finally:
        wall.POLICIES = old


def _visit_table(rows):
    lines = [
        "| visit | t0 | t1 | image | reward |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['visit']} | {row['t0']} | {row['t1']} | "
            f"{row['image']} | {f6(row['reward'])} |"
        )
    return lines


def probe_lines(smoke):
    static = smoke["static"]
    fast = smoke["fast"]
    lines = ["", "forgetting probes, life 0, curious_prog learner, print only", ""]
    ratio = static["ratio"]
    ratio_s = "nan" if ratio is None else f6(ratio)
    lines.append(
        "static stare 6 visits "
        f"visit1 {f6(static['visit1'])} visit6 {f6(static['visit6'])} "
        f"ratio {ratio_s} below_10pct {str(static['below_10pct']).lower()}"
    )
    lines.append("")
    lines.extend(_visit_table(static["visits"]))
    lines.append("")
    lines.append("fast stare, first two replacements")
    lines.append("")
    lines.extend(_visit_table(fast["visits"]))
    lines.append("")
    lines.append("| replacement_t | before_visit | before | after_visit | after | higher |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for pair in fast["replacements"]:
        lines.append(
            f"| {pair['t']} | {pair['before_visit']} | {f6(pair['before'])} | "
            f"{pair['after_visit']} | {f6(pair['after'])} | "
            f"{str(pair['higher']).lower()} |"
        )
    lines.append("")
    return lines


def write_stop(partial, by):
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [
        "stopped before the scored run",
        "",
        "life 0 curious_prog first-pass visit rewards",
        "",
        "| kind | reward |",
        "|---|---:|",
    ]
    for kind in ("static", "tv", "fast", "slow"):
        lines.append(f"| {kind} | {f6(by[kind])} |")
    smallest = min(by, key=lambda k: (by[k], k))
    lines.append("")
    lines.append("tv_smallest false")
    lines.append(f"smallest {smallest}")
    lines.append("")
    lines.extend(probe_lines(partial["smoke"]))
    lines.append("scored lives not run")
    lines.append("")
    text = "\n".join(lines)
    wall.atomic_write(os.path.join(OUT_DIR, "RESULTS.md"), text)
    wall.atomic_write(os.path.join(OUT_DIR, "results.json"), json.dumps({
        "stopped": "tv_not_smallest",
        "first_pass": by,
        "smoke": partial.get("smoke", {}),
    }))
    log("STOP tv is not the smallest; wrote results/w8e/RESULTS.md")


def render(payload):
    old = panels.POLICIES
    panels.POLICIES = POLICIES
    try:
        text = panels.render(payload)
    finally:
        panels.POLICIES = old
    lines = probe_lines(payload["smoke"])
    lines.append(
        "life 0 random round_robin curious_surp match results/w8d exactly; "
        "curious_prog_lifetime matches results/w8d curious_prog exactly"
    )
    lines.append("")
    lines.append("life 0 first-pass visit rewards, curious_prog")
    lines.append("")
    lines.append("| kind | reward |")
    lines.append("|---|---:|")
    by = payload["smoke"]["first_pass_curious_prog"]
    for kind in ("static", "tv", "fast", "slow"):
        lines.append(f"| {kind} | {f6(by[kind])} |")
    lines.append("")
    lines.append(
        "tv_smallest " + str(payload["smoke"]["tv_smallest"]).lower()
    )
    lines.append("")
    text = text.rstrip() + "\n" + "\n".join(lines)
    return text


def assemble(partial, pack, load_start, t_wall):
    n_lives = N_LIVES
    for life in range(n_lives):
        got = set(partial["lives"][str(life)]["policies"])
        if got != set(POLICIES):
            raise RuntimeError(f"life {life} missing policies {set(POLICIES) - got}")
    summary = summarise(partial["lives"], n_lives)
    demos = []
    for policy in panels.DEMO_POLICIES:
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
        "first_pass_complete_curious_prog": str(sum(
            t is not None and t < N_DECISIONS for t in first_pass
        )),
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
        "gamma": str(GAMMA),
        "reservoir": "last 32 glances FIFO per panel for curious_prog and curious_prog_noage; none for curious_surp_forget; curious_prog_lifetime uses W8-1c ObservedReservoir cap 256",
        "counts": "on each sight, the pixel's 8 grey counts are multiplied by gamma 0.8 and then the sight is added; KT alpha 0.5",
        "mover": "W7 path_sweep, one cursor per panel, not reset when the image changes",
        "visit": "8 decisions, every policy",
        "bandit": "4 panels, W5/W6 prog_age_s",
        "ema_beta": str(BETA),
        "epsilon": str(EPS),
        "tau": str(TAU),
        "dwell": str(DWELL),
        "optimistic": "nearest unvisited on a 4-ring, start panel 0, values start at 0",
        "reward": "curious_prog and curious_prog_noage sum equation 5 on the last 32 stored glances under the counts before the forgetting update, then learn and store the glance; curious_surp_forget sums mean KT surprise under the forgetting counts; curious_prog_lifetime is W8-1c curious_prog; curious_surp sums per-glance surprise; EMA once per visit",
        "age": "at the choice, completed decisions minus last_end; last_end is the completed count when the visit ended; greedy step only; stored EMA is not aged",
        "v_max": "largest visit reward so far",
        "noage": "curious_prog_noage ranks the raw EMA on the greedy step",
        "lifetime_arm": "curious_prog_lifetime is W8-1c curious_prog",
        "forget_surp_arm": "curious_surp_forget is report only",
        "bootstrap": "10000 paired percentile, numpy quantile linear, default_rng(0) per contrast",
        "runner": "vps",
        "device": "cpu",
        "torch_threads": "2",
        "torch": panels.torch_version(),
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
    log("DONE wrote results/w8e/RESULTS.md")
    log("W8E_DONE")
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


def setup_torch():
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)
    if torch.get_num_threads() != 2:
        raise RuntimeError(f"torch threads {torch.get_num_threads()}")
    return torch


def prepare(stage):
    setup_torch()
    formula = check_formula()
    panels.check_bandit()
    panels.check_mover()
    if stage == "checks":
        return None, {"formula": formula}, wall.load_average()
    load = wait_for_load()
    log(f"SMOKE start load_average_1m {load:.2f} torch_threads 2")
    pack = wall.prepare_pack()
    smoke = wall.fast_checks(pack)
    kinds, schedule = wall.make_world(0)
    smoke["formula"] = formula
    smoke["static"] = check_static(pack, schedule, kinds)
    smoke["fast"] = check_fast(pack, schedule, kinds)
    smoke["life0_kinds"] = list(kinds)
    return pack, smoke, load


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("checks", "smoke", "all"):
        raise SystemExit("usage: w8e_forget.py [checks|smoke|all]")
    log(f"code_sha256 {code_sha()}")
    pack, smoke, load = prepare(stage)
    if stage == "checks":
        log("checks done")
        return
    previous = load_w8d()
    model, eye_s = wall.cam.load_eye()
    log(f"eye_s {eye_s:.3f}")
    partial = load_partial()
    if partial is None:
        partial = empty_partial(pack, smoke, load)
        partial["eye_s"] = eye_s
        clear_demos()
    else:
        partial["smoke"]["static"] = smoke["static"]
        partial["smoke"]["fast"] = smoke["fast"]
        partial["smoke"]["formula"] = smoke["formula"]
        partial["smoke"]["coverage"] = smoke["coverage"]
        log(f"resuming lives {sorted(int(k) for k in partial['lives'])}")
    t0 = time.perf_counter()
    run_lives(pack, model, partial, [0], previous)
    if stage == "smoke":
        spent = sum(partial["timing"]["0"].values())
        log(f"smoke done life0_s {spent:.1f} estimate_20_s {spent * N_LIVES:.1f}")
        return
    full_load = wait_for_load()
    log(f"FULL start load_average_1m {full_load:.2f}")
    log("W8E_FULL_START")
    run_lives(pack, model, partial, list(range(N_LIVES)), previous)
    assemble(partial, pack, partial.get("load_average_start", load), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
