"""W8-1c: the panel visit reward is progress on the stored sights.

curious_prog and curious_prog_noage keep one ObservedReservoir per panel
(cap 256, seed = life, the same seed W8-1b passed to GlanceReservoir).
The visit reward is the sum of ObservedReservoir.progress over the visit
(equation 5, no mask), scored on the sights stored before this glance.
The glance is included after the score. An empty reservoir scores 0, so
the first glance of each panel is 0. curious_prog_glance is W8-1b's
curious_prog: GlanceReservoir and R._reward, report only.

round_robin, random and curious_surp are the W8-1b code path.
World, lives 0-19, L = 1024, the sweep, the eye and the bars are W8-1b.
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
import w8b_wall as wall
import w8c_panels as panels

ROOT = wall.ROOT
OUT_DIR = os.path.join(ROOT, "results", "w8d")
DEMO_DIR = os.path.join(ROOT, "internal", "w8_demo", "wall_c")
PARTIAL = os.path.join(OUT_DIR, "partial.json")
W8C_JSON = os.path.join(ROOT, "results", "w8c", "results.json")
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

POLICIES = (
    "random",
    "round_robin",
    "curious_prog",
    "curious_surp",
    "curious_prog_noage",
    "curious_prog_glance",
)
CURIOUS = {
    "curious_prog": ("prog", True),
    "curious_surp": ("surp", True),
    "curious_prog_noage": ("prog", False),
    "curious_prog_glance": ("prog", True),
}
UNCHANGED = (
    ("random", "random"),
    ("round_robin", "round_robin"),
    ("curious_surp", "curious_surp"),
)
EXACT_FIELDS = (
    "accuracy",
    "logloss",
    "panel_accuracy",
    "panel_logloss",
    "share",
    "share_n",
    "checks",
    "n_choices",
    "n_walk",
    "n_dwell",
    "n_first",
    "n_eps",
    "n_greedy",
    "first_pass_t",
)


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

    def __init__(self, rewards, tv_visits):
        self.rewards = rewards
        self.tv_visits = tv_visits


def _sight(grey, vis, sx, sy):
    ys = sy[vis]
    xs = sx[vis]
    vs = grey[vis].astype(np.int64)
    return ys, xs, vs


def _checked_progress(reward):
    if isinstance(reward, tuple):
        raise RuntimeError("observed progress returned a split")
    reward = float(reward)
    if not math.isfinite(reward):
        raise RuntimeError("nonfinite observed progress")
    return reward


def observed_reward(counts, reservoir, grey, vis, sx, sy, order="before"):
    """Score ObservedReservoir.progress, then count this glance.

    order "before" scores the sights stored before this glance, then includes
    it. An empty reservoir scores 0, so the first glance into a panel is 0.
    order "after" includes this glance and then scores, which is R._reward.
    curious_prog and curious_prog_noage use "before".
    """
    ys, xs, vs = _sight(grey, vis, sx, sy)
    if order == "before":
        if vs.size == 0 or len(reservoir.slots) == 0:
            reward = 0.0
        else:
            reward = _checked_progress(reservoir.progress(counts, ys, xs, vs))
        reservoir.include(ys, xs, vs)
    elif order == "after":
        reservoir.include(ys, xs, vs)
        if vs.size == 0:
            reward = 0.0
        else:
            reward = _checked_progress(reservoir.progress(counts, ys, xs, vs))
    else:
        raise ValueError(order)
    if vs.size:
        np.add.at(counts, (ys, xs, vs), 1)
    return float(reward)


def glance_reward(counts, reservoir, grey, vis, sx, sy, order="after"):
    """GlanceReservoir reward. order "after" is R._reward, unchanged."""
    if order == "after":
        return float(R._reward("prog", counts, reservoir, grey, vis, sx, sy))
    if order != "before":
        raise ValueError(order)
    ys, xs, vs = _sight(grey, vis, sx, sy)
    if vs.size == 0 or len(reservoir.slots) == 0:
        reward = 0.0
    else:
        reward = float(reservoir.progress(counts, ys, xs, vs))
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite glance progress")
    reservoir.include(ys, xs)
    if vs.size:
        np.add.at(counts, (ys, xs, vs), 1)
    return float(reward)


class ProgressLife(panels.PanelLife):
    """W8-1b's life, with the reservoir chosen by the policy."""

    def __init__(self, life, kinds, schedule, pack, policy, demo):
        if policy == "curious_prog_glance":
            parent = "curious_prog"
            kind = "glance"
        elif policy in ("curious_prog", "curious_prog_noage"):
            parent = policy
            kind = "observed"
        else:
            parent = policy
            kind = None
        super().__init__(life, kinds, schedule, pack, parent, demo)
        self.policy = policy
        self.reservoir_kind = kind
        if kind == "observed":
            if self.mode != "prog":
                raise RuntimeError("observed reservoir on a non-progress policy")
            self.reservoirs = [
                wall.W.ObservedReservoir(int(life), cap=256)
                for _ in range(N_PANELS)
            ]
            for reservoir in self.reservoirs:
                if reservoir.cap != 256:
                    raise RuntimeError("reservoir cap")
        elif kind == "glance":
            if self.mode != "prog" or not self.reservoirs:
                raise RuntimeError("glance arm did not build a reservoir")
            for reservoir in self.reservoirs:
                if type(reservoir) is not wall.W.GlanceReservoir:
                    raise RuntimeError("glance arm reservoir type")

    def observe(self, panel, t):
        if self.reservoir_kind != "observed":
            return super().observe(panel, t)
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
            reservoir = self.reservoirs[panel]
            reward = observed_reward(
                self.counts[panel], reservoir, grey, vis, sx, sy
            )
        self.share_n[kind] += 1
        self.panel_seq.append(int(panel))
        completed = int(t) + 1
        if completed % CHECK_EVERY == 0:
            self.checkpoint(completed)
        return reward


def run_policy(life, kinds, schedule, pack, policy, model, demo):
    run = ProgressLife(life, kinds, schedule, pack, policy, demo)
    sweep = panels.PanelSweep(N_DECISIONS)
    bandit = None
    if policy in CURIOUS:
        _mode, use_age = CURIOUS[policy]
        bandit = panels.PanelBandit(life, use_age)
    rng = np.random.default_rng(8000 + int(life))
    panel = 0
    visit_sum = 0.0
    first_pass = []
    trace_tv = life == 0 and policy == "curious_prog"
    tv_visits = []
    gate_fail = None
    gate_by = None
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
        reward = run.observe(panel, t)
        if trace_tv and kinds[panel] == "tv":
            if t % DWELL == 0:
                tv_visits.append([])
            tv_visits[-1].append({"t": int(t), "reward": float(reward)})
        visit_sum += reward
        if bandit is not None and (t + 1) % DWELL == 0:
            bandit.finish(panel, visit_sum, t + 1)
            if len(first_pass) < N_PANELS:
                first_pass.append({
                    "panel": int(panel),
                    "kind": kinds[panel],
                    "reward": float(visit_sum),
                })
            if trace_tv and gate_by is None and len(first_pass) == N_PANELS:
                assert_first_pass(bandit)
                ok, gate_by = tv_smallest(first_pass)
                if not ok:
                    gate_fail = gate_by
            if (
                gate_fail is not None
                and len(tv_visits) >= 3
                and len(tv_visits[2]) == DWELL
            ):
                raise FirstPassStop(gate_fail, tv_visits[:3])
    if gate_fail is not None:
        complete = [visit for visit in tv_visits if len(visit) == DWELL]
        raise FirstPassStop(gate_fail, complete[:3])
    for p in range(N_PANELS):
        if sweep.cursor[p] != int(run.share_n[kinds[p]]):
            raise RuntimeError("sweep cursor is not the number of glances")
    if sum(sweep.cursor) != N_DECISIONS:
        raise RuntimeError("sweep cursors do not cover the life")
    row = run.finish(model, bandit)
    row["wall_s"] = time.perf_counter() - t0
    row["first_pass"] = first_pass
    if trace_tv:
        row["first_pass_by"] = gate_by
        row["tv_smallest"] = True
    if bandit is not None:
        if bandit.n_choices != N_DECISIONS // DWELL:
            raise RuntimeError("choice count")
        assert_first_pass(bandit)
        if len(first_pass) != N_PANELS:
            raise RuntimeError("first-pass reward count")
    seq = row["panel_seq"]
    for start in range(0, N_DECISIONS, DWELL):
        if any(p != seq[start] for p in seq[start:start + DWELL]):
            raise RuntimeError("a visit left its panel")
    return row


def check_seed_and_split():
    glance = wall.W.GlanceReservoir(0)
    observed = wall.W.ObservedReservoir(0, cap=256)
    if observed.cap != 256 or glance.cap != 256:
        raise RuntimeError("cap is not 256")
    if glance.rng.bit_generator.state != observed.rng.bit_generator.state:
        raise RuntimeError("reservoir seeds differ")
    ys = np.array([4], dtype=np.int64)
    xs = np.array([5], dtype=np.int64)
    gres = wall.W.GlanceReservoir(1)
    ores = wall.W.ObservedReservoir(1, cap=256)
    gc = np.zeros((wall.W.N, wall.W.N, wall.W.G), dtype=np.int32)
    oc = np.zeros_like(gc)
    gres.include(ys, xs)
    ores.include(ys, xs, np.array([1], dtype=np.int64))
    first_g = gres.progress(gc, ys, xs, np.array([1], dtype=np.int64))
    first_o = ores.progress(oc, ys, xs, np.array([1], dtype=np.int64))
    if abs(float(first_g) - float(first_o)) > 1e-9:
        raise RuntimeError(f"first sight disagreed {first_g} {first_o}")
    gc[4, 5, 1] += 1
    oc[4, 5, 1] += 1
    gres.include(ys, xs)
    ores.include(ys, xs, np.array([4], dtype=np.int64))
    mixed_g = float(gres.progress(gc, ys, xs, np.array([4], dtype=np.int64)))
    mixed_o = float(ores.progress(oc, ys, xs, np.array([4], dtype=np.int64)))
    if not mixed_o < mixed_g:
        raise RuntimeError(
            f"mixed greys glance {mixed_g} observed {mixed_o}"
        )
    log(
        f"smoke split first {first_g:.6f} mixed_glance {mixed_g:.6f} "
        f"mixed_observed {mixed_o:.6f}"
    )
    return {"first": float(first_g), "mixed_glance": mixed_g, "mixed_observed": mixed_o}


def check_still(pack):
    """Life 0, static panel, first 64 sweep glances.

    The two progress functions must agree to 1e-9 under each order. The
    printed before/after columns are the observed rewards. There is no bar
    on the difference between the orders.
    """
    kinds, schedule = wall.make_world(0)
    panel = kinds.index("static")
    if schedule[panel].min() != schedule[panel].max():
        raise RuntimeError("static panel changed image")
    index = int(schedule[panel, 0])
    scene = pack["greys"][index]
    occ = pack["occs"][index]
    orders = ("before", "after")
    sweep = {name: panels.PanelSweep(N_DECISIONS) for name in orders}
    glance = {name: wall.W.GlanceReservoir(0) for name in orders}
    observed = {name: wall.W.ObservedReservoir(0, cap=256) for name in orders}
    g_counts = {
        name: np.zeros((wall.W.N, wall.W.N, wall.W.G), dtype=np.int32)
        for name in orders
    }
    o_counts = {name: np.zeros_like(g_counts[name]) for name in orders}
    series = {name: [] for name in orders}
    max_abs = {name: 0.0 for name in orders}
    for t in range(64):
        for name in orders:
            x, y = sweep[name].step(panel)
            grey, vis, sx, sy = wall.glance(pack["maps"], scene, occ, x, y)
            g = glance_reward(
                g_counts[name], glance[name], grey, vis, sx, sy, order=name
            )
            o = observed_reward(
                o_counts[name], observed[name], grey, vis, sx, sy, order=name
            )
            max_abs[name] = max(max_abs[name], abs(g - o))
            series[name].append(float(o))
    for name in orders:
        if not np.array_equal(g_counts[name], o_counts[name]):
            raise RuntimeError(f"still-panel counts diverged order {name}")
    if series["before"][0] != 0.0:
        raise RuntimeError(f"first glance reward {series['before'][0]}")
    agree = max_abs["before"] <= 1e-9 and max_abs["after"] <= 1e-9
    log(
        "smoke still max_abs_before "
        f"{max_abs['before']:.3e} max_abs_after {max_abs['after']:.3e} "
        f"agree {str(agree).lower()}"
    )
    log("life 0 static panel, first 64 glances, both orders")
    log("before: score stored sights, then include the glance")
    log("after: include the glance, then score (W8-1b order)")
    log("t before after")
    for t in range(64):
        log(f"{t} {f6(series['before'][t])} {f6(series['after'][t])}")
    if not agree:
        raise RuntimeError(
            "still progress disagreed "
            f"before {max_abs['before']} after {max_abs['after']}"
        )
    return {
        "max_abs": max_abs["before"],
        "max_abs_before": max_abs["before"],
        "max_abs_after": max_abs["after"],
        "agree": True,
        "n_glances": 64,
        "panel": "static",
        "first_before": series["before"][0],
        "orders": series,
    }


def load_w8c():
    with open(W8C_JSON, encoding="utf-8") as f:
        return json.load(f)


def _same(got, exp):
    if isinstance(got, dict) or isinstance(exp, dict):
        if not isinstance(got, dict) or not isinstance(exp, dict):
            return False
        if set(got) != set(exp):
            return False
        return all(_same(got[k], exp[k]) for k in got)
    if isinstance(got, list) or isinstance(exp, list):
        if not isinstance(got, list) or not isinstance(exp, list):
            return False
        if len(got) != len(exp):
            return False
        return all(_same(a, b) for a, b in zip(got, exp))
    if isinstance(got, (int, float)) and isinstance(exp, (int, float)) and not isinstance(got, bool):
        return float(got) == float(exp)
    return got == exp


def _max_abs(got, exp):
    if isinstance(got, dict) and isinstance(exp, dict):
        vals = [_max_abs(got[k], exp[k]) for k in got if k in exp]
        return max(vals) if vals else 0.0
    if isinstance(got, list) and isinstance(exp, list) and got and isinstance(got[0], (int, float, list, dict)):
        vals = [_max_abs(a, b) for a, b in zip(got, exp)]
        return max(vals) if vals else 0.0
    if isinstance(got, (int, float)) and isinstance(exp, (int, float)) and not isinstance(got, bool):
        return abs(float(got) - float(exp))
    return 0.0 if got == exp else math.inf


def check_exact(partial, w8c, life, pairs):
    got_block = partial["lives"][str(life)]["policies"]
    exp_block = w8c["lives"][str(life)]["policies"]
    for got_name, exp_name in pairs:
        got = got_block[got_name]
        exp = exp_block[exp_name]
        for field in EXACT_FIELDS:
            if field not in got or field not in exp:
                raise RuntimeError(f"life {life} {got_name} missing {field}")
            if not _same(got[field], exp[field]):
                diff = _max_abs(got[field], exp[field])
                raise RuntimeError(
                    f"life {life} {got_name} != w8c {exp_name} field {field} max_abs {diff}"
                )
    log(f"smoke reproduce life {life} exact")


def check_glance_choices(choices, kinds, w8c):
    exp = w8c["life0_choices"]
    if len(choices) < len(exp):
        raise RuntimeError("glance choices shorter than w8c")
    for i, row in enumerate(exp):
        got = choices[i]
        if int(got["t"]) != int(row["t"]) or int(got["panel"]) != int(row["panel"]):
            raise RuntimeError(f"glance choice {i} diverged")
        if got["mode"] != row["mode"] or kinds[got["panel"]] != row["kind"]:
            raise RuntimeError(f"glance choice {i} kind or mode diverged")
    log("smoke glance choices match w8c life 0 first 32")


def assert_first_pass(bandit):
    if bandit.n_first != N_PANELS or bandit.first_pass_t != N_PANELS * DWELL:
        raise RuntimeError(
            f"first pass n {bandit.n_first} t {bandit.first_pass_t}"
        )
    first = [c["panel"] for c in bandit.choices[:N_PANELS]]
    if first != list(range(N_PANELS)):
        raise RuntimeError(f"first pass order {first}")
    if any(c["mode"] != "first" for c in bandit.choices[:N_PANELS]):
        raise RuntimeError("first pass mode")


def tv_smallest(rewards):
    if len(rewards) != N_PANELS:
        raise RuntimeError("first pass is not one visit per panel")
    by = {}
    for row in rewards:
        if row["kind"] in by:
            raise RuntimeError("repeated first-pass kind")
        by[row["kind"]] = float(row["reward"])
    if set(by) != set(KINDS):
        raise RuntimeError(f"first-pass kinds {sorted(by)}")
    others = [by[k] for k in KINDS if k != "tv"]
    ok = by["tv"] < min(others)
    rendered = " ".join(f"{row['kind']}={row['reward']:.10f}" for row in rewards)
    log("life 0 curious_prog first-pass " + rendered)
    log(
        "life 0 curious_prog first-pass smallest "
        + min(by, key=lambda k: (by[k], k))
    )
    return ok, by


def order_lines(still):
    orders = still.get("orders") or {}
    before = orders.get("before") or []
    after = orders.get("after") or []
    lines = [
        "life 0 static panel, first 64 glances, both orders",
        "",
        "before: score stored sights, then include the glance",
        "after: include the glance, then score (W8-1b order)",
        "",
        "| t | before | after |",
        "|---:|---:|---:|",
    ]
    for t, (b, a) in enumerate(zip(before, after)):
        lines.append(f"| {t} | {f6(b)} | {f6(a)} |")
    lines.append("")
    return lines


def write_stop(partial, by, tv_visits):
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [
        "stopped before the scored run",
        "",
        "curious_prog scores ObservedReservoir.progress on the sights stored before the glance, then includes the glance.",
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
    lines.append("tv per-glance rewards, first visits, score then include")
    lines.append("")
    lines.append("| visit | glance | t | reward |")
    lines.append("|---:|---:|---:|---:|")
    reported = list(tv_visits[:3])
    for visit_i, visit in enumerate(reported, start=1):
        for glance_i, step in enumerate(visit, start=1):
            lines.append(
                f"| {visit_i} | {glance_i} | {int(step['t'])} | {f6(step['reward'])} |"
            )
    lines.append("")
    lines.append(f"tv_visits_reported {len(reported)}")
    if len(reported) < 3:
        lines.append("tv_visits_short true")
    still = partial.get("smoke", {}).get("still", {})
    if still:
        lines.append(
            "still_panel life 0 static first 64 glances "
            f"max_abs_before {still.get('max_abs_before', still['max_abs']):.3e} "
            f"max_abs_after {still.get('max_abs_after', still['max_abs']):.3e} "
            f"agree {str(still['agree']).lower()}"
        )
        lines.append("")
        lines.extend(order_lines(still))
    lines.append("scored lives not run")
    lines.append("demo strips not written; the life stopped at the first-pass gate")
    lines.append("")
    text = "\n".join(lines)
    wall.atomic_write(os.path.join(OUT_DIR, "RESULTS.md"), text)
    wall.atomic_write(os.path.join(OUT_DIR, "results.json"), json.dumps({
        "stopped": "tv_not_smallest",
        "first_pass": by,
        "tv_visits": reported,
        "smoke": partial.get("smoke", {}),
    }))
    log("STOP tv is not the smallest; wrote results/w8d/RESULTS.md")


def policy_order(life, partial):
    done = set(partial["lives"].get(str(life), {}).get("policies", {}))
    if life == 0 and "curious_prog" not in done:
        return ("curious_prog",) + tuple(p for p in POLICIES if p != "curious_prog")
    return POLICIES


def run_lives(pack, model, partial, lives, w8c):
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
                partial["smoke"]["first_pass_curious_prog"] = stop.rewards
                partial["smoke"]["tv_smallest"] = False
                partial["smoke"]["tv_visits"] = stop.tv_visits
                partial["stopped"] = "tv_not_smallest"
                save_partial(partial)
                write_stop(partial, stop.rewards, stop.tv_visits)
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
                by = row.pop("first_pass_by")
                ok = row.pop("tv_smallest")
                if by is None or not ok:
                    raise RuntimeError("life 0 curious_prog returned without a passed gate")
                partial["smoke"]["first_pass_curious_prog"] = by
                partial["smoke"]["tv_smallest"] = True
            if life == 0 and policy == "curious_prog_glance":
                check_glance_choices(row["choices"], kinds, w8c)
                partial["smoke"]["first_pass_curious_prog_glance"] = {
                    row["kind"]: float(row["reward"]) for row in row["first_pass"]
                }
            seq = row.pop("panel_seq")
            row.pop("choices")
            first_pass = row.pop("first_pass")
            stored = panels.public_row(row)
            if first_pass:
                stored["first_pass"] = first_pass
            partial["lives"][key]["policies"][policy] = stored
            partial["timing"].setdefault(key, {})[policy] = row["wall_s"]
            if policy in ("curious_prog", "curious_prog_glance"):
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
            pairs = list(UNCHANGED)
            if "curious_prog_glance" in have:
                pairs.append(("curious_prog_glance", "curious_prog"))
            check_exact(partial, w8c, life, pairs)
            if life == 0:
                partial["smoke"]["reproduce_life0"] = True
                save_partial(partial)
                log("GATE life 0 reproduce passed")
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
        log("full run waiting 300s for load_average_1m < 1.0")
        time.sleep(300)


def summarise(store, n_lives):
    old = wall.POLICIES
    wall.POLICIES = POLICIES
    try:
        return wall.summarise(store, n_lives)
    finally:
        wall.POLICIES = old


def render(payload):
    old = panels.POLICIES
    panels.POLICIES = POLICIES
    try:
        text = panels.render(payload)
    finally:
        panels.POLICIES = old
    lines = ["", "progress smoke", ""]
    still = payload["smoke"]["still"]
    lines.append(
        "still_panel life 0 static first 64 glances "
        f"max_abs_before {still.get('max_abs_before', still['max_abs']):.3e} "
        f"max_abs_after {still.get('max_abs_after', still['max_abs']):.3e} "
        f"agree {str(still['agree']).lower()}"
    )
    lines.append("")
    lines.extend(order_lines(still))
    lines.append(
        "life 0 random round_robin curious_surp match results/w8c exactly; "
        "curious_prog_glance matches results/w8c curious_prog exactly"
    )
    lines.append("")
    lines.append("life 0 first-pass visit rewards")
    lines.append("")
    lines.append("| policy | static | tv | fast | slow |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in ("curious_prog", "curious_prog_glance"):
        by = {}
        for row in payload["lives"]["0"]["policies"][policy]["first_pass"]:
            by[row["kind"]] = row["reward"]
        lines.append(
            f"| {policy} | "
            + " | ".join(f6(by[k]) for k in ("static", "tv", "fast", "slow"))
            + " |"
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
        "reservoir": "ObservedReservoir(life) cap 256 per panel for curious_prog and curious_prog_noage; GlanceReservoir(life) per panel for curious_prog_glance; none for curious_surp",
        "mover": "W7 path_sweep, one cursor per panel, not reset when the image changes",
        "visit": "8 decisions, every policy",
        "bandit": "4 panels, W5/W6 prog_age_s",
        "ema_beta": str(BETA),
        "epsilon": str(EPS),
        "tau": str(TAU),
        "dwell": str(DWELL),
        "optimistic": "nearest unvisited on a 4-ring, start panel 0, values start at 0",
        "reward": "curious_prog and curious_prog_noage sum ObservedReservoir.progress on the sights stored before the glance, then include the glance (an empty reservoir scores 0); curious_prog_glance sums GlanceReservoir.progress after including the glance, as W8-1b; curious_surp sums per-glance surprise; EMA once per visit",
        "age": "at the choice, completed decisions minus last_end; last_end is the completed count when the visit ended; greedy step only; stored EMA is not aged",
        "v_max": "largest visit reward so far",
        "noage": "curious_prog_noage ranks the raw EMA on the greedy step",
        "glance_arm": "curious_prog_glance is W8-1b curious_prog",
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
    log("DONE wrote results/w8d/RESULTS.md")
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
    load = wall.load_average()
    log(f"load_average_1m {load:.2f} torch_threads 2 stage {stage}")
    panels.check_bandit()
    panels.check_mover()
    split = check_seed_and_split()
    pack = wall.prepare_pack()
    smoke = wall.fast_checks(pack)
    smoke["still"] = check_still(pack)
    smoke["split"] = split
    return pack, smoke, load


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("checks", "smoke", "all"):
        raise SystemExit("usage: w8d_progress.py [checks|smoke|all]")
    log(f"code_sha256 {code_sha()}")
    pack, smoke, load = prepare(stage)
    if stage == "checks":
        log("checks done")
        return
    w8c = load_w8c()
    model, eye_s = wall.cam.load_eye()
    log(f"eye_s {eye_s:.3f}")
    partial = load_partial()
    if partial is None:
        partial = empty_partial(pack, smoke, load)
        partial["eye_s"] = eye_s
        clear_demos()
    else:
        partial["smoke"]["still"] = smoke["still"]
        partial["smoke"]["split"] = smoke["split"]
        partial["smoke"]["coverage"] = smoke["coverage"]
        log(f"resuming lives {sorted(int(k) for k in partial['lives'])}")
    t0 = time.perf_counter()
    run_lives(pack, model, partial, [0], w8c)
    if stage == "smoke":
        spent = sum(partial["timing"]["0"].values())
        log(f"smoke done life0_s {spent:.1f} estimate_20_s {spent * N_LIVES:.1f}")
        return
    full_load = wait_for_load()
    log(f"FULL start load_average_1m {full_load:.2f}")
    run_lives(pack, model, partial, list(range(N_LIVES)), w8c)
    assemble(partial, pack, partial.get("load_average_start", load), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
