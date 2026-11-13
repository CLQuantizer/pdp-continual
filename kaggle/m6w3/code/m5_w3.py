"""M5-W3: a visit bandit on the W3 ring. Reward is a rate per decision, travel included.

Imports curio, w3_world and w3_headroom and does not edit them. World, learner, corridor
glances, life length and the score W are the W3 ones. No verdict.
"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("CURIO_LR") != "0.004":
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["CURIO_LR"] = "0.004"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import time
from multiprocessing import get_context

import numpy as np

import curio
from w3_headroom import LR, W3Compressor, evaluate, next_action, shortest
from w3_world import World3

DWELL = 8
SEEDS = (6, 7, 8, 9, 10, 11)
CONTROLLERS_512 = ("rand_dwell", "oracle7", "err", "prog")
CONTROLLERS_2048 = ("prog", "oracle7")
# 20 minutes, the spec's cutoff for staying on this machine.
LOCAL_WALL_LIMIT_S = 20 * 60


def nearest_unvisited(src, unvisited, n=9):
    """Nearest unvisited room. Equal distances are broken toward +index (right)."""
    best, best_key = None, None
    for dst in unvisited:
        right = (dst - src) % n
        left = (src - dst) % n
        dist = right if right <= left else left
        key = (dist, 0 if right <= left else 1, dst)
        if best_key is None or key < best_key:
            best, best_key = dst, key
    return best


def _empty_hist():
    return {c: [] for c in list(curio.CH) + ["corridor"]}


def _shares(channels):
    """Decision shares. A stay counts for its room channel. A move counts as corridor."""
    n = len(channels)
    counts = {c: 0 for c in curio.CH}
    corridor = 0
    for ch in channels:
        if ch == "corridor":
            corridor += 1
        else:
            counts[ch] += 1
    share = {c: counts[c] / n for c in curio.CH}
    return share, corridor / n


def _quarters(channels):
    q = len(channels) // 4
    out = []
    for i in range(4):
        share, corridor = _shares(channels[i * q:(i + 1) * q])
        out.append({"share": share, "corridor": corridor})
    return out


def _finish(seed, controller, L, comp, world, hist, channels, t0, eval_on, visits):
    eval_s = 0.0
    if eval_on:
        e0 = time.perf_counter()
        W, held = evaluate(comp, world, hist)
        eval_s = time.perf_counter() - e0
    else:
        W, held = float("nan"), {}
    share, corridor = _shares(channels)
    wall = time.perf_counter() - t0
    return {
        "seed": int(seed),
        "controller": controller,
        "L": int(L),
        "W": float(W),
        "held": {c: float(held[c]) for c in held},
        "share": {c: float(share[c]) for c in curio.CH},
        "corridor_share": float(corridor),
        "quarter": [
            {"share": {c: float(block["share"][c]) for c in curio.CH}, "corridor": float(block["corridor"])}
            for block in _quarters(channels)
        ],
        "room_channel": list(world.rooms),
        "wall_s": wall,
        "eval_s": eval_s,
        "learn_s": wall - eval_s,
        "finite": bool(math.isfinite(W) and all(math.isfinite(v) for v in held.values())),
        "visits": visits,
    }


def run_fixed(seed, policy, L, score=True):
    """rand_dwell / oracle7, the same planner and rng stream as w3_headroom.run_life.

    Policy rng is default_rng(1_000 + seed), also drawn on for N-channel noise.
    That is what makes W match w3_headroom.py. Intermediate checkpoints are skipped;
    evaluate does not change the compressor or the world, so the final W matches.
    """
    t0 = time.perf_counter()
    world = World3(seed)
    rng = np.random.default_rng(1_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = _empty_hist()
    state = {"kind": policy, "room": 0, "plan": []}
    channels = []
    for _ in range(L):
        action = next_action(state, world.rooms, rng)
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        comp.bits(channel, hist[channel], symbols, learn=True)
        hist[channel] += symbols
        channels.append(channel)
    return _finish(seed, policy, L, comp, world, hist, channels, t0, score, None)


def _prog_glance(comp, hist, res, seen, res_rng, channel, symbols):
    """One dwell glance, prog_kind hist. Same order as curio.live.

    curio binds hs = res[c] and only then appends to that list, so the
    per-glance denominator is the reservoir length after this glance is stored.
    """
    reservoir = res[channel]
    before = comp.score_pairs(channel, reservoir) if reservoir else 0.0
    comp.bits(channel, hist[channel], symbols, learn=True)
    after = comp.score_pairs(channel, reservoir) if reservoir else 0.0
    hs = reservoir
    full = hist[channel] + symbols
    max_k = max(curio.ORDERS)
    for i in range(len(hist[channel]), len(full)):
        pair = (tuple(full[max(0, i - max_k):i]), full[i])
        seen[channel] += 1
        if len(res[channel]) < curio.R_RES:
            res[channel].append(pair)
        else:
            k = int(res_rng.integers(seen[channel]))
            if k < curio.R_RES:
                res[channel][k] = pair
    r = (before - after) / max(len(hs), 1)
    hist[channel] += symbols
    return r


def _err_glance(comp, hist, channel, symbols):
    b = comp.bits(channel, hist[channel], symbols, learn=True)
    hist[channel] += symbols
    return b / curio.B


def _learn_corridor(comp, hist, channel, symbols):
    comp.bits(channel, hist[channel], symbols, learn=True)
    hist[channel] += symbols


def _choose_target(src, unvisited, v, rng):
    """Optimistic first pass, then eps-greedy. Tie break matches curio.live's max key."""
    if unvisited:
        target = nearest_unvisited(src, unvisited)
        unvisited.remove(target)
        return target
    if rng.random() < curio.EPS:
        return int(rng.integers(9))
    return max(range(9), key=lambda k: (v[k], rng.random()))


def _ema(v, nvis, room, reward):
    if nvis[room] == 0:
        v[room] = reward
    else:
        v[room] = (1.0 - curio.BETA) * v[room] + curio.BETA * reward
    nvis[room] += 1


def run_bandit(seed, kind, L, score=True, trace_visits=0):
    """err or prog. Choices use default_rng(10_000 + seed).

    N-channel noise uses default_rng(1_000 + seed), the W3 noise stream, and is
    not mixed with choice draws. The progress reservoir uses default_rng(20_000 + seed).
    A visit is the shortest path plus 8 dwells. The visit reward is the sum of the
    dwell glances' per-glance rewards divided by travel plus dwell. A visit cut off
    by the end of life ends there and still updates the EMA.
    """
    if kind not in ("err", "prog"):
        raise ValueError(kind)
    t0 = time.perf_counter()
    world = World3(seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    world_rng = np.random.default_rng(1_000 + seed)
    res_rng = np.random.default_rng(20_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = _empty_hist()
    res = {c: [] for c in curio.CH}
    seen = {c: 0 for c in curio.CH}
    v = [0.0] * 9
    nvis = [0] * 9
    unvisited = set(range(9))
    channels = []
    visits = []
    t = 0
    while t < L:
        target = _choose_target(world.room, unvisited, v, choice_rng)
        path = shortest(world.room, target)
        actions = []
        travel = 0
        while path and t < L:
            action = path.pop(0)
            _room, channel, symbols = world.step(action, world_rng)
            _learn_corridor(comp, hist, channel, symbols)
            channels.append(channel)
            actions.append(action)
            travel += 1
            t += 1
        sum_r = 0.0
        dwell = 0
        while dwell < DWELL and t < L:
            _room, channel, symbols = world.step("stay", world_rng)
            if kind == "prog":
                r = _prog_glance(comp, hist, res, seen, res_rng, channel, symbols)
            else:
                r = _err_glance(comp, hist, channel, symbols)
            sum_r += r
            channels.append(channel)
            dwell += 1
            t += 1
        n_dec = travel + dwell
        reward = sum_r / n_dec
        _ema(v, nvis, target, reward)
        if trace_visits and len(visits) < trace_visits:
            visits.append({
                "target": int(target),
                "channel": world.rooms[target],
                "travel": int(travel),
                "actions": list(actions),
                "dwell": int(dwell),
                "sum_progress": float(sum_r),
                "visit_reward": float(reward),
                "decisions": int(n_dec),
            })
    kept = visits if trace_visits else None
    return _finish(seed, kind, L, comp, world, hist, channels, t0, score, kept)


def run_controller(seed, controller, L, score=True, trace_visits=0):
    if controller in ("rand_dwell", "oracle7"):
        return run_fixed(seed, controller, L, score=score)
    return run_bandit(seed, controller, L, score=score, trace_visits=trace_visits)


def _job(args):
    seed, controller, L = args
    out = run_controller(seed, controller, L, score=True, trace_visits=0)
    print(
        f's{seed} {controller} L={L} W {out["W"]:.4f} '
        f'N {out["share"]["N"]:.3f} Z {out["share"]["Z"]:.3f} '
        f'corridor {out["corridor_share"]:.3f} wall {out["wall_s"]:.1f}s',
        flush=True,
    )
    out["visits"] = None
    return out


def _check_nearest():
    # Start at 0 with everyone unvisited: distance 0 wins.
    if nearest_unvisited(0, set(range(9))) != 0:
        raise SystemExit("nearest: start should stay in room 0")
    # Distance 1 ties between 1 (right) and 8 (left).
    if nearest_unvisited(0, {1, 8}) != 1:
        raise SystemExit("nearest: tie at distance 1 should go right")
    if nearest_unvisited(3, {1, 5}) != 5:
        raise SystemExit("nearest: tie at distance 2 should go right")
    if nearest_unvisited(3, {2}) != 2:
        raise SystemExit("nearest: a unique left neighbour should still be chosen")
    if nearest_unvisited(0, {4, 5}) != 4:
        raise SystemExit("nearest: tie at distance 4 should go right")


def _fmt_trace(visits):
    lines = ["prog seed 0, first 6 visits (target, travel, summed progress, visit reward)"]
    for i, v in enumerate(visits, 1):
        actions = ",".join(v["actions"]) if v["actions"] else "-"
        lines.append(
            f'  visit {i} target {v["target"]} {v["channel"]} '
            f'travel {v["travel"]} [{actions}] dwell {v["dwell"]} '
            f'sum_progress {v["sum_progress"]:.6f} '
            f'visit_reward {v["visit_reward"]:.6f} decisions {v["decisions"]}'
        )
    return lines


def _estimate(timed):
    """Wall of the scored job from the L=64 lives. prog is scaled up.

    L=64 prog is still inside the first pass, so its reservoir is not full on
    every glance. Steady-state scoring touches R_RES pairs; the first-pass
    average is 176 pairs. 256/176 is the scale applied to the prog learn time.
    """
    by = {r["controller"]: r for r in timed}
    jobs = []
    for seed in SEEDS:
        for controller in CONTROLLERS_512:
            jobs.append((seed, controller, 512))
        for controller in CONTROLLERS_2048:
            jobs.append((seed, controller, 2048))
    raw = 0.0
    conservative = 0.0
    prog_scale = curio.R_RES / 176.0
    for _seed, controller, L in jobs:
        row = by[controller]
        learn = row["learn_s"] / 64.0 * L
        one = learn + row["eval_s"]
        raw += one
        if controller == "prog":
            conservative += learn * prog_scale + row["eval_s"]
        else:
            conservative += one
    # imap chunksize 1 splits consecutive jobs across the pool.
    return {
        "serial_raw_s": raw,
        "serial_conservative_s": conservative,
        "pool2_raw_s": raw / 2.0,
        "pool2_conservative_s": conservative / 2.0,
        "prog_reservoir_scale": prog_scale,
        "n_jobs": len(jobs),
    }


def smoke():
    """Three checks from the spec. Exit before any scored seed if one fails."""
    _check_nearest()
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    say("smoke check 3 first: L=64 seed 0, four controllers")
    timed = []
    trace = None
    for controller in CONTROLLERS_512:
        row = run_controller(0, controller, 64, score=True, trace_visits=6 if controller == "prog" else 0)
        say(
            f'  {controller} W {row["W"]:.6f} finite {row["finite"]} '
            f'wall {row["wall_s"]:.2f}s learn {row["learn_s"]:.2f}s eval {row["eval_s"]:.2f}s '
            f'corridor {row["corridor_share"]:.3f}'
        )
        if not row["finite"]:
            raise SystemExit(f"smoke check 3 failed: {controller} W is not finite")
        total = sum(row["share"].values()) + row["corridor_share"]
        if abs(total - 1.0) > 1e-12:
            raise SystemExit(f"smoke check 3 failed: {controller} shares sum to {total}")
        timed.append(row)
        if controller == "prog":
            trace = row["visits"]
    say("smoke check 2: 6-visit prog trace, seed 0")
    if trace is None or len(trace) < 6:
        raise SystemExit("smoke check 2 failed: fewer than 6 visits")
    for line in _fmt_trace(trace):
        say(line)
    # First pass walks every room once, nearest, ties to the right: 0,1,2,3,4,5,...
    for i, room in enumerate(range(6)):
        if trace[i]["target"] != room:
            raise SystemExit(f"smoke check 2 failed: visit {i + 1} target {trace[i]['target']} != {room}")
        if trace[i]["dwell"] != DWELL:
            raise SystemExit(f"smoke check 2 failed: visit {i + 1} dwell {trace[i]['dwell']}")
        if i == 0 and trace[i]["travel"] != 0:
            raise SystemExit("smoke check 2 failed: first visit should not travel")
        if i > 0 and trace[i]["travel"] != 1:
            raise SystemExit(f"smoke check 2 failed: visit {i + 1} travel {trace[i]['travel']}")
        expect = trace[i]["sum_progress"] / trace[i]["decisions"]
        if abs(expect - trace[i]["visit_reward"]) > 1e-12:
            raise SystemExit("smoke check 2 failed: visit reward is not sum_progress / decisions")
        if not math.isfinite(trace[i]["sum_progress"]) or not math.isfinite(trace[i]["visit_reward"]):
            raise SystemExit("smoke check 2 failed: non-finite visit reward")

    say("smoke check 1: rand_dwell and oracle7 vs w3_headroom.run_life, seed 0, L=512")
    import w3_headroom
    repro = {}
    for policy in ("rand_dwell", "oracle7"):
        ref = w3_headroom.run_life(0, policy, 512)
        mine = run_fixed(0, policy, 512, score=True)
        dW = abs(mine["W"] - ref["W"])
        held_d = {c: abs(mine["held"][c] - ref["held"][c]) for c in curio.LEARN}
        d_corr = abs(mine["corridor_share"] - ref["corridor_share"])
        say(
            f'  {policy} m5 W {mine["W"]:.12f} w3 W {ref["W"]:.12f} '
            f'abs {dW:.3e} corridor_abs {d_corr:.3e}'
        )
        say("  held abs " + " ".join(f"{c} {held_d[c]:.3e}" for c in curio.LEARN))
        if dW > 1e-9 or max(held_d.values()) > 1e-9 or d_corr > 1e-12:
            raise SystemExit(f"smoke check 1 failed for {policy}")
        repro[policy] = {
            "W_m5": mine["W"], "W_w3": ref["W"], "abs_W": dW,
            "held_abs": held_d, "corridor_abs": d_corr,
        }
    # Published kernel numbers, same code path.
    published = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/w3/results.json")))
    for policy, got in repro.items():
        hit = [r for r in published["runs"] if r["policy"] == policy and r["L"] == 512 and r["seed"] == 0]
        if len(hit) != 1:
            raise SystemExit(f"published w3 row missing for {policy}")
        d_pub = abs(got["W_m5"] - hit[0]["W"])
        say(f'  {policy} vs published results/w3 abs {d_pub:.3e}')
        if d_pub > 1e-9:
            raise SystemExit(f"smoke check 1 failed against published W for {policy}")
        got["abs_published"] = d_pub
        got["W_published"] = hit[0]["W"]

    est = _estimate(timed)
    say(
        f'estimate serial raw {est["serial_raw_s"]:.0f}s conservative {est["serial_conservative_s"]:.0f}s '
        f'pool2 raw {est["pool2_raw_s"]:.0f}s ({est["pool2_raw_s"] / 60:.1f} min) '
        f'conservative {est["pool2_conservative_s"]:.0f}s ({est["pool2_conservative_s"] / 60:.1f} min)'
    )
    if est["pool2_conservative_s"] > LOCAL_WALL_LIMIT_S:
        say("estimate exceeds 20 min; do not start the scored run here")
        over = True
    else:
        say("estimate is within 20 min; scored run stays on this machine, pool of 2")
        over = False
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "reproduce": repro,
        "trace": trace,
        "L64": [{k: r[k] for k in ("controller", "W", "finite", "wall_s", "learn_s", "eval_s", "corridor_share")}
                for r in timed],
        "estimate": est,
        "over_limit": over,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
    }


def mean_std(xs):
    a = np.asarray(list(xs), np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def _pick(runs, controller, L, seed):
    hit = [r for r in runs if r["controller"] == controller and r["L"] == L and r["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} L={L} seed {seed}, got {len(hit)}")
    return hit[0]


def summarize(runs):
    keys = sorted({(r["controller"], r["L"]) for r in runs}, key=lambda k: (k[1], CONTROLLERS_512.index(k[0]) if k[0] in CONTROLLERS_512 else 9))
    by = {}
    for controller, L in keys:
        rows = sorted((r for r in runs if r["controller"] == controller and r["L"] == L), key=lambda r: r["seed"])
        mu, sd = mean_std(r["W"] for r in rows)
        held_mu, held_sd = {}, {}
        for c in curio.LEARN:
            held_mu[c], held_sd[c] = mean_std(r["held"][c] for r in rows)
        share_mu = {}
        for c in curio.CH:
            share_mu[c] = mean_std(r["share"][c] for r in rows)[0]
        corr_mu = mean_std(r["corridor_share"] for r in rows)[0]
        quarter = []
        for i in range(4):
            q_share = {c: mean_std(r["quarter"][i]["share"][c] for r in rows)[0] for c in curio.CH}
            q_corr = mean_std(r["quarter"][i]["corridor"] for r in rows)[0]
            quarter.append({"share": q_share, "corridor": q_corr})
        by[f"{controller}/L{L}"] = {
            "controller": controller,
            "L": L,
            "n": len(rows),
            "W_mean": mu,
            "W_std_ddof0": sd,
            "held_mean": held_mu,
            "held_std_ddof0": held_sd,
            "share_mean": share_mu,
            "corridor_mean": corr_mu,
            "quarter_mean": quarter,
            "per_seed": [{
                "seed": r["seed"],
                "W": r["W"],
                "held": r["held"],
                "share": r["share"],
                "corridor_share": r["corridor_share"],
                "room_channel": r["room_channel"],
                "wall_s": r["wall_s"],
            } for r in rows],
        }
    gaps = []
    for s in SEEDS:
        prog = _pick(runs, "prog", 512, s)["W"]
        dwell = _pick(runs, "rand_dwell", 512, s)["W"]
        oracle = _pick(runs, "oracle7", 512, s)["W"]
        err = _pick(runs, "err", 512, s)["W"]
        gaps.append({
            "seed": s,
            "prog": prog,
            "rand_dwell": dwell,
            "oracle7": oracle,
            "err": err,
            "prog_minus_rand_dwell": prog - dwell,
            "prog_minus_oracle7": prog - oracle,
            "prog_minus_err": prog - err,
        })
    gaps_long = []
    for s in SEEDS:
        prog = _pick(runs, "prog", 2048, s)["W"]
        oracle = _pick(runs, "oracle7", 2048, s)["W"]
        gaps_long.append({"seed": s, "prog": prog, "oracle7": oracle, "prog_minus_oracle7": prog - oracle})
    return {"by_controller": by, "gaps_L512": gaps, "gaps_L2048": gaps_long}


def _headroom_reference():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/w3/results.json")
    payload = json.load(open(path))
    order = (("rand_walk", 512), ("rand_dwell", 512), ("oracle7", 512), ("oracle_hard", 512), ("oracle7", 2048))
    rows = []
    for policy, L in order:
        per = []
        for s in (0, 1, 2):
            hit = [r for r in payload["runs"] if r["policy"] == policy and r["L"] == L and r["seed"] == s]
            per.append({"seed": s, "W": hit[0]["W"]})
        mu, sd = mean_std(p["W"] for p in per)
        rows.append({"policy": policy, "L": L, "per_seed": per, "W_mean": mu, "W_std_ddof0": sd})
    gaps = []
    for s in (0, 1, 2):
        def one(policy, L, seed=s):
            block = [r for r in rows if r["policy"] == policy and r["L"] == L][0]
            return [p["W"] for p in block["per_seed"] if p["seed"] == seed][0]
        gaps.append({
            "seed": s,
            "rand_dwell_minus_oracle7": one("rand_dwell", 512) - one("oracle7", 512),
            "oracle7_L512_minus_L2048": one("oracle7", 512) - one("oracle7", 2048),
        })
    return {
        "source": "results/w3/results.json, seeds 0-2. W3 headroom check. Not recomputed here.",
        "rows": rows,
        "gaps": gaps,
    }


def _fmt(x):
    return f"{x:.6f}"


def _pm(mu, sd):
    return f"{_fmt(mu)} ± {_fmt(sd)}"


def render_markdown(payload):
    by = payload["summary"]["by_controller"]
    lines = []
    lines.append("# M5-W3 progress controller")
    lines.append("")
    lines.append("Visit controllers on the W3 ring. Bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. Lower is better. Numbers only.")
    lines.append("")
    lines.append("## W at L = 512")
    lines.append("")
    lines.append("| controller | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | seed 11 | mean ± std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for controller in CONTROLLERS_512:
        block = by[f"{controller}/L512"]
        cells = []
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            cells.append(_fmt(hit["W"]))
        lines.append(f'| {controller} | ' + " | ".join(cells) + f' | {_pm(block["W_mean"], block["W_std_ddof0"])} |')
    lines.append("")
    lines.append("## Per-channel held-out bits at L = 512")
    lines.append("")
    lines.append("| controller | seed | G | C1 | C7 | E30 | E110 | M7 | M37 | W |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for controller in CONTROLLERS_512:
        block = by[f"{controller}/L512"]
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            h = hit["held"]
            lines.append(
                f'| {controller} | {s} | ' + " | ".join(_fmt(h[c]) for c in curio.LEARN) + f' | {_fmt(hit["W"])} |'
            )
        lines.append(
            f'| {controller} | mean | ' + " | ".join(_fmt(block["held_mean"][c]) for c in curio.LEARN)
            + f' | {_fmt(block["W_mean"])} |'
        )
    lines.append("")
    lines.append("## Decision shares at L = 512")
    lines.append("")
    lines.append("Mean over seeds. A stay counts for the room's channel. A move counts for the corridor. Shares sum to 1.")
    lines.append("")
    header = "| controller | " + " | ".join(list(curio.CH) + ["corridor"]) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * 10)
    for controller in CONTROLLERS_512:
        block = by[f"{controller}/L512"]
        cells = [f'{block["share_mean"][c]:.4f}' for c in curio.CH]
        cells.append(f'{block["corridor_mean"]:.4f}')
        lines.append(f'| {controller} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## prog shares by quarter of life")
    lines.append("")
    lines.append("Mean over seeds. Quarter i is decisions [(i−1)L/4, iL/4).")
    lines.append("")
    lines.append("| L | quarter | " + " | ".join(list(curio.CH) + ["corridor"]) + " |")
    lines.append("|---:|---:|" + "---:|" * 10)
    for L in (512, 2048):
        block = by[f"prog/L{L}"]
        for i, q in enumerate(block["quarter_mean"], 1):
            cells = [f'{q["share"][c]:.4f}' for c in curio.CH]
            cells.append(f'{q["corridor"]:.4f}')
            lines.append(f'| {L} | {i} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Gaps at L = 512")
    lines.append("")
    lines.append("prog minus the other controller, per seed.")
    lines.append("")
    lines.append("| seed | prog − rand_dwell | prog − oracle7 | prog − err |")
    lines.append("|---:|---:|---:|---:|")
    gaps = payload["summary"]["gaps_L512"]
    for g in gaps:
        lines.append(
            f'| {g["seed"]} | {_fmt(g["prog_minus_rand_dwell"])} | {_fmt(g["prog_minus_oracle7"])} | {_fmt(g["prog_minus_err"])} |'
        )
    mu_d, sd_d = mean_std(g["prog_minus_rand_dwell"] for g in gaps)
    mu_o, sd_o = mean_std(g["prog_minus_oracle7"] for g in gaps)
    mu_e, sd_e = mean_std(g["prog_minus_err"] for g in gaps)
    lines.append(f'| mean | {_fmt(mu_d)} | {_fmt(mu_o)} | {_fmt(mu_e)} |')
    lines.append(f'| std ddof=0 | {_fmt(sd_d)} | {_fmt(sd_o)} | {_fmt(sd_e)} |')
    n_d = sum(g["prog_minus_rand_dwell"] < 0 for g in gaps)
    n_o = sum(g["prog_minus_oracle7"] < 0 for g in gaps)
    n_e = sum(g["prog_minus_err"] < 0 for g in gaps)
    lines.append(f'| prog lower (count) | {n_d} of 6 | {n_o} of 6 | {n_e} of 6 |')
    lines.append("")
    lines.append("## L = 2048")
    lines.append("")
    lines.append("| controller | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | seed 11 | mean ± std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for controller in CONTROLLERS_2048:
        block = by[f"{controller}/L2048"]
        cells = []
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            cells.append(_fmt(hit["W"]))
        lines.append(f'| {controller} | ' + " | ".join(cells) + f' | {_pm(block["W_mean"], block["W_std_ddof0"])} |')
    lines.append("")
    lines.append("| seed | prog − oracle7 |")
    lines.append("|---:|---:|")
    for g in payload["summary"]["gaps_L2048"]:
        lines.append(f'| {g["seed"]} | {_fmt(g["prog_minus_oracle7"])} |')
    mu_l, sd_l = mean_std(g["prog_minus_oracle7"] for g in payload["summary"]["gaps_L2048"])
    n_l = sum(g["prog_minus_oracle7"] < 0 for g in payload["summary"]["gaps_L2048"])
    lines.append(f'| mean | {_fmt(mu_l)} |')
    lines.append(f'| std ddof=0 | {_fmt(sd_l)} |')
    lines.append(f'| prog lower (count) | {n_l} of 6 |')
    lines.append("")
    lines.append("Mean held-out bits at L = 2048.")
    lines.append("")
    lines.append("| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | W |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for controller in CONTROLLERS_2048:
        block = by[f"{controller}/L2048"]
        lines.append(
            f'| {controller} | ' + " | ".join(_fmt(block["held_mean"][c]) for c in curio.LEARN)
            + f' | {_fmt(block["W_mean"])} |'
        )
    lines.append("")
    lines.append("## Noise and dark shares")
    lines.append("")
    lines.append("Mean over seeds of the fraction of decisions whose glance is N (noise) or Z (dark).")
    lines.append("")
    lines.append("| controller | L | N | Z | corridor |")
    lines.append("|---|---:|---:|---:|---:|")
    for L, controllers in ((512, CONTROLLERS_512), (2048, CONTROLLERS_2048)):
        for controller in controllers:
            block = by[f"{controller}/L{L}"]
            lines.append(
                f'| {controller} | {L} | {block["share_mean"]["N"]:.4f} | {block["share_mean"]["Z"]:.4f} | {block["corridor_mean"]:.4f} |'
            )
    lines.append("")
    lines.append("## W3 headroom reference (seeds 0–2)")
    lines.append("")
    lines.append("From results/w3/results.json. Not recomputed in this run.")
    lines.append("")
    lines.append("| policy | L | seed 0 | seed 1 | seed 2 | mean | std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["w3_headroom"]["rows"]:
        cells = [_fmt(p["W"]) for p in row["per_seed"]]
        lines.append(
            f'| {row["policy"]} | {row["L"]} | ' + " | ".join(cells)
            + f' | {_fmt(row["W_mean"])} | {_fmt(row["W_std_ddof0"])} |'
        )
    lines.append("")
    lines.append("| seed | rand_dwell − oracle7 | oracle7 L512 − L2048 |")
    lines.append("|---:|---:|---:|")
    for g in payload["w3_headroom"]["gaps"]:
        lines.append(f'| {g["seed"]} | {_fmt(g["rand_dwell_minus_oracle7"])} | {_fmt(g["oracle7_L512_minus_L2048"])} |')
    mu_h, sd_h = mean_std(g["rand_dwell_minus_oracle7"] for g in payload["w3_headroom"]["gaps"])
    mu_s, sd_s = mean_std(g["oracle7_L512_minus_L2048"] for g in payload["w3_headroom"]["gaps"])
    lines.append(f'| mean | {_fmt(mu_h)} | {_fmt(mu_s)} |')
    lines.append(f'| std ddof=0 | {_fmt(sd_h)} | {_fmt(sd_s)} |')
    lines.append("")
    lines.append("## Ring order")
    lines.append("")
    lines.append("Channel in rooms 0..8, seeds 6–11. One order per seed; controllers share it.")
    lines.append("")
    lines.append("| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        rooms = _pick(payload["runs"], "prog", 512, s)["room_channel"]
        lines.append(f'| {s} | ' + " | ".join(rooms) + " |")
    lines.append("")
    lines.append("## Walls")
    lines.append("")
    lines.append(f'Scored run wall {payload["wall_s"]:.3f} s. Pool of {payload["workers"]}. Smoke wall {payload["smoke_wall_s"]:.3f} s.')
    lines.append("")
    lines.append("| controller | L | seed | wall seconds |")
    lines.append("|---|---:|---:|---:|")
    order = [(c, 512) for c in CONTROLLERS_512] + [(c, 2048) for c in CONTROLLERS_2048]
    runs_sorted = sorted(payload["runs"], key=lambda r: (order.index((r["controller"], r["L"])), r["seed"]))
    for r in runs_sorted:
        lines.append(f'| {r["controller"]} | {r["L"]} | {r["seed"]} | {r["wall_s"]:.3f} |')
    lines.append("")
    lines.append("## Smoke test")
    lines.append("")
    lines.append("```")
    lines.append(payload["smoke_text"].rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- World, corridor glances, `curio.SGDCompressor` at lr 0.004, life length and W are `w3_headroom.py`. W is the mean over the 7 learnable channels of bits/symbol on EVAL=2048 fresh symbols. The H20 world is copied before the continuation. Channels are scored in `curio.LEARN` order.")
    lines.append("- A visit chooses a target room, walks the shortest path (each move is one corridor glance), then stays 8 decisions. A visit cut short by the end of life ends there.")
    lines.append("- Per-glance rewards follow `curio.live` with `prog_kind=hist`. prog scores the old and new compressor on the channel reservoir (`R_RES`, rng `default_rng(20_000 + seed)`), and the denominator is the reservoir length after that glance is stored, because `curio.live` binds `hs = res[c]` before the append. err is the glance's bits per symbol. Corridor glances are learnt and earn nothing.")
    lines.append("- The visit reward is the sum of the dwell glances' per-glance rewards, divided by travel decisions plus dwell decisions.")
    lines.append("- `rand_dwell` and `oracle7` use `w3_headroom.next_action` and `default_rng(1_000 + seed)` for both targets and N-channel noise. The smoke test requires their W on seed 0, L=512, to match `w3_headroom.run_life` to 1e-9, which fixes that stream. The spec's `default_rng(10_000 + seed)` is the choice rng of `err` and `prog`.")
    lines.append("- `err` and `prog` share one bandit. The first pass visits every room once, always the nearest unvisited room, ties broken to the right. Then, with probability EPS=0.1, a uniform room; otherwise the room with the highest EMA, ties broken by `max` on `(value, rng.random())` as in `curio.live`. The EMA uses BETA=0.3 once per visit. Their N-channel noise is `default_rng(1_000 + seed)`, separate from choices.")
    lines.append("- Seeds 6–11. L=512 for every controller. L=2048 for prog and oracle7.")
    lines.append("- PYTHONHASHSEED=0. CPU. Pool of 2. No GPU.")
    lines.append("")
    return "\n".join(lines)


def scored_jobs():
    jobs = [(s, "prog", 2048) for s in SEEDS]
    jobs += [(s, "prog", 512) for s in SEEDS]
    jobs += [(s, "oracle7", 2048) for s in SEEDS]
    jobs += [(s, "err", 512) for s in SEEDS]
    jobs += [(s, "oracle7", 512) for s in SEEDS]
    jobs += [(s, "rand_dwell", 512) for s in SEEDS]
    return jobs


def run_scored(workers, smoke_payload, smoke_wall_s, out_dir):
    t0 = time.perf_counter()
    jobs = scored_jobs()
    print(f"scored jobs {len(jobs)} workers {workers} seeds {list(SEEDS)}", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        runs = list(pool.imap_unordered(_job, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    for r in runs:
        r.pop("visits", None)
    summary = summarize(runs)
    payload = {
        "seeds": list(SEEDS),
        "workers": workers,
        "wall_s": wall,
        "smoke_wall_s": smoke_wall_s,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "runs": runs,
        "summary": summary,
        "w3_headroom": _headroom_reference(),
        "smoke": {k: smoke_payload[k] for k in ("reproduce", "trace", "L64", "estimate", "over_limit", "pythonhashseed", "curio_lr")},
        "smoke_text": smoke_payload["text"],
    }
    # Attach quarters from runs into the saved runs (already there).
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    json.dump(payload, open(json_path, "w"), indent=1)
    md_path = os.path.join(out_dir, "RESULTS.md")
    open(md_path, "w").write(render_markdown(payload))
    print(f"wrote {md_path} and {json_path} wall {wall:.1f}s", flush=True)
    for g in summary["gaps_L512"]:
        print(
            f'seed {g["seed"]} prog-dwell {g["prog_minus_rand_dwell"]:.4f} '
            f'prog-oracle7 {g["prog_minus_oracle7"]:.4f} prog-err {g["prog_minus_err"]:.4f}',
            flush=True,
        )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default="results/m5w3")
    a = ap.parse_args()
    do_smoke = a.smoke or not a.run
    do_run = a.run or not a.smoke
    if a.smoke and not a.run:
        do_run = False
    if a.run and not a.smoke:
        do_smoke = True  # spec: smoke before the scored run
    t_smoke = time.perf_counter()
    result = smoke()
    smoke_wall = time.perf_counter() - t_smoke
    print(f"smoke wall {smoke_wall:.3f}s", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({**result, "smoke_wall_s": smoke_wall}, open(os.path.join(a.out_dir, "smoke.json"), "w"), indent=1)
    if not do_run:
        return
    if result["over_limit"]:
        raise SystemExit("smoke estimate exceeds 20 min; scored run was not started")
    run_scored(a.workers, result, smoke_wall, a.out_dir)


if __name__ == "__main__":
    main()
