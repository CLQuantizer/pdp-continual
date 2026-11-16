"""W3 headroom: fixed policies on the ring of H23c channels. No verdict.

Score W is H23c's: mean bits/symbol of the frozen SGD compressor on a fresh EVAL-symbol
continuation of each of the 7 learnable channels, from a copy of the world at that moment.
"""
import os, sys

# Hash seed and the compressor's default lr are fixed at process start. curio.SGDCompressor
# bakes CURIO_LR into its default argument at import.
if __name__ == "__main__" and (os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("CURIO_LR") != "0.004"):
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CURIO_LR"] = "0.004"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse, copy, json, math, subprocess, time
from multiprocessing import get_context

import numpy as np

import curio
import h20_world
from w3_world import World3

# Mean end-of-life held-out bits of rand7, seeds 6-11, results/h23b/score.json.
# H23c RESULTS.md states those runs are its baselines (same compressor, seeds, budget).
# results/h23c/score.json stores the prog condition only.
HARD_BITS = {
    "G": 3.6930947286329108,
    "C1": 4.094935363238028,
    "C7": 4.254920244396625,
    "E30": 1.351967301484789,
    "E110": 1.2917227584211532,
    "M7": 0.47399050368018897,
    "M37": 2.206473954752524,
}
HARD_SOURCE = (
    "mean held-out bits/symbol of rand7 on seeds 6-11 in results/h23b/score.json; "
    "H23c RESULTS.md states those runs are its baselines. "
    "results/h23c/score.json has prog only."
)
ACTIONS = ("stay", "left", "right")
LR = 0.004


class W3Compressor(curio.SGDCompressor):
    """curio.SGDCompressor plus a corridor channel. Non-corridor calls the parent unchanged."""

    def bits(self, c, hist, seq, learn):
        if c != "corridor":
            return super().bits(c, hist, seq, learn)
        ci = len(curio.CH)
        ctx, tot = list(hist[-max(curio.ORDERS):]), 0.0
        for x in seq:
            f = self.feats(ci, ctx)
            z = self.W[f].sum(0)
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            tot -= math.log2(max(float(p[x]), 1e-12))
            if learn:
                p[x] -= 1.0
                self.W[f] -= self.lr * p
            ctx = (ctx + [x])[-max(curio.ORDERS):]
        return tot


def shortest(src, dst, n=9):
    """Shortest walk on the ring. n is odd, so the shortest direction is unique."""
    right = (dst - src) % n
    left = (src - dst) % n
    if right == 0:
        return []
    if right < left:
        return ["right"] * right
    return ["left"] * left


def _learnable(rooms):
    return [i for i, c in enumerate(rooms) if c in curio.LEARN]


def next_action(state, rooms, rng):
    """One action. Dwell policies draw the next target only when the current plan is empty."""
    if state["kind"] == "rand_walk":
        return ACTIONS[int(rng.integers(3))]
    if not state["plan"]:
        kind = state["kind"]
        learnable = _learnable(rooms)
        if kind == "rand_dwell":
            target = int(rng.integers(9))
        elif kind == "oracle7":
            target = learnable[int(rng.integers(len(learnable)))]
        elif kind == "oracle_hard":
            w = np.array([HARD_BITS[rooms[i]] for i in learnable], np.float64)
            target = int(rng.choice(np.asarray(learnable, np.int64), p=w / w.sum()))
        else:
            raise ValueError(kind)
        state["plan"] = shortest(state["room"], target) + ["stay"] * 8
    return state["plan"].pop(0)


def evaluate(comp, world, hist):
    """Frozen compressor, copy of the H20 world, channels in curio.LEARN order. Same as curio.live."""
    wc = copy.deepcopy(world.h20)
    held = {}
    for c in curio.LEARN:
        seq = [h20_world.ALPHA.index(s) for s in wc.take(c, curio.EVAL)]
        held[c] = comp.score(c, hist.get(c, []), seq) / curio.EVAL
    W = float(np.mean([held[c] for c in curio.LEARN]))
    return W, {c: float(held[c]) for c in curio.LEARN}


def run_life(seed, policy, L, checkpoints=None, score=True):
    """One life. Policy rng is default_rng(1000 + seed), shared with N-channel glances."""
    t0 = time.perf_counter()
    world = World3(seed)
    rng = np.random.default_rng(1_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = {c: [] for c in list(curio.CH) + ["corridor"]}
    state = {"kind": policy, "room": 0, "plan": []}
    room_counts = [0] * 9
    corridor = 0
    if checkpoints is None:
        checkpoints = (L // 4, L // 2, (3 * L) // 4, L)
    want = set(checkpoints) if score else set()
    ckpt, eval_s = [], 0.0
    trace = []
    for t in range(1, L + 1):
        action = next_action(state, world.rooms, rng)
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        comp.bits(channel, hist[channel], symbols, learn=True)
        hist[channel] += symbols
        if channel == "corridor":
            corridor += 1
        else:
            room_counts[room] += 1
        if len(trace) < 20:
            trace.append({"t": t, "action": action, "room": room, "channel": channel})
        if t in want:
            e0 = time.perf_counter()
            W, held = evaluate(comp, world, hist)
            eval_s += time.perf_counter() - e0
            ckpt.append({"t": t, "W": W, "held": held})
    wall = time.perf_counter() - t0
    if score:
        if not ckpt or ckpt[-1]["t"] != L:
            raise RuntimeError(f"missing final checkpoint for {policy} L={L}")
        W, held = ckpt[-1]["W"], ckpt[-1]["held"]
    else:
        W, held = float("nan"), {}
    return {
        "seed": int(seed),
        "policy": policy,
        "L": int(L),
        "W": W,
        "held": held,
        "room_share": {str(i): room_counts[i] / L for i in range(9)},
        "room_channel": list(world.rooms),
        "corridor_share": corridor / L,
        "checkpoints": ckpt,
        "trace20": trace,
        "wall_s": wall,
        "eval_s": eval_s,
        "learn_s": wall - eval_s,
        "finite": bool(math.isfinite(W) and all(math.isfinite(v) for v in held.values())),
    }


def _job(args):
    seed, policy, L = args
    out = run_life(seed, policy, L)
    print(
        f's{seed} {policy} L={L} W {out["W"]:.4f} corridor {out["corridor_share"]:.3f} '
        f'wall {out["wall_s"]:.1f}s',
        flush=True,
    )
    return out


def jobs_for(seeds):
    spec = (
        ("rand_walk", 512),
        ("rand_dwell", 512),
        ("oracle7", 512),
        ("oracle_hard", 512),
        ("oracle7", 2048),
    )
    return [(s, p, L) for s in seeds for p, L in spec]


def mean_std(xs):
    a = np.asarray(xs, np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def summarize(runs):
    """Means over seeds. Gaps are per seed, as pre-registered."""
    keys = sorted({(r["policy"], r["L"]) for r in runs})
    by = {}
    for policy, L in keys:
        rows = [r for r in runs if r["policy"] == policy and r["L"] == L]
        rows = sorted(rows, key=lambda r: r["seed"])
        mu, sd = mean_std([r["W"] for r in rows])
        held_mu = {}
        for c in curio.LEARN:
            held_mu[c] = mean_std([r["held"][c] for r in rows])
        by[f"{policy}/L{L}"] = {
            "policy": policy,
            "L": L,
            "n": len(rows),
            "W_mean": mu,
            "W_std_ddof0": sd,
            "per_seed": [{"seed": r["seed"], "W": r["W"], "held": r["held"],
                          "room_share": r["room_share"], "corridor_share": r["corridor_share"],
                          "room_channel": r["room_channel"]} for r in rows],
            "held_mean": {c: held_mu[c][0] for c in curio.LEARN},
            "held_std_ddof0": {c: held_mu[c][1] for c in curio.LEARN},
        }
    seeds = sorted({r["seed"] for r in runs})
    gaps = []
    for s in seeds:
        def W(policy, L, seed=s):
            hit = [r for r in runs if r["seed"] == seed and r["policy"] == policy and r["L"] == L]
            if len(hit) != 1:
                raise RuntimeError(f"expected one run for {policy} L={L} seed {seed}, got {len(hit)}")
            return hit[0]["W"]
        rw, rd = W("rand_walk", 512), W("rand_dwell", 512)
        o7, oh = W("oracle7", 512), W("oracle_hard", 512)
        o7_long = W("oracle7", 2048)
        gaps.append({
            "seed": s,
            "rand_walk_minus_oracle7": rw - o7,
            "rand_walk_minus_oracle_hard": rw - oh,
            "rand_dwell_minus_oracle7": rd - o7,
            "oracle7_L512_minus_L2048": o7 - o7_long,
            "rand_walk": rw,
            "rand_dwell": rd,
            "oracle7": o7,
            "oracle_hard": oh,
            "oracle7_L2048": o7_long,
        })
    return {"by_policy": by, "gaps": gaps}


def h23c_reference():
    """Published H23c headline numbers, plus the exact H23b baseline means they come from."""
    return {
        "source": "results/h23c/RESULTS.md table. Baselines are H23b, seeds 6-11, 512 glances, sgd, CURIO_LR=0.004.",
        "seeds": [6, 7, 8, 9, 10, 11],
        "rand_W_mean_published": 2.637,
        "rand_W_std_published": 0.052,
        "rand7_W_mean_published": 2.481,
        "rand7_W_std_published": 0.046,
        "rand_W_mean_h23b": 2.6373326405149022,
        "rand_W_std_ddof0_h23b": 0.05156851548617568,
        "rand7_W_mean_h23b": 2.48101497922946,
        "rand7_W_std_ddof0_h23b": 0.04602245123519197,
        "rand7_held_mean_h23b": dict(HARD_BITS),
    }


def teleport_rand(seed, n_gl):
    """No travel. Same channel sequence as curio's rand controller, including its first pass."""
    world = World3(seed)
    rng = np.random.default_rng(10_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = {c: [] for c in curio.CH}
    for g in range(n_gl):
        if g < len(curio.CH):
            c = curio.CH[g]
        else:
            c = curio.CH[int(rng.integers(len(curio.CH)))]
        x = world.glance(c, rng)
        comp.bits(c, hist[c], x, learn=True)
        hist[c] += x
    return evaluate(comp, world, hist)


def smoke():
    """VPS checks. Exit 1 on a failed check. Does not push a kernel."""
    os.environ["CURIO_LR"] = "0.004"
    out = {"pythonhashseed": os.environ.get("PYTHONHASHSEED"), "curio_lr": os.environ.get("CURIO_LR")}
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("smoke requires PYTHONHASHSEED=0 so it matches curio.py in a subprocess")

    W_tel, held_tel = teleport_rand(6, 64)
    ref = curio.live(("rand", 6, 64, "sgd", "window"))
    cli_path = "/tmp/w3_eq_curio.json"
    subprocess.check_call(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "curio.py"),
         "--seeds", "6", "--glances", "64", "--conds", "rand", "--comp", "sgd", "--out", cli_path],
        env={**os.environ, "CURIO_LR": "0.004", "PYTHONHASHSEED": "0"},
    )
    cli = json.load(open(cli_path))
    if len(cli) != 1:
        raise SystemExit(f"curio.py returned {len(cli)} rows, expected 1")
    W_cli = float(cli[0]["W"])
    d_live = abs(W_tel - float(ref["W"]))
    d_cli = abs(W_tel - W_cli)
    held_d = {c: abs(held_tel[c] - float(ref["held"][c])) for c in curio.LEARN}
    print(f"equivalence teleport W {W_tel:.12f}")
    print(f"equivalence curio.live W {float(ref['W']):.12f} abs {d_live:.3e}")
    print(f"equivalence curio.py W {W_cli:.12f} abs {d_cli:.3e}")
    print("equivalence per-channel abs " + " ".join(f"{c} {held_d[c]:.3e}" for c in curio.LEARN))
    if d_live > 1e-9 or d_cli > 1e-9 or max(held_d.values()) > 1e-9:
        raise SystemExit("equivalence failed")
    out["equivalence"] = {
        "seed": 6, "glances": 64, "W_teleport": W_tel, "W_curio_live": float(ref["W"]),
        "W_curio_cli": W_cli, "abs_live": d_live, "abs_cli": d_cli, "held_abs": held_d,
    }

    print("ring orders")
    orders = {}
    for s in (0, 1, 2):
        w = World3(s)
        orders[str(s)] = list(w.rooms)
        print(f"  seed {s} perm {w.perm} rooms {w.rooms}")
        for src in range(9):
            for dst in range(9):
                path = shortest(src, dst)
                room = src
                for a in path:
                    room = (room - 1) % 9 if a == "left" else (room + 1) % 9
                if room != dst or len(path) != min((dst - src) % 9, (src - dst) % 9):
                    raise SystemExit(f"shortest path failed {src}->{dst}")
    out["room_orders"] = orders

    print("rand_walk trace seed 0, 20 decisions (action, room_after, channel)")
    traced = run_life(0, "rand_walk", 20, score=False)
    for row in traced["trace20"]:
        print(f'  ({row["action"]}, {row["room"]}, {row["channel"]})')
    out["trace20"] = traced["trace20"]

    print("L=64 seed 0, four policies")
    timed = []
    for policy in ("rand_walk", "rand_dwell", "oracle7", "oracle_hard"):
        row = run_life(0, policy, 64)
        print(
            f'  {policy} W {row["W"]:.6f} finite {row["finite"]} '
            f'wall {row["wall_s"]:.2f}s learn {row["learn_s"]:.2f}s eval {row["eval_s"]:.2f}s '
            f'corridor {row["corridor_share"]:.3f}'
        )
        if not row["finite"]:
            raise SystemExit(f"{policy} W is not finite")
        share = sum(row["room_share"].values()) + row["corridor_share"]
        if abs(share - 1.0) > 1e-12:
            raise SystemExit(f"{policy} shares sum to {share}")
        timed.append(row)
    per_dec = max(r["learn_s"] / 64 for r in timed)
    per_eval = max(r["eval_s"] / 4 for r in timed)

    def life_s(L):
        return per_dec * L + per_eval * 4

    serial = 3 * (4 * life_s(512) + life_s(2048))
    wall_est = serial / 4.0
    print(
        f"estimate per decision {per_dec:.4f}s per eval {per_eval:.4f}s "
        f"serial {serial:.0f}s pool4 wall {wall_est:.0f}s ({wall_est / 60:.1f} min)"
    )
    out["L64"] = [{k: r[k] for k in ("policy", "W", "finite", "wall_s", "learn_s", "eval_s", "corridor_share")}
                  for r in timed]
    out["estimate"] = {
        "per_decision_s": per_dec,
        "per_eval_s": per_eval,
        "serial_s": serial,
        "pool4_wall_s": wall_est,
        "pool4_wall_min": wall_est / 60.0,
    }
    if wall_est > 90 * 60:
        out["split"] = True
        print("estimate exceeds 90 min; split by seed")
    else:
        out["split"] = False
        print("estimate is within 90 min; one kernel")
    return out


def kernel_main(out, kernel_id, seeds=(0, 1, 2), workers=4):
    os.environ["CURIO_LR"] = "0.004"
    t0 = time.perf_counter()
    jobs = jobs_for(seeds)
    print(f"kernel {kernel_id} jobs {len(jobs)} workers {workers} seeds {list(seeds)}", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        runs = pool.map(_job, jobs)
    wall = time.perf_counter() - t0
    orders = {str(s): World3(s).rooms for s in seeds}
    payload = {
        "kernel_id": kernel_id,
        "seeds": list(seeds),
        "wall_s": wall,
        "hard_bits": HARD_BITS,
        "hard_source": HARD_SOURCE,
        "room_orders": orders,
        "runs": runs,
        "summary": summarize(runs),
        "h23c_reference": h23c_reference(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    payload["prefix_check"] = []
    for s in sorted({r["seed"] for r in runs}):
        short = [r for r in runs if r["seed"] == s and r["policy"] == "oracle7" and r["L"] == 512]
        long = [r for r in runs if r["seed"] == s and r["policy"] == "oracle7" and r["L"] == 2048]
        ck = [c for c in long[0]["checkpoints"] if c["t"] == 512]
        diff = abs(short[0]["W"] - ck[0]["W"])
        print(f"prefix check seed {s} |oracle7 L512 - L2048 at t=512| {diff:.3e}", flush=True)
        payload["prefix_check"].append({"seed": s, "abs_W": diff})
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.json")
    json.dump(payload, open(path, "w"), indent=1)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    for g in payload["summary"]["gaps"]:
        print(
            f'seed {g["seed"]} rw-o7 {g["rand_walk_minus_oracle7"]:.4f} '
            f'rw-oh {g["rand_walk_minus_oracle_hard"]:.4f} '
            f'rd-o7 {g["rand_dwell_minus_oracle7"]:.4f} '
            f'o7 512-2048 {g["oracle7_L512_minus_L2048"]:.4f}',
            flush=True,
        )
    return payload


def _fmt(x):
    return f"{x:.6f}"


def render_markdown(payload, smoke_text):
    """Numbers only. The orchestrator scores the bars."""
    ref = payload["h23c_reference"]
    lines = []
    lines.append("# W3 headroom")
    lines.append("")
    lines.append("Fixed policies on the ring of H23c channels. Bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. Lower is better.")
    lines.append("")
    lines.append("## W")
    lines.append("")
    lines.append("| policy | L | seed 0 | seed 1 | seed 2 | mean | std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    order = (("rand_walk", 512), ("rand_dwell", 512), ("oracle7", 512), ("oracle_hard", 512), ("oracle7", 2048))
    by = payload["summary"]["by_policy"]
    for policy, L in order:
        block = by[f"{policy}/L{L}"]
        cells = []
        for s in (0, 1, 2):
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            cells.append(_fmt(hit["W"]))
        lines.append(f'| {policy} | {L} | ' + " | ".join(cells) + f' | {_fmt(block["W_mean"])} | {_fmt(block["W_std_ddof0"])} |')
    lines.append("")
    lines.append("## Per-channel bits at the end of life")
    lines.append("")
    lines.append("| policy | L | seed | G | C1 | C7 | E30 | E110 | M7 | M37 | W |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    runs = sorted(payload["runs"], key=lambda r: (order.index((r["policy"], r["L"])), r["seed"]))
    for r in runs:
        h = r["held"]
        lines.append(
            f'| {r["policy"]} | {r["L"]} | {r["seed"]} | '
            + " | ".join(_fmt(h[c]) for c in curio.LEARN)
            + f' | {_fmt(r["W"])} |'
        )
    lines.append("")
    lines.append("## Room shares and corridor share")
    lines.append("")
    lines.append("A stay counts for the room it observes. A move counts for the corridor. Shares sum to 1.")
    lines.append("")
    lines.append("| policy | L | seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | corridor |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        sh = r["room_share"]
        lines.append(
            f'| {r["policy"]} | {r["L"]} | {r["seed"]} | '
            + " | ".join(f'{sh[str(i)]:.4f}' for i in range(9))
            + f' | {r["corridor_share"]:.4f} |'
        )
    lines.append("")
    lines.append("Room order is the channel in ring rooms 0..8.")
    lines.append("")
    lines.append("| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    for s in (0, 1, 2):
        rooms = payload["room_orders"][str(s)]
        lines.append(f'| {s} | ' + " | ".join(rooms) + " |")
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("| seed | rand_walk − oracle7 | rand_walk − oracle_hard | rand_dwell − oracle7 | oracle7 L512 − L2048 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for g in payload["summary"]["gaps"]:
        lines.append(
            f'| {g["seed"]} | {_fmt(g["rand_walk_minus_oracle7"])} | {_fmt(g["rand_walk_minus_oracle_hard"])} | '
            f'{_fmt(g["rand_dwell_minus_oracle7"])} | {_fmt(g["oracle7_L512_minus_L2048"])} |'
        )
    mu_rw, sd_rw = mean_std([g["rand_walk_minus_oracle7"] for g in payload["summary"]["gaps"]])
    mu_rh, sd_rh = mean_std([g["rand_walk_minus_oracle_hard"] for g in payload["summary"]["gaps"]])
    mu_dw, sd_dw = mean_std([g["rand_dwell_minus_oracle7"] for g in payload["summary"]["gaps"]])
    mu_sc, sd_sc = mean_std([g["oracle7_L512_minus_L2048"] for g in payload["summary"]["gaps"]])
    lines.append(f'| mean | {_fmt(mu_rw)} | {_fmt(mu_rh)} | {_fmt(mu_dw)} | {_fmt(mu_sc)} |')
    lines.append(f'| std ddof=0 | {_fmt(sd_rw)} | {_fmt(sd_rh)} | {_fmt(sd_dw)} | {_fmt(sd_sc)} |')
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append("W after L/4, L/2, 3L/4 and L decisions.")
    lines.append("")
    lines.append("| policy | L | seed | t1 | W1 | t2 | W2 | t3 | W3 | t4 | W4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        cells = []
        for c in r["checkpoints"]:
            cells.append(str(c["t"]))
            cells.append(_fmt(c["W"]))
        lines.append(f'| {r["policy"]} | {r["L"]} | {r["seed"]} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## H23c reference")
    lines.append("")
    lines.append("H23c scored seeds 6–11, 512 glances, no travel. Published table in results/h23c/RESULTS.md.")
    lines.append("")
    lines.append("| cond | W mean published | std published | W mean from the H23b baseline file | std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f'| rand | {ref["rand_W_mean_published"]:.3f} | {ref["rand_W_std_published"]:.3f} | '
        f'{_fmt(ref["rand_W_mean_h23b"])} | {_fmt(ref["rand_W_std_ddof0_h23b"])} |'
    )
    lines.append(
        f'| rand7 | {ref["rand7_W_mean_published"]:.3f} | {ref["rand7_W_std_published"]:.3f} | '
        f'{_fmt(ref["rand7_W_mean_h23b"])} | {_fmt(ref["rand7_W_std_ddof0_h23b"])} |'
    )
    lines.append("")
    lines.append("oracle_hard draws its target with probability proportional to these rand7 held-out bits:")
    lines.append("")
    lines.append("| channel | bits |")
    lines.append("|---|---:|")
    for c in curio.LEARN:
        lines.append(f'| {c} | {HARD_BITS[c]:.6f} |')
    lines.append("")
    lines.append(f"Source: {payload['hard_source']}")
    lines.append("")
    if payload.get("prefix_check"):
        lines.append("## Prefix check")
        lines.append("")
        lines.append("oracle7 at L=512 and the t=512 checkpoint of oracle7 at L=2048 see the same decisions.")
        lines.append("")
        lines.append("| seed | abs W |")
        lines.append("|---:|---:|")
        for row in payload["prefix_check"]:
            lines.append(f'| {row["seed"]} | {row["abs_W"]:.3e} |')
        lines.append("")
    lines.append("## Walls and kernel")
    lines.append("")
    lines.append(f'Kernel `{payload["kernel_id"]}`. Script wall {payload["wall_s"]:.3f} s.')
    if payload.get("kernel_status"):
        lines.append("")
        lines.append("```")
        lines.append(payload["kernel_status"].rstrip())
        lines.append("```")
    lines.append("")
    lines.append("| policy | L | seed | wall seconds |")
    lines.append("|---|---:|---:|---:|")
    for r in runs:
        lines.append(f'| {r["policy"]} | {r["L"]} | {r["seed"]} | {r["wall_s"]:.3f} |')
    lines.append("")
    lines.append("## Smoke test")
    lines.append("")
    lines.append("```")
    lines.append(smoke_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- 9 rooms on a ring. Channel order is `default_rng(world_seed + 70000).permutation(9)` into `curio.CH`.")
    lines.append("- Actions stay, left, right. A move shows 64 space symbols labelled corridor and does not observe the rooms it passes.")
    lines.append("- Life length is the number of decisions. One decision is one glance. The agent starts in room 0.")
    lines.append("- Compressor is `curio.SGDCompressor` with lr 0.004. Corridor uses the same update with channel index 9.")
    lines.append("- Policy rng is `default_rng(1000 + world_seed)`, also used for N-channel noise. Each policy builds its own rng.")
    lines.append("- rand_walk is uniform over the three actions. rand_dwell picks a room uniformly from 9, walks the shortest path, stays 8 decisions, and repeats.")
    lines.append("- oracle7 is rand_dwell over the 7 learnable rooms. oracle_hard draws that target with probability proportional to the rand7 bits above.")
    lines.append("- Shortest path is unique because 9 is odd.")
    lines.append("- W is the mean over the 7 learnable channels of bits/symbol on EVAL=2048 fresh symbols. The world is copied before the continuation. Channels are scored in `curio.LEARN` order.")
    lines.append("- Checkpoints are L/4, L/2, 3L/4 and L. The compressor keeps learning after a checkpoint.")
    lines.append("- Seeds 0, 1 and 2. L=512 for every policy. oracle7 also runs at L=2048.")
    lines.append("- PYTHONHASHSEED=0. CPU. Pool of 4.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--kernel-json", default="")
    ap.add_argument("--smoke-log", default="")
    ap.add_argument("--out-dir", default="results/w3")
    a = ap.parse_args()
    if a.smoke:
        text_path = "/tmp/w3_smoke.log"
        # Smoke prints as it goes; the caller tees stdout. Also return json on stdout's sibling.
        result = smoke()
        json.dump(result, open("/tmp/w3_smoke.json", "w"), indent=1)
        print("smoke json /tmp/w3_smoke.json")
    elif a.report:
        payload = json.load(open(a.kernel_json))
        smoke_text = open(a.smoke_log).read() if a.smoke_log else ""
        os.makedirs(a.out_dir, exist_ok=True)
        # Drop the 20-step traces from the published json; the smoke log keeps the seed-0 trace.
        for r in payload["runs"]:
            r.pop("trace20", None)
        json.dump(payload, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1)
        open(os.path.join(a.out_dir, "RESULTS.md"), "w").write(render_markdown(payload, smoke_text))
        print(f'wrote {a.out_dir}/RESULTS.md and results.json')
    else:
        raise SystemExit("pass --smoke or --report")
