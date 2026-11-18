"""W7 scored run. One process. Seeds 10–29. Writes results/w7/."""
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import w7_world as W

SEEDS = list(range(10, 30))
LIVES = (25, 64)
OPACITIES = (512, 768, 922)
POLICIES = ("still", "random", "sweep", "curious_prog", "curious_surp")
VISIT_SEEDS = (10, 11, 12)
PRIMARY = 768
BAR1 = -0.30
BAR2 = -0.30
BAR3 = 0.10
BAR1_NEED = 18
BAR2_NEED = 16
SHIFT_PARALLAX = (W.SCENE_SHIFT, W.DOT_SHIFT)
SHIFT_GLUED = (1, 1)


def path_still(budget):
    return [(0, 0)] * int(budget)


def path_random(budget, seed):
    rng = np.random.default_rng(8_000 + int(seed))
    x = y = 0
    path = []
    for _ in range(int(budget)):
        a = int(rng.integers(0, 4))
        if a == 0:
            x = (x + 1) % W.P
        elif a == 1:
            x = (x - 1) % W.P
        elif a == 2:
            y = (y + 1) % W.P
        else:
            y = (y - 1) % W.P
        path.append((x, y))
    return path


def path_sweep(budget):
    """Snake through a lattice whose stride is the dot diameter, then repeat."""
    stride = 2 * W.RADIUS + 1
    xs = list(range(0, W.P, stride))
    ys = list(range(0, W.P, stride))
    lattice = []
    for iy, y in enumerate(ys):
        row = xs if iy % 2 == 0 else list(reversed(xs))
        for x in row:
            lattice.append((x, y))
    if len(lattice) != 25:
        raise RuntimeError("sweep lattice is not 25")
    path = []
    while len(path) < int(budget):
        path.extend(lattice)
    return path[: int(budget)]


def region_id(px, py):
    side = W.P // W.REG
    return (int(py) // W.REG) * side + (int(px) // W.REG)


def lawn_step(px, py):
    rx, ry = int(px) // W.REG, int(py) // W.REG
    x0, y0 = rx * W.REG, ry * W.REG
    nx = int(px) + 1
    ny = int(py)
    if nx >= x0 + W.REG:
        nx = x0
        ny = int(py) + 1
        if ny >= y0 + W.REG:
            ny = y0
    return nx % W.P, ny % W.P


def step_toward(px, py, tx, ty):
    dx = (int(tx) - int(px) + W.P // 2) % W.P - W.P // 2
    dy = (int(ty) - int(py) + W.P // 2) % W.P - W.P // 2
    if dx == 0 and dy == 0:
        return lawn_step(px, py)
    if abs(dx) >= abs(dy):
        return (int(px) + (1 if dx > 0 else -1)) % W.P, int(py)
    return int(px), (int(py) + (1 if dy > 0 else -1)) % W.P


def _reward(mode, counts, reservoir, grey, vis, sx, sy):
    ys = sy[vis]
    xs = sx[vis]
    vs = grey[vis].astype(np.int64)
    if mode == "surp":
        if vs.size == 0:
            return 0.0
        bits = float(W.kt_from_counts(counts[ys, xs], vs).mean())
        np.add.at(counts, (ys, xs, vs), 1)
        return bits
    reservoir.include(ys, xs)
    if vs.size == 0:
        return 0.0
    reward = reservoir.progress(counts, ys, xs, vs)
    np.add.at(counts, (ys, xs, vs), 1)
    return reward


def path_region(scene, occ, budget, scene_shift, dot_shift, mode, seed):
    """Sixteen 8×8 regions. Replan on arrival, shortest step to the centre, lawnmower inside.

    surp: EMA of the glance's mean KT bits, unvisited at log2(8). The sketch's policy.
    prog: EMA of reservoir compression progress, unvisited at +inf (first sample sets it).
    """
    if mode not in ("surp", "prog"):
        raise ValueError(mode)
    side = W.P // W.REG
    nreg = side * side
    centers = []
    for ry in range(side):
        for rx in range(side):
            centers.append((rx * W.REG + W.REG // 2, ry * W.REG + W.REG // 2))
    counts = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    if mode == "surp":
        ema = np.full(nreg, np.log2(W.G))
        reservoir = None
    else:
        ema = np.full(nreg, np.inf)
        reservoir = W.GlanceReservoir(seed)
    visits = np.zeros(nreg, dtype=np.int32)
    px = py = 0
    target = None
    path = []
    for _ in range(int(budget)):
        here = region_id(px, py)
        if target is None or here == target:
            order = np.lexsort((np.arange(nreg), visits, -ema))
            target = int(order[0])
            if here == target:
                px, py = lawn_step(px, py)
            else:
                px, py = step_toward(px, py, centers[target][0], centers[target][1])
        else:
            px, py = step_toward(px, py, centers[target][0], centers[target][1])
        path.append((px, py))
        grey, vis, sx, sy = W.frame(scene, occ, px, py, scene_shift, dot_shift)
        landed = region_id(px, py)
        reward = _reward(mode, counts, reservoir, grey, vis, sx, sy)
        if mode == "prog" and int(visits[landed]) == 0:
            ema[landed] = reward
        else:
            ema[landed] = (1.0 - W.BETA) * ema[landed] + W.BETA * reward
        visits[landed] += 1
    if mode == "prog":
        if reservoir.n_seen != len(path) or len(reservoir.slots) != len(path):
            raise RuntimeError("reservoir dropped a glance inside the cap")
    batch, _visits = W.accumulate(scene, occ, path, scene_shift, dot_shift)
    if not np.array_equal(batch, counts):
        raise RuntimeError("online integrator drifted from the path")
    return path


def make_path(policy, budget, seed, scene, occ, scene_shift, dot_shift):
    if policy == "still":
        return path_still(budget)
    if policy == "random":
        return path_random(budget, seed)
    if policy == "sweep":
        return path_sweep(budget)
    if policy == "curious_prog":
        return path_region(scene, occ, budget, scene_shift, dot_shift, "prog", seed)
    if policy == "curious_surp":
        return path_region(scene, occ, budget, scene_shift, dot_shift, "surp", seed)
    raise ValueError(policy)


def region_sequence(path):
    return [region_id(px, py) for px, py in path]


def first_visits(regions):
    seen = []
    used = set()
    for r in regions:
        if r not in used:
            used.add(r)
            seen.append(int(r))
    return seen


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def f6(x):
    return f"{x:.6f}"


def pm(mu, sd):
    return f"{mu:.6f} ± {sd:.6f}"


def collect(scene):
    rows = []
    visits = []
    for seed in SEEDS:
        field = W.make_field(seed)
        probes = W.probe_points(seed)
        for n_opaque in OPACITIES:
            occ = W.occluder(field, n_opaque)
            clear = (W.N * W.N - n_opaque) / float(W.N * W.N)
            for shift_name, (ss, sd) in (("parallax", SHIFT_PARALLAX), ("glued", SHIFT_GLUED)):
                paths = {
                    policy: make_path(policy, LIVES[-1], seed, scene, occ, ss, sd)
                    for policy in POLICIES
                }
                for policy, path in paths.items():
                    if len(path) != LIVES[-1]:
                        raise RuntimeError("path length")
                if shift_name == "parallax" and n_opaque == PRIMARY and seed in VISIT_SEEDS:
                    for policy in ("curious_prog", "curious_surp"):
                        regions = region_sequence(paths[policy])
                        for life in LIVES:
                            chunk = regions[:life]
                            visits.append({
                                "policy": policy,
                                "seed": int(seed),
                                "opacity": PRIMARY,
                                "life": int(life),
                                "shift": "parallax",
                                "regions": chunk,
                                "first": first_visits(chunk),
                            })
                for life in LIVES:
                    got = {}
                    for policy, path in paths.items():
                        row = W.evaluate(scene, occ, path[:life], probes, ss, sd)
                        row.update({
                            "seed": int(seed),
                            "opacity": int(n_opaque),
                            "life": int(life),
                            "shift": shift_name,
                            "policy": policy,
                        })
                        rows.append(row)
                        got[policy] = row
                    if shift_name == "glued":
                        probes_g = [got[p]["probe"] for p in POLICIES]
                        logos = [got[p]["logo"] for p in POLICIES]
                        covers = [got[p]["cover"] for p in POLICIES]
                        if max(probes_g) - min(probes_g) > 1e-9 or max(logos) - min(logos) > 1e-9:
                            raise RuntimeError("glued integrator depended on the path")
                        if any(abs(c - clear) > 1e-12 for c in covers):
                            raise RuntimeError("glued coverage moved")
                    still_cover = got["still"]["cover"]
                    if abs(still_cover - clear) > 1e-12:
                        raise RuntimeError("still coverage is not the clear fraction")
    return rows, visits


def subset(rows, **want):
    out = []
    for row in rows:
        if all(row[k] == v for k, v in want.items()):
            out.append(row)
    return out


def by_policy_seed(rows, policy):
    found = [r for r in rows if r["policy"] == policy]
    found.sort(key=lambda r: r["seed"])
    if [r["seed"] for r in found] != SEEDS:
        raise RuntimeError(f"missing seeds for {policy}")
    return found


def scoring_block(rows, life):
    block = subset(rows, shift="parallax", opacity=PRIMARY, life=life)
    cols = {}
    for policy in ("curious_prog", "random", "sweep", "still"):
        cols[policy] = [r["probe"] for r in by_policy_seed(block, policy)]
    d_pr = [a - b for a, b in zip(cols["curious_prog"], cols["random"])]
    d_ps = [a - b for a, b in zip(cols["curious_prog"], cols["sweep"])]
    d_rs = [a - b for a, b in zip(cols["random"], cols["still"])]
    per_seed = []
    for i, seed in enumerate(SEEDS):
        per_seed.append({
            "seed": int(seed),
            "curious_prog": cols["curious_prog"][i],
            "random": cols["random"][i],
            "sweep": cols["sweep"][i],
            "still": cols["still"][i],
            "curious_prog_minus_random": d_pr[i],
            "curious_prog_minus_sweep": d_ps[i],
            "random_minus_still": d_rs[i],
        })
    return {
        "life": int(life),
        "per_seed": per_seed,
        "mean_curious_prog_minus_random": float(np.mean(d_pr)),
        "mean_curious_prog_minus_sweep": float(np.mean(d_ps)),
        "mean_random_minus_still": float(np.mean(d_rs)),
        "count_curious_prog_minus_random_le_-0.30": int(np.sum(np.asarray(d_pr) <= BAR2)),
        "count_curious_prog_minus_sweep_le_+0.10": int(np.sum(np.asarray(d_ps) <= BAR3)),
        "count_random_minus_still_le_-0.30": int(np.sum(np.asarray(d_rs) <= BAR1)),
    }


def grid(rows, shift, key):
    out = []
    for n_opaque in OPACITIES:
        for life in LIVES:
            for policy in POLICIES:
                got = by_policy_seed(subset(rows, shift=shift, opacity=n_opaque, life=life), policy)
                mu, sd = mean_std([r[key] for r in got])
                out.append({
                    "shift": shift,
                    "opacity": int(n_opaque),
                    "life": int(life),
                    "policy": policy,
                    "mean": mu,
                    "std": sd,
                })
    return out


def render(payload):
    lines = []
    lines.append("opacity 768. integrator viewpoint probe, bits per scored grey. population std.")
    lines.append("")
    for block in payload["scoring"]:
        life = block["life"]
        lines.append(f"L={life}")
        lines.append("")
        lines.append("| seed | curious_prog-random | curious_prog-sweep | random-still |")
        lines.append("|---:|---:|---:|---:|")
        for row in block["per_seed"]:
            lines.append(
                f"| {row['seed']} | {f6(row['curious_prog_minus_random'])} | "
                f"{f6(row['curious_prog_minus_sweep'])} | {f6(row['random_minus_still'])} |"
            )
        lines.append(
            f"| mean | {f6(block['mean_curious_prog_minus_random'])} | "
            f"{f6(block['mean_curious_prog_minus_sweep'])} | {f6(block['mean_random_minus_still'])} |"
        )
        lines.append("")
        lines.append(
            f"L={life} count curious_prog-random <= -0.30: "
            f"{block['count_curious_prog_minus_random_le_-0.30']}/20"
        )
        lines.append(
            f"L={life} count curious_prog-sweep <= +0.10: "
            f"{block['count_curious_prog_minus_sweep_le_+0.10']}/20"
        )
        lines.append(
            f"L={life} count random-still <= -0.30: "
            f"{block['count_random_minus_still_le_-0.30']}/20"
        )
        lines.append("")
    b25 = payload["scoring"][0]
    b64 = payload["scoring"][1]
    lines.append(
        f"bar1 random-still <= -0.30 at L=25: {b25['count_random_minus_still_le_-0.30']}/20 required >= {BAR1_NEED}"
    )
    lines.append(
        f"bar2 curious_prog-random <= -0.30 at L=25: "
        f"{b25['count_curious_prog_minus_random_le_-0.30']}/20 required >= {BAR2_NEED}"
    )
    lines.append(
        f"bar3 mean curious_prog-sweep at L=64: {f6(b64['mean_curious_prog_minus_sweep'])} required <= +0.10"
    )
    lines.append("")
    lines.append("parallax integrator")
    lines.append("")
    lines.append("| opacity | L | policy | probe | coverage | logo |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    index = {(r["opacity"], r["life"], r["policy"]): r for r in payload["parallax_probe"]}
    cover = {(r["opacity"], r["life"], r["policy"]): r for r in payload["parallax_cover"]}
    logo = {(r["opacity"], r["life"], r["policy"]): r for r in payload["parallax_logo"]}
    for n_opaque in OPACITIES:
        for life in LIVES:
            for policy in POLICIES:
                key = (n_opaque, life, policy)
                lines.append(
                    f"| {n_opaque} | {life} | {policy} | {pm(index[key]['mean'], index[key]['std'])} | "
                    f"{pm(cover[key]['mean'], cover[key]['std'])} | {pm(logo[key]['mean'], logo[key]['std'])} |"
                )
    lines.append("")
    lines.append("glued integrator. dot shift = logo shift = 1.")
    lines.append("")
    lines.append("| opacity | L | policy | probe | coverage | logo |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    index = {(r["opacity"], r["life"], r["policy"]): r for r in payload["glued_probe"]}
    cover = {(r["opacity"], r["life"], r["policy"]): r for r in payload["glued_cover"]}
    logo = {(r["opacity"], r["life"], r["policy"]): r for r in payload["glued_logo"]}
    for n_opaque in OPACITIES:
        for life in LIVES:
            for policy in POLICIES:
                key = (n_opaque, life, policy)
                lines.append(
                    f"| {n_opaque} | {life} | {policy} | {pm(index[key]['mean'], index[key]['std'])} | "
                    f"{pm(cover[key]['mean'], cover[key]['std'])} | {pm(logo[key]['mean'], logo[key]['std'])} |"
                )
    lines.append("")
    lines.append("viewtable probe, parallax. memorised views, no motion registration.")
    lines.append("")
    lines.append("| opacity | L | policy | probe |")
    lines.append("|---:|---:|---|---:|")
    for row in payload["viewtable_probe"]:
        lines.append(
            f"| {row['opacity']} | {row['life']} | {row['policy']} | {pm(row['mean'], row['std'])} |"
        )
    lines.append("")
    lines.append("visit order. curious_prog and curious_surp. opacity 768. parallax. seeds 10 11 12.")
    lines.append("")
    for item in payload["visits"]:
        regions = " ".join(str(r) for r in item["regions"])
        first = " ".join(str(r) for r in item["first"])
        lines.append(
            f"{item['policy']} seed {item['seed']} L={item['life']} first: {first}"
        )
        lines.append(
            f"{item['policy']} seed {item['seed']} L={item['life']} regions: {regions}"
        )
    lines.append("")
    lines.append(f"wall_s {payload['wall_s']:.3f}")
    lines.append("")
    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    W.self_check()
    scene = W.make_scene()
    rows, visits = collect(scene)
    scoring = [scoring_block(rows, life) for life in LIVES]
    payload = {
        "seeds": SEEDS,
        "lives": list(LIVES),
        "opacities": list(OPACITIES),
        "policies": list(POLICIES),
        "primary_opacity": PRIMARY,
        "std": "population",
        "probe": "integrator viewpoint, bits per scored grey, holes excluded",
        "scoring": scoring,
        "parallax_probe": grid(rows, "parallax", "probe"),
        "parallax_cover": grid(rows, "parallax", "cover"),
        "parallax_logo": grid(rows, "parallax", "logo"),
        "glued_probe": grid(rows, "glued", "probe"),
        "glued_cover": grid(rows, "glued", "cover"),
        "glued_logo": grid(rows, "glued", "logo"),
        "viewtable_probe": grid(rows, "parallax", "view"),
        "visits": visits,
        "rows": [
            {
                "seed": r["seed"],
                "opacity": r["opacity"],
                "life": r["life"],
                "shift": r["shift"],
                "policy": r["policy"],
                "probe": r["probe"],
                "view": r["view"],
                "cover": r["cover"],
                "logo": r["logo"],
                "n_pix": r["n_pix"],
            }
            for r in rows
        ],
    }
    payload["wall_s"] = time.perf_counter() - t0
    text = render(payload)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w7")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "RESULTS.md")
    js_path = os.path.join(out_dir, "results.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    payload["wall_s"] = time.perf_counter() - t0
    text = render(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    b25, b64 = scoring
    print(
        f"bar1 {b25['count_random_minus_still_le_-0.30']}/20 "
        f"bar2 {b25['count_curious_prog_minus_random_le_-0.30']}/20 "
        f"bar3 {b64['mean_curious_prog_minus_sweep']:.6f} "
        f"wall_s {payload['wall_s']:.3f}"
    )


if __name__ == "__main__":
    main()
