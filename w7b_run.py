"""W7b scored run. One process. Seeds 30–49. Opacity 768. Writes results/w7b/.

Noise off, the world is W7. The smoke test replays seed 10 through that path.
"""
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import w7_run as R
import w7_world as W

SEEDS = list(range(30, 50))
LIVES = (25, 64)
OPACITY = 768
POLICIES = R.POLICIES
SHIFT = (W.SCENE_SHIFT, W.DOT_SHIFT)
BAR_GAP = -0.20
BAR_NEED = 16
FAIR = 1.0 / 16.0
BAR_SHARE = 1.5 * FAIR


def region_id(px, py):
    return R.region_id(px, py)


def _parts(bits, hit):
    noise = float(bits[hit].mean()) if np.any(hit) else None
    logo = float(bits[~hit].mean()) if np.any(~hit) else None
    return noise, logo, int(np.sum(hit)), int(np.sum(~hit))


def path_curious(scene, occ, budget, scene_shift, dot_shift, mode, seed):
    """W7 region policy. Glances in the noise block carry fresh uniform greys."""
    if mode not in ("surp", "prog"):
        raise ValueError(mode)
    noise = W.NoiseRegion(seed)
    side = W.P // W.REG
    nreg = side * side
    centers = []
    for ry in range(side):
        for rx in range(side):
            centers.append((rx * W.REG + W.REG // 2, ry * W.REG + W.REG // 2))
    counts = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    reservoir = W.ObservedReservoir(seed)
    if mode == "surp":
        ema = np.full(nreg, np.log2(W.G))
    else:
        ema = np.full(nreg, np.inf)
    visits = np.zeros(nreg, dtype=np.int32)
    px = py = 0
    target = None
    path = []
    trace = []
    for _ in range(int(budget)):
        here = region_id(px, py)
        if target is None or here == target:
            order = np.lexsort((np.arange(nreg), visits, -ema))
            target = int(order[0])
            if here == target:
                px, py = R.lawn_step(px, py)
            else:
                px, py = R.step_toward(px, py, centers[target][0], centers[target][1])
        else:
            px, py = R.step_toward(px, py, centers[target][0], centers[target][1])
        path.append((px, py))
        grey, vis, sx, sy = noise.frame(scene, occ, px, py, scene_shift, dot_shift)
        landed = region_id(px, py)
        ys = sy[vis]
        xs = sx[vis]
        vs = grey[vis].astype(np.int64)
        hit = noise.mask[ys, xs] if vs.size else np.zeros(0, dtype=bool)
        if vs.size == 0:
            reward = 0.0
            noise_s = logo_s = None
            n_hit = n_logo = 0
            nprog = lprog = 0.0
            reservoir.include(ys, xs, vs)
        else:
            bits = W.kt_from_counts(counts[ys, xs], vs)
            noise_s, logo_s, n_hit, n_logo = _parts(bits, hit)
            if mode == "surp":
                reward = float(bits.mean())
            reservoir.include(ys, xs, vs)
            total, nprog, lprog = reservoir.progress(counts, ys, xs, vs, mask=noise.mask)
            if mode == "prog":
                reward = total
            np.add.at(counts, (ys, xs, vs), 1)
        if mode == "prog" and int(visits[landed]) == 0:
            ema[landed] = reward
        else:
            ema[landed] = (1.0 - W.BETA) * ema[landed] + W.BETA * reward
        visits[landed] += 1
        trace.append({
            "region": int(landed),
            "reward": float(reward),
            "noise_surprise": noise_s,
            "logo_surprise": logo_s,
            "noise_progress": float(nprog),
            "logo_progress": float(lprog),
            "n_noise": int(n_hit),
            "n_logo": int(n_logo),
        })
    if reservoir.n_seen != len(path) or len(reservoir.slots) != len(path):
        raise RuntimeError("reservoir dropped a glance inside the cap")
    batch, _visits = W.accumulate(scene, occ, path, scene_shift, dot_shift)
    if not np.array_equal(batch[~noise.mask], counts[~noise.mask]):
        raise RuntimeError("online integrator drifted on logo pixels")
    return path, trace, noise.region, noise.mask


def make_path(policy, budget, seed, scene, occ, scene_shift, dot_shift):
    if policy in ("still", "random", "sweep"):
        return R.make_path(policy, budget, seed, scene, occ, scene_shift, dot_shift), None
    if policy == "curious_prog":
        return path_curious(scene, occ, budget, scene_shift, dot_shift, "prog", seed)
    if policy == "curious_surp":
        return path_curious(scene, occ, budget, scene_shift, dot_shift, "surp", seed)
    raise ValueError(policy)


def noise_count(path, life, region):
    n = 0
    for px, py in path[:life]:
        if region_id(px, py) == region:
            n += 1
    return n


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def f6(x):
    return f"{x:.6f}"


def pm(mu, sd):
    return f"{mu:.6f} ± {sd:.6f}"


def _mean_defined(xs):
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def trace_summary(trace, life):
    chunk = trace[:life]
    return {
        "noise_surprise": _mean_defined([s["noise_surprise"] for s in chunk]),
        "logo_surprise": _mean_defined([s["logo_surprise"] for s in chunk]),
        "noise_progress": float(np.mean([s["noise_progress"] for s in chunk])),
        "logo_progress": float(np.mean([s["logo_progress"] for s in chunk])),
        "visible_noise_fraction": float(np.mean([
            s["n_noise"] / (s["n_noise"] + s["n_logo"])
            for s in chunk
            if s["n_noise"] + s["n_logo"]
        ])),
    }


def by_policy(rows, policy):
    found = [r for r in rows if r["policy"] == policy]
    found.sort(key=lambda r: r["seed"])
    if [r["seed"] for r in found] != SEEDS:
        raise RuntimeError(f"missing seeds for {policy}")
    return found


def grid(rows, key):
    out = []
    for life in LIVES:
        block = [r for r in rows if r["life"] == life]
        for policy in POLICIES:
            got = by_policy(block, policy)
            vals = [r[key] for r in got]
            if any(v is None for v in vals):
                continue
            mu, sd = mean_std(vals)
            out.append({
                "life": int(life),
                "policy": policy,
                "key": key,
                "mean": mu,
                "std": sd,
            })
    return out


def render(payload):
    lines = []
    lines.append(
        "opacity 768. integrator viewpoint probe on non-noise logo pixels, "
        "bits per scored grey. population std."
    )
    lines.append(
        f"noise region: default_rng(9500 + seed).integers(1, 16). "
        f"fair share {FAIR:.6f}. bar2 limit {BAR_SHARE:.6f}."
    )
    lines.append(
        "Policy columns count glances whose viewpoint lies in that region. "
        "The mean row is the mean share."
    )
    lines.append("")
    for block in payload["scoring"]:
        life = block["life"]
        lines.append(f"L={life}")
        lines.append("")
        lines.append(
            "| seed | region | curious_prog-curious_surp | curious_prog-random | "
            "still | random | sweep | prog | surp |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in block["per_seed"]:
            cells = []
            for policy in POLICIES:
                n = row["noise_glances"][policy]
                cells.append(f"{n}/{life}")
            lines.append(
                f"| {row['seed']} | {row['noise_region']} | "
                f"{f6(row['curious_prog_minus_curious_surp'])} | "
                f"{f6(row['curious_prog_minus_random'])} | "
                + " | ".join(cells)
                + " |"
            )
        share_cells = []
        for policy in POLICIES:
            share_cells.append(f6(block["mean_noise_share"][policy]))
        lines.append(
            f"| mean |  | {f6(block['mean_curious_prog_minus_curious_surp'])} | "
            f"{f6(block['mean_curious_prog_minus_random'])} | "
            + " | ".join(share_cells)
            + " |"
        )
        lines.append("")
        lines.append(
            f"L={life} count curious_prog-curious_surp <= -0.20: "
            f"{block['count_curious_prog_minus_curious_surp_le_-0.20']}/20"
        )
        lines.append(
            f"L={life} mean curious_prog noise share: {f6(block['mean_noise_share']['curious_prog'])}"
        )
        lines.append("")
    b64 = payload["scoring"][1]
    lines.append(
        f"bar1 curious_prog-curious_surp <= -0.20 at L=64: "
        f"{b64['count_curious_prog_minus_curious_surp_le_-0.20']}/20 required >= {BAR_NEED}"
    )
    lines.append(
        f"bar2 mean curious_prog noise share at L=64: "
        f"{f6(b64['mean_noise_share']['curious_prog'])} required <= {BAR_SHARE:.6f}"
    )
    lines.append(
        f"L=64 seeds curious_prog path equals curious_surp path: "
        f"{payload['paths_equal_l64']}/20"
    )
    lines.append("")
    lines.append("policies. probe, coverage and logo bits on non-noise pixels.")
    lines.append("")
    lines.append("| L | policy | probe | coverage | logo | noise_share |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    index = {(r["life"], r["policy"]): r for r in payload["probe"]}
    cover = {(r["life"], r["policy"]): r for r in payload["cover"]}
    logo = {(r["life"], r["policy"]): r for r in payload["logo"]}
    share = {(r["life"], r["policy"]): r for r in payload["noise_share"]}
    for life in LIVES:
        for policy in POLICIES:
            key = (life, policy)
            lines.append(
                f"| {life} | {policy} | {pm(index[key]['mean'], index[key]['std'])} | "
                f"{pm(cover[key]['mean'], cover[key]['std'])} | "
                f"{pm(logo[key]['mean'], logo[key]['std'])} | "
                f"{pm(share[key]['mean'], share[key]['std'])} |"
            )
    lines.append("")
    lines.append(
        "visible pixels. noise surprise is mean KT bits on noise-region pixels. "
        "logo surprise is the same on the other visible pixels. "
        "progress is mean eq. 5 bits per glance, split by those pixels."
    )
    lines.append("")
    lines.append(
        "| L | policy | noise_surprise | logo_surprise | noise_progress | logo_progress | visible_noise_fraction |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    bundles = {
        (r["life"], r["policy"], r["key"]): r for r in payload["pixel"]
    }
    for life in LIVES:
        for policy in ("curious_prog", "curious_surp"):
            cells = []
            for key_name in (
                "noise_surprise",
                "logo_surprise",
                "noise_progress",
                "logo_progress",
                "visible_noise_fraction",
            ):
                row = bundles[(life, policy, key_name)]
                cells.append(pm(row["mean"], row["std"]))
            lines.append(f"| {life} | {policy} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"smoke_seed10_max_abs_err {payload['smoke_max_abs_err']:.12f}")
    lines.append(f"wall_s {payload['wall_s']:.3f}")
    lines.append("")
    return "\n".join(lines)


def _replay_match(scene, occ, path, ss, sd, seed):
    """ObservedReservoir.progress matches GlanceReservoir.progress on a fixed logo."""
    parent = W.GlanceReservoir(seed)
    child = W.ObservedReservoir(seed)
    counts_p = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    counts_c = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    worst = 0.0
    for px, py in path:
        grey, vis, sx, sy = W.frame(scene, occ, px, py, ss, sd)
        ys = sy[vis]
        xs = sx[vis]
        vs = grey[vis].astype(np.int64)
        parent.include(ys, xs)
        child.include(ys, xs, vs)
        if vs.size == 0:
            rp = 0.0
            rc = 0.0
        else:
            rp = parent.progress(counts_p, ys, xs, vs)
            rc = child.progress(counts_c, ys, xs, vs)
            np.add.at(counts_p, (ys, xs, vs), 1)
            np.add.at(counts_c, (ys, xs, vs), 1)
        worst = max(worst, abs(rp - rc))
    return worst


def _contradict_progress():
    """A third sight that disagrees is not the deterministic shortcut."""
    counts = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    res = W.ObservedReservoir(0)
    short = W.GlanceReservoir(0)
    seq = [0, 0, 1]
    for v in seq:
        ys = np.array([0])
        xs = np.array([0])
        vs = np.array([v])
        res.include(ys, xs, vs)
        short.include(ys, xs)
        got = res.progress(counts, ys, xs, vs)
        shortcut = short.progress(counts, ys, xs, vs)
        np.add.at(counts, (ys, xs, vs), 1)
    # Last step is the contradictory one. Recompute it explicitly.
    # counts now include all three. Rebuild the last step's expected value.
    counts2 = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    counts2[0, 0, 0] = 2
    # reservoir greys 0, 0, 1. New grey 1. t=2, c0=2, c1=0.
    b0 = -np.log2((2.0 + W.ALPHA) / (2.0 + W.ALPHA * W.G))
    a0 = -np.log2((2.0 + W.ALPHA) / (3.0 + W.ALPHA * W.G))
    b1 = -np.log2((0.0 + W.ALPHA) / (2.0 + W.ALPHA * W.G))
    a1 = -np.log2((1.0 + W.ALPHA) / (3.0 + W.ALPHA * W.G))
    expect = 2.0 * (b0 - a0) + (b1 - a1)
    if abs(got - expect) > 1e-9:
        raise RuntimeError(f"eq5 contradict {got} != {expect}")
    if not (got < shortcut - 1.0):
        raise RuntimeError(f"noise progress did not fall below the shortcut ({got} vs {shortcut})")
    return abs(got - expect)


def smoke():
    W.self_check()
    if W.noise_region_index(30) != 4:
        raise RuntimeError("seed 30 noise region is not 4")
    for seed in SEEDS:
        if W.noise_region_index(seed) == 0:
            raise RuntimeError("origin region")
    scene = W.make_scene()
    nr = W.NoiseRegion(30)
    nr2 = W.NoiseRegion(30)
    if nr.region != 4 or int(nr.mask.sum()) != 64:
        raise RuntimeError("noise mask")
    occ = W.occluder(W.make_field(30), OPACITY)
    g0, _vis, sx, sy = W.frame(scene, occ, 3, 5, *SHIFT)
    g1, vis1, sx1, sy1 = nr.frame(scene, occ, 3, 5, *SHIFT)
    g2, _v2, _sx2, _sy2 = nr2.frame(scene, occ, 3, 5, *SHIFT)
    hit = nr.mask[sy, sx]
    if not np.array_equal(g1, g2):
        raise RuntimeError("noise stream is not a function of the seed")
    if not np.array_equal(g0[~hit], g1[~hit]):
        raise RuntimeError("noise rewrote a logo pixel")
    if int(hit.sum()) != 64:
        raise RuntimeError("viewport did not cover the noise region once")
    if g1[hit].min() < 0 or g1[hit].max() >= W.G:
        raise RuntimeError("noise grey outside 0..7")
    g3, _v3, sy3, sx3 = nr.frame(scene, occ, 3, 5, *SHIFT)
    if np.array_equal(g1[hit], g3[nr.mask[sy3, sx3]]):
        raise RuntimeError("second glance repeated the noise")
    contra_err = _contradict_progress()
    field = W.make_field(10)
    probes = W.probe_points(10)
    occ10 = W.occluder(field, OPACITY)
    tour = R.make_path("curious_prog", 64, 10, scene, occ10, *SHIFT)
    match = _replay_match(scene, occ10, tour, SHIFT[0], SHIFT[1], 10)
    if match > 1e-9:
        raise RuntimeError(f"eq5 drifted from the deterministic progress by {match}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w7", "results.json"), encoding="utf-8") as f:
        prev = json.load(f)
    want = [r for r in prev["rows"] if r["seed"] == 10]
    got = {}
    for n_opaque in (512, 768, 922):
        occ_i = W.occluder(field, n_opaque)
        for shift_name, (ss, sd) in (("parallax", R.SHIFT_PARALLAX), ("glued", R.SHIFT_GLUED)):
            paths = {
                policy: R.make_path(policy, 64, 10, scene, occ_i, ss, sd)
                for policy in POLICIES
            }
            for life in LIVES:
                for policy, path in paths.items():
                    row = W.evaluate(scene, occ_i, path[:life], probes, ss, sd)
                    got[(n_opaque, life, shift_name, policy)] = row
    max_err = 0.0
    if len(want) != 3 * 2 * 2 * 5:
        raise RuntimeError(f"expected 60 seed-10 rows, found {len(want)}")
    for r in want:
        g = got[(r["opacity"], r["life"], r["shift"], r["policy"])]
        for key in ("probe", "view", "logo", "cover"):
            max_err = max(max_err, abs(g[key] - r[key]))
        if g["n_pix"] != r["n_pix"]:
            raise RuntimeError(f"n_pix {g['n_pix']} != {r['n_pix']}")
    if max_err > 1e-9:
        raise RuntimeError(f"seed 10 noise-off err {max_err}")
    return max_err, match, contra_err


def pixel_grid(rows):
    out = []
    keys = (
        "noise_surprise",
        "logo_surprise",
        "noise_progress",
        "logo_progress",
        "visible_noise_fraction",
    )
    for life in LIVES:
        block = [r for r in rows if r["life"] == life]
        for policy in ("curious_prog", "curious_surp"):
            got = by_policy(block, policy)
            for key in keys:
                mu, sd = mean_std([r[key] for r in got])
                out.append({
                    "life": int(life),
                    "policy": policy,
                    "key": key,
                    "mean": mu,
                    "std": sd,
                })
    return out


def main():
    t0 = time.perf_counter()
    smoke_err, match_err, contra_err = smoke()
    scene = W.make_scene()
    rows = []
    ss, sd = SHIFT
    path_equal = []
    for seed in SEEDS:
        field = W.make_field(seed)
        probes = W.probe_points(seed)
        occ = W.occluder(field, OPACITY)
        region = W.noise_region_index(seed)
        mask = W.region_mask(region)
        paths = {}
        traces = {}
        for policy in POLICIES:
            got = make_path(policy, LIVES[-1], seed, scene, occ, ss, sd)
            if policy in ("curious_prog", "curious_surp"):
                path, trace, reg, mask_p = got
                if reg != region or not np.array_equal(mask_p, mask):
                    raise RuntimeError("noise region disagreed with the seed draw")
                traces[policy] = trace
            else:
                path = got[0]
            if len(path) != LIVES[-1]:
                raise RuntimeError("path length")
            paths[policy] = path
        same = paths["curious_prog"] == paths["curious_surp"]
        path_equal.append(bool(same))
        for life in LIVES:
            for policy, path in paths.items():
                row = W.evaluate(scene, occ, path[:life], probes, ss, sd, skip=mask)
                n = noise_count(path, life, region)
                row.update({
                    "seed": int(seed),
                    "opacity": OPACITY,
                    "life": int(life),
                    "policy": policy,
                    "noise_region": int(region),
                    "noise_glances": int(n),
                    "noise_share": n / float(life),
                    "path_equals_surp": bool(same) if policy == "curious_prog" else None,
                })
                if policy in traces:
                    row.update(trace_summary(traces[policy], life))
                rows.append(row)
    scoring = []
    for life in LIVES:
        block_rows = [r for r in rows if r["life"] == life]
        cols = {policy: by_policy(block_rows, policy) for policy in POLICIES}
        d_ps = [
            a["probe"] - b["probe"]
            for a, b in zip(cols["curious_prog"], cols["curious_surp"])
        ]
        d_pr = [
            a["probe"] - b["probe"]
            for a, b in zip(cols["curious_prog"], cols["random"])
        ]
        per_seed = []
        for i, seed in enumerate(SEEDS):
            per_seed.append({
                "seed": int(seed),
                "noise_region": cols["still"][i]["noise_region"],
                "curious_prog": cols["curious_prog"][i]["probe"],
                "curious_surp": cols["curious_surp"][i]["probe"],
                "random": cols["random"][i]["probe"],
                "curious_prog_minus_curious_surp": d_ps[i],
                "curious_prog_minus_random": d_pr[i],
                "noise_glances": {policy: cols[policy][i]["noise_glances"] for policy in POLICIES},
                "noise_share": {policy: cols[policy][i]["noise_share"] for policy in POLICIES},
                "paths_equal": bool(path_equal[i]),
            })
        scoring.append({
            "life": int(life),
            "per_seed": per_seed,
            "mean_curious_prog_minus_curious_surp": float(np.mean(d_ps)),
            "mean_curious_prog_minus_random": float(np.mean(d_pr)),
            "count_curious_prog_minus_curious_surp_le_-0.20": int(np.sum(np.asarray(d_ps) <= BAR_GAP)),
            "mean_noise_share": {
                policy: float(np.mean([row["noise_share"][policy] for row in per_seed]))
                for policy in POLICIES
            },
            "std_noise_share": {
                policy: float(np.std([row["noise_share"][policy] for row in per_seed], ddof=0))
                for policy in POLICIES
            },
        })
    payload = {
        "seeds": SEEDS,
        "lives": list(LIVES),
        "opacity": OPACITY,
        "policies": list(POLICIES),
        "std": "population",
        "probe_def": "integrator viewpoint, non-noise logo pixels, holes excluded",
        "noise_region": "default_rng(9500+seed).integers(1, 16)",
        "noise_greys": "default_rng(9600+seed), one stream per policy, uniform 0..7",
        "fair_share": FAIR,
        "bar2_limit": BAR_SHARE,
        "scoring": scoring,
        "probe": grid(rows, "probe"),
        "cover": grid(rows, "cover"),
        "logo": grid(rows, "logo"),
        "noise_share": grid(rows, "noise_share"),
        "pixel": pixel_grid(rows),
        "paths_equal_l64": int(np.sum(path_equal)),
        "eq5_deterministic_max_abs_err": match_err,
        "eq5_contradict_max_abs_err": contra_err,
        "smoke_max_abs_err": smoke_err,
        "rows": [
            {
                "seed": r["seed"],
                "life": r["life"],
                "policy": r["policy"],
                "noise_region": r["noise_region"],
                "probe": r["probe"],
                "logo": r["logo"],
                "cover": r["cover"],
                "n_pix": r["n_pix"],
                "noise_glances": r["noise_glances"],
                "noise_share": r["noise_share"],
                "noise_surprise": r.get("noise_surprise"),
                "logo_surprise": r.get("logo_surprise"),
                "noise_progress": r.get("noise_progress"),
                "logo_progress": r.get("logo_progress"),
                "visible_noise_fraction": r.get("visible_noise_fraction"),
            }
            for r in rows
        ],
    }
    payload["wall_s"] = time.perf_counter() - t0
    text = render(payload)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w7b")
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
    b64 = scoring[1]
    print(
        f"bar1 {b64['count_curious_prog_minus_curious_surp_le_-0.20']}/20 "
        f"bar2 {b64['mean_noise_share']['curious_prog']:.6f} "
        f"paths_equal {payload['paths_equal_l64']}/20 "
        f"smoke {smoke_err:.3e} "
        f"wall_s {payload['wall_s']:.3f}"
    )


if __name__ == "__main__":
    main()
