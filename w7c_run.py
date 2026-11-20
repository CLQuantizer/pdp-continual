"""W7c: viewpoint glare, unchanged W7 region bandits, one CPU process.

Tests principles 3 and 4: compression-progress reward and action selection.
Raw paths and observed-frame KT counts include all glare observations. The
reservoir stores actual greys (W7b's ObservedReservoir), including the current
frame as in W7-0. No reservoir replacement occurs at L <= 64.
"""
import json
import time
from pathlib import Path

import numpy as np

import w7_run as R
import w7_world as W

SEEDS = list(range(50, 70))
LIVES = (25, 64)
POLICIES = R.POLICIES
OPACITY = 768
SHIFT = (W.SCENE_SHIFT, W.DOT_SHIFT)
BAR_SURP = 3 / 16
BAR_PROG = 1.5 / 16
BAR_BITS = -0.20
BAR_NEED = 16
OUT = Path(__file__).resolve().parent / "results" / "w7c"


def path_curious(scene, occ, budget, mode, seed, glare=True, shifts=SHIFT):
    """Same movement, EMA, initialization and tie breaks as R.path_region.

    Only the observation source and the history's actual-grey accounting differ.
    """
    ss, sd = shifts
    source = W.GlareRegion(seed) if glare else None
    side = W.P // W.REG
    nreg = side * side
    centers = [(rx * W.REG + W.REG // 2, ry * W.REG + W.REG // 2)
               for ry in range(side) for rx in range(side)]
    counts = np.zeros((W.N, W.N, W.G), dtype=np.int32)
    if mode == "surp":
        ema = np.full(nreg, np.log2(W.G))
        reservoir = None
    elif mode == "prog":
        ema = np.full(nreg, np.inf)
        reservoir = W.ObservedReservoir(seed)
    else:
        raise ValueError(mode)
    visits = np.zeros(nreg, dtype=np.int32)
    px = py = 0
    target = None
    path, trace = [], []
    for _ in range(int(budget)):
        here = R.region_id(px, py)
        if target is None or here == target:
            order = np.lexsort((np.arange(nreg), visits, -ema))
            target = int(order[0])
            if here == target:
                px, py = R.lawn_step(px, py)
            else:
                px, py = R.step_toward(px, py, *centers[target])
        else:
            px, py = R.step_toward(px, py, *centers[target])
        path.append((px, py))
        grey, vis, sx, sy = W.frame(scene, occ, px, py, ss, sd, glare_region=source)
        landed = R.region_id(px, py)
        ys, xs = sy[vis], sx[vis]
        vs = grey[vis].astype(np.int64)
        surprise = float(W.kt_from_counts(counts[ys, xs], vs).mean()) if vs.size else 0.0
        if mode == "prog":
            reservoir.include(ys, xs, vs)
            reward = reservoir.progress(counts, ys, xs, vs) if vs.size else 0.0
        else:
            reward = surprise
        np.add.at(counts, (ys, xs, vs), 1)
        if mode == "prog" and int(visits[landed]) == 0:
            ema[landed] = reward
        else:
            ema[landed] = (1.0 - W.BETA) * ema[landed] + W.BETA * reward
        visits[landed] += 1
        trace.append({"region": landed, "reward": reward, "surprise": surprise})
    if mode == "prog":
        assert reservoir.n_seen == len(path) == len(reservoir.slots)
    replay, _ = W.accumulate(scene, occ, path, ss, sd,
                             glare_region=W.GlareRegion(seed) if glare else None)
    if not np.array_equal(replay, counts):
        raise RuntimeError("online integrator differs from glare replay")
    return path, trace


def smoke_off():
    W.self_check()
    scene = W.make_scene()
    baseline = json.loads((OUT.parent / "w7" / "results.json").read_text())
    wanted = [r for r in baseline["rows"] if r["seed"] == 10]
    assert len(wanted) == 60
    got = {}
    for opacity in (512, 768, 922):
        occ = W.occluder(W.make_field(10), opacity)
        for name, shifts in (("parallax", R.SHIFT_PARALLAX), ("glued", R.SHIFT_GLUED)):
            for policy in POLICIES:
                reference = R.make_path(policy, 64, 10, scene, occ, *shifts)
                if policy.startswith("curious_"):
                    path, _ = path_curious(scene, occ, 64, policy.split("_")[1],
                                           10, glare=False, shifts=shifts)
                    assert path == reference, (opacity, name, policy, "off-path mismatch")
                else:
                    path = reference
                for life in LIVES:
                    got[opacity, name, policy, life] = W.evaluate(
                        scene, occ, path[:life], W.probe_points(10), *shifts,
                        glare_region=None)
    worst = 0.0
    for row in wanted:
        replay = got[row["opacity"], row["shift"], row["policy"], row["life"]]
        for key in ("probe", "view", "logo", "cover"):
            worst = max(worst, abs(row[key] - replay[key]))
        assert row["n_pix"] == replay["n_pix"]
    assert worst <= 1e-9, worst
    print(f"smoke_seed10_max_abs_err {worst:.12f}", flush=True)
    return {"seed": 10, "rows": len(wanted), "max_abs_err": worst,
            "curious_glare_off_paths_match_w7": True}


def frame_lines(grey, vis):
    # Lossless 32x32 display in the requested eight lines: four rows per line.
    rows = ["".join(str(int(g)) if v else "." for g, v in zip(gr, vr))
            for gr, vr in zip(grey, vis)]
    return [" | ".join(rows[i:i + 4]) for i in range(0, W.N, 4)]


def smoke_on():
    seed = 50
    scene = W.make_scene()
    occ = W.occluder(W.make_field(seed), OPACITY)
    glare = W.GlareRegion(seed)
    ry, rx = divmod(glare.region, W.P // W.REG)
    inside = (rx * W.REG + W.REG // 2, ry * W.REG + W.REG // 2)
    grey, vis, sx, sy = W.frame(scene, occ, *inside, *SHIFT, glare_region=glare)
    clean, cv, cx, cy = W.frame(scene, occ, *inside, *SHIFT)
    assert np.array_equal(vis, cv) and np.array_equal(sx, cx) and np.array_equal(sy, cy)
    assert np.array_equal(grey[~vis], clean[~vis])
    expected = np.random.default_rng(9700 + seed).integers(0, 8, size=int(vis.sum()))
    assert np.array_equal(grey[vis], expected)
    second, *_ = W.frame(scene, occ, *inside, *SHIFT, glare_region=glare)
    assert not np.array_equal(grey[vis], second[vis])
    outside = (0, 0)
    outer = W.frame(scene, occ, *outside, *SHIFT, glare_region=glare)
    normal = W.frame(scene, occ, *outside, *SHIFT)
    assert all(np.array_equal(a, b) for a, b in zip(outer, normal))
    # An outside glance must not advance the glare stream.
    stream_a, stream_b = W.GlareRegion(seed), W.GlareRegion(seed)
    W.frame(scene, occ, *outside, *SHIFT, glare_region=stream_a)
    a = W.frame(scene, occ, *inside, *SHIFT, glare_region=stream_a)
    b = W.frame(scene, occ, *inside, *SHIFT, glare_region=stream_b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    item = {"seed": seed, "region": glare.region, "inside_viewpoint": inside,
            "outside_viewpoint": outside, "inside": frame_lines(grey, vis),
            "outside": frame_lines(outer[0], outer[1])}
    print(f"seed 50 glare region {glare.region}", flush=True)
    print("32x32 frames: eight lines; | separates four consecutive raster rows; . = hole")
    for label in ("inside", "outside"):
        print(f"{label} {item[label + '_viewpoint']}")
        print("\n".join(item[label]))
    return item


def collect_paths(scene):
    paths, traces = {}, {}
    for seed in SEEDS:
        occ = W.occluder(W.make_field(seed), OPACITY)
        for policy in POLICIES:
            if policy.startswith("curious_"):
                path, trace = path_curious(scene, occ, 64, policy.split("_")[1], seed)
                traces[seed, policy] = trace
            else:
                path = R.make_path(policy, 64, seed, scene, occ, *SHIFT)
            paths[seed, policy] = path
    return paths, traces


def collect_rows(scene, paths):
    rows = []
    for seed in SEEDS:
        occ = W.occluder(W.make_field(seed), OPACITY)
        region = W.GlareRegion(seed).region
        for life in LIVES:
            for policy in POLICIES:
                path = paths[seed, policy][:life]
                row = W.evaluate(scene, occ, path, W.probe_points(seed), *SHIFT,
                                 glare_region=W.GlareRegion(seed))
                count = sum(R.region_id(x, y) == region for x, y in path)
                row.update(seed=seed, life=life, policy=policy, glare_region=region,
                           glare_glances=count, glare_share=count / life)
                assert row["n_pix"] == W.N_PROBE * (W.N * W.N - OPACITY)
                rows.append(row)
    return rows


def scoring(rows, life):
    block = {(r["seed"], r["policy"]): r for r in rows if r["life"] == life}
    seeds = []
    for seed in SEEDS:
        prog, surp, rand = [block[seed, p] for p in ("curious_prog", "curious_surp", "random")]
        seeds.append({"seed": seed, "glare_region": prog["glare_region"],
                      "prog_minus_surp": prog["probe"] - surp["probe"],
                      "prog_minus_random": prog["probe"] - rand["probe"],
                      "glare_glances": {p: block[seed, p]["glare_glances"] for p in POLICIES},
                      "glare_share": {p: block[seed, p]["glare_share"] for p in POLICIES}})
    return {"life": life, "per_seed": seeds,
            "mean_prog_minus_surp": float(np.mean([r["prog_minus_surp"] for r in seeds])),
            "mean_prog_minus_random": float(np.mean([r["prog_minus_random"] for r in seeds])),
            "mean_glare_share": {p: float(np.mean([r["glare_share"][p] for r in seeds])) for p in POLICIES},
            "count_gap_le_minus_0.20": sum(r["prog_minus_surp"] <= BAR_BITS for r in seeds),
            "count_surp_share_ge_3_over_16": sum(r["glare_share"]["curious_surp"] >= BAR_SURP for r in seeds),
            "count_prog_share_le_1.5_over_16": sum(r["glare_share"]["curious_prog"] <= BAR_PROG for r in seeds)}


def summaries(rows):
    result = []
    for life in LIVES:
        for policy in POLICIES:
            block = [r for r in rows if r["life"] == life and r["policy"] == policy]
            summary = {"life": life, "policy": policy}
            for key in ("probe", "cover", "logo", "glare_share"):
                vals = np.array([r[key] for r in block])
                summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
            result.append(summary)
    return result


def render(payload):
    lines = []
    for b in payload.get("scoring", []):
        life = b["life"]
        lines += [f"L = {life}. Probe differences in bits per clear grey; policy columns are glare glances / L (share).", "",
                  "| seed | glare region | prog − surp | prog − random | still | random | sweep | curious_prog | curious_surp |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in b["per_seed"]:
            cells = [f"{r['glare_glances'][p]}/{life} ({r['glare_share'][p]:.6f})" for p in POLICIES]
            lines.append(f"| {r['seed']} | {r['glare_region']} | {r['prog_minus_surp']:.6f} | {r['prog_minus_random']:.6f} | " + " | ".join(cells) + " |")
        cells = [f"{b['mean_glare_share'][p]:.6f}" for p in POLICIES]
        lines += [f"| mean | | {b['mean_prog_minus_surp']:.6f} | {b['mean_prog_minus_random']:.6f} | " + " | ".join(cells) + " |", ""]
    if payload.get("scoring"):
        b = payload["scoring"][-1]
        lines += ["L = 64 pre-registered bars (unchanged):", "",
                  "| bar | observed | required | result |", "|---|---:|---:|---|",
                  f"| 1. Mean surprise glare share | {b['mean_glare_share']['curious_surp']:.6f} | ≥ {BAR_SURP:.6f} | {'PASS' if payload['bars']['surprise_trapped'] else 'FAIL'} |",
                  f"| 2. Mean progress glare share | {b['mean_glare_share']['curious_prog']:.6f} | ≤ {BAR_PROG:.6f} | {'PASS' if payload['bars']['progress_escapes'] else 'FAIL'} |",
                  f"| 3. prog − surp ≤ −0.20 bits, seed count | {b['count_gap_le_minus_0.20']}/20 | ≥ {BAR_NEED}/20 | {'PASS' if payload['bars']['sees_more'] else 'FAIL'} |", "",
                  f"Per-seed threshold counts (bars 1–2 use means): surprise share ≥ 3/16: {b['count_surp_share_ge_3_over_16']}/20; progress share ≤ 1.5/16: {b['count_prog_share_le_1.5_over_16']}/20.", ""]
        if not payload['bars']['surprise_trapped']:
            lines += ["Bar 1 fails: the region bandit's surprise reward already saturates, and the noisy-TV contrast does not exist in this world (pre-registered interpretation).", ""]
        lines += ["Opacity 768/1024. Integrator viewpoint probe on all logo pixels; holes excluded. Mean ± population std (ddof = 0), seeds 50–69. Coverage is the share of logo pixels observed at least once, including glare observations.", "",
                  "| L | policy | probe bits | coverage | logo bits | glare share |", "|---:|---|---:|---:|---:|---:|"]
        for r in payload["summary"]:
            cells = [f"{r[k]['mean']:.6f} ± {r[k]['std']:.6f}" for k in ("probe", "cover", "logo", "glare_share")]
            lines.append(f"| {r['life']} | {r['policy']} | " + " | ".join(cells) + " |")
        lines += ["", "Principles tested: reward compression progress (3), act to maximize it (4). Each policy gets a fresh default_rng(9700 + seed) stream, consumed only on clear cells during glare glances. Regions use default_rng(9600 + seed).integers(1, 16). Paths run to 64; L=25 uses their prefixes. Glare counts are stored in the integrator. Probes use the original logo, including viewpoints inside the glare region.", ""]
    else:
        lines += ["The test is void again: curious_prog and curious_surp paths are identical on all 20 seeds. Stopped before scoring.", ""]
    lines += [f"Smoke: seed 10 glare-off max absolute error = {payload['smoke_off']['max_abs_err']:.12f} across 60 W7-0 rows; curious glare-off paths also match W7-0.",
              f"L=64 seeds with different curious_prog and curious_surp paths: {payload['paths_differ_l64']}/20 (smoke count; no bar).", "",
              "Seed 50 glare frames: 32×32, eight lines per frame. Each line groups four consecutive raster rows separated by |. A dot is an opaque hole.", "",
              f"Glare region: {payload['smoke_on']['region']}."]
    for label in ("inside", "outside"):
        lines += ["", f"{label}, viewpoint {payload['smoke_on'][label + '_viewpoint']}:", "", "```", *payload['smoke_on'][label], "```"]
    lines += ["", f"Total wall time: {payload['wall_s']:.3f} s (smokes, paths, scoring and report preparation; one CPU process).", ""]
    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    off, on = smoke_off(), smoke_on()
    scene = W.make_scene()
    paths, traces = collect_paths(scene)
    differ = sum(paths[s, "curious_prog"] != paths[s, "curious_surp"] for s in SEEDS)
    print(f"paths_differ_l64 {differ}/20", flush=True)
    payload = {"seeds": SEEDS, "lives": LIVES, "policies": POLICIES, "opacity": OPACITY,
               "bar_thresholds": {"surprise_mean_share_min": BAR_SURP,
                                  "progress_mean_share_max": BAR_PROG,
                                  "probe_gap_max_bits": BAR_BITS,
                                  "probe_gap_seed_count_min": BAR_NEED},
               "std_ddof": 0, "smoke_off": off, "smoke_on": on, "paths_differ_l64": differ,
               "void": differ == 0, "paths": [{"seed": s, "policy": p, "path": paths[s, p]}
                                               for s in SEEDS for p in POLICIES],
               "traces": [{"seed": s, "policy": p, "glances": tr} for (s, p), tr in traces.items()]}
    if differ:
        rows = collect_rows(scene, paths)
        scores = [scoring(rows, life) for life in LIVES]
        b = scores[-1]
        payload.update(rows=rows, scoring=scores, summary=summaries(rows), bars={
            "surprise_trapped": b["mean_glare_share"]["curious_surp"] >= BAR_SURP,
            "progress_escapes": b["mean_glare_share"]["curious_prog"] <= BAR_PROG,
            "sees_more": b["count_gap_le_minus_0.20"] >= BAR_NEED})
        print(json.dumps({"bars": payload["bars"], "mean_glare_share_l64": b["mean_glare_share"],
                          "gap_count_l64": b["count_gap_le_minus_0.20"]}), flush=True)
    else:
        print("The test is void again. Stopping before scoring.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload["wall_s"] = time.perf_counter() - t0
    (OUT / "RESULTS.md").write_text(render(payload))
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    payload["wall_s"] = time.perf_counter() - t0
    (OUT / "RESULTS.md").write_text(render(payload))
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wall_s {payload['wall_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
