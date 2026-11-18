"""W7 sketch: headroom of the dotted window.

One process, CPU only. Prints the probe table. Does not write a results file.

The scene is a fixed 32×32 logo. A near dot field covers a set fraction of it.
Viewpoint actions move the viewer on a 32×32 torus. Per step the logo shifts by
SCENE_SHIFT pixels and the dots by DOT_SHIFT, so the holes move across the logo.
A glance is the full frame: holes change, the logo underneath does not.

Two predictors, neither of which is a curious controller:

- integrator: KT counts in logo coordinates, using the true shifts. This is the
  ceiling used to size the world. It is not the proposed learner.
- view table: KT counts keyed by (viewpoint, raster cell). Efference is only a
  key. No scene coordinate and no shift. This is the order-0 form of the
  proposed learner.

The probe asks both to name the unoccluded pixels at viewpoints drawn once per
seed, shared by every policy. Holes are not scored. Logo bits score every logo
pixel under the integrator only, so a glued mask (equal shift) cannot look solved.
"""
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

N = 32
G = 8
P = 32
REG = 8
SCENE_SHIFT = 1
DOT_SHIFT = 2
RADIUS = 3
DENSITIES = (0.50, 0.75, 0.90)
SEEDS = (0, 1, 2)
BUDGETS = (25, 64)
N_PROBE = 32
BETA = 0.5
ALPHA = 0.5
# Flat tape a cortex can read. Roles are disjoint ranges.
# 0..7 grey, 8 hole, 9..13 actions, 14..45 px, 46..77 py.
HOLE = G
ACT0 = G + 1
PX0 = ACT0 + 5
PY0 = PX0 + P
ALPHABET = PY0 + P
ACTIONS = ("stay", "+x", "-x", "+y", "-y")


def box_blur(field, radius):
    acc = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
    return acc


def make_scene():
    """Four 16×16 glyphs, edge to edge. Greys 0..7 all occur."""
    scene = np.zeros((N, N), dtype=np.int16)

    def block(y0, x0, fill):
        g = np.full((16, 16), fill, dtype=np.int16)
        return g, y0, x0

    ring, y, x = block(0, 0, 1)
    yy, xx = np.ogrid[0:16, 0:16]
    ring[(yy - 8) ** 2 + (xx - 8) ** 2 <= 36] = 6
    ring[(yy - 8) ** 2 + (xx - 8) ** 2 <= 9] = 1
    scene[y:y + 16, x:x + 16] = ring

    plus = np.full((16, 16), 2, dtype=np.int16)
    plus[7:10, 2:14] = 5
    plus[2:14, 7:10] = 5
    scene[0:16, 16:32] = plus

    diag = np.full((16, 16), 0, dtype=np.int16)
    for k in range(16):
        diag[k, k] = 7
        if k + 1 < 16:
            diag[k, k + 1] = 4
        if k + 4 < 16:
            diag[k, k + 4] = 7
    scene[16:32, 0:16] = diag

    steps = np.full((16, 16), 3, dtype=np.int16)
    for k in range(0, 16, 4):
        steps[k:k + 2, :] = 3
        steps[k + 2:k + 4, : 8 + k // 2] = 4
        steps[k + 2:k + 4, 8 + k // 2:] = 7
    scene[16:32, 16:32] = steps
    if scene.min() < 0 or scene.max() >= G:
        raise RuntimeError("glyph grey outside 0..7")
    if len(set(int(v) for v in np.unique(scene))) < G:
        raise RuntimeError("logo does not use every grey")
    return scene


def make_field(seed):
    rng = np.random.default_rng(9_000 + int(seed))
    return box_blur(rng.random((N, N)), RADIUS)


def occluder_from(acc, density):
    """Opaque on the highest `density` fraction of the blurred field.

    Ties in the blurred field break by raster index (mergesort is stable).
    """
    target = int(round(float(density) * N * N))
    if target <= 0 or target >= N * N:
        raise RuntimeError("density leaves no hole or no dot")
    order = np.argsort(acc.ravel(), kind="mergesort")
    opaque = np.zeros(N * N, dtype=bool)
    opaque[order[-target:]] = True
    occ = opaque.reshape(N, N)
    if int(occ.sum()) != target:
        raise RuntimeError("occluder missed its fraction")
    return occ


def maps(px, py, scene_shift, dot_shift):
    ii, jj = np.indices((N, N))
    sy = (ii + int(py) * scene_shift) % N
    sx = (jj + int(px) * scene_shift) % N
    oy = (ii + int(py) * dot_shift) % N
    ox = (jj + int(px) * dot_shift) % N
    return sx, sy, ox, oy


def frame(scene, occ, px, py, scene_shift, dot_shift):
    """Grey values and visibility of the 32×32 glance. Visibility is the holes."""
    sx, sy, ox, oy = maps(px, py, scene_shift, dot_shift)
    vis = ~occ[oy, ox]
    grey = scene[sy, sx]
    return grey, vis, sx, sy


def glance_tape(action, px, py, grey, vis):
    """One glance as a symbol tape. (px, py) is repeated at the start of every row
    so a short context still contains the efference copy."""
    if action not in ACTIONS:
        raise ValueError(action)
    pix = np.where(vis, grey, HOLE).astype(np.int16)
    out = [ACT0 + ACTIONS.index(action), PX0 + int(px), PY0 + int(py)]
    for i in range(N):
        out.append(PX0 + int(px))
        out.append(PY0 + int(py))
        out.extend(int(v) for v in pix[i])
    if len(out) != 3 + N * (2 + N):
        raise RuntimeError("glance length drifted")
    if min(out) < 0 or max(out) >= ALPHABET:
        raise RuntimeError("symbol outside the alphabet")
    return out


def kt_from_counts(counts, values):
    """counts (n, G), values (n,). KT bits of each value. Unseen rows are log2(G)."""
    tot = counts.sum(axis=1)
    got = counts[np.arange(len(values)), values]
    return -np.log2((got + ALPHA) / (tot + ALPHA * G))


def logo_bits(counts, scene):
    flat_c = counts.reshape(-1, G)
    flat_v = scene.reshape(-1).astype(np.int64)
    return float(kt_from_counts(flat_c, flat_v).mean())


def coverage(counts):
    return float((counts.sum(axis=2) > 0).mean())


def accumulate(scene, occ, path, scene_shift, dot_shift):
    """Integrator counts and per-viewpoint visit counts along a path."""
    counts = np.zeros((N, N, G), dtype=np.int32)
    visits = np.zeros((P, P), dtype=np.int32)
    pool = np.zeros((N, N, G), dtype=np.int32)
    for px, py in path:
        grey, vis, sx, sy = frame(scene, occ, px, py, scene_shift, dot_shift)
        ys = sy[vis]
        xs = sx[vis]
        vs = grey[vis].astype(np.int64)
        np.add.at(counts, (ys, xs, vs), 1)
        ii, jj = np.where(vis)
        np.add.at(pool, (ii, jj, vs), 1)
        visits[int(py), int(px)] += 1
    return counts, visits, pool


def probe_points(seed):
    rng = np.random.default_rng(7_000 + int(seed))
    pick = rng.choice(P * P, size=N_PROBE, replace=False)
    return [(int(i % P), int(i // P)) for i in pick]


def score_probe(scene, occ, counts, visits, pool, probes, scene_shift, dot_shift):
    """Mean KT bits on unoccluded probe pixels. Three readouts, one probe set."""
    integ = []
    view = []
    pooled = []
    n_pix = 0
    for qx, qy in probes:
        grey, vis, sx, sy = frame(scene, occ, qx, qy, scene_shift, dot_shift)
        if not np.any(vis):
            continue
        vs = grey[vis].astype(np.int64)
        n_pix += int(vs.size)
        integ.append(kt_from_counts(counts[sy[vis], sx[vis]], vs))
        k = int(visits[qy, qx])
        if k == 0:
            view.append(np.full(vs.size, math.log2(G)))
        else:
            p = (k + ALPHA) / (k + ALPHA * G)
            view.append(np.full(vs.size, -math.log2(p)))
        ii, jj = np.where(vis)
        pooled.append(kt_from_counts(pool[ii, jj], vs))
    if n_pix == 0:
        raise RuntimeError("probe saw no unoccluded pixel")
    return (
        float(np.concatenate(integ).mean()),
        float(np.concatenate(view).mean()),
        float(np.concatenate(pooled).mean()),
        n_pix,
    )


def path_still(budget):
    return [(0, 0)] * int(budget)


def path_random(budget, seed):
    rng = np.random.default_rng(8_000 + int(seed))
    x = y = 0
    path = []
    for _ in range(int(budget)):
        a = int(rng.integers(0, 4))
        if a == 0:
            x = (x + 1) % P
        elif a == 1:
            x = (x - 1) % P
        elif a == 2:
            y = (y + 1) % P
        else:
            y = (y - 1) % P
        path.append((x, y))
    return path


def path_sweep(budget):
    """Snake through a lattice whose stride is the dot diameter, then repeat."""
    stride = 2 * RADIUS + 1
    xs = list(range(0, P, stride))
    ys = list(range(0, P, stride))
    lattice = []
    for iy, y in enumerate(ys):
        row = xs if iy % 2 == 0 else list(reversed(xs))
        for x in row:
            lattice.append((x, y))
    if len(lattice) < 4:
        raise RuntimeError("sweep lattice is too small")
    path = []
    while len(path) < int(budget):
        path.extend(lattice)
    return path[: int(budget)]


def region_id(px, py):
    side = P // REG
    return (int(py) // REG) * side + (int(px) // REG)


def lawn_step(px, py):
    rx, ry = int(px) // REG, int(py) // REG
    x0, y0 = rx * REG, ry * REG
    nx = int(px) + 1
    ny = int(py)
    if nx >= x0 + REG:
        nx = x0
        ny = int(py) + 1
        if ny >= y0 + REG:
            ny = y0
    return nx % P, ny % P


def step_toward(px, py, tx, ty):
    dx = (int(tx) - int(px) + P // 2) % P - P // 2
    dy = (int(ty) - int(py) + P // 2) % P - P // 2
    if dx == 0 and dy == 0:
        return lawn_step(px, py)
    if abs(dx) >= abs(dy):
        return (int(px) + (1 if dx > 0 else -1)) % P, int(py)
    return int(px), (int(py) + (1 if dy > 0 else -1)) % P


def path_curious(scene, occ, budget, scene_shift, dot_shift):
    """Walk toward the region whose glances are still surprising under the integrator.

    Progress is the prequential bits of the glance just taken (high while the
    holes there still show unknown logo pixels). Unvisited regions stay at the
    prior, log2(G), so the first tour covers regions. This uses the ceiling
    compressor. It is the policy a curious agent would follow if it could integrate.
    """
    side = P // REG
    nreg = side * side
    centers = []
    for ry in range(side):
        for rx in range(side):
            centers.append((rx * REG + REG // 2, ry * REG + REG // 2))
    counts = np.zeros((N, N, G), dtype=np.int32)
    ema = np.full(nreg, math.log2(G))
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
        grey, vis, sx, sy = frame(scene, occ, px, py, scene_shift, dot_shift)
        landed = region_id(px, py)
        if np.any(vis):
            vs = grey[vis].astype(np.int64)
            bits = float(kt_from_counts(counts[sy[vis], sx[vis]], vs).mean())
            ys = sy[vis]
            xs = sx[vis]
            np.add.at(counts, (ys, xs, vs), 1)
        else:
            bits = 0.0
        ema[landed] = (1.0 - BETA) * ema[landed] + BETA * bits
        visits[landed] += 1
    return path


def observed_frames(scene, occ, path, scene_shift, dot_shift):
    """The tape's grey/hole frames. A shift hypothesis does not get to re-render these."""
    frames = []
    for px, py in path:
        grey, vis, _sx, _sy = frame(scene, occ, px, py, scene_shift, dot_shift)
        frames.append((int(px), int(py), grey, vis))
    return frames


def register_bits(frames, hyp_shift):
    """KT bits of the observed greys binned at logo coordinate (i, j) + viewpoint * hyp_shift.

    The true scene shift agrees with itself. A wrong integer piles different greys
    onto one logo pixel. This assumes a rigid shift. It is not the proposed learner.
    """
    counts = np.zeros((N, N, G), dtype=np.int32)
    total = 0.0
    n = 0
    for px, py, grey, vis in frames:
        ii, jj = np.nonzero(vis)
        vs = grey[vis].astype(np.int64)
        sy = (ii + py * int(hyp_shift)) % N
        sx = (jj + px * int(hyp_shift)) % N
        total += float(kt_from_counts(counts[sy, sx], vs).sum())
        n += int(vs.size)
        np.add.at(counts, (sy, sx, vs), 1)
    if n == 0:
        raise RuntimeError("no grey to register")
    return total / n


def mask_conflict(frames, hyp_shift):
    """Fraction of dot-field cells that the tape calls both hole and clear."""
    clear = np.zeros((N, N), dtype=bool)
    hole = np.zeros((N, N), dtype=bool)
    for px, py, _grey, vis in frames:
        ii, jj = np.indices((N, N))
        oy = (ii + py * int(hyp_shift)) % N
        ox = (jj + px * int(hyp_shift)) % N
        clear[oy[vis], ox[vis]] = True
        hole[oy[~vis], ox[~vis]] = True
    return float((clear & hole).mean())


def unique_views(path):
    return len(set(path))


def run_policy(scene, occ, path, probes, scene_shift, dot_shift):
    counts, visits, pool = accumulate(scene, occ, path, scene_shift, dot_shift)
    integ, view, pooled, n_pix = score_probe(
        scene, occ, counts, visits, pool, probes, scene_shift, dot_shift
    )
    return {
        "integ": integ,
        "view": view,
        "pool": pooled,
        "logo": logo_bits(counts, scene),
        "cover": coverage(counts),
        "views": unique_views(path),
        "n_pix": n_pix,
    }


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def fmt(mu, sd):
    return f"{mu:.3f} ± {sd:.3f}"


def check_tape(scene, occ):
    grey, vis, _sx, _sy = frame(scene, occ, 3, 5, SCENE_SHIFT, DOT_SHIFT)
    tape = glance_tape("+x", 3, 5, grey, vis)
    if tape[0] != ACT0 + 1 or tape[1] != PX0 + 3 or tape[2] != PY0 + 5:
        raise RuntimeError("header is not action, px, py")
    # First pixel of row 0 sits after the repeated (px, py).
    pix0 = tape[5]
    expect = int(grey[0, 0]) if vis[0, 0] else HOLE
    if pix0 != expect:
        raise RuntimeError("raster order drifted")
    # Efference repeated on the next row.
    row1 = 3 + (2 + N)
    if tape[row1] != PX0 + 3 or tape[row1 + 1] != PY0 + 5:
        raise RuntimeError("row efference missing")


def spatial_bits(scene, order):
    """One-pass KT codelength of the logo in row-major order. A floor for a spatial cortex."""
    flat = scene.reshape(-1).astype(np.int64)
    tab = {}
    total = 0.0
    ctx = []
    for v in flat:
        key = tuple(ctx[-order:]) if order else ()
        cnt, n = tab.get(key, ({}, 0))
        total += -math.log2((cnt.get(int(v), 0) + ALPHA) / (n + ALPHA * G))
        cnt[int(v)] = cnt.get(int(v), 0) + 1
        tab[key] = (cnt, n + 1)
        ctx.append(int(v))
    return total / flat.size


def policies_for(budget, seed, scene, occ, scene_shift, dot_shift):
    return {
        "still": path_still(budget),
        "random": path_random(budget, seed),
        "sweep": path_sweep(budget),
        "curious": path_curious(scene, occ, budget, scene_shift, dot_shift),
    }


def summarize(rows, key):
    return mean_std([r[key] for r in rows])


def main():
    t0 = time.perf_counter()
    scene = make_scene()
    check_tape(scene, occluder_from(make_field(0), 0.50))
    if ALPHABET != 78:
        raise RuntimeError("alphabet size drifted")
    if glance_tape("+x", 0, 0, scene, np.ones((N, N), dtype=bool)).__len__() != 3 + N * (N + 2):
        raise RuntimeError("glance length drifted")

    s1 = spatial_bits(scene, 1)
    s2 = spatial_bits(scene, 2)
    print(f"logo spatial KT  order1 {s1:.3f}  order2 {s2:.3f}  bits/pixel  (one pass, full logo)")
    print(f"alphabet {ALPHABET}  glance {3 + N * (N + 2)}  greys 0..{G - 1}  hole {HOLE}")
    print(f"shifts scene {SCENE_SHIFT} dot {DOT_SHIFT}  radius {RADIUS}  seeds {SEEDS}  probe {N_PROBE}")
    print()

    # Visibility correlation at density 0.75, seed 0: a step vs a lattice stride.
    occ = occluder_from(make_field(0), 0.75)
    clear = ~occ
    shift1 = np.roll(clear, 1, axis=1)
    stride = 2 * RADIUS + 1
    shift_s = np.roll(clear, stride, axis=1)
    p = float(clear.mean())
    both1 = float((clear & shift1).mean())
    boths = float((clear & shift_s).mean())
    print(
        f"seed0 density 0.75 clear {p:.3f}  "
        f"P(both, shift 1) {both1:.3f}  indep {p * p:.3f}  "
        f"P(both, stride {stride}) {boths:.3f}"
    )
    print()

    header = (
        f"{'T':>4} {'dens':>5} {'policy':>8} {'shift':>7} "
        f"{'cover':>14} {'logo':>14} {'probe_int':>14} {'probe_view':>14} {'probe_pool':>14} {'views':>8}"
    )
    print(header)
    records = []
    for budget in BUDGETS:
        for density in DENSITIES:
            for shift_name, ss, sd in (
                ("parallax", SCENE_SHIFT, DOT_SHIFT),
                ("glued", 1, 1),
            ):
                if shift_name == "glued" and density != 0.75:
                    continue
                by_policy = {name: [] for name in ("still", "random", "sweep", "curious")}
                id_hits = []
                for seed in SEEDS:
                    occ = occluder_from(make_field(seed), density)
                    probes = probe_points(seed)
                    paths = policies_for(budget, seed, scene, occ, ss, sd)
                    for name, path in paths.items():
                        row = run_policy(scene, occ, path, probes, ss, sd)
                        by_policy[name].append(row)
                    if shift_name == "parallax" and density == 0.75:
                        frames = observed_frames(scene, occ, paths["sweep"], ss, sd)
                        reg = [(register_bits(frames, h), h) for h in range(0, 8)]
                        reg.sort()
                        mask = [(mask_conflict(frames, h), h) for h in range(0, 8)]
                        mask.sort()
                        id_hits.append((reg[0], reg[1], mask[0], mask[1]))
                for name in ("still", "random", "sweep", "curious"):
                    rows = by_policy[name]
                    cover = summarize(rows, "cover")
                    logo = summarize(rows, "logo")
                    integ = summarize(rows, "integ")
                    view = summarize(rows, "view")
                    pool = summarize(rows, "pool")
                    views = summarize(rows, "views")
                    records.append({
                        "T": budget,
                        "density": density,
                        "policy": name,
                        "shift": shift_name,
                        "cover": cover,
                        "logo": logo,
                        "integ": integ,
                        "view": view,
                        "pool": pool,
                        "views": views,
                    })
                    print(
                        f"{budget:4d} {density:5.2f} {name:>8} {shift_name:>7} "
                        f"{fmt(*cover):>14} {fmt(*logo):>14} {fmt(*integ):>14} "
                        f"{fmt(*view):>14} {fmt(*pool):>14} {fmt(*views):>8}"
                    )
                if id_hits:
                    parts = []
                    for reg_best, reg_second, mask_best, mask_second in id_hits:
                        parts.append(
                            f"logo-shift {reg_best[1]} bits {reg_best[0]:.3f} "
                            f"margin {reg_second[0] - reg_best[0]:.3f} over {reg_second[1]}; "
                            f"dot-shift {mask_best[1]} conflict {mask_best[0]:.3f} "
                            f"next {mask_second[1]} conflict {mask_second[0]:.3f}"
                        )
                    print("  identifiability on sweep, density 0.75: " + "; ".join(parts))
        print()
    def mean_of(budget, density, policy, shift, key):
        for rec in records:
            if (
                rec["T"] == budget
                and rec["density"] == density
                and rec["policy"] == policy
                and rec["shift"] == shift
            ):
                return rec[key][0]
        raise RuntimeError(f"missing row {budget} {density} {policy} {shift}")

    for density in DENSITIES:
        still = mean_of(25, density, "still", "parallax", "integ")
        walk = mean_of(25, density, "random", "parallax", "integ")
        sweep = mean_of(25, density, "sweep", "parallax", "integ")
        if not (still > walk + 0.05 and walk > sweep + 0.05):
            raise RuntimeError(
                f"T=25 density {density}: integrator probe is not still > random > sweep "
                f"({still:.3f}, {walk:.3f}, {sweep:.3f})"
            )
        view_still = mean_of(25, density, "still", "parallax", "view")
        if view_still < math.log2(G) - 1e-9:
            raise RuntimeError("stand-still view table beat the prior on held-out viewpoints")
    for policy in ("still", "random", "sweep", "curious"):
        cover = mean_of(25, 0.75, policy, "glued", "cover")
        if abs(cover - 0.25) > 1e-9:
            raise RuntimeError(f"glued mask coverage moved with policy {policy}: {cover}")
        logo = mean_of(25, 0.75, policy, "glued", "logo")
        if logo < 2.0:
            raise RuntimeError(f"glued mask logo bits collapsed for {policy}")
    elapsed = time.perf_counter() - t0
    print(f"elapsed_s {elapsed:.2f}")
    if elapsed > 600:
        raise RuntimeError("sketch exceeded 10 minutes")


if __name__ == "__main__":
    main()
