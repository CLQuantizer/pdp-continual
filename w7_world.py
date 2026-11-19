"""W7 world, integrator, and viewpoint probe.

The scene is a fixed 32×32 logo. One dot field per seed covers a set number of
cells. Viewpoint actions move the agent on a 32×32 torus. The logo shifts by
SCENE_SHIFT and the dots by DOT_SHIFT, so the holes slide across the logo.

The integrator is a KT counter in logo coordinates. It reads the agent's
viewpoint and applies the true shifts. The view table memorizes (viewpoint,
raster cell) and does not register motion. The probe asks either learner for
the unoccluded greys at held-out viewpoints. Holes are not scored.

NoiseRegion is off unless a caller builds one. Frames, counts and the probe
then match the deterministic logo bit for bit.
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

N = 32
G = 8
P = 32
REG = 8
ALPHA = 0.5
BETA = 0.5
SCENE_SHIFT = 1
DOT_SHIFT = 2
RADIUS = 3
N_PROBE = 32
RESERVOIR = 256


def box_blur(field, radius):
    acc = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
    return acc


def make_scene():
    """Four 16×16 glyphs, edge to edge. Greys 0..7 all occur."""
    scene = np.zeros((N, N), dtype=np.int16)

    ring = np.full((16, 16), 1, dtype=np.int16)
    yy, xx = np.ogrid[0:16, 0:16]
    ring[(yy - 8) ** 2 + (xx - 8) ** 2 <= 36] = 6
    ring[(yy - 8) ** 2 + (xx - 8) ** 2 <= 9] = 1
    scene[0:16, 0:16] = ring

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


def occluder(acc, n_opaque):
    """Opaque on the highest `n_opaque` cells. Ties break by raster index."""
    target = int(n_opaque)
    if target <= 0 or target >= N * N:
        raise RuntimeError("occluder leaves no hole or no dot")
    order = np.argsort(acc.ravel(), kind="mergesort")
    opaque = np.zeros(N * N, dtype=bool)
    opaque[order[-target:]] = True
    occ = opaque.reshape(N, N)
    if int(occ.sum()) != target:
        raise RuntimeError("occluder missed its count")
    return occ


def maps(px, py, scene_shift, dot_shift):
    ii, jj = np.indices((N, N))
    sy = (ii + int(py) * int(scene_shift)) % N
    sx = (jj + int(px) * int(scene_shift)) % N
    oy = (ii + int(py) * int(dot_shift)) % N
    ox = (jj + int(px) * int(dot_shift)) % N
    return sx, sy, ox, oy


def frame(scene, occ, px, py, scene_shift, dot_shift):
    """Grey, visibility, and logo coordinates of one 32×32 glance."""
    sx, sy, ox, oy = maps(px, py, scene_shift, dot_shift)
    vis = ~occ[oy, ox]
    grey = scene[sy, sx]
    return grey, vis, sx, sy


def kt_from_counts(counts, values):
    """KT bits of each value. counts (n, G), values (n,). Unseen rows are log2(G)."""
    tot = counts.sum(axis=1)
    got = counts[np.arange(len(values)), values]
    return -np.log2((got + ALPHA) / (tot + ALPHA * G))


def logo_bits(counts, scene, skip=None):
    if skip is None:
        flat_c = counts.reshape(-1, G)
        flat_v = scene.reshape(-1).astype(np.int64)
    else:
        flat_c = counts[~skip]
        flat_v = scene[~skip].astype(np.int64)
    return float(kt_from_counts(flat_c, flat_v).mean())


def coverage(counts, skip=None):
    seen = counts.sum(axis=2) > 0
    if skip is None:
        return float(seen.mean())
    return float(seen[~skip].mean())


def probe_points(seed):
    rng = np.random.default_rng(7_000 + int(seed))
    pick = rng.choice(P * P, size=N_PROBE, replace=False)
    return [(int(i % P), int(i // P)) for i in pick]


def accumulate(scene, occ, path, scene_shift, dot_shift):
    """Integrator counts and per-viewpoint visit counts along a path."""
    counts = np.zeros((N, N, G), dtype=np.int32)
    visits = np.zeros((P, P), dtype=np.int32)
    for px, py in path:
        grey, vis, sx, sy = frame(scene, occ, px, py, scene_shift, dot_shift)
        if np.any(vis):
            ys = sy[vis]
            xs = sx[vis]
            vs = grey[vis].astype(np.int64)
            np.add.at(counts, (ys, xs, vs), 1)
        visits[int(py), int(px)] += 1
    return counts, visits


def score_probe(scene, occ, counts, visits, probes, scene_shift, dot_shift, skip=None):
    """Mean KT bits on unoccluded probe pixels. Integrator and view table.

    skip is a logo-coordinate mask. Those pixels are not scored.
    """
    integ = []
    view = []
    for qx, qy in probes:
        grey, vis, sx, sy = frame(scene, occ, qx, qy, scene_shift, dot_shift)
        if skip is not None:
            vis = vis & ~skip[sy, sx]
        if not np.any(vis):
            continue
        vs = grey[vis].astype(np.int64)
        integ.append(kt_from_counts(counts[sy[vis], sx[vis]], vs))
        k = int(visits[qy, qx])
        if k == 0:
            view.append(np.full(vs.size, math.log2(G)))
        else:
            p = (k + ALPHA) / (k + ALPHA * G)
            view.append(np.full(vs.size, -math.log2(p)))
    if not integ:
        raise RuntimeError("probe saw no unoccluded pixel")
    return float(np.concatenate(integ).mean()), float(np.concatenate(view).mean()), int(np.concatenate(integ).size)


def evaluate(scene, occ, path, probes, scene_shift, dot_shift, skip=None):
    counts, visits = accumulate(scene, occ, path, scene_shift, dot_shift)
    probe, view, n_pix = score_probe(
        scene, occ, counts, visits, probes, scene_shift, dot_shift, skip=skip
    )
    return {
        "probe": probe,
        "view": view,
        "logo": logo_bits(counts, scene, skip=skip),
        "cover": coverage(counts, skip=skip),
        "n_pix": n_pix,
    }


def region_mask(region):
    """Boolean logo mask for one 8×8 region. Index is row-major, x within y."""
    region = int(region)
    side = N // REG
    if region < 0 or region >= side * side:
        raise RuntimeError("region index")
    ry, rx = divmod(region, side)
    mask = np.zeros((N, N), dtype=bool)
    mask[ry * REG:(ry + 1) * REG, rx * REG:(rx + 1) * REG] = True
    if int(mask.sum()) != REG * REG:
        raise RuntimeError("region mask")
    return mask


def noise_region_index(seed):
    """One 8×8 scene region. Never the origin's region, index 0.

    Single draw: default_rng(9500 + seed).integers(1, 16), uniform on 1..15.
    """
    rng = np.random.default_rng(9_500 + int(seed))
    region = int(rng.integers(1, 16))
    if region <= 0 or region >= (N // REG) * (N // REG):
        raise RuntimeError("noise region landed on the origin")
    return region


def apply_noise(grey, sy, sx, mask, rng):
    """Fresh uniform greys on the masked logo pixels. One draw per hit cell, row-major."""
    grey = np.array(grey, copy=True)
    hit = mask[sy, sx]
    n = int(hit.sum())
    if n:
        grey[hit] = rng.integers(0, G, size=n)
    return grey


class NoiseRegion:
    """8×8 block of the scene redrawn as uniform greys 0..7 on every glance.

    The region index comes from noise_region_index. Greys use a separate stream,
    default_rng(9600 + seed), so the region draw stays a single integers() call.
    Construct one per policy; each stream starts at the same seed.
    """

    GREY_STREAM = 9_600

    def __init__(self, seed):
        self.seed = int(seed)
        self.region = noise_region_index(self.seed)
        self.mask = region_mask(self.region)
        self.rng = np.random.default_rng(self.GREY_STREAM + self.seed)

    def frame(self, scene, occ, px, py, scene_shift, dot_shift):
        grey, vis, sx, sy = frame(scene, occ, px, py, scene_shift, dot_shift)
        grey = apply_noise(grey, sy, sx, self.mask, self.rng)
        return grey, vis, sx, sy


class GlanceReservoir:
    """Uniform reservoir of glances, cap 256, current glance fed in before the score.

    `default_rng(9000 + seed)` draws a replacement index only after the cap fills.
    Multiplicity is in logo coordinates: how many stored glances showed that pixel.
    """

    def __init__(self, seed, cap=RESERVOIR):
        self.cap = int(cap)
        self.rng = np.random.default_rng(9_000 + int(seed))
        self.n_seen = 0
        self.slots = []
        self.mult = np.zeros((N, N), dtype=np.int32)

    def include(self, ys, xs):
        ys = np.asarray(ys, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        self.n_seen += 1
        if self.n_seen <= self.cap:
            self.slots.append((ys, xs))
            if ys.size:
                np.add.at(self.mult, (ys, xs), 1)
            return
        j = int(self.rng.integers(0, self.n_seen))
        if j < self.cap:
            oys, oxs = self.slots[j]
            if oys.size:
                np.add.at(self.mult, (oys, oxs), -1)
            self.slots[j] = (ys, xs)
            if ys.size:
                np.add.at(self.mult, (ys, xs), 1)

    def progress(self, counts, ys, xs, vs):
        """Codelength of the reservoir before this glance is learned, minus after.

        Only pixels this glance shows change. Every stored sight of such a pixel,
        including this glance, is scored at the same old and new KT probability.
        """
        if len(vs) == 0:
            return 0.0
        c = counts[ys, xs, vs].astype(np.float64)
        t = counts[ys, xs].sum(axis=1).astype(np.float64)
        m = self.mult[ys, xs].astype(np.float64)
        before = -np.log2((c + ALPHA) / (t + ALPHA * G))
        after = -np.log2((c + 1.0 + ALPHA) / (t + 1.0 + ALPHA * G))
        return float(np.sum(m * (before - after)))


class ObservedReservoir(GlanceReservoir):
    """Same reservoir, but each sight keeps its grey and progress is eq. 5.

    On a deterministic pixel every stored grey matches the current one, so the
    sum equals GlanceReservoir.progress. A pixel whose greys disagree, the
    noise patch, is scored at the greys actually stored.
    """

    def include(self, ys, xs, vs):
        ys = np.asarray(ys, dtype=np.int64).copy()
        xs = np.asarray(xs, dtype=np.int64).copy()
        vs = np.asarray(vs, dtype=np.int64).copy()
        if vs.shape != ys.shape or xs.shape != ys.shape:
            raise RuntimeError("reservoir sight shape")
        self.n_seen += 1
        if self.n_seen <= self.cap:
            self.slots.append((ys, xs, vs))
            if ys.size:
                np.add.at(self.mult, (ys, xs), 1)
            return
        j = int(self.rng.integers(0, self.n_seen))
        if j < self.cap:
            oys, oxs, _ovs = self.slots[j]
            if oys.size:
                np.add.at(self.mult, (oys, oxs), -1)
            self.slots[j] = (ys, xs, vs)
            if ys.size:
                np.add.at(self.mult, (ys, xs), 1)

    def progress(self, counts, ys, xs, vs, mask=None):
        """Codelength of stored sights before this glance is learned, minus after.

        mask on logo coordinates splits the sum into (total, noise, logo).
        Without it, the return is the total only.
        """
        empty = (0.0, 0.0, 0.0) if mask is not None else 0.0
        if len(vs) == 0:
            return empty
        parts_k = []
        parts_v = []
        for sy, sx, sv in self.slots:
            if sy.size == 0:
                continue
            parts_k.append(sy * N + sx)
            parts_v.append(sv)
        if not parts_k:
            return empty
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
            return empty
        ak = ak[ok]
        av = av[ok]
        vn = v_s[inb[ok]]
        yy = ak // N
        xx = ak % N
        c_rows = counts[yy, xx]
        t = c_rows.sum(axis=1).astype(np.float64)
        c = c_rows[np.arange(av.size), av].astype(np.float64)
        c_after = c + (av == vn).astype(np.float64)
        before = -np.log2((c + ALPHA) / (t + ALPHA * G))
        after = -np.log2((c_after + ALPHA) / (t + 1.0 + ALPHA * G))
        delta = before - after
        total = float(np.sum(delta))
        if mask is None:
            return total
        noise = mask[yy, xx]
        return total, float(np.sum(delta[noise])), float(np.sum(delta[~noise]))


def self_check():
    scene = make_scene()
    if scene.shape != (N, N):
        raise RuntimeError("scene shape")
    if set(int(v) for v in np.unique(scene)) != set(range(G)):
        raise RuntimeError("scene greys")
    field = make_field(0)
    for n_opaque in (512, 768, 922):
        occ = occluder(field, n_opaque)
        if int(occ.sum()) != n_opaque:
            raise RuntimeError("opaque count")
    for px, py, ss, sd in ((0, 0, 1, 2), (3, 5, 1, 2), (31, 17, 1, 1)):
        sx, sy, ox, oy = maps(px, py, ss, sd)
        if len(set(zip(sy.ravel().tolist(), sx.ravel().tolist()))) != N * N:
            raise RuntimeError("logo map is not bijective")
        if len(set(zip(oy.ravel().tolist(), ox.ravel().tolist()))) != N * N:
            raise RuntimeError("dot map is not bijective")
    # One new pixel, current glance included: log2(8) - (-log2(1.5/5)).
    counts = np.zeros((N, N, G), dtype=np.int32)
    res = GlanceReservoir(0)
    res.include(np.array([4]), np.array([5]))
    got = res.progress(counts, np.array([4]), np.array([5]), np.array([0]))
    expect = -math.log2(0.5 / 4.0) - (-math.log2(1.5 / 5.0))
    if abs(got - expect) > 1e-12:
        raise RuntimeError(f"progress identity {got} != {expect}")
    # Second sight of the same pixel: both reservoir copies move from count 1 to 2.
    np.add.at(counts, (np.array([4]), np.array([5]), np.array([0])), 1)
    res.include(np.array([4]), np.array([5]))
    got2 = res.progress(counts, np.array([4]), np.array([5]), np.array([0]))
    b1 = -math.log2(1.5 / 5.0)
    b2 = -math.log2(2.5 / 6.0)
    if abs(got2 - 2.0 * (b1 - b2)) > 1e-12:
        raise RuntimeError("second-sight progress")
    ema = np.full(4, np.inf)
    visits = np.zeros(4, dtype=np.int32)
    order = np.lexsort((np.arange(4), visits, -ema))
    if int(order[0]) != 0:
        raise RuntimeError("unvisited tie-break")
    print("w7_world self_check ok")


if __name__ == "__main__":
    self_check()
