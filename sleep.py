"""Wake/sleep forward-forward on Split-MNIST.

WAKE   The stream arrives one example at a time. The FF network learns from
       it as usual (positive + hard negative), and a small, bounded fast
       store (the "hippocampus") records it.
SLEEP  Every `sleep_every` wake examples, input switches off. The network
       trains only on what the fast store generates:
         - replay: samples from every class seen so far, class-balanced, as
           positives (this is what interleaves old with new);
         - generated negatives: the same replayed samples paired with the
           wrong label the network itself finds most plausible;
         - synaptic downscaling: weak weights shrink to exactly 0;
         - homeostasis runs on replayed data too, reviving silent units.
       Sleep is triggered by a clock, not by task boundaries, so the
       learner never knows when the task changes.

Fast stores, both bounded and both reported in floats stored:
  proto      K prototypes per class learned online by competitive learning
             (nearest prototype moves toward the input), plus a per-pixel
             variance. Replay = prototype + Gaussian noise. Stores 2*784
             floats per prototype: a compressed memory, no raw images.
  reservoir  M raw images per class, kept by reservoir sampling. The usual
             "experience replay" baseline. Stores 784 floats per image.
  K prototypes cost the same as M = 2K raw images.
  pixels     no store: an unlabelled running mean/variance of every pixel,
             where the network's own dreams start from (dreams.py).

Usage
  python sleep.py                         # all conditions, seed 0
  python sleep.py --seeds 0 1 2 --conditions off proto10 reservoir20
"""
import argparse
import json
import os
import time

import numpy as np

import learners as L
from bench import TASKS, accuracy, task_il_accuracy
from data import load_mnist

D = 784


class ProtoStore:
    def __init__(self, k, rng, noise=1.0, dim=D, clip=(0, 1), surprise=0.0):
        self.k, self.rng, self.noise = k, rng, noise
        self.surprise = surprise  # step for an example the network got wrong (mismatch-gated plasticity)
        self.dim, self.clip = dim, clip  # clip=(0, None) for ReLU codes
        self.mu = {}    # class -> (n_protos, 784)
        self.var = {}   # class -> (n_protos, 784)
        self.n = {}     # class -> (n_protos,)

    def add(self, x, y, surprise=False):
        x = x[0]
        if y not in self.mu:
            self.mu[y], self.var[y], self.n[y] = x[None].copy(), np.full((1, self.dim), 0.01, np.float32), np.ones(1)
            return
        mu, var, n = self.mu[y], self.var[y], self.n[y]
        if len(mu) < self.k:
            self.mu[y] = np.vstack([mu, x])
            self.var[y] = np.vstack([var, np.full(self.dim, 0.01, np.float32)])
            self.n[y] = np.append(n, 1.0)
            return
        j = np.argmin(((mu - x) ** 2).sum(1))
        n[j] += 1
        lr = max(1.0 / n[j], 0.01)  # running mean, with a floor so old protos still track drift
        if surprise:
            lr = max(lr, self.surprise)
        d = x - mu[j]
        mu[j] += lr * d
        var[j] += lr * (d * d - var[j])

    def sample(self, m):
        classes = np.array(sorted(self.mu))
        ys = self.rng.choice(classes, m)
        xs = np.empty((m, self.dim), np.float32)
        for c in np.unique(ys):
            idx = np.flatnonzero(ys == c)
            xs[idx] = self.sample_class(c, len(idx))
        return xs, ys

    def has(self, c):
        return c in self.mu

    def sample_class(self, c, m):
        p = self.n[c] / self.n[c].sum()
        j = self.rng.choice(len(p), m, p=p)
        eps = self.rng.standard_normal((m, self.dim)).astype(np.float32)
        return np.clip(self.mu[c][j] + self.noise * np.sqrt(self.var[c][j]) * eps, *self.clip)

    def floats(self):
        # mean + variance per prototype; without replay noise the variance is never used
        return sum((2 if self.noise else 1) * m.size for m in self.mu.values())


class ReservoirStore:
    """k items per class by reservoir sampling. bits: store each item at that
    many bits, scaled by its own max (one float) with stochastic rounding."""

    def __init__(self, m, rng, bits=None, surprise=0.0):
        self.m, self.rng, self.bits = m, rng, bits
        self.surprise = surprise  # prob. that an example the network got wrong overwrites a random slot
        self.buf, self.seen = {}, {}
        if bits:
            self.qrng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])

    def _quant(self, x):
        L, s = 2 ** self.bits - 1, max(float(x.max()), 1e-12)
        return (np.floor(x / s * L + self.qrng.random(x.shape)) * (s / L)).astype(x.dtype)

    def add(self, x, y, surprise=False):
        b = self.buf.setdefault(y, [])
        self.seen[y] = self.seen.get(y, 0) + 1
        if self.bits:
            x = self._quant(x)
        if len(b) < self.m:
            b.append(x[0].copy())
        elif surprise and self.rng.random() < self.surprise:
            b[self.rng.integers(self.m)] = x[0].copy()
        else:
            j = self.rng.integers(self.seen[y])
            if j < self.m:
                b[j] = x[0].copy()

    def sample(self, m):
        classes = np.array(sorted(self.buf))
        ys = self.rng.choice(classes, m)
        xs = np.stack([self.buf[c][self.rng.integers(len(self.buf[c]))] for c in ys])
        return xs, ys

    def has(self, c):
        return c in self.buf

    def sample_class(self, c, m):
        return np.stack([self.buf[c][j] for j in self.rng.integers(len(self.buf[c]), size=m)])

    def floats(self):
        per = b0 = next(iter(self.buf.values()))[0].size if self.buf else 0
        if self.bits:
            per = b0 * self.bits / 32 + 1
        return sum(len(b) * per for b in self.buf.values())


class PPCAStore:
    """Per class: running mean + a rank-r principal subspace learned online
    with Sanger's rule (local, Hebbian), its variances, and a residual
    variance. Samples z = mu + U (sqrt(lam) * e1) + sqrt(resid) * e2, so the
    samples' second moment matches the data's along the top r directions
    (a probabilistic-PCA prior) instead of being diagonal."""

    def __init__(self, r, rng, dim):
        self.r, self.rng, self.dim = r, rng, dim
        self.mu, self.U, self.lam, self.res, self.n, self.d2 = {}, {}, {}, {}, {}, {}

    def has(self, c):
        return c in self.mu

    def add(self, x, y):
        x = x[0].astype(np.float64)
        if y not in self.mu:
            q, _ = np.linalg.qr(self.rng.standard_normal((self.dim, self.r)))
            self.mu[y], self.U[y], self.n[y] = x.copy(), q, 0
            self.lam[y], self.res[y], self.d2[y] = np.full(self.r, 1e-2), 1e-2, 1.0
            return
        self.n[y] += 1
        rate = max(1.0 / self.n[y], 1e-3)
        self.mu[y] += rate * (x - self.mu[y])
        d = x - self.mu[y]
        U = self.U[y]
        p = U.T @ d
        self.d2[y] += rate * (d @ d - self.d2[y])
        eta = min(max(1.0 / self.n[y], 5e-3), 0.1) / (self.d2[y] + 1e-3)
        U += eta * (np.outer(d, p) - U @ np.triu(np.outer(p, p)))  # Sanger: U[:, i] learns the i-th component
        self.lam[y] += rate * (p * p - self.lam[y])
        e = d - U @ p
        self.res[y] += rate * (e @ e / self.dim - self.res[y])

    def sample_class(self, c, m):
        e1 = self.rng.standard_normal((m, self.r)) * np.sqrt(self.lam[c])
        e2 = self.rng.standard_normal((m, self.dim)) * np.sqrt(max(self.res[c], 0))
        return np.maximum(self.mu[c] + e1 @ self.U[c].T + e2, 0).astype(np.float32)

    def sample(self, m):
        classes = np.array(sorted(self.mu))
        ys = self.rng.choice(classes, m)
        xs = np.empty((m, self.dim), np.float32)
        for c in np.unique(ys):
            idx = np.flatnonzero(ys == c)
            xs[idx] = self.sample_class(c, len(idx))
        return xs, ys

    def floats(self):
        return sum(self.dim * (self.r + 1) + self.r + 1 for _ in self.mu)


class PixelStats:
    """Unlabelled running mean and variance of every pixel (2*784 floats).
    The only memory a dreaming network keeps: where dreams start from and
    how far each pixel may move. Holds nothing about any class."""

    def __init__(self, rng, recent=0):
        self.rng, self.n = rng, 0
        self.mu, self.var = np.zeros(D, np.float32), np.zeros(D, np.float32)
        self.recent = np.zeros((recent, D), np.float32)  # optional "day residue": last few inputs, unlabelled

    def add(self, x, y):
        if len(self.recent):
            self.recent[self.n % len(self.recent)] = x[0]
        self.n += 1
        d = x[0] - self.mu
        self.mu += d / self.n
        self.var += (d * (x[0] - self.mu) - self.var) / self.n

    def start(self, m):
        """Where dreams start: recent inputs if kept, else pixel noise."""
        if len(self.recent):
            return self.recent[self.rng.integers(min(self.n, len(self.recent)), size=m)].copy()
        return self.noise(m)

    def noise(self, m):
        sd = np.sqrt(self.var)
        return np.clip(self.mu + sd * self.rng.standard_normal((m, D)), 0, 1).astype(np.float32)

    def floats(self):
        return 2 * D + self.recent.size


class TopDown:
    """Helmholtz-style generative path, learned locally while awake.

    Each FF layer l gets top-down weights G_l that predict the activity of
    the layer below (the image, for layer 1) from its own activity. Awake,
    after every input, each G_l takes one normalised delta-rule step on its
    own prediction error: purely local, nothing crosses more than one layer
    and no gradient flows through the recognition weights. The top layer
    keeps a small class-conditional prior over its own codes (k code
    prototypes per class, mean + variance, in code space).

    To dream class c: sample a top code from c's prior, then run it down
    through G_top ... G_1 to an image. No image is ever stored.

    Rules for G (all local to one pair of layers):
      nlms   normalised delta rule, fixed step: tracks recent data, so new
             tasks overwrite the decoding of old codes.
      count  delta rule with a per-unit step 1/(total activity of that unit
             so far): each unit's top-down weights become a running average
             over everything it ever fired for. Accumulates, barely forgets.
      ridge  running sums of h h^T and h x^T; G is the least-squares solve,
             done at the start of each sleep. Order-independent.
      outstar  Grossberg's outstar: unit j's weights are the running
             activity-weighted mean of what lay below whenever j fired,
             G_j += (h_j / N_j)(x - G_j). Each unit uses its own error, not
             the shared one, so units can't overwrite each other. Decoding
             is the activity-weighted average sum_j a_j G_j / sum_j a_j with
             a = h^p. No solve; extra memory: N (one count per unit).
      contrastive  wake-sleep Hebbian/anti-Hebbian. Awake, each synapse
             accumulates its own Hebbian sum C_ij += h_i x_j. Asleep, codes
             are sampled from the prior and decoded, x^ = h G, and each
             synapse moves by its own pre * post terms:
             G_ij += eta (C_ij / n - mean(h_i x^_j)). The fixed point is the
             least-squares decoder under the prior's code statistics, reached
             without h h^T or a solve. Extra memory: C (same size as G).
      fold   contrastive with C folded into G: one number per synapse. Awake,
             each synapse adds its Hebbian term straight into its weight,
             G_ij += r_t h_i x_j with r_t = kappa / (t E|h|^2) (1/t: every
             example weighs about the same, as in a running mean). Asleep,
             the anti-Hebbian term owed for that wake is paid back in one
             burst on decoded prior codes, G_ij -= eta mean(h_i x^_j), with
             total weight sum r_t. Same fixed point as contrastive when the
             stream is stationary; no extra memory, only two scalars.

    A cheap C for the contrastive rule (cbits and/or topk): write
    C/n = diag(N/n) F, where N_i is unit i's total activity and F_ij the
    activity-weighted mean of x_j whenever i fired (Grossberg's outstar
    weights). F lies in [0, 1] whatever n is, so it quantises without a
    scale. Awake, F_i += (h_i / N_i)(x - F_i), then:
      cbits  F is stored in cbits bits, with stochastic rounding (unbiased
             even when the step is far below one level).
      topk   each row keeps only its k largest entries; a new pixel enters
             only by beating the row's smallest stored value.
    Memory: F at cbits (or 32) bits per stored value, a 16-bit index per
    value when sparse, plus N (one float per unit).

    Priors over top codes (what dreams start from; for contrastive, also
    what the negative phase samples): "proto" = k code prototypes/class with
    diagonal variance; "reservoir" = k real codes/class kept by reservoir
    sampling; "ppca" = class mean + rank-k subspace (Sanger's rule). `neg`
    optionally gives the negative phase its own prior.
    """

    def __init__(self, m, levels, k, rng, mu=0.05, rule="nlms", lam=1e-2, p=2, iters=100,
                 prior="proto", neg=None, kappa=1.0, cbits=None, topk=None, solve=None, pbits=None, decay=0.0,
                 noise=0.0, step=0.2, center=False, jacobi=False, keep=None, polyak=False, expand=0, expand_k=0, sharpen=0.0,
                 surprise=0.0):
        self.m, self.levels, self.mu, self.rule, self.lam = m, levels, mu, rule, lam
        self.kappa, self.owed, self.h2 = kappa, 0.0, 0.0  # fold: rate, unpaid anti-Hebbian weight, mean |hb|^2
        self.p, self.iters, self.n = p, iters, 0
        H = m.W[0].shape[1]
        sizes = [D] + [H] * levels
        # expand: the image decoder reads [code, expand fixed random features of the code] (see _phi)
        assert not expand or levels == 1
        self.expand, self.expand_k = expand, expand_k
        self.sharpen = sharpen  # dreams only: centre-surround (lateral inhibition) gain, see decode
        ins = [H + expand] + sizes[2:]  # what each level's decoder reads
        self.G = [np.zeros((b, a), np.float32) for a, b in zip(sizes[:-1], ins)]  # G_l: layer l -> l-1
        self.c = [np.zeros(a, np.float32) for a in sizes[:-1]]
        self.N = [np.zeros(b, np.float32) for b in ins]  # per-unit total activity ("count")
        if rule == "ridge":
            self.A = [np.zeros((b + 1, b + 1)) for b in ins]
        self.cbits, self.topk, self.solve = cbits, topk, solve  # solve="prior": diagnostic exact fixed point
        # contrastive sleep: eta E|hb|^2 = step; decay / noise act like ridge's lambda (relative to the
        # mean eigenvalue E|hb|^2 / (H+1)): decay shrinks each synapse, noise jitters the negative codes
        self.decay, self.noise, self.step = decay, noise, step
        # center: each presynaptic unit uses z_i - (its mean wake activity) (covariance rule), bias unpenalised.
        # jacobi: per-presynaptic-unit step 1/E[u_i^2 + lam], times one global gain (1/largest eigenvalue of
        #         the normalised statistics, estimated once per sleep). keep: dropout on negative codes (each
        #         unit kept with prob keep, rescaled). polyak: average the weights over the second half of a sleep.
        self.center, self.jacobi, self.keep, self.polyak = center, jacobi, keep, polyak
        self.cheap = rule == "contrastive" and (cbits or topk)
        if self.cheap:
            assert levels == 1 or not (center or jacobi or topk)
            self.Fl = [np.zeros((b + 1, a), np.float32) for a, b in zip(sizes[:-1], ins)]
            self.NFl = [np.zeros(b + 1) for b in ins]
            # value scale of F: pixels lie in [0, 1]; a hidden layer uses its running max activity
            self.Fs = [1.0] + [1e-12] * (levels - 1)
            self.F, self.NF = self.Fl[0], self.NFl[0]
            self.qrng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])  # leaves rng's stream alone
        elif rule in ("ridge", "contrastive"):
            self.C = [np.zeros((b + 1, a)) for a, b in zip(sizes[:-1], ins)]
        if expand:  # fixed random fan-out, drawn after qrng so the other streams are unchanged
            self.R = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]).standard_normal(
                (H, expand)).astype(np.float32) / np.sqrt(H)
        make = lambda kind, kk: (ProtoStore(kk, rng, dim=H, clip=(0, None)) if kind == "proto"
                                 else ReservoirStore(kk, rng, pbits, surprise) if kind == "reservoir" else PPCAStore(kk, rng, H))
        self.prior = make(prior, k)
        self.neg = make(*neg) if neg else self.prior
        self.err = 1.0  # running reconstruction error of the image (diagnostic)

    def add(self, x, y, surprise=False):
        acts = [x] + [h for _, h in self.m._layers(self.m.embed(x, np.array([y])))][:self.levels]
        self.n += 1
        for l, (G, c) in enumerate(zip(self.G, self.c)):
            below, h = acts[l], acts[l + 1]
            pred = self._down(l, h)
            if l == 0 and self.expand:
                h = self._phi(h)
            e = below - pred
            if l == 0:
                e[:, :10] = 0  # label pixels aren't part of the image
                self.err = 0.999 * self.err + 0.001 * float((e * e).mean())
            if self.rule == "nlms":
                G += (self.mu / (float((h * h).sum()) + 1.0)) * h.T @ e
                c += self.mu * 0.1 * e[0]
            elif self.rule == "outstar":
                self.N[l] += h[0]
                r = h[0] / np.maximum(self.N[l], 1e-8)
                G += r[:, None] * (below[0] - G)  # each unit: running mean of what it fired for
            elif self.cheap:
                self._accumulate(np.append(h[0], 1.0), below[0], l)
            elif self.rule == "contrastive":
                self.C[l] += np.outer(np.append(h[0], 1.0), below[0])
            elif self.rule == "fold":
                self.h2 += (float((h * h).sum()) + 1.0 - self.h2) / self.n
                r = self.kappa / (self.n * self.h2)
                G += r * np.outer(h[0], below[0]).astype(np.float32)
                c += r * below[0]
                self.owed += r
            elif self.rule == "count":
                self.N[l] += h[0]
                r = h[0] / (self.N[l] + 1.0)  # per-unit step: 1 / (activity so far)
                G += (r / max(1.0, float(r @ h[0])))[:, None] * e  # cap so the steps together never overshoot
                c += 0.01 * e[0]
            else:
                hb = np.append(h[0], 1.0)
                self.A[l] += np.outer(hb, hb)
                self.C[l] += np.outer(hb, below[0])
        kw = dict(surprise=surprise) if surprise else {}
        self.prior.add(acts[-1], y, **kw)
        if self.neg is not self.prior:
            self.neg.add(acts[-1], y, **kw)

    def _accumulate(self, hb, x, l=0):
        """Cheap C: outstar update of F, then quantise and/or keep top-k per row."""
        on = hb > 0
        NF, Fl = self.NFl[l], self.Fl[l]
        NF += hb
        if l:
            self.Fs[l] = max(self.Fs[l], float(x.max()))
        s = self.Fs[l]
        r = (hb[on] / NF[on]).astype(np.float32)[:, None]
        F = Fl[on] + r * (x - Fl[on])
        if self.cbits:
            L = 2 ** self.cbits - 1
            F = np.floor(F / s * L + self.qrng.random(F.shape, np.float32)) / L * s  # s = 1: same bits as before
        if self.topk:
            drop = np.argpartition(F, -self.topk, axis=1)[:, :-self.topk]
            np.put_along_axis(F, drop, 0.0, axis=1)
        Fl[on] = np.clip(F, 0, s)

    def _target(self, l=0):
        """C / n, the mean Hebbian term each synapse is pulled toward in sleep."""
        if self.cheap:
            return (self.NFl[l] / self.n)[:, None] * self.Fl[l]
        return self.C[l] / self.n

    def _down(self, l, h):
        """Top-down prediction of layer l-1 (l=0: the image) from layer l."""
        if l == 0 and self.expand:
            h = self._phi(h)
        if self.rule == "outstar":
            a = h ** self.p
            out = a @ self.G[l] / (a.sum(1, keepdims=True) + 1e-8)
        else:
            out = h @ self.G[l] + self.c[l]
        return np.maximum(out, 0) if l > 0 else np.clip(out, 0, 1)

    def consolidate(self):
        """In sleep. ridge: solve G from the sums. contrastive: anti-Hebbian
        steps on the prior's own decoded codes, toward the Hebbian sums."""
        if self.rule == "contrastive" and self.n:
            if self.levels == 2:
                # top level first, on stored top codes; then layer 1 -> image on the layer-1 codes the top
                # level now dreams (wake-sleep: the lower decoder learns on the statistics it will decode)
                assert self.keep or self.polyak
                draw = lambda k: self.neg.sample(k)[0]
                self._sleep_steps(self.G[1], self.c[1], self._target(1), draw)
                return self._sleep_steps(self.G[0], self.c[0], self._target(0), lambda k: self._down(1, draw(k)))
            G, c, target = self.G[0], self.c[0], self._target()
            if self.solve == "prior":
                # not local: the exact point the sleep steps head for, using every stored code
                Z = np.vstack([np.array(b) for b in self.neg.buf.values()])
                Zb = np.hstack([Z, np.ones((len(Z), 1))])
                S = Zb.T @ Zb / len(Z)
                Gb = np.linalg.solve(S + self.lam * np.trace(S) / len(S) * np.eye(len(S)), target)
                G[:], c[:] = Gb[:-1], Gb[-1]
                return
            if self.center or self.jacobi or self.keep or self.polyak:
                return self._sleep_steps(G, c, target, (lambda k: self._phi(self.neg.sample(k)[0])) if self.expand else None)
            for _ in range(self.iters):
                z, _ = self.neg.sample(64)
                hb = np.hstack([z, np.ones((len(z), 1), np.float32)])
                h2 = float((hb * hb).sum(1).mean())
                lam = h2 / hb.shape[1]  # mean eigenvalue of the code second moment
                if self.noise:
                    z = z + np.sqrt(self.noise * lam) * self.qrng.standard_normal(z.shape).astype(np.float32)
                    hb = np.hstack([z, np.ones((len(z), 1), np.float32)])
                xh = z @ G + c  # linear dream (unclipped) for the update
                eta = self.step / h2  # default 0.2: a fifth of the stable step
                dG = target - hb.T @ xh / len(z)
                if self.decay:
                    dG -= self.decay * lam * np.vstack([G, c[None]])
                G += eta * dG[:-1].astype(np.float32)
                c += eta * dG[-1].astype(np.float32)
            return
        if self.rule == "fold" and self.owed:
            assert self.levels == 1
            G, c = self.G[0], self.c[0]
            K = max(self.iters, int(np.ceil(self.owed * self.h2 / 0.2)))  # keep eta E|hb|^2 <= 0.2
            eta = self.owed / K
            for _ in range(K):
                z, _ = self.neg.sample(64)
                xh = z @ G + c
                G -= (eta / len(z)) * (z.T @ xh)
                c -= eta * xh.mean(0)
            self.owed = 0.0
            return
        if self.rule != "ridge":
            return
        for l in range(self.levels):
            A = self.A[l]
            Gb = np.linalg.solve(A + self.lam * np.trace(A) / len(A) * np.eye(len(A)), self.C[l])
            self.G[l][:] = Gb[:-1]
            self.c[l][:] = Gb[-1]

    def _sleep_steps(self, G, c, target, draw=None):
        """Contrastive sleep with centring, Jacobi steps, dropout negatives and/or
        Polyak averaging. Same update as consolidate's loop otherwise. draw(k):
        k negative codes for this level (default: the prior's top codes)."""
        draw = draw or (lambda k: self.neg.sample(k)[0])
        m = (self.NF[:-1] / self.n).astype(np.float32) if self.center else np.zeros(G.shape[0], np.float32)
        T = np.vstack([target[:-1] - np.outer(m, target[-1]), target[-1]])  # E[u x^T], E[x]
        cc = c + m @ G  # same decoder, centred bias
        z = draw(1024)
        ub = np.hstack([z - m, np.ones((len(z), 1), np.float32)])
        S = ub.T @ ub / len(ub)
        lam = float(np.trace(S)) / len(S)
        pen = np.full(len(S), self.decay * lam, np.float32)
        if self.center:
            pen[-1] = 0.0
        if self.jacobi:
            dg = np.diag(S) + pen + 1e-6 * lam
            Sn = (S + np.diag(pen)) / np.sqrt(np.outer(dg, dg))
            v = np.ones(len(S))
            for _ in range(50):  # power iteration: the one global gain
                v = Sn @ v
                v /= np.linalg.norm(v)
            eta = (self.step / float(v @ Sn @ v) / dg).astype(np.float32)[:, None]
        else:
            eta = np.float32(self.step / float(np.trace(S)))
        start, Ga, ca, na = self.iters // 2, np.zeros_like(G), np.zeros_like(c), 0
        for k in range(self.iters):
            z = draw(64)
            if self.keep:
                z = z * (self.qrng.random(z.shape, np.float32) < self.keep) / self.keep
            u = z - m
            xh = u @ G + cc
            dG = T[:-1] - u.T @ xh / len(u) - pen[:-1, None] * G
            dc = T[-1] - xh.mean(0) - pen[-1] * cc
            G += (eta[:-1] if self.jacobi else eta) * dG
            cc += (eta[-1] if self.jacobi else eta) * dc
            if self.polyak and k >= start:
                Ga += G; ca += cc; na += 1
        if self.polyak:
            G[:], cc = Ga / na, ca / na
        c[:] = cc - m @ G

    def _phi(self, h):
        """[code, sparse random features of the code]: a fixed fan-out to expand units (granule cells),
        k-winners-take-all (expand_k active, threshold at the next one: Golgi inhibition), then divisive
        normalisation so the features carry the same total activity as the code."""
        a = h @ self.R
        th = -np.partition(-a, self.expand_k, axis=1)[:, self.expand_k:self.expand_k + 1]
        v = np.maximum(a - th, 0)
        v *= np.linalg.norm(h, axis=1, keepdims=True) / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return np.hstack([h, v]).astype(np.float32)

    def decode(self, z):
        h = z
        for l in reversed(range(self.levels)):
            h = self._down(l, h)
        if self.sharpen:
            # centre-surround: each pixel minus the mean of its 3x3 neighbourhood, added back with gain
            # sharpen (an unsharp mask; retinal / LGN lateral inhibition). Edges and gaps get their contrast back.
            img = h.reshape(-1, 28, 28)
            pad = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode="edge")
            blur = sum(pad[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)) / 9
            h = np.clip(img + self.sharpen * (img - blur), 0, 1).reshape(len(h), -1)
        return h.astype(np.float32)

    def extra_floats(self):
        """Learning memory beyond the weights G themselves."""
        return {"nlms": 0, "count": sum(n.size for n in self.N), "outstar": sum(n.size for n in self.N),
                "ridge": sum(a.size for a in getattr(self, "A", [])) + sum(c.size for c in getattr(self, "C", [])),
                "contrastive": self._c_floats(), "fold": 2}[self.rule]

    def _c_floats(self):
        if not self.cheap:
            return sum(c.size for c in self.C)
        bits = (self.cbits or 32) + (16 if self.topk else 0)
        return sum(F.shape[0] * (self.topk or F.shape[1]) * bits / 32 + NF.size + (l > 0)  # + a hidden layer's scale
                   for l, (F, NF) in enumerate(zip(self.Fl, self.NFl)))

    def sample(self, m):
        z, ys = self.prior.sample(m)
        return self.decode(z), ys

    def floats(self):
        # the generator weights are part of the network, reported separately
        return self.prior.floats() + (self.neg.floats() if self.neg is not self.prior else 0)


# name -> settings. store=None means no sleep at all.
# replay: "store" = positives sampled from the store (prototype + noise or raw
#         images); "dream" = inputs the network generates itself (FF.dream),
#         starting from unlabelled pixel noise. The set of labels seen so far
#         is known (10 bits).
# split:  Hinton's split. Wake trains positives only, sleep negatives only:
#         "wrong" = replayed/dreamed input with the most plausible wrong label;
#         "own" = a dream with the very label it was dreamed for (the
#         network's own generations are the negatives).
CONDITIONS = {
    "off":          dict(store=None),
    "proto10":      dict(store="proto", size=10, downscale=0.05),
    "proto10_nods": dict(store="proto", size=10, downscale=0.0),
    "proto1":       dict(store="proto", size=1, downscale=0.05),
    "proto50":      dict(store="proto", size=50, downscale=0.05),
    "reservoir20":  dict(store="reservoir", size=20, downscale=0.05),
    "reservoir100": dict(store="reservoir", size=100, downscale=0.05),
    "dream":             dict(store="pixels", replay="dream", downscale=0.0),
    "split_proto":       dict(store="proto", size=10, split="wrong", downscale=0.0),
    "split_dream":       dict(store="pixels", replay="dream", split="wrong", downscale=0.0),
    "split_dream_own":   dict(store="pixels", replay="dream", split="own", downscale=0.0),
    "proto1_nods":       dict(store="proto", size=1, downscale=0.0),
    "gen1_k1":           dict(store="gen", levels=1, size=1, downscale=0.0),
    "gen1_k10":          dict(store="gen", levels=1, size=10, downscale=0.0),
    "gen2_k1":           dict(store="gen", levels=2, size=1, downscale=0.0),
    "gen2_k10":          dict(store="gen", levels=2, size=10, downscale=0.0),
    "gen1_k10_count":    dict(store="gen", levels=1, size=10, rule="count", downscale=0.0),
    "gen1_k10_ridge":    dict(store="gen", levels=1, size=10, rule="ridge", downscale=0.0),
    "gen2_k10_count":    dict(store="gen", levels=2, size=10, rule="count", downscale=0.0),
    "gen2_k10_ridge":    dict(store="gen", levels=2, size=10, rule="ridge", downscale=0.0),
    "gen1_k1_ridge":     dict(store="gen", levels=1, size=1, rule="ridge", downscale=0.0),
    "gen1_k10_outstar1": dict(store="gen", levels=1, size=10, rule="outstar", p=1, downscale=0.0),
    "gen1_k10_outstar2": dict(store="gen", levels=1, size=10, rule="outstar", p=2, downscale=0.0),
    "gen1_k10_outstar4": dict(store="gen", levels=1, size=10, rule="outstar", p=4, downscale=0.0),
    "gen1_k1_outstar1":  dict(store="gen", levels=1, size=1, rule="outstar", p=1, downscale=0.0),
    "gen1_k10_contrast": dict(store="gen", levels=1, size=10, rule="contrastive", downscale=0.0),
    "gen1_k10_contrast300": dict(store="gen", levels=1, size=10, rule="contrastive", iters=300, downscale=0.0),
    "c_res20":       dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", downscale=0.0),
    "c_res20_300":   dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", iters=300, downscale=0.0),
    "c_proto_neg10": dict(store="gen", levels=1, size=10, rule="contrastive", neg=("reservoir", 10), downscale=0.0),
    "c_ppca20":      dict(store="gen", levels=1, size=20, rule="contrastive", prior="ppca", downscale=0.0),
    "c_ppca20_300":  dict(store="gen", levels=1, size=20, rule="contrastive", prior="ppca", iters=300, downscale=0.0),
    "c_res10":       dict(store="gen", levels=1, size=10, rule="contrastive", prior="reservoir", downscale=0.0),
    "reservoir20_nods": dict(store="reservoir", size=20, downscale=0.0),
    "cq8_res10": dict(store="gen", levels=1, size=10, rule="contrastive", prior="reservoir", cbits=8, topk=None, downscale=0.0),
    "cq8_p8res20": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0),
    "cq8_p8res40": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0),
    "cq8_p8res80": dict(store="gen", levels=1, size=80, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0),
    "proto9_nods": dict(store="proto", size=9, downscale=0.0),
    "proto10_nonoise": dict(store="proto", size=10, downscale=0.0, pnoise=0.0),
    "proto9_nonoise": dict(store="proto", size=9, downscale=0.0, pnoise=0.0),
    "proto9_nn_sg": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, surprise=0.2),
    "proto10_nn_sg": dict(store="proto", size=10, downscale=0.0, pnoise=0.0, surprise=0.2),
    "proto9_nn_st": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, sleep_surprise=200),
    "proto9_nn_st50": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, sleep_surprise=50),
    "proto9_nn_st25": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, sleep_surprise=25),
    "proto9_nn_sg_st50": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, surprise=0.2, sleep_surprise=50),
    "proto9_nn_sg_st": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, surprise=0.2, sleep_surprise=200),
    "proto9_nn_trace": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, trace=True),
    **{f"proto9_nn_dr{t}": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, sleep_drift=t / 1000)
       for t in (5, 10, 20, 30, 40)},
    **{f"proto9_nn_pr{t}": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, sleep_probe=t / 100)
       for t in (10, 20, 30, 50)},
    **{f"proto9_nn_mix{t}": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, mix=t / 10) for t in (2, 4, 8, 20)},
    **{f"proto9_nn_mix8_wl{t}": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, mix=0.8, wake_lr=t / 10)
       for t in (1, 3)},
    **{f"proto{k}_nn_wl{t}": dict(store="proto", size=k, downscale=0.0, pnoise=0.0, wake_lr=t / 10)
       for k in (9, 10) for t in (1, 3)},
    # varied replay against awake replay's recency bias (jN = shift replays by up to N pixels)
    "proto9_mix8": dict(store="proto", size=9, downscale=0.0, mix=0.8),                  # pixel noise
    "proto9_nn_j2": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, jitter=2),
    "proto9_nn_mix8_j2": dict(store="proto", size=9, downscale=0.0, pnoise=0.0, mix=0.8, jitter=2),
    "proto40_nn": dict(store="proto", size=40, downscale=0.0, pnoise=0.0),
    "proto40_nn_mix8": dict(store="proto", size=40, downscale=0.0, pnoise=0.0, mix=0.8),
    "proto9_mix8_j2": dict(store="proto", size=9, downscale=0.0, mix=0.8, jitter=2),
    "proto9_mix8_j2_wl3": dict(store="proto", size=9, downscale=0.0, mix=0.8, jitter=2, wake_lr=0.3),
    "res9": dict(store="reservoir", size=9, downscale=0.0),
    "res9_mix8": dict(store="reservoir", size=9, downscale=0.0, mix=0.8),
    "p40_3k_sg": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0,
                      keep=0.8, step=1.0, iters=3000, polyak=True, surprise=0.5),
    "proto5_nonoise": dict(store="proto", size=5, downscale=0.0, pnoise=0.0),
    "proto3_nonoise": dict(store="proto", size=3, downscale=0.0, pnoise=0.0),
    "cq8_p8res40_d1": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=1.0, step=1.0, iters=300),
    "cq8_p8res40_d3": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=3.0, step=1.0, iters=300),
    "cq8_p8res40_d03": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=0.3, step=1.0, iters=1000),
    "cq8_p8res40_n1": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, noise=1.0, step=1.0, iters=300),
    "cq8_p8res40_n01_3k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, noise=0.1, step=1.0, iters=3000),
    "cq8_p8res40_n03_3k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, noise=0.3, step=1.0, iters=3000),
    "cq8_p8res40_d03_3k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=0.3, step=1.0, iters=3000),
    "p40_d03_300": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=0.3, step=1.0, iters=300),
    "p40_cen_d03_300": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=0.3, step=1.0, iters=300, center=True),
    "p40_jac_d03_300": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, decay=0.3, step=1.0, iters=300, center=True, jacobi=True),
    "p40_drop08_1k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.8, step=1.0, iters=1000),
    "p40_drop08_avg_1k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.8, step=1.0, iters=1000, polyak=True),
    "p40_drop05_avg_1k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.5, step=1.0, iters=1000, polyak=True),
    "p40_drop08_avg_3k": dict(store="gen", levels=1, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.8, step=1.0, iters=3000, polyak=True),
    "g2_drop08_avg_1k": dict(store="gen", levels=2, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.8, step=1.0, iters=1000, polyak=True),
    "g2_drop08_avg_3k": dict(store="gen", levels=2, size=40, rule="contrastive", prior="reservoir", cbits=8, pbits=8, downscale=0.0, keep=0.8, step=1.0, iters=3000, polyak=True),
    "ridge2_p8res40": dict(store="gen", levels=2, size=40, rule="ridge", prior="reservoir", pbits=8, downscale=0.0),
    "nl1000_3k": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="contrastive", cbits=8, keep=0.8, step=1.0, iters=3000, polyak=True, expand=1000, expand_k=100),
    "nl2000_3k": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="contrastive", cbits=8, keep=0.8, step=1.0, iters=3000, polyak=True, expand=2000, expand_k=100),
    "ridge_nl1000": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="ridge", expand=1000, expand_k=100),
    "ridge_sharp1": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="ridge", sharpen=1.0),
    "p40_3k_sharp1": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="contrastive", cbits=8, keep=0.8, step=1.0, iters=3000, polyak=True, sharpen=1.0),
    "p40_3k_sharp2": dict(store="gen", levels=1, size=40, prior="reservoir", pbits=8, downscale=0.0, rule="contrastive", cbits=8, keep=0.8, step=1.0, iters=3000, polyak=True, sharpen=2.0),
    "ridge_p8res40": dict(store="gen", levels=1, size=40, rule="ridge", prior="reservoir", pbits=8, downscale=0.0),
    "ridge_res10": dict(store="gen", levels=1, size=10, rule="ridge", prior="reservoir", downscale=0.0),
    "cq8_res10_exact": dict(store="gen", levels=1, size=10, rule="contrastive", prior="reservoir", cbits=8, solve="prior", downscale=0.0),
    "cq8_res10_1000": dict(store="gen", levels=1, size=10, rule="contrastive", prior="reservoir", cbits=8, iters=1000, downscale=0.0),
    "cq8_res5": dict(store="gen", levels=1, size=5, rule="contrastive", prior="reservoir", cbits=8, topk=None, downscale=0.0),
    "cq8": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=8, topk=None, downscale=0.0),
    "cq4": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=4, topk=None, downscale=0.0),
    "cq2": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=2, topk=None, downscale=0.0),
    "ck200": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=None, topk=200, downscale=0.0),
    "ck100": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=None, topk=100, downscale=0.0),
    "ck50": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=None, topk=50, downscale=0.0),
    "ck100_q8": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=8, topk=100, downscale=0.0),
    "ck50_q4": dict(store="gen", levels=1, size=20, rule="contrastive", prior="reservoir", cbits=4, topk=50, downscale=0.0),
    "fold_res20_k025": dict(store="gen", levels=1, size=20, rule="fold", kappa=0.25, prior="reservoir", downscale=0.0),
    "fold_res20_k1":  dict(store="gen", levels=1, size=20, rule="fold", kappa=1.0, prior="reservoir", downscale=0.0),
    "fold_res20_k4":  dict(store="gen", levels=1, size=20, rule="fold", kappa=4.0, prior="reservoir", downscale=0.0),
    "fold_res20_k16": dict(store="gen", levels=1, size=20, rule="fold", kappa=16.0, prior="reservoir", downscale=0.0),
    "ridge_res20":   dict(store="gen", levels=1, size=20, rule="ridge", prior="reservoir", downscale=0.0),
}
LABELS = {
    "off": "No sleep (baseline)",
    "proto10": "Sleep, 10 prototypes/class",
    "proto10_nods": "Sleep, 10 prototypes/class, no downscaling",
    "proto1": "Sleep, 1 prototype/class",
    "proto50": "Sleep, 50 prototypes/class",
    "reservoir20": "Sleep, 20 raw images/class (same memory as 10 protos)",
    "reservoir100": "Sleep, 100 raw images/class",
    "dream": "Sleep, network's own dreams as positives (no store)",
    "split_proto": "Split: wake +, sleep − (10 prototypes/class, wrong label)",
    "split_dream": "Split: wake +, sleep − (dreams, wrong label)",
    "split_dream_own": "Split: wake +, sleep − (dreams, own label: Hinton)",
    "proto1_nods": "Sleep, 1 prototype/class, no downscaling",
    "gen1_k1": "Top-down dreams from layer 1, 1 code/class",
    "gen1_k10": "Top-down dreams from layer 1, 10 codes/class",
    "gen2_k1": "Top-down dreams from layer 2, 1 code/class",
    "gen2_k10": "Top-down dreams from layer 2, 10 codes/class",
    "gen1_k10_count": "Top-down dreams from layer 1, 10 codes/class, count rule",
    "gen1_k10_ridge": "Top-down dreams from layer 1, 10 codes/class, ridge rule",
    "gen2_k10_count": "Top-down dreams from layer 2, 10 codes/class, count rule",
    "gen2_k10_ridge": "Top-down dreams from layer 2, 10 codes/class, ridge rule",
    "gen1_k1_ridge": "Top-down dreams from layer 1, 1 code/class, ridge rule",
    "gen1_k10_outstar1": "Top-down, outstar rule, decode weight h",
    "gen1_k10_outstar2": "Top-down, outstar rule, decode weight h²",
    "gen1_k10_outstar4": "Top-down, outstar rule, decode weight h⁴",
    "gen1_k1_outstar1": "Top-down, outstar rule, decode weight h, 1 code/class",
    "gen1_k10_contrast": "Top-down, contrastive wake-sleep rule, 100 steps/sleep",
    "gen1_k10_contrast300": "Top-down, contrastive wake-sleep rule, 300 steps/sleep",
    "c_res20": "Contrastive, prior = 20 real codes/class, 100 steps",
    "c_res20_300": "Contrastive, prior = 20 real codes/class, 300 steps",
    "c_proto_neg10": "Contrastive, dreams from prototypes, negatives from 10 real codes/class",
    "c_ppca20": "Contrastive, prior = class mean + rank-20 PCA, 100 steps",
    "c_ppca20_300": "Contrastive, prior = class mean + rank-20 PCA, 300 steps",
    "ridge_res20": "Ridge, prior = 20 real codes/class",
    "c_res10": "Contrastive, prior = 10 real codes/class, 100 steps",
    "reservoir20_nods": "Sleep, 20 raw images/class, no downscaling",
    "cq8_res10": "Contrastive, C 8-bit, prior = 10 real codes/class",
    "cq8_p8res20": "Contrastive, C 8-bit, prior = 20 real codes/class at 8 bits",
    "cq8_p8res40": "Contrastive, C 8-bit, prior = 40 real codes/class at 8 bits",
    "cq8_p8res80": "Contrastive, C 8-bit, prior = 80 real codes/class at 8 bits",
    "proto9_nods": "Sleep, 9 prototypes/class, no downscaling",
    "proto10_nonoise": "Sleep, 10 prototypes/class, replayed without pixel noise",
    "proto5_nonoise": "Sleep, 5 prototypes/class, replayed without pixel noise",
    "proto3_nonoise": "Sleep, 3 prototypes/class, replayed without pixel noise",
    "ridge_sharp1": "Ridge, 40 8-bit codes/class, dreams sharpened (centre-surround gain 1)",
    "p40_3k_sharp1": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 3000 steps, dreams sharpened (gain 1)",
    "p40_3k_sharp2": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 3000 steps, dreams sharpened (gain 2)",
    "proto9_nonoise": "Sleep, 9 prototypes/class, replayed without pixel noise",
    "proto9_nn_sg": "9 prototypes, no noise, surprise-gated update (step 0.2 when the network is wrong)",
    "proto10_nn_sg": "10 prototypes, no noise, surprise-gated update (step 0.2 when the network is wrong)",
    "proto9_nn_st": "9 prototypes, no noise, extra sleep after 200 surprises",
    "proto9_nn_st50": "9 prototypes, no noise, extra sleep after 50 surprises",
    "proto9_nn_st25": "9 prototypes, no noise, extra sleep after 25 surprises",
    "proto9_nn_sg_st50": "9 prototypes, no noise, surprise-gated update + extra sleep after 50 surprises",
    "proto9_nn_sg_st": "9 prototypes, no noise, surprise-gated update + extra sleep after 200 surprises",
    "proto9_nn_trace": "9 prototypes, no noise, clock 2000, interference traced (no extra sleeps)",
    **{f"proto9_nn_dr{t}": f"9 prototypes, no noise, extra sleep at weight drift {t / 1000:g}" for t in (5, 10, 20, 30, 40)},
    **{f"proto9_nn_pr{t}": f"9 prototypes, no noise, extra sleep at probe drop {t}%" for t in (10, 20, 30, 50)},
    **{f"proto9_nn_mix{t}": f"9 prototypes, no noise, awake replay: {t / 10:g} replayed items in each wake update"
       for t in (2, 4, 8, 20)},
    **{f"proto9_nn_mix8_wl{t}": f"9 prototypes, no noise, awake replay 0.8 + wake lr x{t / 10:g}" for t in (1, 3)},
    **{f"proto{k}_nn_wl{t}": f"{k} prototypes, no noise, wake lr x{t / 10:g}" for k in (9, 10) for t in (1, 3)},
    "proto9_mix8": "9 prototypes + pixel noise, awake replay 0.8",
    "proto9_nn_j2": "9 prototypes, no noise, replays shifted up to 2 px",
    "proto9_nn_mix8_j2": "9 prototypes, no noise, awake replay 0.8, replays shifted up to 2 px",
    "proto40_nn": "40 prototypes, no noise",
    "proto40_nn_mix8": "40 prototypes, no noise, awake replay 0.8",
    "proto9_mix8_j2": "9 prototypes + pixel noise, awake replay 0.8, shifted up to 2 px",
    "proto9_mix8_j2_wl3": "9 prototypes + pixel noise, awake replay 0.8, shifted up to 2 px, wake lr x0.3",
    "res9": "9 real images/class (reservoir), clock only",
    "res9_mix8": "9 real images/class (reservoir), awake replay 0.8",
    "p40_3k_sg": "Local decoder, 3000 steps, surprise-gated prior (wrong examples overwrite a slot w.p. 0.5)",
    "cq8_p8res40_d1": "Contrastive, 40 8-bit codes/class, C 8-bit, weight decay 1, 300 steps",
    "cq8_p8res40_d3": "Contrastive, 40 8-bit codes/class, C 8-bit, weight decay 3, 300 steps",
    "cq8_p8res40_d03": "Contrastive, 40 8-bit codes/class, C 8-bit, weight decay 0.3, 1000 steps",
    "cq8_p8res40_n1": "Contrastive, 40 8-bit codes/class, C 8-bit, noisy negatives (≈ decay 1), 300 steps",
    "cq8_p8res40_n01_3k": "Contrastive, 40 8-bit codes/class, C 8-bit, noisy negatives 0.1, 3000 steps",
    "cq8_p8res40_n03_3k": "Contrastive, 40 8-bit codes/class, C 8-bit, noisy negatives 0.3, 3000 steps",
    "cq8_p8res40_d03_3k": "Contrastive, 40 8-bit codes/class, C 8-bit, weight decay 0.3, 3000 steps",
    "p40_d03_300": "Contrastive, 40 8-bit codes/class, C 8-bit, decay 0.3, 300 steps",
    "p40_cen_d03_300": "Contrastive, 40 8-bit codes/class, C 8-bit, centred, decay 0.3, 300 steps",
    "p40_jac_d03_300": "Contrastive, 40 8-bit codes/class, C 8-bit, centred + Jacobi steps, decay 0.3, 300 steps",
    "p40_drop08_1k": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout negatives keep 0.8, 1000 steps",
    "p40_drop08_avg_1k": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout negatives keep 0.8, Polyak, 1000 steps",
    "p40_drop08_avg_3k": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout negatives keep 0.8, Polyak, 3000 steps",
    "g2_drop08_avg_1k": "Two-level contrastive (layer 2 -> 1 -> image), 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 1000 steps",
    "g2_drop08_avg_3k": "Two-level contrastive (layer 2 -> 1 -> image), 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 3000 steps",
    "ridge2_p8res40": "Two-level ridge (layer 2 -> 1 -> image), 40 8-bit codes/class",
    "nl1000_3k": "Contrastive, code + 1000 sparse random features (100 active), 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 3000 steps",
    "nl2000_3k": "Contrastive, code + 2000 sparse random features (100 active), 40 8-bit codes/class, C 8-bit, dropout 0.8, Polyak, 3000 steps",
    "ridge_nl1000": "Ridge, code + 1000 sparse random features (100 active), 40 8-bit codes/class",
    "p40_drop05_avg_1k": "Contrastive, 40 8-bit codes/class, C 8-bit, dropout negatives keep 0.5, Polyak, 1000 steps",
    "ridge_p8res40": "Ridge, 40 8-bit codes/class",
    "ridge_res10": "Ridge, prior = 10 real codes/class",
    "cq8_res10_exact": "Contrastive's fixed point solved exactly (not local), C 8-bit, 10 codes/class",
    "cq8_res10_1000": "Contrastive, C 8-bit, 10 codes/class, 1000 steps",
    "cq8_res5": "Contrastive, C 8-bit, prior = 5 real codes/class",
    "cq8": "Contrastive, C 8-bit",
    "cq4": "Contrastive, C 4-bit",
    "cq2": "Contrastive, C 2-bit",
    "ck200": "Contrastive, C top-200/row",
    "ck100": "Contrastive, C top-100/row",
    "ck50": "Contrastive, C top-50/row",
    "ck100_q8": "Contrastive, C 8-bit, top-100/row",
    "ck50_q4": "Contrastive, C 4-bit, top-50/row",
    "fold_res20_k025": "Contrastive folded into G (1 number/synapse), κ=0.25",
    "fold_res20_k1": "Contrastive folded into G (1 number/synapse), κ=1",
    "fold_res20_k4": "Contrastive folded into G (1 number/synapse), κ=4",
    "fold_res20_k16": "Contrastive folded into G (1 number/synapse), κ=16",
}


def jitter(x, j, rng):
    """Shift each image by a random (dy, dx) in [-j, j] pixels, zero-filled (the label pixels are rewritten
    by embed anyway). Cheap variety for replay: nothing more is stored."""
    if not j:
        return x
    out = np.zeros_like(x).reshape(-1, 28, 28)
    img = x.reshape(-1, 28, 28).copy()
    img[:, 0, :10] = 0
    for i, (dy, dx) in enumerate(rng.integers(-j, j + 1, (len(x), 2))):
        ys, yd = (slice(0, 28 - dy), slice(dy, 28)) if dy >= 0 else (slice(-dy, 28), slice(0, 28 + dy))
        xs, xd = (slice(0, 28 - dx), slice(dx, 28)) if dx >= 0 else (slice(-dx, 28), slice(0, 28 + dx))
        out[i, yd, xd] = img[i, ys, xs]
    return out.reshape(len(x), -1)


def sleep_once(m, cfg, store, batches, bs, seen, dream_steps, dream_pool, rng):
    """One sleep. Returns (examples replayed, dream p(label) mean or None)."""
    split = cfg.get("split")
    if cfg.get("replay") == "dream":
        # Imagine a pool for this sleep (dream_pool per class seen), replay
        # from it, then forget it: nothing outlives the sleep.
        ys = np.repeat(seen, dream_pool)
        xs, conf = m.dream(ys, store.start(len(ys)), seen, steps=cfg.get("steps", dream_steps),
                           scale=np.sqrt(store.var), blur=cfg.get("blur", 0))
        sample = lambda: (lambda j: (xs[j], ys[j]))(rng.integers(len(ys), size=bs))
    else:
        if isinstance(store, TopDown):
            store.consolidate()
        conf, sample = None, lambda: store.sample(bs)
    for _ in range(batches):
        x, y = sample()
        x = jitter(x, cfg.get("jitter", 0), rng)
        if not split:
            m.fit_batch(x, y)  # replayed positives + generated negatives
        elif split == "wrong":
            m.fit_phase(x, m._negative_labels(x, y), positive=False)
        else:
            m.fit_phase(x, y, positive=False)
    return batches * bs, (None if conf is None else float(conf.mean()))


def run(cond, seed, data, sleep_every=2000, sleep_batches=200, sleep_bs=32, lr=1e-3,
        dream_steps=20, dream_pool=16, return_store=False, swap=None, anytime=0):
    """One run. return_store: also hand back (network, store) after the run, for offline analysis.
    swap: dict(a, b, n, check) -> after the five tasks, the world changes: n more wake examples of
    digits a and b only, with their labels swapped, on the same sleep clock (see swap_phase).
    anytime: every `anytime` wake examples (between sleeps, wherever the clock is), class-IL on a fixed
    2000-image test subset restricted to the classes seen so far. Final accuracy is measured right after
    the last clock sleep (60000 is a multiple of 2000), so it is a best case; this is the average case."""
    xtr, ytr, xte, yte = data
    cfg = CONDITIONS[cond]
    rng = np.random.default_rng(seed)
    m = L.ForwardForward(lr=lr, loss="symba", homeo=0.001, seed=seed)
    store = None
    if cfg["store"] == "proto":
        store = ProtoStore(cfg["size"], np.random.default_rng(seed + 1000), noise=cfg.get("pnoise", 1.0),
                           surprise=cfg.get("surprise", 0.0))
    elif cfg["store"] == "reservoir":
        store = ReservoirStore(cfg["size"], np.random.default_rng(seed + 1000))
    elif cfg["store"] == "gen":
        store = TopDown(m, cfg["levels"], cfg["size"], np.random.default_rng(seed + 1000), rule=cfg.get("rule", "nlms"),
                        p=cfg.get("p", 2), iters=cfg.get("iters", 100), prior=cfg.get("prior", "proto"),
                        neg=cfg.get("neg"), kappa=cfg.get("kappa", 1.0),
                        cbits=cfg.get("cbits"), topk=cfg.get("topk"), solve=cfg.get("solve"), pbits=cfg.get("pbits"),
                        decay=cfg.get("decay", 0.0), noise=cfg.get("noise", 0.0), step=cfg.get("step", 0.2),
                        center=cfg.get("center", False), jacobi=cfg.get("jacobi", False),
                        keep=cfg.get("keep"), polyak=cfg.get("polyak", False),
                        expand=cfg.get("expand", 0), expand_k=cfg.get("expand_k", 0),
                        sharpen=cfg.get("sharpen", 0.0), surprise=cfg.get("surprise", 0.0))
    elif cfg["store"] == "pixels":
        store = PixelStats(np.random.default_rng(seed + 1000), cfg.get("recent", 0))
    sleep_rng, mix_rng = np.random.default_rng(seed + 2000), np.random.default_rng(seed + 5000)

    test_sets = [np.isin(yte, t) for t in TASKS]
    R, R_task, conf_tasks, t0 = [], [], [], time.time()
    wake, replayed, sleeps, seen, confs = 0, 0, 0, set(), []
    # surprise: the network's prediction (before learning) contradicts the label. It speeds up store
    # turnover (cfg surprise) and/or triggers a sleep after sleep_surprise of them (cfg sleep_surprise).
    gated = cfg.get("surprise") or cfg.get("sleep_surprise")
    surprises, since = 0, 0
    # interference ("sleep pressure"), checked every `check` wake examples, reset by each sleep:
    #   drift  relative weight change since the last sleep, sum ||W - W_sleep||^2 / sum ||W_sleep||^2
    #          (Tononi & Cirelli: pressure builds with net synaptic change)
    #   probe  accuracy drop on a fixed probe of stored prototypes, re-drawn at each sleep: the
    #          network checks a few memories while awake (awake replay) and sleeps when they fail
    # cfg sleep_drift / sleep_probe = thresholds (extra sleeps on top of the clock); trace = record both.
    P = dict(check=cfg.get("check", 100), gap=cfg.get("min_gap", 200), last=0, snap=None, px=None, py=None, base=None,
             trace=[], probes=0, rng=np.random.default_rng(seed + 3000))
    watch = cfg.get("sleep_drift") or cfg.get("sleep_probe") or cfg.get("trace")

    def reset_pressure():
        P["snap"], P["last"] = [W.copy() for W in m.W], wake
        if isinstance(store, ProtoStore) and store.mu:
            mu = np.concatenate([store.mu[c] for c in sorted(store.mu)])
            ys = np.concatenate([np.full(len(store.mu[c]), c) for c in sorted(store.mu)])
            j = P["rng"].choice(len(mu), min(cfg.get("probe_n", 30), len(mu)), replace=False)
            P["px"], P["py"] = mu[j].copy(), ys[j]
            P["base"] = float((m.scores(P["px"]).argmax(1) == P["py"]).mean())

    def pressure():
        drift = sum(float(((W - W0) ** 2).sum()) for W, W0 in zip(m.W, P["snap"])) / \
            sum(float((W0 ** 2).sum()) for W0 in P["snap"])
        drop = None
        if P["px"] is not None:
            drop = P["base"] - float((m.scores(P["px"]).argmax(1) == P["py"]).mean())
            P["probes"] += len(P["px"])
        return drift, drop

    def awake(x, y):
        """One wake example, and a sleep when the clock (or accumulated surprise) says so."""
        nonlocal wake, replayed, sleeps, surprises, since
        sup = bool(gated) and int(m.scores(x).argmax(1)[0]) != int(y[0])
        surprises += sup
        if cfg.get("wake_lr"):  # complementary learning systems: the cortex learns slowly while awake
            m.opt.lr = lr * cfg["wake_lr"]
        if cfg.get("split"):
            m.fit_phase(x, y, positive=True)  # wake: positives only
        elif cfg.get("mix") and store is not None and seen:
            # awake replay: the wake example shares its update with `mix` replayed items on average
            # (integer part always, fraction by chance), so old classes are in every gradient step
            k = int(cfg["mix"]) + int(mix_rng.random() < cfg["mix"] % 1)
            if k:
                xr, yr = store.sample(k)
                xr = jitter(xr, cfg.get("jitter", 0), mix_rng)
                m.fit_batch(np.concatenate([x, xr]), np.concatenate([y, yr]))
                replayed += k
            else:
                m.fit_batch(x, y)
        else:
            m.fit_batch(x, y)               # wake learning
        m.opt.lr = lr
        seen.add(int(y[0]))
        if anytime and (wake + 1) % anytime == 0:  # wake not yet incremented: count this example
            j = any_idx[np.isin(yte[any_idx], sorted(seen))]
            any_trace.append((wake + 1, float((m.scores(xte[j]).argmax(1) == yte[j]).mean())))
        if store is not None:
            store.add(x, int(y[0]), **(dict(surprise=sup) if cfg.get("surprise") else {}))  # hippocampus records
            wake += 1
            since += sup
            urge = False
            if watch and P["snap"] is not None and wake % P["check"] == 0:
                drift, drop = pressure()
                if cfg.get("trace"):
                    P["trace"].append((wake, drift, drop))
                urge = wake - P["last"] >= P["gap"] and (
                    (cfg.get("sleep_drift") and drift >= cfg["sleep_drift"]) or
                    (cfg.get("sleep_probe") and drop is not None and drop >= cfg["sleep_probe"]))
            if wake % sleep_every == 0 or urge or (cfg.get("sleep_surprise") and since >= cfg["sleep_surprise"]):
                n, conf = sleep_once(m, cfg, store, sleep_batches, sleep_bs, sorted(seen),
                                     dream_steps, dream_pool, sleep_rng)
                replayed += n
                if conf is not None:
                    confs.append(conf)
                if cfg["downscale"]:
                    m.downscale(cfg["downscale"])
                sleeps += 1
                since = 0
                if watch:
                    reset_pressure()

    any_idx, any_trace = np.random.default_rng(seed + 4000).choice(len(xte), 2000, replace=False), []
    if watch:
        reset_pressure()
    for t in TASKS:
        for i in rng.permutation(np.flatnonzero(np.isin(ytr, t))):
            awake(xtr[i:i + 1], ytr[i:i + 1])
        R.append([accuracy(m, xte[s], yte[s]) for s in test_sets])
        R_task.append([task_il_accuracy(m, xte[s], yte[s], c) for s, c in zip(test_sets, TASKS)])
        conf_tasks.append(confusion(m, xte, yte))
    R = np.array(R)
    out = dict(
        class_il_final=accuracy(m, xte, yte),
        task_il_final=float(np.mean(R_task[-1])),
        forgetting=float(np.mean([R[:, j].max() - R[-1, j] for j in range(len(TASKS) - 1)])),
        R=R.tolist(),
        store_floats=store.floats() if store else 0,
        sleeps=sleeps,
        replayed=replayed,
        surprises=surprises,
        probes=P["probes"],
        weight_sparsity=m.weight_sparsity(),
        dead_units=m.dead_fraction(xte[:2000], yte[:2000]),
        dream_conf=float(np.mean(confs)) if confs else None,
        dreams=final_dreams(m, store, xte, yte, dream_steps) if cfg.get("replay") == "dream"
        else (store.consolidate(), topdown_dreams(store, xte, yte))[1] if cfg["store"] == "gen" else None,
        confusion=conf_tasks[-1],  # test, rows = true class, cols = predicted
        confusion_tasks=conf_tasks,  # the same after each task
        replay=replay_stats(m, store, xte, yte) if cfg["store"] in ("proto", "gen") else None,  # after the dreams
        gen_params=sum(G.size for G in store.G) if cfg["store"] == "gen" else 0,
        gen_extra=store.extra_floats() if cfg["store"] == "gen" else 0,
        recon_mse=store.err if cfg["store"] == "gen" else None,
        seconds=time.time() - t0,
    )
    out["pre_swap_trace_len"] = len(P["trace"])
    if anytime:
        out["anytime"] = float(np.mean([a for _, a in any_trace]))
        out["anytime_late"] = float(np.mean([a for w, a in any_trace if w > 24000]))  # tasks 3-5
        out["anytime_min"] = float(min(a for _, a in any_trace))
        out["anytime_trace"] = any_trace
    if swap:
        out["swap"] = swap_phase(m, store, awake, swap, xtr, ytr, xte, yte, rng)
        out["swap_sleeps"], out["swap_replayed"] = sleeps - out["sleeps"], replayed - out["replayed"]
        out["swap_probes"] = P["probes"] - out["probes"]
    if cfg.get("trace"):
        out["trace"] = P["trace"]
    return (out, m, store) if return_store else out


def swap_phase(m, store, awake, sw, xtr, ytr, xte, yte, rng):
    """The world changes: digits a and b swap labels, and only they are seen for n wake examples.
    Every `check` examples, on the test set: new = fraction of a/b predicted with the new (swapped)
    label, old = still predicted with the old label, others = class-IL accuracy on the other 8
    digits; store_new = fraction of the store's replays labelled a (or b) that look like the other
    digit (nearest real class mean), i.e. how far the store itself has taken in the new mapping."""
    a, b, n, check = sw["a"], sw["b"], sw["n"], sw["check"]
    flip = {a: b, b: a}
    idx = rng.permutation(np.flatnonzero(np.isin(ytr, [a, b])))
    idx = np.resize(idx, n)  # cycle if n exceeds the available examples
    sel, rest = np.isin(yte, [a, b]), ~np.isin(yte, [a, b])
    means = np.stack([xte[yte == c].mean(0) for c in range(10)])
    z = lambda v: (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-8)

    def look_like(c, k=200):
        if isinstance(store, TopDown):
            xs = store.decode(store.prior.sample_class(c, k)) if store.prior.has(c) else None
        elif isinstance(store, ProtoStore):
            xs = store.sample_class(c, k)
        else:
            return None
        return float(((z(xs[:, 10:]) @ z(means[:, 10:]).T).argmax(1) == flip[c]).mean())

    def probe(t):
        pred = np.concatenate([m.scores(xte[i:i + 2000]).argmax(1) for i in range(0, len(xte), 2000)])
        ya = yte[sel]
        sn = [look_like(c) for c in (a, b)]
        return dict(t=t, new=float((pred[sel] == np.vectorize(flip.get)(ya)).mean()),
                    old=float((pred[sel] == ya).mean()), others=float((pred[rest] == yte[rest]).mean()),
                    store_new=None if sn[0] is None else float(np.mean(sn)))

    traj = [probe(0)]
    for k, i in enumerate(idx, 1):
        awake(xtr[i:i + 1], np.array([flip[int(ytr[i])]]))
        if k % check == 0:
            traj.append(probe(k))
    return traj


def confusion(m, x, y, chunk=2000):
    pred = np.concatenate([m.scores(x[i:i + chunk]).argmax(1) for i in range(0, len(x), chunk)])
    return np.bincount(y * 10 + pred, minlength=100).reshape(10, 10).tolist()


def replay_stats(m, store, xte, yte, n=2000):
    """How the final network sees what it replays: confusion of replayed inputs (rows = the label they're
    replayed with), per-class pixel spread (mean per-pixel std) of replays vs real test images, and the
    correlation of each class's mean replay with the real class means."""
    xs, ys = store.sample(n)
    spread = lambda x, y: [float(x[y == c, 10:].std(0).mean()) for c in range(10)]
    return dict(confusion=confusion(m, xs, ys), spread=spread(xs, ys), real_spread=spread(xte, yte),
                corr=dream_corr(xs, ys, xte, yte))


def dream_corr(xs, ys, xte, yte):
    dm = np.stack([xs[ys == c].mean(0) for c in range(10)])
    rm = np.stack([xte[yte == c].mean(0) for c in range(10)])
    z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)
    return (z(dm[:, 10:]) @ z(rm[:, 10:]).T / (D - 10)).tolist()


def topdown_dreams(store, xte, yte, per=10):
    """Final top-down dreams, 10 per class (as final_dreams)."""
    ys = np.repeat(np.arange(10), per)
    xs = np.concatenate([store.decode(store.prior.sample_class(c, per)) if store.prior.has(c)
                         else np.zeros((per, D), np.float32) for c in range(10)])
    return dict(images=np.round(xs, 3).tolist(), corr=dream_corr(xs, ys, xte, yte))


def final_dreams(m, store, xte, yte, steps, per=10):
    """Dreams of the final network, 10 per class, and how much each class's
    average dream correlates with the real average test digit of every
    class (evaluation only; dreams never see test data)."""
    ys = np.repeat(np.arange(10), per)
    xs, _ = m.dream(ys, store.noise(len(ys)), np.arange(10), steps=steps, scale=np.sqrt(store.var))
    return dict(images=np.round(xs, 3).tolist(), corr=dream_corr(xs, ys, xte, yte))


def summarise(results, seeds):
    def ms(v, scale=100, fmt=".2f"):
        v = np.array(v) * scale
        return f"{v.mean():{fmt}}" + (f" ± {v.std():{fmt}}" if len(v) > 1 else "")

    out = [f"Seeds: {seeds}. Split-MNIST, class-incremental, one pass, batch 1 while awake.\n",
           "| Condition | Class-IL final | Task-IL final | Forgetting | Store (floats) | Replayed / wake examples | Weight sparsity | Dead units |",
           "|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        rs = results.get(c)
        if not rs:
            continue
        out.append(
            f"| {LABELS[c]} | {ms([r['class_il_final'] for r in rs])} | {ms([r['task_il_final'] for r in rs])} | "
            f"{ms([r['forgetting'] for r in rs])} | {int(np.mean([r['store_floats'] for r in rs])):,} | "
            f"{np.mean([r['replayed'] for r in rs]) / 60000:.1f}× | "
            f"{ms([r['weight_sparsity'] for r in rs], fmt='.0f')}% | {ms([r['dead_units'] for r in rs], fmt='.0f')}% |")
    out += [f"\n### Class-IL accuracy per task after each task (seed {seeds[0]})\n"]
    for c in CONDITIONS:
        rs = results.get(c)
        if not rs:
            continue
        out += [f"**{LABELS[c]}**\n", "| after \\ on | " + " | ".join(f"{a}/{b}" for a, b in TASKS) + " |",
                "|---" * (len(TASKS) + 1) + "|"]
        for i, row in enumerate(rs[0]["R"]):
            out.append(f"| {TASKS[i][0]}/{TASKS[i][1]} | " + " | ".join(
                f"{v * 100:.0f}" if j <= i else "·" for j, v in enumerate(row)) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="results/sleep")
    ap.add_argument("--merge", nargs="+", metavar="DIR", help="combine results.json from separate runs into --out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results, seeds = {}, list(args.seeds)
    if args.merge:
        seeds = []
        for d in args.merge:
            run_ = json.load(open(os.path.join(d, "results.json")))
            seeds = sorted(set(seeds) | set(run_["seeds"]))
            for c, rs in run_["results"].items():
                results.setdefault(c, []).extend(rs)
    else:
        data = load_mnist()
        for c in args.conditions:
            for s in args.seeds:
                r = run(c, s, data)
                results.setdefault(c, []).append(r)
                print(f"{c:13s} seed={s} classIL={r['class_il_final']:.4f} taskIL={r['task_il_final']:.4f} "
                      f"forget={r['forgetting']:.3f} store={r['store_floats']} sleeps={r['sleeps']} "
                      f"sparsity={r['weight_sparsity']:.2f} dead={r['dead_units']:.2f} {r['seconds']:.0f}s", flush=True)
                json.dump(dict(seeds=args.seeds, results=results), open(os.path.join(args.out, "results.json"), "w"), indent=1)
        json.dump(dict(seeds=seeds, results=results), open(os.path.join(args.out, "results.json"), "w"), indent=1)
    md = summarise(results, seeds)
    open(os.path.join(args.out, "RESULTS.md"), "w").write("# Wake/sleep forward-forward\n\n" + md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
