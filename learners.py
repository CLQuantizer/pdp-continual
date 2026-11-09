"""Five ways to learn MNIST, in plain NumPy, behind one interface.

Every learner has:
    fit_batch(x, y)   one learning step on a batch (batch size 1 = pure online)
    scores(x)         (n, 10) class scores; prediction is argmax
    end_training(x, y) optional hook after the last epoch of the *standard*
                      protocol (only Hebbian uses it, to fit its readout on the
                      final features). Never called in the online protocols.

Data convention: x is (n, 784) float32 in [0, 1], y is (n,) int labels.
Weights are stored (fan_in, fan_out), so a layer is  h_next = h @ W + b.
"""
import numpy as np

N_CLASSES = 10
F = np.float32


def onehot(y, n=N_CLASSES):
    out = np.zeros((len(y), n), F)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def he_init(rng, fan_in, fan_out):
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(F)


class Adam:
    """Per-parameter Adam, or plain SGD when adam=False. Updates arrays in place."""

    def __init__(self, params, lr, adam=True, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.adam = params, lr, adam
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def reset(self):
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        if not self.adam:
            for p, g in zip(self.params, grads):
                p -= self.lr * g
            return
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# --------------------------------------------------------------------------
# 1-3. MLP trained by backprop, feedback alignment, or direct feedback alignment
# --------------------------------------------------------------------------
class MLP:
    """784-H-H-10 ReLU MLP with softmax cross-entropy. Only the backward pass differs.

    rule="bp":  delta_l = (delta_{l+1} @ W_{l+1}.T) * relu'      exact gradient
    rule="fa":  delta_l = (delta_{l+1} @ B_{l+1})   * relu'      fixed random B
    rule="dfa": delta_l = (e @ D_l)                 * relu'      output error e sent
                                                                  straight to each layer
    """

    def __init__(self, rule="bp", hidden=256, lr=1e-3, adam=True, seed=0):
        assert rule in ("bp", "fa", "dfa")
        rng = np.random.default_rng(seed)
        self.rule = rule
        sizes = [784, hidden, hidden, N_CLASSES]
        self.W = [he_init(rng, a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, F) for b in sizes[1:]]
        # Feedback matrices: same shape as W.T, same scale, never learned.
        # B[i] replaces W[i].T when sending error from layer i+1 down to layer i.
        self.B = [he_init(rng, b, a) for a, b in zip(sizes[:-1], sizes[1:])]
        # DFA: one matrix per hidden layer, from the 10 outputs to that layer.
        self.D = [he_init(rng, N_CLASSES, hidden) for _ in range(2)]
        self.opt = Adam(self.W + self.b, lr, adam)

    def _forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return hs

    def scores(self, x):
        return self._forward(x)[-1]

    def fit_batch(self, x, y):
        hs = self._forward(x)
        e = (softmax(hs[-1]) - onehot(y)) / len(x)  # dLoss/dlogits
        deltas = [None] * len(self.W)
        deltas[-1] = e
        for i in range(len(self.W) - 2, -1, -1):  # hidden layers, top down
            mask = (hs[i + 1] > 0).astype(F)
            if self.rule == "bp":
                deltas[i] = (deltas[i + 1] @ self.W[i + 1].T) * mask
            elif self.rule == "fa":
                deltas[i] = (deltas[i + 1] @ self.B[i + 1]) * mask
            else:
                deltas[i] = (e @ self.D[i]) * mask
        gW = [hs[i].T @ deltas[i] for i in range(len(self.W))]
        gb = [d.sum(0) for d in deltas]
        self.opt.step(gW + gb)


# --------------------------------------------------------------------------
# 4. Forward-forward (Hinton 2022), supervised variant
# --------------------------------------------------------------------------
class ForwardForward:
    """The label is written into the first 10 pixels (always blank in MNIST).
    Positive data = image + correct label, negative = image + a wrong label.
    Each layer independently pushes its goodness g = mean(h^2) above `theta`
    for positive data and below it for negative data. Layer inputs are
    L2-normalised so a layer can't just pass its own goodness upward.
    Nothing crosses a layer boundary backward. To classify, try all 10 labels
    and pick the one with the largest total goodness.
    """

    def __init__(self, hidden=500, n_layers=2, theta=2.0, lr=3e-3, adam=True,
                 hard_neg=True, neg_temp=1.0, loss="threshold", alpha=4.0,
                 homeo=0.0, target=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.rng, self.theta = rng, theta
        self.hard_neg, self.neg_temp = hard_neg, neg_temp
        self.loss, self.alpha = loss, alpha
        self.homeo, self.target = homeo, target
        sizes = [784] + [hidden] * n_layers
        self.W = [he_init(rng, a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, F) for b in sizes[1:]]
        self.opt = Adam(self.W + self.b, lr, adam)

    @staticmethod
    def embed(x, labels):
        x = x.copy()
        x[:, :N_CLASSES] = 0.0
        x[np.arange(len(x)), labels] = 1.0
        return x

    @staticmethod
    def _norm(h):
        # Direction only, scaled to unit RMS (not unit length) so that
        # activations, and hence goodness, start near theta rather than ~0.
        return h / (np.sqrt((h * h).mean(1, keepdims=True)) + 1e-4)

    def _layers(self, x):
        """Yields (layer input, activation) per layer."""
        h = x
        for W, b in zip(self.W, self.b):
            inp = self._norm(h)
            h = np.maximum(inp @ W + b, 0)
            yield inp, h

    def goodness(self, x):
        return sum((h * h).mean(1) for _, h in self._layers(x))

    def scores(self, x):
        out = np.empty((len(x), N_CLASSES), F)
        for c in range(N_CLASSES):
            out[:, c] = self.goodness(self.embed(x, np.full(len(x), c)))
        return out

    def _negative_labels(self, x, y):
        if not self.hard_neg:
            return (y + self.rng.integers(1, N_CLASSES, len(y))) % N_CLASSES
        # Sample a wrong label in proportion to how good the network currently
        # thinks it is (Hinton 2022), so negatives target actual confusions.
        p = softmax(self.scores(x) * self.neg_temp)
        p[np.arange(len(y)), y] = 0.0
        p /= p.sum(1, keepdims=True)
        r = self.rng.random((len(y), 1))
        return np.minimum((p.cumsum(1) < r).sum(1), N_CLASSES - 1)

    def fit_batch(self, x, y):
        wrong = self._negative_labels(x, y)
        xx = np.concatenate([self.embed(x, y), self.embed(x, wrong)])
        n = len(x)
        sign = np.concatenate([np.ones(n, F), -np.ones(n, F)])
        gW, gb, active = [], [], []
        for inp, h in self._layers(xx):
            g = (h * h).mean(1)
            if self.loss == "threshold":
                # softplus(sign * (theta - g)); dloss/dg = -sign * sigmoid(sign*(theta-g))
                s = sign * (self.theta - g)
                dg = -sign / (1 + np.exp(-s)) / len(xx)
            else:
                # "symba": softplus(-alpha * (g_pos - g_neg)) for each image's
                # (pos, neg) pair. Only the gap matters, so nothing pushes
                # goodness down overall.
                d = g[:n] - g[n:]
                w = self.alpha * 0.5 * (1 - np.tanh(0.5 * self.alpha * d)) / n  # alpha*sigmoid(-alpha*d), overflow-safe
                dg = np.concatenate([-w, w])
            dz = (2.0 / h.shape[1]) * h * dg[:, None]  # h=0 where relu is off, so mask is implicit
            gW.append(inp.T @ dz)
            gb.append(dz.sum(0))
            active.append((h[:n] > 0).mean(0))
        self.opt.step(gW + gb)
        if self.homeo:
            # Intrinsic plasticity: nudge each unit's bias toward firing on a
            # `target` fraction of positive inputs. Acts on b directly (not via
            # the gradient), so it can revive a unit that has gone silent.
            for b, a in zip(self.b, active):
                b += self.homeo * (self.target - a)

    def downscale(self, delta):
        """Synaptic downscaling (Tononi & Cirelli): shrink every weight toward
        zero by delta * (mean |w| of its layer); weights smaller than that
        become exactly 0. Uniform scaling alone would change nothing here
        (inputs are normalised and labels are compared by goodness), so it is
        the removal of weak synapses that does the work."""
        for W in self.W:
            t = delta * np.abs(W).mean()
            np.copyto(W, np.sign(W) * np.maximum(np.abs(W) - t, 0))

    def shrink_perturb(self, lam, sigma, rng):
        """Ash & Adams 2020 shrink-and-perturb: W <- lam * W + N(0, sigma^2).
        Preserves ReLU hypotheses under no-bias scaling; raises entropy of the
        last-layer scores. Caller should reset the Adam moments afterward."""
        for W in self.W:
            W *= lam
            W += sigma * rng.standard_normal(W.shape).astype(W.dtype)
        for b in self.b:
            b *= lam
            b += sigma * rng.standard_normal(b.shape).astype(b.dtype)

    def weight_sparsity(self):
        return float(np.mean([(W == 0).mean() for W in self.W]))

    def dead_fraction(self, x, y):
        """Fraction of units that never fire on x with its true labels y."""
        return float(np.mean([((h > 0).sum(0) == 0).mean() for _, h in self._layers(self.embed(x, y))]))

    def fit_phase(self, x, labels, positive):
        """One phase of Hinton's wake/sleep split: only positives (push
        goodness of (x, labels) above theta) or only negatives (push it
        below), with the original threshold loss since there is no pair to
        compare. Homeostasis runs on positive phases only."""
        xx = self.embed(x, labels)
        sign = 1.0 if positive else -1.0
        gW, gb, active = [], [], []
        for inp, h in self._layers(xx):
            g = (h * h).mean(1)
            dg = -sign / (1 + np.exp(-np.clip(sign * (self.theta - g), -30, 30))) / len(xx)
            dz = (2.0 / h.shape[1]) * h * dg[:, None]
            gW.append(inp.T @ dz)
            gb.append(dz.sum(0))
            active.append((h > 0).mean(0))
        self.opt.step(gW + gb)
        if self.homeo and positive:
            for b, a in zip(self.b, active):
                b += self.homeo * (self.target - a)

    def _goodness_grad(self, x, coef):
        """Per-row goodness of x and d/dx of sum_i coef_i * goodness(x_i),
        through the frozen network. Used only to generate dreams; no weight
        learns from it."""
        cache, h, g = [], x, 0.0
        for W, b in zip(self.W, self.b):
            r = np.sqrt((h * h).mean(1, keepdims=True))
            inp = h / (r + 1e-4)
            z = np.maximum(inp @ W + b, 0)
            cache.append((h, r, inp, z))
            g = g + (z * z).mean(1)
            h = z
        gh = np.zeros_like(h)
        for W, (hp, r, inp, z) in zip(reversed(self.W), reversed(cache)):
            gh = gh + coef[:, None] * (2.0 / z.shape[1]) * z  # this layer's own goodness
            ginp = (gh * (z > 0)) @ W.T
            re = r + 1e-4
            gh = ginp / re - hp * (ginp * hp).sum(1, keepdims=True) / (re * re * hp.shape[1] * (r + 1e-12))
        return g, gh

    def dream(self, labels, x0, classes, steps=20, lr=0.05, scale=1.0, blur=0):
        """The network imagines an input for each label: start from x0 and
        climb log p(label | x) = log softmax over `classes` of goodness, by
        normalised gradient ascent on the pixels (step scaled per pixel by
        `scale`, e.g. the pixel std). Returns dreams and final p(label)."""
        x, classes = x0.copy(), np.asarray(classes)
        t = np.searchsorted(classes, labels)
        n = len(x)
        for _ in range(steps):
            xx = np.concatenate([self.embed(x, np.full(n, c)) for c in classes])
            S = self.goodness(xx)
            p = softmax(S.reshape(len(classes), n).T)
            coef = (np.eye(len(classes), dtype=F)[t] - p).T.reshape(-1)
            _, gx = self._goodness_grad(xx, coef.astype(F))
            gx = gx.reshape(len(classes), n, -1).sum(0)
            gx[:, :N_CLASSES] = 0.0
            for _ in range(blur):  # 3x3 box blur of the step: favours strokes over pixel noise
                im = np.pad(gx.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))
                gx = sum(im[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)).reshape(n, -1) / 9
                gx[:, :N_CLASSES] = 0.0
            gx *= scale
            x = np.clip(x + lr * gx / (np.sqrt((gx * gx).mean(1, keepdims=True)) + 1e-8), 0, 1)
        return x.astype(F), p[np.arange(n), t]


# --------------------------------------------------------------------------
# Ridge readout shared by 5 and 6: accumulate statistics, solve on demand
# --------------------------------------------------------------------------
class RidgeReadout:
    """Least-squares map from features to one-hot labels.

    Keeps only A = sum h h^T and C = sum h y^T. Each example adds a rank-1
    term, so this is a genuine online learner, and because a sum doesn't care
    about order the solution after any stream equals the batch solution.
    Rank-1 updates are buffered and flushed as one matmul (identical result).
    """

    def __init__(self, dim, lam=1e-2, flush=512):
        self.A = np.zeros((dim + 1, dim + 1))
        self.C = np.zeros((dim + 1, N_CLASSES))
        self.lam, self.flush = lam, flush
        self.buf_h, self.buf_y, self.Wout = [], [], None

    @staticmethod
    def _bias(h):
        return np.hstack([h, np.ones((len(h), 1), h.dtype)])

    def update(self, h, y):
        self.buf_h.append(h)
        self.buf_y.append(y)
        if sum(len(b) for b in self.buf_y) >= self.flush:
            self._flush()

    def _flush(self):
        if not self.buf_y:
            return
        H = self._bias(np.concatenate(self.buf_h)).astype(np.float64)
        self.A += H.T @ H
        self.C += H.T @ onehot(np.concatenate(self.buf_y)).astype(np.float64)
        self.buf_h, self.buf_y, self.Wout = [], [], None

    def reset(self):
        self.A[:] = 0
        self.C[:] = 0
        self.buf_h, self.buf_y, self.Wout = [], [], None

    def scores(self, h):
        self._flush()
        if self.Wout is None:
            n = self.A.shape[0]
            reg = self.lam * max(np.trace(self.A) / n, 1e-12)
            self.Wout = np.linalg.solve(self.A + reg * np.eye(n), self.C).astype(F)
        return self._bias(h) @ self.Wout


# --------------------------------------------------------------------------
# 5. Competitive Hebbian features (SoftHebb-style) + ridge readout
# --------------------------------------------------------------------------
class SoftHebb:
    """Unsupervised layer of H units with soft winner-take-all.

    u = W x (x and rows of W on the unit sphere), y = softmax(u / tau).
    Update per unit k:  dW_k = lr * y_k * (x - u_k W_k)    (Moraitis et al. 2022)
    That is Oja's rule gated by how much unit k won: winners move toward the
    input, losers barely move, so units spread out over the data.
    Homeostasis: each unit has a threshold t_k that rises when it wins more
    than its 1/H share and falls when it wins less; competition uses u - t.
    Without it one unit captures every input and the rest stay random.
    Labels are never seen by this layer, only by the readout.
    Features for the readout: h = relu(u - mean_k u), a 'triangle' code.
    """

    def __init__(self, hidden=2000, tau=0.02, lr=0.05, homeo=0.01, lam=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((hidden, 784)).astype(F)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.t = np.zeros(hidden, F)
        self.tau, self.lr, self.homeo = tau, lr, homeo
        self.readout = RidgeReadout(hidden, lam)

    @staticmethod
    def _unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def features(self, x, u=None):
        if u is None:
            u = self._unit(x) @ self.W.T
        return np.maximum(u - u.mean(1, keepdims=True), 0)

    def hebb(self, x):
        xn = self._unit(x)
        u = xn @ self.W.T
        y = softmax((u - self.t) / self.tau)
        dW = y.T @ xn - (y * u).sum(0)[:, None] * self.W
        # Summed, not averaged, over the batch: each example moves its own
        # winner, exactly as if the examples had arrived one at a time.
        self.W += self.lr * dW
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)  # Oja keeps |W|~1; this guards big steps
        H = len(self.t)
        self.t += self.homeo * self.tau * (H * y.mean(0) - 1)
        return u

    def fit_batch(self, x, y):
        u = self.hebb(x)
        self.readout.update(self.features(x, u), y)

    def end_training(self, x, y):
        # Readout statistics gathered while features were still moving are
        # stale; refit them on the final features (no feature learning here).
        self.readout.reset()
        for i in range(0, len(x), 1000):
            self.readout.update(self.features(x[i:i + 1000]), y[i:i + 1000])

    def scores(self, x):
        return self.readout.scores(self.features(x))


# --------------------------------------------------------------------------
# 6. Random fixed features + ridge readout ("extreme learning machine")
# --------------------------------------------------------------------------
class RandomFeatures:
    """h = relu(x @ R + c) with R, c random and never changed. Only the readout learns."""

    def __init__(self, hidden=2000, lam=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.R = (rng.standard_normal((784, hidden)) / np.sqrt(784 * 0.1)).astype(F)
        self.c = rng.standard_normal(hidden).astype(F) * 0.5
        self.readout = RidgeReadout(hidden, lam)

    def features(self, x):
        return np.maximum(x @ self.R + self.c, 0)

    def fit_batch(self, x, y):
        self.readout.update(self.features(x), y)

    def scores(self, x):
        return self.readout.scores(self.features(x))
