"""FF-confirmed one-shot remap (loop 02:35 hyp 1, pre-registered 03:15 UTC+8 31 Oct).

Lisman & Grace 2005: a one-shot write needs an input-side mismatch, i.e. the cortex recognises the input as the new
label. Here an in-radius conflicting hit toward y counts 2 (full evidence, H = 1) when the FF cortex confirms y on
that input (FF argmax == y, and for fQ also FF margin >= its rolling quantile q = Q/100); otherwise it counts 1
(H = 2). Store default a25w250, K = 2 (fQk2); f-1 = baseline h2k2. FF rows joined per arrival from gate_probe logs.
Test: f50k2.  Prediction: held-out bg 0.02 swap >= 74 under mq30_s2, pre-swap remaps <= 5 in both worlds.
Kill: pre-swap remaps > 10 in either world, or swap < 73.
"""
VDOC = """Follow-up v variants (03:20 UTC+8, Kumaran & Maguire 2007 associative mismatch): a conflict counts 2 when the
FF cortex confidently predicts the store's OLD class s on that input (FF argmax == s, margin >= rolling qQ), i.e. a
familiar item in a new pairing. Test: v50k2. Same prediction and kill as f50k2.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import life as LF
from data import load_mnist
import remap_probe as RP
import presence_bg as PB
import fast_remap as FR
from gate_probe import rolling_quantile
from loose_gate import score


class FFStore(FR.FStore):
    def __init__(self, *a, K=None, **kw):
        self.mode = "v" if "v" in K else "f"
        base, fk = K.split(self.mode); q, Kp = fk.split("k")
        super().__init__(*a, K=f"{base}h2k{Kp}", **kw)
        self.q = int(q); self.conf = False; self.ffp = -1; self.ffhi = False; self.vconf = 0

    def observe(self, x, y):
        self.win.append((self.t, y))
        while self.win and self.win[0][0] < self.t - self.W:
            self.win.popleft()
        if not self.mu:
            return
        s, j, d = self.nearest(x)
        r = self.rad[s][j]
        if s == y:
            self.rad[s][j] = d if r is None else r + 0.1 * (d - r); self.hist[s][j].append(0); return
        inr = r is not None and d <= r
        conf = self.conf if self.mode == "f" else (self.ffp == s and self.ffhi)
        self.vconf += bool(inr and conf)
        self.hist[s][j].append((1, y, 2 if conf else 1) if inr else 2)
        if not inr:
            return
        hot = [i for i, h in enumerate(self.hist[s]) if sum(e[2] for e in h if isinstance(e, tuple) and e[1] == y) >= 2]
        if len(hot) >= self.K and len(self.mu[s]) > len(hot):
            if PB.absent_rel(self, s):
                self.blocked += 1; return
            for i in sorted(hot, reverse=True):
                self.log.append((self.t, s, y)); self.move(s, i, y)


def run(args):
    seed, days, w, Ks, F, FFM = args
    w = dict(LF.WORLD, **w)
    xtr, ytr, _, _ = load_mnist("mnist")
    world = LF.World(xtr, ytr, days * w["day"], seed, w)
    stores = {K: FFStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=K) for K in Ks}
    qs = {q: FFM >= rolling_quantile(FFM, q / 100) for q in {s.q for s in stores.values()} if q > 0}
    out = {K: [] for K in Ks}; i = 0; nconf = {K: [0, 0] for K in Ks}
    for t in range(days * w["day"]):
        a = world.arrive(t)
        if a is None:
            continue
        x, c, y, ins = a
        for K, st in stores.items():
            out[K].append((t, y) + RP.answer(st, x)); st.t = t
            st.ffp = F[i]; st.ffhi = st.q == 0 or (st.q > 0 and bool(qs[st.q][i]))
            st.conf = st.q >= 0 and F[i] == y and st.ffhi
            nconf[K][t >= world.swap_at] += st.conf
            st.observe(x, y); st.add(x, y)
        i += 1
    sw = world.swap_at
    return w["bg"], seed, {K: np.array(v) for K, v in out.items()}, {K: dict(n=s.remaps, pre_swap=sum(t < sw for t, _, _ in s.log), post_lat=[t - sw for t, _, _ in s.log if t >= sw][:6]) for K, s in stores.items()}


def main():
    Ks = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a25w250f-1k2", "a25w250f50k2", "a25w250f0k2", "a25w250f50k3"]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/ff_remap"; os.makedirs(out_dir, exist_ok=True)
    logs = {}
    for path, bg in (("results/gate_probe", 0.02), ("results/gate_probe_s345", 0.02), ("results/gate_probe_bg02", 0.2)):
        for r in json.load(open(os.path.join(path, "results.json")))["raw"][:6 if bg == 0.2 else 3]:
            logs[(bg, r["seed"])] = r
    load_mnist("mnist")
    jobs = []
    for bg, s in sorted(logs):
        R = np.array(logs[(bg, s)]["rows"]); jobs.append((s, 16, dict(bg=bg), Ks, R[:, 5].astype(int), R[:, 7]))
    with Pool(min(12, os.cpu_count())) as p:
        res = p.map(run, jobs, chunksize=1)
    allres = {}
    for bg, seed, out, rm in res:
        r = logs[(bg, seed)]; R = np.array(r["rows"]); w = dict(LF.WORLD, bg=bg)
        assert np.array_equal(out[Ks[0]][:, 1], R[:, 2])
        allres[f"{bg}_{seed}"] = {K: dict(routes=score(R, v[:, 2], v[:, 3], w, r["swap_at"], r["novel_at"]), **rm[K]) for K, v in out.items()}
    md = ["# FF-confirmed one-shot remap (`ff_remap.py`)\n", __doc__, VDOC if "v" in "".join(Ks) else ""]
    for bg, seeds, name in ((0.02, [3, 4, 5], "held-out seeds 3-5"), (0.02, [0, 1, 2], "seeds 0-2"), (0.2, range(6), "seeds 0-5 (all held out)")):
        keys = [f"{bg}_{s}" for s in seeds]
        md += [f"\n## bg {bg}, {name}, mean ± std %\n", "| Store | Route | Lifetime | Returns | Swap first 300 | Store answers | Remaps (pre-swap) |", "|---|---|---|---|---|---|---|"]
        for K in Ks:
            for g in ("store", "mq30_s2"):
                m = lambda k: f"{np.nanmean([allres[s][K]['routes'][g][k] for s in keys])*100:.1f} ± {np.nanstd([allres[s][K]['routes'][g][k] for s in keys])*100:.1f}"
                md.append(f"| {K} | {g} | {m('lifetime')} | {m('returns')} | {m('swap')} | {m('fired')} | {np.mean([allres[s][K]['n'] for s in keys]):.0f} ({np.mean([allres[s][K]['pre_swap'] for s in keys]):.0f}) |")
        md.append("\nFirst post-swap remap latencies (ticks): " + "; ".join(f"{K}: " + ", ".join(str(allres[s][K]['post_lat'][:2]) for s in keys) for K in Ks))
    json.dump(allres, open(os.path.join(out_dir, "results.json"), "w"))
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
