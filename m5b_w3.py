"""M5b-W3: life length on the W3 ring. Imports m5_w3 and does not edit it.

Adds W9, the frozen-compressor score over all 9 room channels. No verdict.
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

import m5_w3

SEEDS = (12, 13, 14, 15, 16, 17)
CONTROLLERS = ("rand_dwell", "oracle7", "prog")
LENGTHS = (512, 1024, 2048)
# Fresh N continuation for W9. Not the life rng (1_000 + seed), so every
# controller on a seed is scored on the same noise sequence.
N_EVAL_BASE = 30_000
# m5_w3 scales L=64 prog learn time by R_RES/176 because that life is still
# the first pass: the reservoir length averaged over an 8-dwell visit is 176.
LOCAL_WALL_LIMIT_S = 30 * 60
ROOT = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(ROOT, "results", "m5w3", "results.json")

_orig_finish = None


def score_nz(comp, hist, seed):
    """Held-out bits/symbol of N and Z on the frozen compressor. Does not learn.

    Z is EVAL copies of the space symbol, the same glance curio uses for the
    dark room. N is EVAL uniform symbols. The 7 learnable channels are not
    touched here; their rates stay the ones evaluate() already returned.
    """
    curio = m5_w3.curio
    space = curio.ALPHA.index(" ")
    z_seq = [space] * curio.EVAL
    rng = m5_w3.np.random.default_rng(N_EVAL_BASE + int(seed))
    n_seq = [int(v) for v in rng.integers(0, curio.A, curio.EVAL)]
    return {
        "N": comp.score("N", hist.get("N", []), n_seq) / curio.EVAL,
        "Z": comp.score("Z", hist.get("Z", []), z_seq) / curio.EVAL,
    }


def _finish_with_w9(seed, controller, L, comp, world, hist, channels, t0, eval_on, visits):
    """m5_w3._finish, then W9 on the same frozen compressor."""
    row = _orig_finish(seed, controller, L, comp, world, hist, channels, t0, eval_on, visits)
    curio = m5_w3.curio
    if not eval_on:
        row["W9"] = float("nan")
        row["held9"] = {}
        return row
    e0 = time.perf_counter()
    extra = score_nz(comp, hist, seed)
    dt = time.perf_counter() - e0
    held9 = {c: float(row["held"][c]) for c in curio.LEARN}
    held9["N"] = float(extra["N"])
    held9["Z"] = float(extra["Z"])
    w9 = float(m5_w3.np.mean([held9[c] for c in curio.CH]))
    row["held9"] = held9
    row["W9"] = w9
    row["eval_s"] = float(row["eval_s"] + dt)
    row["wall_s"] = float(row["wall_s"] + dt)
    row["learn_s"] = float(row["wall_s"] - row["eval_s"])
    row["finite"] = bool(
        row["finite"]
        and math.isfinite(w9)
        and all(math.isfinite(held9[c]) for c in curio.CH)
    )
    return row


def install():
    global _orig_finish
    if _orig_finish is None:
        _orig_finish = m5_w3._finish
        m5_w3._finish = _finish_with_w9


def _check_row(row, tag):
    curio = m5_w3.curio
    if set(row["held9"]) != set(curio.CH):
        raise SystemExit(f"{tag}: held9 channels are {sorted(row['held9'])}")
    w = float(m5_w3.np.mean([row["held"][c] for c in curio.LEARN]))
    w9 = float(m5_w3.np.mean([row["held9"][c] for c in curio.CH]))
    if abs(w - row["W"]) > 1e-12:
        raise SystemExit(f"{tag}: W {row['W']} is not the mean of the 7 held channels ({w})")
    if abs(w9 - row["W9"]) > 1e-12:
        raise SystemExit(f"{tag}: W9 {row['W9']} is not the mean of the 9 held channels ({w9})")
    for c in curio.LEARN:
        if abs(row["held9"][c] - row["held"][c]) > 1e-12:
            raise SystemExit(f"{tag}: held9[{c}] drifted from the W held-out rate")
    total = sum(row["share"].values()) + row["corridor_share"]
    if abs(total - 1.0) > 1e-12:
        raise SystemExit(f"{tag}: shares sum to {total}")


def _ref_prog_w():
    payload = json.load(open(REF_PATH))
    hit = [r for r in payload["runs"] if r["controller"] == "prog" and r["seed"] == 6 and r["L"] == 512]
    if len(hit) != 1:
        raise SystemExit(f"expected one M5-W3 prog seed 6 L=512 row, got {len(hit)}")
    return float(hit[0]["W"])


def _estimate(timed):
    by = {r["controller"]: r for r in timed}
    prog_scale = m5_w3.curio.R_RES / 176.0
    raw = 0.0
    conservative = 0.0
    n_jobs = 0
    for controller in CONTROLLERS:
        row = by[controller]
        for L in LENGTHS:
            learn = row["learn_s"] / 64.0 * L
            one = learn + row["eval_s"]
            for _seed in SEEDS:
                n_jobs += 1
                raw += one
                if controller == "prog":
                    conservative += learn * prog_scale + row["eval_s"]
                else:
                    conservative += one
    return {
        "serial_raw_s": raw,
        "serial_conservative_s": conservative,
        "pool2_raw_s": raw / 2.0,
        "pool2_conservative_s": conservative / 2.0,
        "prog_reservoir_scale": prog_scale,
        "n_jobs": n_jobs,
        "limit_s": LOCAL_WALL_LIMIT_S,
    }


def smoke():
    """Two checks from the spec. Exit before any scored seed if one fails."""
    install()
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    say("smoke check 1: prog seed 6 L=512 vs results/m5w3/results.json")
    ref = _ref_prog_w()
    repro = m5_w3.run_controller(6, "prog", 512, score=True, trace_visits=0)
    dW = abs(repro["W"] - ref)
    say(
        f'  prog W {repro["W"]:.12f} ref {ref:.12f} abs {dW:.3e} '
        f'W9 {repro["W9"]:.6f} finite {repro["finite"]} wall {repro["wall_s"]:.2f}s'
    )
    _check_row(repro, "smoke check 1")
    if dW > 1e-9:
        raise SystemExit(f"smoke check 1 failed: abs W {dW:.3e} > 1e-9")
    if not repro["finite"]:
        raise SystemExit("smoke check 1 failed: prog W or W9 is not finite")

    say("smoke check 2: seed 12 L=64, rand_dwell oracle7 prog, W and W9 finite")
    timed = []
    for controller in CONTROLLERS:
        row = m5_w3.run_controller(12, controller, 64, score=True, trace_visits=0)
        say(
            f'  {controller} W {row["W"]:.6f} W9 {row["W9"]:.6f} finite {row["finite"]} '
            f'wall {row["wall_s"]:.2f}s learn {row["learn_s"]:.2f}s eval {row["eval_s"]:.2f}s '
            f'N {row["share"]["N"]:.3f} Z {row["share"]["Z"]:.3f} corridor {row["corridor_share"]:.3f}'
        )
        _check_row(row, f"smoke check 2 {controller}")
        if not row["finite"] or not math.isfinite(row["W"]) or not math.isfinite(row["W9"]):
            raise SystemExit(f"smoke check 2 failed: {controller} W or W9 is not finite")
        timed.append(row)

    est = _estimate(timed)
    say(
        f'estimate serial raw {est["serial_raw_s"]:.0f}s conservative {est["serial_conservative_s"]:.0f}s '
        f'pool2 raw {est["pool2_raw_s"]:.0f}s ({est["pool2_raw_s"] / 60:.1f} min) '
        f'conservative {est["pool2_conservative_s"]:.0f}s ({est["pool2_conservative_s"] / 60:.1f} min)'
    )
    over = est["pool2_conservative_s"] > LOCAL_WALL_LIMIT_S
    if over:
        say("estimate exceeds 30 min; do not start the scored run here")
    else:
        say("estimate is within 30 min; scored run stays on this machine, pool of 2")
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "reproduce": {
            "seed": 6,
            "L": 512,
            "controller": "prog",
            "W": repro["W"],
            "W_ref": ref,
            "abs_W": dW,
            "W9": repro["W9"],
            "wall_s": repro["wall_s"],
            "finite": repro["finite"],
        },
        "L64": [{
            "controller": r["controller"],
            "seed": r["seed"],
            "L": r["L"],
            "W": r["W"],
            "W9": r["W9"],
            "finite": r["finite"],
            "wall_s": r["wall_s"],
            "learn_s": r["learn_s"],
            "eval_s": r["eval_s"],
            "share_N": r["share"]["N"],
            "share_Z": r["share"]["Z"],
            "corridor_share": r["corridor_share"],
        } for r in timed],
        "estimate": est,
        "over_limit": over,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "n_eval_rng_base": N_EVAL_BASE,
    }


def _pick(runs, controller, L, seed):
    hit = [r for r in runs if r["controller"] == controller and r["L"] == L and r["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} L={L} seed {seed}, got {len(hit)}")
    return hit[0]


def _summarize(runs):
    curio = m5_w3.curio
    by = {}
    for L in LENGTHS:
        for controller in CONTROLLERS:
            rows = sorted(
                (r for r in runs if r["controller"] == controller and r["L"] == L),
                key=lambda r: r["seed"],
            )
            if len(rows) != len(SEEDS):
                raise RuntimeError(f"{controller} L={L}: {len(rows)} rows")
            w_mu, w_sd = m5_w3.mean_std(r["W"] for r in rows)
            w9_mu, w9_sd = m5_w3.mean_std(r["W9"] for r in rows)
            held_mu, held_sd = {}, {}
            for c in curio.LEARN:
                held_mu[c], held_sd[c] = m5_w3.mean_std(r["held"][c] for r in rows)
            held9_mu, held9_sd = {}, {}
            for c in curio.CH:
                held9_mu[c], held9_sd[c] = m5_w3.mean_std(r["held9"][c] for r in rows)
            share_mu = {c: m5_w3.mean_std(r["share"][c] for r in rows)[0] for c in curio.CH}
            corr_mu = m5_w3.mean_std(r["corridor_share"] for r in rows)[0]
            quarter = []
            for i in range(4):
                q_share = {c: m5_w3.mean_std(r["quarter"][i]["share"][c] for r in rows)[0] for c in curio.CH}
                q_corr = m5_w3.mean_std(r["quarter"][i]["corridor"] for r in rows)[0]
                quarter.append({"share": q_share, "corridor": q_corr})
            by[f"{controller}/L{L}"] = {
                "controller": controller,
                "L": L,
                "n": len(rows),
                "W_mean": w_mu,
                "W_std_ddof0": w_sd,
                "W9_mean": w9_mu,
                "W9_std_ddof0": w9_sd,
                "held_mean": held_mu,
                "held_std_ddof0": held_sd,
                "held9_mean": held9_mu,
                "held9_std_ddof0": held9_sd,
                "share_mean": share_mu,
                "corridor_mean": corr_mu,
                "quarter_mean": quarter,
                "per_seed": [{
                    "seed": r["seed"],
                    "W": r["W"],
                    "W9": r["W9"],
                    "held": r["held"],
                    "held9": r["held9"],
                    "share": r["share"],
                    "corridor_share": r["corridor_share"],
                    "room_channel": r["room_channel"],
                    "wall_s": r["wall_s"],
                } for r in rows],
            }
    gaps = []
    for L in LENGTHS:
        for s in SEEDS:
            prog = _pick(runs, "prog", L, s)
            dwell = _pick(runs, "rand_dwell", L, s)
            oracle = _pick(runs, "oracle7", L, s)
            gaps.append({
                "L": L,
                "seed": s,
                "prog": prog["W"],
                "rand_dwell": dwell["W"],
                "oracle7": oracle["W"],
                "prog_minus_rand_dwell": prog["W"] - dwell["W"],
                "prog_minus_oracle7": prog["W"] - oracle["W"],
            })
    return {"by_controller": by, "gaps": gaps}


def _job(args):
    seed, controller, L = args
    out = m5_w3.run_controller(seed, controller, L, score=True, trace_visits=0)
    print(
        f's{seed} {controller} L={L} W {out["W"]:.4f} W9 {out["W9"]:.4f} '
        f'N {out["share"]["N"]:.3f} Z {out["share"]["Z"]:.3f} '
        f'corridor {out["corridor_share"]:.3f} wall {out["wall_s"]:.1f}s',
        flush=True,
    )
    out["visits"] = None
    return out


def _fmt(x):
    return f"{x:.6f}"


def _pm(mu, sd):
    return f"{_fmt(mu)} ± {_fmt(sd)}"


def _seed_header():
    return " | ".join(f"seed {s}" for s in SEEDS)


def render_markdown(payload):
    curio = m5_w3.curio
    by = payload["summary"]["by_controller"]
    lines = []
    lines.append("# M5b-W3 life length")
    lines.append("")
    lines.append(
        "Visit controllers on the W3 ring at three life lengths. "
        "W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. "
        "W9 is that mean over all 9 room channels (the 7, plus N and Z). Lower is better. Numbers only."
    )
    lines.append("")
    for title, key_mu, key_sd, field in (
        ("W", "W_mean", "W_std_ddof0", "W"),
        ("W9", "W9_mean", "W9_std_ddof0", "W9"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for L in LENGTHS:
            lines.append(f"### L = {L}")
            lines.append("")
            lines.append(f'| controller | {_seed_header()} | mean ± std ddof=0 |')
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for controller in CONTROLLERS:
                block = by[f"{controller}/L{L}"]
                cells = []
                for s in SEEDS:
                    hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                    cells.append(_fmt(hit[field]))
                lines.append(
                    f'| {controller} | ' + " | ".join(cells)
                    + f' | {_pm(block[key_mu], block[key_sd])} |'
                )
            lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("prog minus the other controller, on W, per seed and L.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | prog − rand_dwell | prog − oracle7 |")
        lines.append("|---:|---:|---:|")
        gaps = [g for g in payload["summary"]["gaps"] if g["L"] == L]
        for g in gaps:
            lines.append(
                f'| {g["seed"]} | {_fmt(g["prog_minus_rand_dwell"])} | {_fmt(g["prog_minus_oracle7"])} |'
            )
        mu_d, sd_d = m5_w3.mean_std(g["prog_minus_rand_dwell"] for g in gaps)
        mu_o, sd_o = m5_w3.mean_std(g["prog_minus_oracle7"] for g in gaps)
        n_d = sum(g["prog_minus_rand_dwell"] < 0 for g in gaps)
        n_o = sum(g["prog_minus_oracle7"] < 0 for g in gaps)
        lines.append(f'| mean | {_fmt(mu_d)} | {_fmt(mu_o)} |')
        lines.append(f'| std ddof=0 | {_fmt(sd_d)} | {_fmt(sd_o)} |')
        lines.append(f'| prog lower (count) | {n_d} of 6 | {n_o} of 6 |')
        lines.append("")
    lines.append("## Decision shares")
    lines.append("")
    lines.append(
        "Mean over seeds. A stay counts for the room's channel. A move counts for the corridor. Shares sum to 1."
    )
    lines.append("")
    header = "| controller | L | " + " | ".join(list(curio.CH) + ["corridor"]) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * 10)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [f'{block["share_mean"][c]:.4f}' for c in curio.CH]
            cells.append(f'{block["corridor_mean"]:.4f}')
            lines.append(f'| {controller} | {L} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## prog shares by quarter of life")
    lines.append("")
    lines.append("Mean over seeds. Quarter i is decisions [(i−1)L/4, iL/4).")
    lines.append("")
    lines.append("| L | quarter | " + " | ".join(list(curio.CH) + ["corridor"]) + " |")
    lines.append("|---:|---:|" + "---:|" * 10)
    for L in LENGTHS:
        block = by[f"prog/L{L}"]
        for i, q in enumerate(block["quarter_mean"], 1):
            cells = [f'{q["share"][c]:.4f}' for c in curio.CH]
            cells.append(f'{q["corridor"]:.4f}')
            lines.append(f'| {L} | {i} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Mean held-out bits")
    lines.append("")
    lines.append("Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.")
    lines.append("")
    lines.append("| controller | L | " + " | ".join(list(curio.CH)) + " | W | W9 |")
    lines.append("|---|---:|" + "---:|" * 11)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [_fmt(block["held9_mean"][c]) for c in curio.CH]
            lines.append(
                f'| {controller} | {L} | ' + " | ".join(cells)
                + f' | {_fmt(block["W_mean"])} | {_fmt(block["W9_mean"])} |'
            )
    lines.append("")
    lines.append("## Ring order")
    lines.append("")
    lines.append("Channel in rooms 0..8, seeds 12–17. One order per seed; controllers share it.")
    lines.append("")
    lines.append("| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        rooms = _pick(payload["runs"], "prog", 512, s)["room_channel"]
        lines.append(f'| {s} | ' + " | ".join(rooms) + " |")
    lines.append("")
    lines.append("## Walls")
    lines.append("")
    lines.append(
        f'Scored run wall {payload["wall_s"]:.3f} s. Pool of {payload["workers"]}. '
        f'Smoke wall {payload["smoke_wall_s"]:.3f} s.'
    )
    lines.append("")
    lines.append("| controller | L | seed | wall seconds |")
    lines.append("|---|---:|---:|---:|")
    order = [(c, L) for L in LENGTHS for c in CONTROLLERS]
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
    lines.append(
        "- Controllers `rand_dwell`, `oracle7` and `prog` are `m5_w3.run_controller`. "
        "`m5_w3.py` is imported and not edited. World, corridor glances, compressor, "
        "visit length and W are the M5-W3 ones."
    )
    lines.append(
        "- W is `w3_headroom.evaluate`: mean bits/symbol of the frozen compressor on "
        "EVAL=2048 fresh symbols of the 7 learnable channels, from a copy of the H20 world, "
        "channels in `curio.LEARN` order."
    )
    lines.append(
        "- W9 is the mean of those 7 rates plus N and Z, scored on the same frozen compressor "
        "with learning off. Z is EVAL copies of the space symbol. N is EVAL uniform symbols "
        f"from `default_rng({N_EVAL_BASE} + seed)`. That stream is not the life rng, so the "
        "three controllers on a seed share one N continuation. Corridor is not in W9."
    )
    lines.append(
        "- A stay counts for the room channel. A move counts as corridor. "
        "Quarter i is decisions [(i−1)L/4, iL/4). Shares are means over seeds."
    )
    lines.append("- Seeds 12–17. L ∈ {512, 1024, 2048}. Nothing retuned.")
    lines.append("- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Pool of 2. No GPU.")
    lines.append("")
    return "\n".join(lines)


def _check_runs(runs):
    curio = m5_w3.curio
    if len(runs) != len(SEEDS) * len(CONTROLLERS) * len(LENGTHS):
        raise SystemExit(f"expected {len(SEEDS) * len(CONTROLLERS) * len(LENGTHS)} runs, got {len(runs)}")
    for r in runs:
        _check_row(r, f's{r["seed"]} {r["controller"]} L={r["L"]}')
        if not r["finite"]:
            raise SystemExit(f's{r["seed"]} {r["controller"]} L={r["L"]} is not finite')
        if r["seed"] not in SEEDS or r["controller"] not in CONTROLLERS or r["L"] not in LENGTHS:
            raise SystemExit(f'unexpected run s{r["seed"]} {r["controller"]} L={r["L"]}')
        if abs(float(m5_w3.np.mean([r["held9"][c] for c in curio.CH])) - r["W9"]) > 1e-12:
            raise SystemExit("W9 check failed after the pool")


def scored_jobs():
    """Long prog lives first, so the pool of 2 does not finish on a short tail."""
    jobs = []
    for L in (2048, 1024, 512):
        for seed in SEEDS:
            jobs.append((seed, "prog", L))
    for L in (2048, 1024, 512):
        for controller in ("oracle7", "rand_dwell"):
            for seed in SEEDS:
                jobs.append((seed, controller, L))
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
    _check_runs(runs)
    summary = _summarize(runs)
    payload = {
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "controllers": list(CONTROLLERS),
        "workers": workers,
        "wall_s": wall,
        "smoke_wall_s": smoke_wall_s,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "n_eval_rng_base": N_EVAL_BASE,
        "runs": runs,
        "summary": summary,
        "smoke": {
            k: smoke_payload[k]
            for k in ("reproduce", "L64", "estimate", "over_limit", "pythonhashseed", "curio_lr", "n_eval_rng_base")
        },
        "smoke_text": smoke_payload["text"],
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    md_path = os.path.join(out_dir, "RESULTS.md")
    json.dump(payload, open(json_path, "w"), indent=1)
    open(json_path, "a").write("\n")
    open(md_path, "w").write(render_markdown(payload))
    print(f"wrote {md_path} and {json_path} wall {wall:.1f}s", flush=True)
    for L in LENGTHS:
        gaps = [g for g in summary["gaps"] if g["L"] == L]
        mu_d, sd_d = m5_w3.mean_std(g["prog_minus_rand_dwell"] for g in gaps)
        mu_o, sd_o = m5_w3.mean_std(g["prog_minus_oracle7"] for g in gaps)
        print(
            f'L={L} mean prog-dwell {_fmt(mu_d)} ± {_fmt(sd_d)} '
            f'mean prog-oracle7 {_fmt(mu_o)} ± {_fmt(sd_o)}',
            flush=True,
        )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default="results/m5bw3")
    a = ap.parse_args()
    if a.workers != 2:
        raise SystemExit("pool is 2 on this machine")
    do_smoke = a.smoke or not a.run
    do_run = a.run or not a.smoke
    if a.smoke and not a.run:
        do_run = False
    if a.run and not a.smoke:
        do_smoke = True
    install()
    t_smoke = time.perf_counter()
    result = smoke()
    smoke_wall = time.perf_counter() - t_smoke
    print(f"smoke wall {smoke_wall:.3f}s", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(
        {**result, "smoke_wall_s": smoke_wall},
        open(os.path.join(a.out_dir, "smoke.json"), "w"),
        indent=1,
    )
    if not do_run:
        return
    if result["over_limit"]:
        raise SystemExit("smoke estimate exceeds 30 min; scored run was not started")
    run_scored(a.workers, result, smoke_wall, a.out_dir)


if __name__ == "__main__":
    main()
