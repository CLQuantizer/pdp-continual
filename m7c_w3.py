"""M7c-W3: longer lives of per-channel curiosity on the bank, fixed sleep.

Imports m7b_w3 and does not modify it. Numbers only. No verdict.
"""
import os
import sys

if __name__ == "__main__" and os.environ.get("_M7C_MAIN") != "1":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CURIO_LR"] = "0.004"
    os.environ["_M7C_MAIN"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import m7b_w3 as m7b

SEEDS = tuple(range(42, 54))
CONTROLLERS = ("rand_dwell_s", "oracle7_s", "prog_chan_s")
LIVES = (2048, 4096)
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "curio.py",
    "h20_world.py",
    "h30w_ring.py",
    "h30x_trace.py",
    "h30y_roots.py",
    "m7_w3.py",
    "m7b_w3.py",
    "m7c_w3.py",
    "w3_headroom.py",
    "w3_world.py",
)
# M7b rand_dwell learn_s, seeds 36-41. Slowest over fastest.
M7B_LEARN_SLOWDOWN = 105.110 / 38.169


def _threads():
    m7b._threads()


def _job(args):
    seed, controller, L = args
    _threads()
    if controller not in CONTROLLERS:
        raise ValueError(controller)
    dose = m7b.sleep_decisions(L)
    print(f"s{seed} {controller} L={L} start dose {dose}", flush=True)
    row = m7b.run_controller(seed, controller, L, score=True, check_sleep=True)
    row.pop("visits", None)
    print(
        f's{seed} {controller} L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
        f'sleep {row["sleep_share"]:.3f} corridor {row["corridor_share"]:.3f} '
        f'wall {row["wall_s"]:.1f}s learn {row["learn_s"]:.1f}s score {row["score_s"]:.1f}s',
        flush=True,
    )
    return row


def _life_body(controller, L, learn_per, full_s, channel_s):
    """In-loop seconds. Exam rescores are charged at a full reservoir."""
    n_sleep = m7b.sleep_decisions(L)
    waking = L - n_sleep
    visits = n_sleep / m7b.m7.DWELL
    if controller == "prog_chan_s":
        return (
            waking * (2.0 * channel_s + learn_per)
            + n_sleep * (full_s + learn_per)
            + visits * full_s
        )
    if controller in ("rand_dwell_s", "oracle7_s"):
        return waking * learn_per + n_sleep * (full_s + learn_per) + visits * full_s
    raise ValueError(controller)


def _kernel_wall(job_seconds, workers):
    """Longest-first assignment onto `workers` cores. Returns the makespan."""
    load = [0.0] * int(workers)
    for cost in sorted(job_seconds, reverse=True):
        slot = min(range(len(load)), key=lambda i: load[i])
        load[slot] += float(cost)
    return float(max(load))


def kernel_estimate(learn_per, full_s, channel_s, eval_s, measured_512_wall):
    """Wall of one seed: 3 controllers x 2 lives, pool of 4.

    The 512-decision prog_chan_s life calibrates the full-store model. A linear
    scaling of that life is kept beside it. The reported kernel wall is the
    larger of the two. A slow-seed figure stretches the learn term by the M7b
    max/min learn ratio and is not the split threshold.
    """
    predicted_512 = _life_body("prog_chan_s", 512, learn_per, full_s, channel_s) + eval_s
    if predicted_512 <= 0.0:
        raise RuntimeError("512-life prediction is not positive")
    scale = max(1.0, float(measured_512_wall) / predicted_512)

    def jobs(learn, extra):
        costs = []
        detail = []
        for L in LIVES:
            for controller in CONTROLLERS:
                cost = scale * (_life_body(controller, L, learn, full_s, channel_s) + extra)
                costs.append(cost)
                detail.append({"controller": controller, "L": L, "seconds": float(cost)})
        return costs, detail

    costs, detail = jobs(learn_per, eval_s)
    linear = []
    for L in LIVES:
        stretched = float(measured_512_wall) * (L / 512.0)
        for _controller in CONTROLLERS:
            linear.append(stretched)
    slow_learn = learn_per * M7B_LEARN_SLOWDOWN
    slow_costs, _slow_detail = jobs(slow_learn, eval_s)
    combined = _kernel_wall(costs, WORKERS)
    linear_wall = _kernel_wall(linear, WORKERS)
    per_life = {}
    for L in LIVES:
        life_costs = [item["seconds"] for item in detail if item["L"] == L]
        per_life[str(L)] = float(_kernel_wall(life_costs, WORKERS))
    # The split decision is the calibrated full-store makespan. Linear and
    # slow-seed figures are logged beside it.
    return {
        "learn_per_decision_s": float(learn_per),
        "full_rescore_s": float(full_s),
        "channel_rescore_s": float(channel_s),
        "eval_s": float(eval_s),
        "measured_prog_chan_s_512_wall_s": float(measured_512_wall),
        "predicted_prog_chan_s_512_s": float(predicted_512),
        "scale": float(scale),
        "jobs": detail,
        "model_kernel_s": float(combined),
        "linear_kernel_s": float(linear_wall),
        "kernel_s": float(combined),
        "per_life_kernel_s": per_life,
        "slow_seed_kernel_s": float(_kernel_wall(slow_costs, WORKERS)),
        "m7b_learn_slowdown": float(M7B_LEARN_SLOWDOWN),
        "workers": WORKERS,
        "split": bool(combined > KERNEL_WALL_LIMIT_S),
    }


def _published_l64():
    """M7b's on-disk rand_dwell_s W at seed 36, L = 64. None if it is absent."""
    path = ROOT / "results" / "m7bw3" / "smoke.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    lines = payload.get("log") or []
    header = "smoke check 4: five controllers, seed 36, L=64, W and W9 finite"
    if header not in lines:
        return None
    for line in lines[lines.index(header):]:
        marker = "rand_dwell_s W "
        if marker not in line:
            continue
        parts = line.strip().split()
        # "rand_dwell_s W <W> W9 <W9> ..."
        if len(parts) < 5 or parts[0] != "rand_dwell_s" or parts[1] != "W" or parts[3] != "W9":
            continue
        return {
            "W": float(parts[2]),
            "W9": float(parts[4]),
            "source": "results/m7bw3/smoke.json",
            "line": line.strip(),
        }
    return None


def _check_row(row, seed, controller, L):
    if row["seed"] != int(seed) or row["controller"] != controller or row["L"] != int(L):
        raise RuntimeError(f"row identity {row['seed']} {row['controller']} {row['L']}")
    if not row["finite"]:
        raise RuntimeError(f"{controller} seed {seed} L={L} W is not finite")
    if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
        raise RuntimeError(f"{controller} seed {seed} L={L} store check failed")
    dose = m7b.sleep_decisions(L)
    if row["sleep_decisions"] != dose:
        raise RuntimeError(
            f"{controller} seed {seed} L={L} sleep {row['sleep_decisions']} != dose {dose}"
        )
    share = sum(row["share"].values()) + row["corridor_share"] + row["sleep_share"]
    if abs(share - 1.0) > 1e-6:
        raise RuntimeError(f"{controller} seed {seed} L={L} shares sum to {share}")


def smoke(out_dir=None):
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "m7cw3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    t0 = time.perf_counter()
    payload = {
        "ok": False,
        "reason": None,
        "torch": m7b.torch.__version__,
        "threads": int(m7b.torch.get_num_threads()),
        "model_seed": m7b.m7.MODEL_SEED,
        "kernels_pushed": False,
        "sleep_share_target": m7b.SLEEP_SHARE,
    }
    log(
        f"torch={m7b.torch.__version__} threads={m7b.torch.get_num_threads()} "
        f"model_seed={m7b.m7.MODEL_SEED}"
    )
    try:
        for L in (64, 512, 2048, 4096):
            dose = m7b.sleep_decisions(L)
            log(f"dose L{L} {dose}/{L} share {dose / L:.6f}")

        published = _published_l64()
        log("smoke check 1: rand_dwell_s seed 36 L=64 versus M7b's W")
        got = m7b.run_controller(36, "rand_dwell_s", 64, score=True, check_sleep=False)
        if published is None:
            log("  no L=64 rand_dwell_s W on disk; rerunning m7b_w3.run_controller")
            ref = m7b.run_controller(36, "rand_dwell_s", 64, score=True, check_sleep=False)
            ref_w = float(ref["W"])
            source = "rerun m7b_w3.run_controller seed 36 L=64"
        else:
            ref_w = float(published["W"])
            source = published["source"]
            log(f"  published line: {published['line']}")
        delta = abs(float(got["W"]) - ref_w)
        ok = delta <= BITS_LIMIT
        log(
            f"  source {source} ref W {ref_w:.6f} live W {got['W']:.10f} "
            f"W9 {got['W9']:.10f} abs_dW {delta:.3e} limit {BITS_LIMIT:.1e} ok {ok}"
        )
        if not ok:
            raise RuntimeError(
                f"rand_dwell_s seed 36 L=64 W differs from M7b by {delta:.3e}"
            )
        if got["sleep_decisions"] != m7b.sleep_decisions(64):
            raise RuntimeError("rand_dwell_s seed 36 L=64 sleep dose mismatch")
        payload["check1"] = {
            "ref_W": ref_w,
            "live_W": float(got["W"]),
            "live_W9": float(got["W9"]),
            "abs_dW": float(delta),
            "source": source,
        }

        log("smoke check 2: rand_dwell_s, oracle7_s, prog_chan_s, seed 42, L=64, finite W")
        for controller in CONTROLLERS:
            row = m7b.run_controller(42, controller, 64, score=True, check_sleep=False)
            log(
                f"  {controller} W {row['W']:.6f} W9 {row['W9']:.6f} finite {row['finite']} "
                f"sleep {row['sleep_share']:.4f} decisions {row['sleep_decisions']} "
                f"corridor {row['corridor_share']:.4f} wall {row['wall_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{controller} seed 42 L=64 W is not finite")
            if row["sleep_decisions"] != m7b.sleep_decisions(64):
                raise RuntimeError(f"{controller} seed 42 L=64 sleep dose mismatch")

        log("smoke check 3: time prog_chan_s seed 42 L=512 and estimate one kernel")
        timed = m7b.run_controller(42, "prog_chan_s", 512, score=True, check_sleep=False)
        if not timed["finite"]:
            raise RuntimeError("prog_chan_s seed 42 L=512 W is not finite")
        log(
            f"  prog_chan_s seed 42 L=512 W {timed['W']:.6f} W9 {timed['W9']:.6f} "
            f"wall {timed['wall_s']:.2f}s learn {timed['learn_s']:.2f}s "
            f"score {timed['score_s']:.2f}s eval {timed['eval_s']:.2f}s "
            f"sleep {timed['sleep_decisions']}"
        )
        probe = m7b.m7.BankCortex(model_seed=m7b.m7.MODEL_SEED, freeze_channel=False)
        full_s = m7b.m7.time_full_rescore(probe.model, probe.channel_emb)
        channel_s = m7b._time_channel_rescore(probe.model, probe.channel_emb)
        learn_per = timed["learn_s"] / 512.0
        est = kernel_estimate(learn_per, full_s, channel_s, timed["eval_s"], timed["wall_s"])
        log(
            f"  learn_per_decision {learn_per:.4f}s full_rescore_288 {full_s:.4f}s "
            f"channel_rescore_32 {channel_s:.4f}s scale {est['scale']:.3f}"
        )
        log(
            f"  model kernel {est['model_kernel_s']:.1f}s linear kernel {est['linear_kernel_s']:.1f}s "
            f"kernel {est['kernel_s']:.1f}s ({est['kernel_s'] / 60:.2f} min) "
            f"slow-seed {est['slow_seed_kernel_s']:.1f}s ({est['slow_seed_kernel_s'] / 60:.2f} min) "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s split {est['split']}"
        )
        for L in LIVES:
            log(f"  one-life kernel L={L} {est['per_life_kernel_s'][str(L)]:.1f}s")
        if est["split"]:
            over = [
                L for L in LIVES
                if est["per_life_kernel_s"][str(L)] > KERNEL_WALL_LIMIT_S
            ]
            if over:
                raise RuntimeError(
                    "one life still estimates over 90 min after splitting: " + ", ".join(str(L) for L in over)
                )
            log("  estimate is over 90 min; kernels will be one life each")
        else:
            log("  estimate is under 90 min; one kernel per seed runs both lives")
        payload["estimate"] = est
        payload["timed_512"] = {
            "seed": 42,
            "controller": "prog_chan_s",
            "L": 512,
            "W": float(timed["W"]),
            "W9": float(timed["W9"]),
            "wall_s": float(timed["wall_s"]),
            "learn_s": float(timed["learn_s"]),
            "score_s": float(timed["score_s"]),
            "eval_s": float(timed["eval_s"]),
        }
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["reason"] = str(exc)
        log(f"SMOKE_FAIL {exc}")
    payload["wall_s"] = time.perf_counter() - t0
    payload["log"] = lines
    m7b.m7.dump(out_dir / "smoke.json", payload)
    (out_dir / "smoke.log").write_text("\n".join(lines) + "\n")
    if not payload["ok"]:
        print(f"SMOKE_FAIL {payload['reason']}. Kernels not pushed.", flush=True)
        raise SystemExit(2)
    print(f"smoke_ok wall {payload['wall_s']:.1f}s", flush=True)
    return payload


def _kernel_slug(seed, life):
    if life is None:
        return f"m7c-w3-s{seed}"
    return f"m7c-w3-s{seed}-l{life}"


def _write_kernel(directory, seed, lives):
    directory.mkdir(parents=True, exist_ok=True)
    slug = _kernel_slug(seed, None if len(lives) == 2 else lives[0])
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "m7c_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["ezzzzzx/pdp-code-m7cw3"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    lives_src = ", ".join(str(L) for L in lives)
    text = f'''"""M7c-W3 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M7C_READY") != "1":
    os.environ["_M7C_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m7c_w3.py"))
    if not hits:
        raise SystemExit("m7c_w3.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m7c_w3
    m7c_w3.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        lives=({lives_src},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "m7c_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and the kernel folders. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    estimate = smoke_payload["estimate"]
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "m7cw3"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code m7cw3",
        "id": "ezzzzzx/pdp-code-m7cw3",
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    split = bool(estimate["split"])
    ids = []
    for seed in SEEDS:
        if split:
            for L in LIVES:
                folder = kaggle_dir / f"s{seed}-l{L}"
                ids.append(_write_kernel(folder, seed, (L,)))
        else:
            folder = kaggle_dir / f"s{seed}"
            ids.append(_write_kernel(folder, seed, LIVES))
    print(f"staged {len(ids)} kernels split {split}", flush=True)
    for kernel_id in ids:
        print(kernel_id, flush=True)
    return ids


def kernel_main(seed, out, workers=4, kernel_id=None, lives=None):
    if __name__ != "m7c_w3":
        raise SystemExit("import m7c_w3 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 42-53")
    if int(workers) < 1 or int(workers) > 4:
        raise SystemExit("kernel pool is at most 4")
    if lives is None:
        lives = LIVES
    lives = tuple(int(L) for L in lives)
    if lives not in (LIVES, (2048,), (4096,)):
        raise SystemExit(f"lives {lives} are not an M7c kernel")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Longest jobs first. prog_chan_s rescores a channel on every waking step,
    # so it starts ahead of the two sleep controllers at the same L.
    order = [(seed, controller, L) for L in lives for controller in CONTROLLERS]
    order.sort(key=lambda item: (
        -item[2],
        0 if item[1] == "prog_chan_s" else 1,
        item[1],
    ))
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} jobs {len(order)} "
        f"lives {lives}",
        flush=True,
    )
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [pool.apply_async(_job, (job,)) for job in order]
        rows = [item.get() for item in asyncs]
    wall = time.perf_counter() - t0
    expected = len(lives) * len(CONTROLLERS)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, got {len(rows)}")
    for row in rows:
        _check_row(row, seed, row["controller"], row["L"])
    by_key = {(row["controller"], row["L"]): row for row in rows}
    if len(by_key) != expected:
        raise SystemExit("duplicate controller rows")
    ordered = [by_key[(controller, L)] for L in lives for controller in CONTROLLERS]
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "lives": list(lives),
        "workers": int(workers),
        "wall_s": wall,
        "torch": m7b.torch.__version__,
        "model_seed": m7b.m7.MODEL_SEED,
        "sleep_share_target": m7b.SLEEP_SHARE,
        "runs": ordered,
    }
    path = out / f"m7c_s{seed}.json"
    m7b.m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pm(mu, sd, digits=6):
    if mu is None:
        return "—"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def _pick(runs, controller, seed, L):
    hit = [
        row for row in runs
        if row["controller"] == controller and row["seed"] == seed and row["L"] == L
    ]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} seed {seed} L={L}, got {len(hit)}")
    return hit[0]


def _rows_for(runs, controller, L):
    rows = [row for row in runs if row["controller"] == controller and row["L"] == L]
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != list(SEEDS):
        raise RuntimeError(f"{controller} L={L} seeds {[row['seed'] for row in rows]}")
    return rows


def summarize(runs):
    curio = m7b.curio
    by_L = {}
    for L in LIVES:
        by = {}
        for controller in CONTROLLERS:
            rows = _rows_for(runs, controller, L)
            w_mu, w_sd = m7b.m7.mean_std(row["W"] for row in rows)
            w9_mu, w9_sd = m7b.m7.mean_std(row["W9"] for row in rows)
            share_mu = {
                name: m7b.m7.mean_std(row["share"][name] for row in rows)[0]
                for name in curio.CH
            }
            corr_mu = m7b.m7.mean_std(row["corridor_share"] for row in rows)[0]
            sleep_mu, sleep_sd = m7b.m7.mean_std(row["sleep_share"] for row in rows)
            forgetting = {}
            for name in curio.LEARN:
                chosen = [
                    row["forgetting"][name]
                    for row in rows
                    if row["forgetting"][name]["delta_bps"] is not None
                ]
                mu, sd = m7b.m7.mean_std(item["delta_bps"] for item in chosen)
                forgetting[name] = {
                    "n": len(chosen),
                    "delta_bps_mean": mu,
                    "delta_bps_std_ddof0": sd,
                    "after_bps_mean": m7b.m7.mean_std(item["after_bps"] for item in chosen)[0],
                    "end_bps_mean": m7b.m7.mean_std(item["end_bps"] for item in chosen)[0],
                }
            defined = [
                row["forgetting"][name]["delta_bps"]
                for row in rows
                for name in curio.LEARN
                if row["forgetting"][name]["delta_bps"] is not None
            ]
            mu, sd = m7b.m7.mean_std(defined)
            by[controller] = {
                "n": len(rows),
                "W_mean": w_mu,
                "W_std_ddof0": w_sd,
                "W9_mean": w9_mu,
                "W9_std_ddof0": w9_sd,
                "share_mean": share_mu,
                "corridor_mean": corr_mu,
                "sleep_mean": sleep_mu,
                "sleep_std_ddof0": sleep_sd,
                "forgetting_channel": forgetting,
                "forgetting_mean": mu,
                "forgetting_std_ddof0": sd,
                "forgetting_n": len(defined),
            }
        gaps = []
        for seed in SEEDS:
            chan = _pick(runs, "prog_chan_s", seed, L)
            rand_s = _pick(runs, "rand_dwell_s", seed, L)
            oracle = _pick(runs, "oracle7_s", seed, L)
            gaps.append({
                "seed": seed,
                "L": L,
                "prog_chan_s_minus_rand_dwell_s": chan["W"] - rand_s["W"],
                "oracle7_s_minus_rand_dwell_s": oracle["W"] - rand_s["W"],
                "prog_chan_s_minus_oracle7_s": chan["W"] - oracle["W"],
            })
        gap_summary = {}
        for key in (
            "prog_chan_s_minus_rand_dwell_s",
            "oracle7_s_minus_rand_dwell_s",
            "prog_chan_s_minus_oracle7_s",
        ):
            values = [row[key] for row in gaps]
            mu, sd = m7b.m7.mean_std(values)
            gap_summary[key] = {
                "mean": mu,
                "std_ddof0": sd,
                "left_lower": int(sum(value < 0.0 for value in values)),
                "n": len(values),
            }
        by_L[str(L)] = {"by_controller": by, "gaps": gaps, "gap_summary": gap_summary}
    return by_L


def _header(seeds):
    return "| controller | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean ± std ddof=0 |"


def _sep(n):
    return "|---|" + "---:|" * n + "---:|"


def render_markdown(result):
    curio = m7b.curio
    seeds = list(result["seeds"])
    runs = result["runs"]
    summary = result["summary"]
    lines = [
        "# M7c-W3 curiosity on the bank, fixed sleep, longer lives",
        "",
        "The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh "
        "continuation of the 7 learnable channels, from a zero state, with that channel's embedding. "
        "W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam "
        "bits/symbol minus the exam bits/symbol just after the last visit to that channel. "
        "Positive forgetting is a worse code. Numbers only.",
        "",
        "Controllers are `m7b_w3`'s `rand_dwell_s`, `oracle7_s` and `prog_chan_s`. "
        "Sleep dose is 8 times the integer nearest to 0.20 L / 8, in visits of 8. "
        f"At L = 2048 that is {m7b.sleep_decisions(2048)} decisions, "
        f"share {m7b.sleep_decisions(2048) / 2048:.6f}. "
        f"At L = 4096 that is {m7b.sleep_decisions(4096)} decisions, "
        f"share {m7b.sleep_decisions(4096) / 4096:.6f}. "
        "`prog_chan_s` room reward is the sum, over that room's exam glances, of bits before minus "
        "bits after, summed over the visit's decisions and divided by those decisions. Travel counts. "
        "Sleep visits are not bandit updates.",
        "",
    ]
    for L in LIVES:
        block = summary[str(L)]["by_controller"]
        lines.extend([f"## W, L = {L}", "", _header(seeds), _sep(len(seeds))])
        for controller in CONTROLLERS:
            cells = [_fmt(_pick(runs, controller, seed, L)["W"]) for seed in seeds]
            cells.append(_pm(block[controller]["W_mean"], block[controller]["W_std_ddof0"]))
            lines.append("| " + controller + " | " + " | ".join(cells) + " |")
        lines.append("")
        lines.extend([f"## W9, L = {L}", "", _header(seeds), _sep(len(seeds))])
        for controller in CONTROLLERS:
            cells = [_fmt(_pick(runs, controller, seed, L)["W9"]) for seed in seeds]
            cells.append(_pm(block[controller]["W9_mean"], block[controller]["W9_std_ddof0"]))
            lines.append("| " + controller + " | " + " | ".join(cells) + " |")
        lines.append("")
        lines.extend([
            f"## Gaps, L = {L}",
            "",
            "On W. A negative gap means the first controller is lower.",
            "",
            "| seed | prog_chan_s − rand_dwell_s | oracle7_s − rand_dwell_s | prog_chan_s − oracle7_s |",
            "|---:|---:|---:|---:|",
        ])
        for gap in summary[str(L)]["gaps"]:
            lines.append(
                f"| {gap['seed']} | {_fmt(gap['prog_chan_s_minus_rand_dwell_s'])} | "
                f"{_fmt(gap['oracle7_s_minus_rand_dwell_s'])} | "
                f"{_fmt(gap['prog_chan_s_minus_oracle7_s'])} |"
            )
        labels = (
            ("mean", "mean"),
            ("std ddof=0", "std_ddof0"),
        )
        gap_summary = summary[str(L)]["gap_summary"]
        for label, key in labels:
            cells = [
                _fmt(gap_summary[name][key])
                for name in (
                    "prog_chan_s_minus_rand_dwell_s",
                    "oracle7_s_minus_rand_dwell_s",
                    "prog_chan_s_minus_oracle7_s",
                )
            ]
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
        counts = [
            f"{gap_summary[name]['left_lower']} of {gap_summary[name]['n']}"
            for name in (
                "prog_chan_s_minus_rand_dwell_s",
                "oracle7_s_minus_rand_dwell_s",
                "prog_chan_s_minus_oracle7_s",
            )
        ]
        lines.append("| left lower (count) | " + " | ".join(counts) + " |")
        lines.append("")
        lines.extend([
            f"## Forgetting, L = {L}",
            "",
            "Exam-store bits/symbol just after the last visit that dwelled in the channel, "
            "minus nothing: the difference is end minus that snapshot. A channel with no exam "
            "glance at that visit is left out of the mean. n is the number of seed-channel pairs "
            "in the mean.",
            "",
            "| controller | mean delta bps | std ddof=0 | n |",
            "|---|---:|---:|---:|",
        ])
        for controller in CONTROLLERS:
            item = block[controller]
            lines.append(
                f"| {controller} | {_fmt(item['forgetting_mean'])} | "
                f"{_fmt(item['forgetting_std_ddof0'])} | {item['forgetting_n']} |"
            )
        lines.extend([
            "",
            "| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for controller in CONTROLLERS:
            for name in curio.LEARN:
                item = block[controller]["forgetting_channel"][name]
                lines.append(
                    f"| {controller} | {name} | {item['n']} | {_fmt(item['after_bps_mean'])} | "
                    f"{_fmt(item['end_bps_mean'])} | {_fmt(item['delta_bps_mean'])} | "
                    f"{_fmt(item['delta_bps_std_ddof0'])} |"
                )
        lines.append("")
        lines.extend([
            f"## Decision shares, L = {L}",
            "",
            "Mean over seeds. A stay counts for the room channel. A move counts for the corridor. "
            "A sleep decision counts for sleep. Shares sum to 1.",
            "",
            "| controller | " + " | ".join(curio.CH) + " | corridor | sleep |",
            "|---|" + "---:|" * (len(curio.CH) + 2),
        ])
        for controller in CONTROLLERS:
            item = block[controller]
            cells = [_fmt(item["share_mean"][name], 4) for name in curio.CH]
            cells.append(_fmt(item["corridor_mean"], 4))
            cells.append(_fmt(item["sleep_mean"], 4))
            lines.append("| " + controller + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([
        "## Walls",
        "",
        "| kernel | seed | lives | wall seconds |",
        "|---|---:|---|---:|",
    ])
    for kernel in result["kernels"]:
        lives = ", ".join(str(L) for L in kernel["lives"])
        lines.append(
            f"| {kernel['kernel_id']} | {kernel['seed']} | {lives} | {_fmt(kernel['wall_s'], 3)} |"
        )
    lines.extend([
        "",
        "| controller | L | seed | wall seconds | learn seconds | score seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in runs:
        lines.append(
            f"| {row['controller']} | {row['L']} | {row['seed']} | {_fmt(row['wall_s'], 3)} | "
            f"{_fmt(row['learn_s'], 3)} | {_fmt(row['score_s'], 3)} |"
        )
    lines.extend(["", "## Smoke test", "", "```"])
    lines.extend(result["smoke_log"])
    lines.extend(["```", "", "## Protocol", ""])
    lines.append(
        "- `BankCortex`, the stores, the sleep decision, W, W9 and forgetting are `m7_w3`'s, "
        "reached through `m7b_w3`. Model seed 0. Glance length 64. Reservoirs hold 32 glances per "
        "channel. Exam coin `default_rng(40_000 + seed)`, exam replacement `default_rng(20_000 + seed)`, "
        "replay replacement `default_rng(50_000 + seed)`, sleep draw `default_rng(30_000 + seed)`."
    )
    lines.append(
        "- `rand_dwell_s` and `oracle7_s` are `m7b_w3.run_fixed_sleep` with `w3_headroom.next_action` "
        "and `default_rng(1_000 + seed)`. A sleep visit starts only when the current plan is empty, "
        "which is the end of a visit."
    )
    lines.append(
        "- `prog_chan_s` uses the 9-room bandit (`default_rng(10_000 + seed)`, optimistic first pass, "
        "EPS 0.1, BETA 0.3). Its room reward sums bits saved on the destination room's exam glances "
        "over the visit, including corridor steps, and divides by the visit's decisions. Sleep does "
        "not call the EMA."
    )
    lines.append(
        "- The dose is 8 times the integer nearest to 0.20 L / 8. Placement is `m7_w3.sleep_plan` "
        "and `m7_w3.sleep_due`. World seeds "
        + ", ".join(str(seed) for seed in seeds)
        + ". L = 2048 and L = 4096. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, "
        "no internet. Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/m7c_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-l*/m7c_s{seed}.json"))
        if not hits:
            raise RuntimeError(f"no results file for seed {seed}")
        found[seed] = hits
    return found


def assemble(out_dir, smoke_payload=None):
    out_dir = Path(out_dir)
    if smoke_payload is None:
        smoke_payload = json.loads((out_dir / "smoke.json").read_text())
    if not smoke_payload.get("ok"):
        raise RuntimeError("smoke did not pass")
    runs = []
    kernels = []
    for seed, hits in _load_kernel_payloads(out_dir).items():
        seen_lives = []
        for path in hits:
            payload = json.loads(path.read_text())
            if payload["seed"] != seed:
                raise RuntimeError(f"{path} seed {payload['seed']} != {seed}")
            for row in payload["runs"]:
                _check_row(row, seed, row["controller"], row["L"])
                runs.append(row)
            lives = [int(L) for L in payload["lives"]]
            seen_lives.extend(lives)
            kernels.append({
                "kernel_id": payload["kernel_id"],
                "seed": seed,
                "lives": lives,
                "wall_s": payload["wall_s"],
                "workers": payload["workers"],
                "path": str(path.resolve().relative_to(ROOT)),
            })
        if sorted(seen_lives) != sorted(LIVES):
            raise RuntimeError(f"seed {seed} lives {seen_lives} != {list(LIVES)}")
    if len(runs) != len(SEEDS) * len(LIVES) * len(CONTROLLERS):
        raise RuntimeError(
            f"expected {len(SEEDS) * len(LIVES) * len(CONTROLLERS)} runs, got {len(runs)}"
        )
    summary = summarize(runs)
    kernels.sort(key=lambda item: (item["seed"], item["lives"]))
    result = {
        "seeds": list(SEEDS),
        "L": list(LIVES),
        "controllers": list(CONTROLLERS),
        "model_seed": m7b.m7.MODEL_SEED,
        "sleep_share_target": m7b.SLEEP_SHARE,
        "sleep_decisions": {str(L): m7b.sleep_decisions(L) for L in LIVES},
        "kernels": kernels,
        "summary": summary,
        "smoke": {key: smoke_payload[key] for key in smoke_payload if key != "log"},
        "smoke_log": smoke_payload["log"],
        "runs": runs,
    }
    text = render_markdown(result)
    m7b.m7.dump(out_dir / "results.json", result)
    (out_dir / "RESULTS.md").write_text(text)
    print(f"wrote {out_dir / 'RESULTS.md'} and results.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "stage", "assemble"))
    parser.add_argument("--out", default=str(ROOT / "results" / "m7cw3"))
    args = parser.parse_args()
    if args.command == "smoke":
        payload = smoke(args.out)
        stage(payload)
        return
    smoke_payload = json.loads((Path(args.out) / "smoke.json").read_text())
    if args.command == "stage":
        stage(smoke_payload)
        return
    assemble(args.out, smoke_payload)


if __name__ == "__main__":
    main()
