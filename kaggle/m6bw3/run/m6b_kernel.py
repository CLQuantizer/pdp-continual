"""M6b-W3 scored lives, seeds 24-29. Private CPU kernel, no internet."""
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M6B_READY") != "1":
    os.environ["_M6B_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m6b_w3.py"))
    if not hits:
        raise SystemExit("m6b_w3.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m6b_w3
    kernel_id = "ezzzzzx/m6b-w3"
    out = "/kaggle/working"
    t0 = time.perf_counter()
    for seed in m6b_w3.SEEDS:
        m6b_w3.kernel_main(seed=seed, out=out, workers=4, kernel_id=kernel_id)
    wall = time.perf_counter() - t0
    summary = {"kernel_id": kernel_id, "seeds": list(m6b_w3.SEEDS), "wall_s": wall}
    path = Path(out) / "m6b_kernel.json"
    path.write_text(__import__("json").dumps(summary, indent=1) + "\n")
    print(f"kernel {kernel_id} wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
