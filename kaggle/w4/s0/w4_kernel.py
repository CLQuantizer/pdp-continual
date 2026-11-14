"""W4 seed 0. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_W4_KERNEL") != "1":
    os.environ["_W4_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("w4_headroom.py"))
    if not hits:
        raise SystemExit("w4_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import w4_headroom
    w4_headroom.kernel_main(
        seed=0,
        out="/kaggle/working",
        workers=4,
        kernel_id="ezzzzzx/w4-s0",
        jobs=(("rand_dwell_s", 2048), ("oracle14_s", 2048), ("oracle7a_s", 2048), ("oracle14_s", 8192),),
    )


if __name__ == "__main__":
    main()
