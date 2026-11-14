"""W5 seed 2. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_W5_KERNEL") != "1":
    os.environ["_W5_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("w5_headroom.py"))
    if not hits:
        raise SystemExit("w5_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import w5_headroom
    w5_headroom.kernel_main(
        seed=2,
        out="/kaggle/working",
        workers=4,
        kernel_id="ezzzzzx/w5-s2",
        jobs=(("rand_dwell_s", 128), ("oracle14_s", 128), ("oracle_fresh_s", 128), ("rand_dwell_s", 512), ("oracle14_s", 512), ("oracle_fresh_s", 512), ("rand_dwell_s", None),),
    )


if __name__ == "__main__":
    main()
