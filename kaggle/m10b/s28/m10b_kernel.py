"""M10b-W6 seed 28. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M10B_KERNEL") != "1":
    os.environ["_M10B_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m10b_w6.py"))
    if not hits:
        raise SystemExit("m10b_w6.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m10b_w6
    m10b_w6.kernel_main(
        seed=28,
        out="/kaggle/working",
        workers=4,
        kernel_id="ezzzzzx/m10b-w6-s28",
        jobs=(("rand_dwell_s", 128), ("prog_age_s", 128), ("fwd_own_s", 128), ("fwd_inter_s", 128), ("rand_dwell_s", 512), ("prog_age_s", 512), ("fwd_own_s", 512), ("fwd_inter_s", 512),),
    )


if __name__ == "__main__":
    main()
