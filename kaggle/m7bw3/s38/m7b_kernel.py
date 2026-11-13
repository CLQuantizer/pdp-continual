"""M7b-W3 seed 38. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M7B_READY") != "1":
    os.environ["_M7B_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m7b_w3.py"))
    if not hits:
        raise SystemExit("m7b_w3.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m7b_w3
    m7b_w3.kernel_main(
        seed=38,
        out="/kaggle/working",
        workers=4,
        kernel_id="ezzzzzx/m7b-w3-s38",
    )


if __name__ == "__main__":
    main()
