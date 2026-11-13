"""W3 headroom, seeds 0-2. Private CPU kernel, no internet."""
import os
import sys

# One re-exec so PYTHONHASHSEED is fixed before curio's hashes are drawn.
if os.environ.get("W3_REEXEC") != "1":
    os.environ["W3_REEXEC"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CURIO_LR"] = "0.004"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    from pathlib import Path

    hits = sorted(Path("/kaggle/input").rglob("w3_headroom.py"))
    if not hits:
        raise SystemExit("w3_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CURIO_LR"] = "0.004"
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    print("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"), flush=True)
    import w3_headroom
    w3_headroom.kernel_main(
        out="/kaggle/working",
        kernel_id="ezzzzzx/w3-headroom",
        seeds=(0, 1, 2),
        workers=4,
    )


if __name__ == "__main__":
    main()
