"""H27b smoke kernel. Private, CPU, no internet. Seed 21, T = 32, all 8 conditions."""
import os
import sys
from pathlib import Path

SEEDS = [21]
GLANCES = 32
PHASE = "smoke"


def code_dir():
    hits = sorted(Path("/kaggle/input").rglob("h27b_matched_dose.py"))
    if not hits:
        raise SystemExit("h27b_matched_dose.py not found under /kaggle/input")
    return str(hits[0].parent)


def main():
    code = code_dir()
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    print("py", sorted(p.name for p in Path(code).glob("*.py")), flush=True)
    import h27b_matched_dose
    h27b_matched_dose.kernel_main(seeds=SEEDS, glances=GLANCES, phase=PHASE,
                                  out="/kaggle/working", processes=2)


if __name__ == "__main__":
    main()
