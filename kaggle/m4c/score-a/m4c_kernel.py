"""M4c score kernel A. Private, CPU, no internet. Seeds 9 and 10, T = 512."""
import os
import sys
from pathlib import Path

SEEDS = [9, 10]
GLANCES = 512
PHASE = "score"


def code_dir():
    hits = sorted(Path("/kaggle/input").rglob("m4c_score.py"))
    if not hits:
        raise SystemExit("m4c_score.py not found under /kaggle/input")
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
    import m4c_score
    m4c_score.kernel_main(seeds=SEEDS, glances=GLANCES, phase=PHASE,
                          out="/kaggle/working", processes=2)


if __name__ == "__main__":
    main()
