"""H27 score kernel C. Private, CPU, no internet. Seeds 19 and 20, T = 512."""
import os
import sys
from pathlib import Path

SEEDS = [19, 20]
GLANCES = 512
PHASE = "score"


def code_dir():
    hits = sorted(Path("/kaggle/input").rglob("h27_bounded_replay.py"))
    if not hits:
        raise SystemExit("h27_bounded_replay.py not found under /kaggle/input")
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
    import h27_bounded_replay
    h27_bounded_replay.kernel_main(seeds=SEEDS, glances=GLANCES, phase=PHASE,
                                   out="/kaggle/working", processes=2)


if __name__ == "__main__":
    main()
