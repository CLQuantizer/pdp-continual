"""MNIST / Fashion-MNIST loader: downloads the original IDX files once, caches them as .npz."""
import gzip
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MIRRORS = {
    "mnist": ["https://ossci-datasets.s3.amazonaws.com/mnist/",
              "https://storage.googleapis.com/cvdf-datasets/mnist/"],
    # Zalando's official Fashion-MNIST files (same IDX format and names)
    "fashion": ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"],
}
# which dataset load_mnist() returns by default (dreams.py --data sets it)
DEFAULT = os.environ.get("PDP_DATA", "mnist")
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _download(name, dataset):
    sub = DATA_DIR if dataset == "mnist" else os.path.join(DATA_DIR, dataset)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, name)
    if os.path.exists(path):
        return path
    for base in MIRRORS[dataset]:
        try:
            urllib.request.urlretrieve(base + name, path)
            return path
        except Exception as e:  # try the next mirror
            print(f"  {base}{name} failed: {e}")
    raise RuntimeError(f"could not download {name}")


def _parse(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    if magic == 2051:  # images
        n, r, c = (int.from_bytes(raw[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
    if magic == 2049:  # labels
        return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    raise ValueError(f"bad magic {magic} in {path}")


def load_mnist(dataset=None):
    """Returns x_train (60000,784), y_train, x_test (10000,784), y_test. Pixels in [0,1].
    dataset: "mnist" or "fashion" (default: $PDP_DATA, else mnist)."""
    dataset = dataset or DEFAULT
    cache = os.path.join(DATA_DIR, "mnist.npz" if dataset == "mnist" else f"{dataset}.npz")
    if not os.path.exists(cache):
        os.makedirs(DATA_DIR, exist_ok=True)
        arrays = {k: _parse(_download(v, dataset)) for k, v in FILES.items()}
        np.savez_compressed(cache, **arrays)
    d = np.load(cache)
    f = lambda x: x.astype(np.float32) / 255.0
    return f(d["x_train"]), d["y_train"], f(d["x_test"]), d["y_test"]


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_mnist()
    print(xtr.shape, ytr.shape, xte.shape, yte.shape, xtr.min(), xtr.max(), np.bincount(ytr))
