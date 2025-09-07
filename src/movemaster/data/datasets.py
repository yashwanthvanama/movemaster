import glob
import os
from typing import Iterator, Tuple
import numpy as np


def iter_npz_shards(data_dir: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for p in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        data = np.load(p)
        yield data["features"], data["labels"]
