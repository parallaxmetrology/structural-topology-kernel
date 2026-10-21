"""
Public API around the frozen kernel (`_stk50.py`): image loading, single-image and
folder measurement. Numbers are pinned by tests/test_golden.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from . import _stk50

COORDINATES: Tuple[str, ...] = tuple(_stk50.KERNEL_COORDS)
VERSION = "stk-50-v1"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def prepare_micrograph(image: ImageLike, size: int = 512, crop_info_bar: bool = True,
                       bar_threshold: int = 40, bar_search: int = 80) -> np.ndarray:
    """Grayscale field in [0, 1], as in the published benchmarks.

    Converts to 8-bit grayscale, removes a dark information/scale bar at the bottom
    (rows whose mean is below `bar_threshold` within the last `bar_search` rows),
    resizes to `size` x `size` with Lanczos, and min-max normalizes.
    """
    if isinstance(image, np.ndarray):
        a = image if image.ndim == 2 else image.mean(axis=2)
        a = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    else:
        im = image if isinstance(image, Image.Image) else Image.open(image)
        a = np.array(im.convert("L"))
    H = a.shape[0]
    crop = H
    if crop_info_bar:
        rm = a.mean(1)
        for i in range(H - 1, max(H - bar_search, 0), -1):
            if rm[i] < bar_threshold:
                crop = i
    o = np.array(Image.fromarray(a[:crop, :]).resize((size, size), Image.LANCZOS)).astype(np.float32)
    return (o - o.min()) / (o.max() - o.min() + 1e-9)


def measure_field(field: np.ndarray) -> Dict[str, float]:
    """The 50 coordinates of a 2-D field (normalized to [0, 1] internally). Use this for
    binary masks, EBSD-derived maps or anything already preprocessed."""
    out = _stk50.compute_full_kernel(np.asarray(field, dtype=np.float64))
    return {k: out[k] for k in COORDINATES}


def measure(image: ImageLike, size: int = 512, crop_info_bar: bool = True) -> Dict[str, float]:
    """The 50 coordinates of a micrograph file or image, prepared as in the benchmarks."""
    return measure_field(prepare_micrograph(image, size=size, crop_info_bar=crop_info_bar))


def find_images(*paths: Union[str, os.PathLike]) -> List[Tuple[Path, str]]:
    found = {}
    for p in map(Path, paths):
        if p.is_dir():
            for q in p.rglob("*"):
                rel = q.relative_to(p)
                if q.suffix.lower() in IMAGE_EXTS and not any(s.startswith(".") for s in rel.parts):
                    found.setdefault(q, rel.as_posix())
        elif p.suffix.lower() in IMAGE_EXTS:
            found.setdefault(p, p.name)
    return sorted(found.items())


def measure_many(paths: Union[str, os.PathLike, Iterable], progress: bool = False, **kw) -> pd.DataFrame:
    """One row per image: `relpath` plus the 50 coordinates. Unreadable files get an
    `error` instead of stopping the run."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = find_images(*paths)
    rows = []
    for i, (p, rel) in enumerate(files, 1):
        if progress and (i % 50 == 0 or i == len(files)):
            print(f"[{i}/{len(files)}]", flush=True)
        try:
            rows.append({"relpath": rel, **measure(p, **kw)})
        except Exception as e:
            rows.append({"relpath": rel, "error": f"{type(e).__name__}: {str(e).replace(str(p), rel)}"})
    return pd.DataFrame(rows)
