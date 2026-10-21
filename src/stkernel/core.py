"""
Public API around the frozen kernel (`_stk50.py`): image loading, single-image and
folder measurement. Numbers are pinned by tests/test_golden.py.
"""
from __future__ import annotations

import os
import warnings
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


def _to_gray_uint8(a: np.ndarray) -> np.ndarray:
    """An array as the 8-bit grayscale image PIL's convert("L") would give for the same
    picture: uint8 kept, other integer types scaled by their range, floats in [0, 1]
    scaled by 255, other floats taken as 0-255; color uses ITU-R 601 luma like PIL."""
    a = np.asarray(a)
    if a.ndim == 3:
        if a.dtype == np.uint8 and a.shape[2] in (3, 4):
            return np.array(Image.fromarray(np.ascontiguousarray(a[..., :3])).convert("L"))
        a = a[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114]) if a.shape[2] >= 3 else a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image or an H x W x 3/4 color image, got shape {a.shape}")
    if a.dtype == np.uint8:
        return a
    if a.dtype == bool:
        return a.astype(np.uint8) * 255
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float64) * (255.0 / np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float64)
        if np.nanmin(a) >= 0 and np.nanmax(a) <= 1.0:
            a = a * 255.0
    return np.clip(np.round(np.nan_to_num(a)), 0, 255).astype(np.uint8)


def prepare_micrograph(image: ImageLike, size: int = 512, crop_info_bar: bool = True,
                       bar_threshold: int = 40, bar_search: int = 80) -> np.ndarray:
    """Grayscale field in [0, 1], as in the published benchmarks.

    Converts to 8-bit grayscale, removes a dark information/scale bar at the bottom
    (rows whose mean is below `bar_threshold` within the last `bar_search` rows),
    resizes to `size` x `size` with Lanczos, and min-max normalizes.
    """
    if isinstance(image, np.ndarray):
        a = _to_gray_uint8(image)
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
    with warnings.catch_warnings():  # the frozen kernel uses a scikit-image argument deprecated in 0.26
        warnings.filterwarnings("ignore", message="Parameter `min_size` is deprecated", category=FutureWarning)
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
