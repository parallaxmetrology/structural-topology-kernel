"""Recompute the Haralick (7x7 grid x 4 GLCM properties, 64 levels) and HOG
(8 orientations, 32 px cells, 2x2 blocks) baselines from the UHCS micrographs, with the
same preparation as STK (info-bar crop, 512 px, min-max). Writes
data/uhcsdb/baselines_har_hog.npz. Usage: python benchmarks/uhcsdb_baselines.py MICROGRAPH_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog

from stkernel import prepare_micrograph

ROOT = Path(__file__).resolve().parents[1]


def haralick(img, grid=7, levels=64):
    H, W = img.shape
    th, tw = H // grid, W // grid
    i8 = (img * (levels - 1)).clip(0, levels - 2).astype(np.uint8)
    f = []
    for r in range(grid):
        for c in range(grid):
            g = graycomatrix(i8[r * th:(r + 1) * th, c * tw:(c + 1) * tw], [1], [0], levels=levels, symmetric=True, normed=True)
            f += [graycoprops(g, p)[0, 0] for p in ["energy", "homogeneity", "contrast", "correlation"]]
    return np.array(f)


def main(img_dir):
    df = pd.read_csv(ROOT / "data" / "uhcsdb" / "stk50_features.csv")
    har, hg = [], []
    for i, p in enumerate(df["path"], 1):
        img = prepare_micrograph(Path(img_dir) / p)
        har.append(haralick(img))
        hg.append(hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True))
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)
    np.savez_compressed(ROOT / "data" / "uhcsdb" / "baselines_har_hog.npz", har=np.array(har), hog=np.array(hg), cls=df["class"].values.astype(str))


if __name__ == "__main__":
    main(sys.argv[1])
