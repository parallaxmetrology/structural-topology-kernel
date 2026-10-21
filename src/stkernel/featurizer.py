"""
matminer-style featurizer. If matminer is installed, `STKFeaturizer` is a real
`matminer.featurizers.base.BaseFeaturizer` (so `featurize_dataframe`, parallel
featurization and citations work as usual); otherwise it falls back to a small
compatible base with the same methods.

    from stkernel import STKFeaturizer
    df = STKFeaturizer().featurize_dataframe(df, "image")   # column of paths, PIL images or arrays
"""
from __future__ import annotations

import numpy as np

from .core import COORDINATES, measure, measure_field

try:  # pragma: no cover - exercised only when matminer is installed
    from matminer.featurizers.base import BaseFeaturizer as _Base
except Exception:  # matminer not installed
    _Base = None


class _FallbackBase:
    def featurize_many(self, entries, ignore_errors=False, **_):
        out = []
        for e in entries:
            try:
                out.append(self.featurize(*e) if isinstance(e, tuple) else self.featurize(e))
            except Exception:
                if not ignore_errors:
                    raise
                out.append([np.nan] * len(self.feature_labels()))
        return out

    def featurize_dataframe(self, df, col_id, ignore_errors=False, inplace=False, **_):
        import pandas as pd
        rows = self.featurize_many(df[col_id].tolist(), ignore_errors=ignore_errors)
        feats = pd.DataFrame(rows, columns=self.feature_labels(), index=df.index)
        if inplace:
            for c in feats:
                df[c] = feats[c]
            return df
        return pd.concat([df, feats], axis=1)

    def fit(self, X, y=None, **_):
        return self

    def transform(self, X):
        return np.array(self.featurize_many(X))


class STKFeaturizer(_Base or _FallbackBase):
    """Structural Topology Kernel: 50 training-free structural coordinates of a
    micrograph or 2-D field.

    Args:
        prefix: prepended to feature labels (default "stk_").
        prepared: True if inputs are already 2-D fields in [0, 1] (masks, EBSD maps);
            False to load and prepare micrographs as in the published benchmarks.
        size, crop_info_bar: passed to `prepare_micrograph`.
    """

    def __init__(self, prefix="stk_", prepared=False, size=512, crop_info_bar=True):
        self.prefix, self.prepared, self.size, self.crop_info_bar = prefix, prepared, size, crop_info_bar

    def featurize(self, image):
        c = measure_field(image) if self.prepared else measure(image, size=self.size, crop_info_bar=self.crop_info_bar)
        return [c[k] for k in COORDINATES]

    def feature_labels(self):
        return [self.prefix + k for k in COORDINATES]

    def citations(self):
        return ["@software{parrish_stk_2026, author = {Parrish, Russell}, title = {Structural Topology Kernel}, "
                "year = {2026}, url = {https://github.com/parallaxmetrology/structural-topology-kernel}}"]

    def implementors(self):
        return ["Russell Parrish"]
