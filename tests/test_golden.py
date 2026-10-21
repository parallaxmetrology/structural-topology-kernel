"""The package must reproduce the original research kernel."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stkernel as stk
from synthetic import FIELDS

ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "tests" / "golden" / "stk50.json").read_text())
# Reference platform (where goldens were made) must match to float precision. Elsewhere, OpenCV
# Canny/contours and BLAS can differ slightly; see docs/known-issues.md.
REFERENCE = platform.system() == "Darwin" and platform.machine() == "arm64"
RTOL, ATOL = (1e-9, 1e-12) if REFERENCE else (0.05, 0.02)


def close(a, b):
    if b is None:
        return a is None or (isinstance(a, float) and np.isnan(a))
    return np.isclose(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", sorted(GOLD["micrographs"]))
def test_micrographs(name):
    got = stk.measure(ROOT / "examples" / "images" / name)
    bad = [k for k, v in GOLD["micrographs"][name].items() if not close(got[k], v)]
    assert len(bad) <= (0 if REFERENCE else 4), bad


@pytest.mark.parametrize("name", sorted(GOLD["fields"]))
def test_fields(name):
    got = stk.measure_field(FIELDS[name]())
    bad = [k for k, v in GOLD["fields"][name].items() if not close(got[k], v)]
    assert len(bad) <= (0 if REFERENCE else 4), bad


def test_samples_match_published_feature_table():
    table = pd.read_csv(ROOT / "data" / "uhcsdb" / "stk50_features.csv").set_index("path")
    for name, gold in GOLD["micrographs"].items():
        row = table.loc[name.removeprefix("uhcs_")]
        for k, v in gold.items():
            assert (v is None and np.isnan(row[k])) or np.isclose(row[k], v, rtol=1e-9, atol=1e-12), (name, k)


def test_coordinates():
    assert len(stk.COORDINATES) == 50 and len(set(stk.COORDINATES)) == 50


def test_featurizer_dataframe():
    df = pd.DataFrame({"image": [ROOT / "examples" / "images" / "uhcs_micrograph1.tif"]})
    out = stk.STKFeaturizer().featurize_dataframe(df, "image")
    assert out.shape[1] == 51 and "stk_nils" in out.columns
