"""Inputs: an image passed as a file, a PIL image or any common array form gives the
same coordinates. CLI output is valid JSON."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import stkernel as stk
from stkernel.cli import main

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "examples" / "images" / "uhcs_micrograph1.tif"


@pytest.fixture(scope="module")
def ref():
    return stk.measure(P)


@pytest.mark.parametrize("make", [
    lambda a: a,                                   # uint8
    lambda a: a / 255.0,                           # float in [0, 1]
    lambda a: a.astype(np.float32),                # float 0-255
    lambda a: a.astype(np.uint16) * 257,           # 16-bit
    lambda a: np.stack([a] * 3, -1),               # RGB uint8
    lambda a: Image.fromarray(a),                  # PIL
], ids=["uint8", "float01", "float255", "uint16", "rgb", "pil"])
def test_input_forms_agree(ref, make):
    a = np.array(Image.open(P).convert("L"))
    got = stk.measure(make(a))
    assert all(np.isclose(got[k], ref[k], rtol=1e-9, atol=1e-12, equal_nan=True) for k in ref)


def test_no_deprecation_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        stk.measure(P)


def test_cli_json(capsys):
    main(["measure", str(P)])
    d = json.loads(capsys.readouterr().out)
    assert len(d) == 50


def test_cli_folder(tmp_path):
    out = tmp_path / "f.csv"
    main(["measure", str(ROOT / "examples" / "images"), "-o", str(out)])
    assert out.read_text().count("\n") == 8
