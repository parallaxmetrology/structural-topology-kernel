"""Regenerate tests/golden/stk50.json by running the ORIGINAL research kernel
(vtl_materials_kernel.py, passed by path) on the sample micrographs and on
synthetic fields. Usage: python tools/golden_from_original.py /path/to/vtl_materials_kernel.py"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from synthetic import FIELDS  # noqa: E402

spec = importlib.util.spec_from_file_location("orig", sys.argv[1])
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

from stkernel.core import prepare_micrograph  # noqa: E402  (loader only; the kernel is the original)


def clean(d):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in d.items() if k in orig.KERNEL_COORDS}


out = {"micrographs": {}, "fields": {}}
for p in sorted((ROOT / "examples" / "images").glob("uhcs_*")):
    out["micrographs"][p.name] = clean(orig.compute_full_kernel(prepare_micrograph(p)))
for name, make in FIELDS.items():
    out["fields"][name] = clean(orig.compute_full_kernel(make()))
(ROOT / "tests" / "golden" / "stk50.json").write_text(json.dumps(out, indent=1))
print(len(out["micrographs"]), "micrographs,", len(out["fields"]), "fields")
