"""The shipped feature tables reproduce the recorded results (fast benchmarks only;
UHCSDB needs the downloaded baselines and runs in benchmarks/uhcsdb.py)."""
import json
import platform
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "results" / "recorded"
# exact on the reference platform; elsewhere BLAS/solver differences move scores slightly (docs/known-issues.md §10)
TOL = 2e-4 if (platform.system() == "Darwin" and platform.machine() == "arm64") else 5e-3


def run(name):
    runpy.run_path(str(ROOT / "benchmarks" / f"{name}.py"), run_name="__main__")
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def test_bainite():
    got = run("bainite")
    rec = json.loads((REC / "arm15_bainite_benchmark.json").read_text())
    assert got["macro_auc"] == pytest.approx(rec["vtl50_macro_auc"], abs=TOL)


def test_permeability():
    got = run("permeability")
    rec = json.loads((REC / "arm19_porous_gate.json").read_text())["R2"]
    assert got["R2"]["STK-50"]["lin"] == pytest.approx(rec["VTL-50"]["lin"], abs=TOL)
    assert got["R2"]["incumbent (porosity+SS+Euler+S2)"]["lin"] == pytest.approx(rec["full incumbent (por+SS+Euler+S2)"]["lin"], abs=TOL)
    boot = json.loads((REC / "arm19_bootstrap_validation.json").read_text())["permeability_deltaR2_vtl_beyond_strengthened_incumbent"]
    assert got["beyond_strengthened_incumbent"]["lin"]["delta"] == pytest.approx(boot["linear"]["point"], abs=TOL)


@pytest.mark.skipif(not (ROOT / "data/uhcsdb/baselines_har_hog.npz").exists(), reason="python tools/fetch_data.py baselines")
def test_uhcsdb():
    got = run("uhcsdb")
    rec = json.loads((REC / "arm13_grouped_benchmark.json").read_text())
    assert got["headline"]["STK-50"]["grouped"] == pytest.approx(rec["VTL-50"]["grouped"], abs=TOL)
