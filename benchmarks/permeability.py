"""
2-D permeability from binary porous media (Röding et al., Zenodo 10.5281/zenodo.17711512):
does STK-50 predict log-permeability, and does it add anything beyond standard
descriptors? 2,000 train / 2,000 test images, features shipped in data/permeability/.

    python benchmarks/permeability.py

Incumbent = porosity, specific surface, Euler number, radial two-point statistics S2
(16 radii). Strengthened incumbent adds 7 connectivity features of the pore phase
(components, skeleton density, distance-transform stats, throat radius).
Models: linear regression and gradient boosting. Bootstrap: 2,000 resamples of the test set.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
F = np.load(ROOT / "data" / "permeability" / "features_2000.npz", allow_pickle=True)
Ftr, Fte, ytr, yte, Mtr, Mte = (F[k] for k in ["Ftr", "Fte", "ytr", "yte", "Mtr", "Mte"])


def predict(a_tr, a_te):
    out = {}
    for nm, m in [("lin", make_pipeline(StandardScaler(), LinearRegression())),
                  ("gbm", HistGradientBoostingRegressor(random_state=0, max_iter=400))]:
        m.fit(np.nan_to_num(a_tr), ytr)
        out[nm] = m.predict(np.nan_to_num(a_te))
    return out


sets = {"porosity only": np.arange(0, 1), "incumbent (porosity+SS+Euler+S2)": np.arange(0, 19),
        "STK-50": np.arange(19, 69), "incumbent + STK-50": np.arange(0, 69)}
res = {"R2": {}}
print("== R² on 2,000 test images (log10 permeability) ==")
for nm, cols in sets.items():
    P = predict(Ftr[:, cols], Fte[:, cols])
    res["R2"][nm] = {k: float(r2_score(yte, v)) for k, v in P.items()}
    print(f"  {nm:36s} linear {res['R2'][nm]['lin']:.3f}   boosted {res['R2'][nm]['gbm']:.3f}")

print("\n== Does STK add beyond a connectivity-aware incumbent? (bootstrap 95% CI) ==")
s_tr, s_te = np.column_stack([Ftr[:, :19], Mtr]), np.column_stack([Fte[:, :19], Mte])
Pstr = predict(s_tr, s_te)
Psv = predict(np.column_stack([s_tr, Ftr[:, 19:69]]), np.column_stack([s_te, Fte[:, 19:69]]))
rng = np.random.default_rng(0)
rng.choice(int(F["n_train_pool"]), 2000, replace=False)  # same generator state as the recorded run
boot = {}
n = len(yte)
for mdl in ["lin", "gbm"]:
    ds = []
    for _ in range(2000):
        idx = rng.choice(n, n, replace=True)
        ds.append(r2_score(yte[idx], Psv[mdl][idx]) - r2_score(yte[idx], Pstr[mdl][idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    pt = r2_score(yte, Psv[mdl]) - r2_score(yte, Pstr[mdl])
    boot[mdl] = {"strengthened": float(r2_score(yte, Pstr[mdl])), "strengthened_plus_stk": float(r2_score(yte, Psv[mdl])),
                 "delta": float(pt), "ci95": [float(lo), float(hi)]}
    print(f"  {'linear' if mdl == 'lin' else 'boosted':8s} strengthened {boot[mdl]['strengthened']:.3f} -> +STK {boot[mdl]['strengthened_plus_stk']:.3f}"
          f"   ΔR² {pt:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
res["beyond_strengthened_incumbent"] = boot
out = ROOT / "results" / "permeability.json"
out.write_text(json.dumps(res, indent=2))

rec = json.loads((ROOT / "results/recorded/arm19_porous_gate.json").read_text())["R2"]
rb = json.loads((ROOT / "results/recorded/arm19_bootstrap_validation.json").read_text())["permeability_deltaR2_vtl_beyond_strengthened_incumbent"]
print("\n== Against the recorded results ==")
for nm, rn in [("porosity only", "porosity only"), ("incumbent (porosity+SS+Euler+S2)", "full incumbent (por+SS+Euler+S2)"),
               ("STK-50", "VTL-50"), ("incumbent + STK-50", "incumbent + VTL")]:
    for k in ["lin", "gbm"]:
        a, b = res["R2"][nm][k], rec[rn][k]
        print(f"  {nm:36s} {k}  now {a:.4f}  recorded {b:.4f}  {'same' if abs(a - b) < 5e-4 else 'DIFFERS'}")
for k, rk in [("lin", "linear"), ("gbm", "gbm")]:
    a, b = boot[k]["delta"], rb[rk]["point"]
    print(f"  ΔR² beyond strengthened {k}  now {a:+.4f}  recorded {b:+.4f}  {'same' if abs(a - b) < 6e-4 else 'DIFFERS'}")
print(f"\nwrote {out.relative_to(ROOT)}")
