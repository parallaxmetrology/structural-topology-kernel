"""
Bainite SEM (Figshare 10.6084/m9.figshare.19242903, MIT): a second steel dataset, three
bainite subclasses, 386 test-set images from 26 specimens, grouped by specimen.

    python benchmarks/bainite.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stkernel import COORDINATES

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "bainite" / "stk50_features.csv")
classes = sorted(set(d["class"]))
yi = np.array([classes.index(c) for c in d["class"]])
groups = d["specimen"].astype(str).values
ng = d["specimen"].nunique()
P = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                      np.nan_to_num(d[list(COORDINATES)].values), yi, cv=GroupKFold(min(5, ng)), groups=groups,
                      method="predict_proba")
auc = float(np.mean([roc_auc_score((yi == k).astype(int), P[:, k]) for k in range(len(classes))]))
acc = float(accuracy_score(yi, P.argmax(1)))
print(f"{len(d)} images, {ng} specimens, classes {dict(pd.Series(d['class']).value_counts())}")
print(f"STK-50 grouped 3-class macro AUC {auc:.4f}, accuracy {acc:.4f}")
rec = json.loads((ROOT / "results/recorded/arm15_bainite_benchmark.json").read_text())
print(f"recorded: AUC {rec['vtl50_macro_auc']:.4f}, accuracy {rec['vtl50_accuracy']:.4f}")
(ROOT / "results" / "bainite.json").write_text(json.dumps({"macro_auc": auc, "accuracy": acc, "n": len(d), "specimens": ng}, indent=2))
