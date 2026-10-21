"""
UHCSDB (ultrahigh carbon steel, 961 SEM micrographs, 7 microconstituent classes):
STK-50 vs Haralick GLCM and HOG, grouped by specimen, plus the stress tests.

    python tools/fetch_data.py baselines      # HOG + Haralick features (22 MB)
    python benchmarks/uhcsdb.py               # ~1 min; compares with results/recorded/

Everything runs from feature tables. To recompute the STK features from the images
(UHCS data, NIST handle 11256/940), use `stk measure micrographs/ -o features.csv`.

Classifier for every descriptor: standardize, then class-balanced logistic regression;
score = 7-class one-vs-rest macro AUC from out-of-fold probabilities.

Grouping. 803 of the 961 micrographs have a specimen key in the UHCSDB database (43
specimens); 158 don't. The published run treats each unkeyed micrograph as its own
specimen. GroupKFold breaks ties between equal-size groups by label, so the exact
fold assignment depends on the labels given to those 158; the labels used for each
recorded result are reproduced here (`singleton_i` for the headline, `s_i` for the
stress tests). The "grouping sensitivity" section shows how much the choice matters.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stkernel import COORDINATES

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "uhcsdb"
OUT = ROOT / "results" / "uhcsdb.json"
ACQ = ["tonal_shadow", "tonal_mid", "tonal_high", "ds_low", "ds_mid", "ds_high", "nils", "ler", "edge_density"]

feat = pd.read_csv(DATA / "stk50_features.csv")
grp = pd.read_csv(DATA / "grouping.csv")[["path", "sample_key"]]
d = feat.merge(grp, on="path", how="left")
classes = sorted(set(d["class"]))
yi = np.array([classes.index(c) for c in d["class"]])
STK = d[list(COORDINATES)].values
bpath = DATA / "baselines_har_hog.npz"
if not bpath.exists():
    raise SystemExit("missing baselines: run  python tools/fetch_data.py baselines")
base = np.load(bpath, allow_pickle=True)
assert (base["cls"] == d["class"].values).all()
HOG, HAR = base["hog"], base["har"]
S2 = np.load(DATA / "s2_radial.npz", allow_pickle=True)["S2"]


def groups(scheme):
    """Specimen keys; unkeyed micrographs get their own group (labelled `scheme`+i),
    or one shared group ('one'), or are dropped ('drop' -> returns a mask)."""
    sk = d["sample_key"].astype("object").copy()
    miss = sk.isna()
    if scheme == "one":
        sk[miss] = "unknown"
    elif scheme != "drop":
        sk[miss] = [f"{scheme}{i}" for i in range(int(miss.sum()))]
    return sk.astype(str).values, ~miss.values


def proba(X, y, g, pca=None, grouped=True):
    steps = [StandardScaler()] + ([PCA(min(pca, X.shape[1]), random_state=0)] if pca else [])
    steps.append(LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = GroupKFold(5) if grouped else StratifiedKFold(5, shuffle=True, random_state=0)
    kw = {"groups": g} if grouped else {}
    return cross_val_predict(make_pipeline(*steps), np.nan_to_num(X), y, cv=cv, method="predict_proba", **kw)


def macro_auc(P, y):
    return float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(P.shape[1]) if (y == k).any()]))


res = {}

print("== Headline: grouped by specimen, 7-class macro AUC ==")
g, _ = groups("singleton_")
head = {}
for name, X in [("STK-50", STK), ("HOG-7200", HOG), ("Haralick-196", HAR)]:
    head[name] = {"grouped": macro_auc(proba(X, yi, g), yi), "ungrouped": macro_auc(proba(X, yi, g, grouped=False), yi)}
    print(f"  {name:13s} grouped {head[name]['grouped']:.4f}   per-image (leaky) {head[name]['ungrouped']:.4f}")
P = proba(STK, yi, g)
head["stk_per_class_grouped"] = {c: float(roc_auc_score((yi == k).astype(int), P[:, k])) for k, c in enumerate(classes)}
res["headline"] = head

print("\n== Grouping sensitivity (unkeyed micrographs) ==")
sens = {}
for label, scheme in [("each its own specimen", "singleton_"), ("each its own specimen, other labels", "img_"),
                      ("one shared unknown specimen", "one"), ("dropped (803 images, 43 specimens)", "drop")]:
    gg, keep = groups(scheme)
    if scheme == "drop":
        row = {n: macro_auc(proba(X[keep], yi[keep], gg[keep]), yi[keep]) for n, X in [("STK-50", STK), ("HOG-7200", HOG), ("Haralick-196", HAR)]}
    else:
        row = {n: macro_auc(proba(X, yi, gg), yi) for n, X in [("STK-50", STK), ("HOG-7200", HOG), ("Haralick-196", HAR)]}
    sens[label] = row
    print(f"  {label:38s} STK {row['STK-50']:.3f}  HOG {row['HOG-7200']:.3f}  Haralick {row['Haralick-196']:.3f}")
res["grouping_sensitivity"] = sens

print("\n== Stress tests (grouped) ==")
g, _ = groups("s_")
struct = [c for c in COORDINATES if c not in ACQ]
full = macro_auc(proba(STK, yi, g), yi)
st = macro_auc(proba(d[struct].values, yi, g), yi)
ac = macro_auc(proba(d[ACQ].values, yi, g), yi)
union = np.column_stack([S2, HAR, HOG])
u_full = macro_auc(proba(union, yi, g), yi)
u_pca = macro_auc(proba(union, yi, g, pca=50), yi)
res["ablation"] = {"full": full, "structural_only_41": st, "acquisition_only_9": ac}
res["union"] = {"stk50": full, "union_S2_Haralick_HOG_full": u_full, "union_pca50": u_pca}
print(f"  without tonal/spectral/edge channels (41 coords): {st:.3f}  (full {full:.3f}; those 9 alone {ac:.3f})")
print(f"  union S2+Haralick+HOG: {union.shape[1]} dims {u_full:.3f}, PCA-50 {u_pca:.3f}  vs STK-50 {full:.3f}")

print("\n== Label ambiguity (no classifier: distances to class centroids in standardized STK space) ==")
Xs = StandardScaler().fit_transform(np.nan_to_num(STK))
cent = {c: Xs[d["class"].values == c].mean(0) for c in classes}


def between(mixed, p1, p2):
    axis = cent[p2] - cent[p1]
    return float(np.dot(cent[mixed] - cent[p1], axis) / np.dot(axis, axis))


cont = {f"{m} on {a} -> {b}": between(m, a, b) for m, a, b in [
    ("pearlite+spheroidite", "pearlite", "spheroidite"),
    ("pearlite+widmanstatten", "pearlite", "spheroidite+widmanstatten"),
    ("spheroidite+widmanstatten", "spheroidite", "network")]}
for k, v in cont.items():
    print(f"  {k:62s} {v:+.2f}  ({'between' if 0 < v < 1 else 'outside'})")
D = cdist(Xs, np.array([cent[c] for c in classes]))
own = D[np.arange(len(yi)), yi]
other = D.copy()
other[np.arange(len(yi)), yi] = np.inf
margin = other.min(1) - own
by_class = {c: float((margin[yi == k] < 0).mean()) for k, c in enumerate(classes)}
print(f"  closer to another class's centroid than their own: {100 * (margin < 0).mean():.1f}% overall")
for c, v in by_class.items():
    print(f"    {c:28s} {100 * v:5.1f}%")
pred = {}
for name, X in [("HOG", HOG), ("Haralick", HAR)]:
    wrong = (proba(X, yi, g).argmax(1) != yi).astype(int)
    pred[name] = float(roc_auc_score(wrong, -margin))
    print(f"  STK ambiguity predicts {name} errors (grouped CV): AUC {pred[name]:.3f}")
res["ambiguity"] = {"continuum": cont, "rate_overall": float((margin < 0).mean()), "rate_by_class": by_class,
                    "predicts_incumbent_error_auc": pred}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))

rec = json.loads((ROOT / "results/recorded/arm13_grouped_benchmark.json").read_text())
rb = json.loads((ROOT / "results/recorded/arm13b_ablation_union.json").read_text())
rc = json.loads((ROOT / "results/recorded/arm13c_c2_grouped.json").read_text())
ra = json.loads((ROOT / "results/recorded/arm12_label_fuzziness.json").read_text())
checks = [("STK-50 grouped", head["STK-50"]["grouped"], rec["VTL-50"]["grouped"]),
          ("HOG grouped", head["HOG-7200"]["grouped"], rec["HOG-7200"]["grouped"]),
          ("Haralick grouped", head["Haralick-196"]["grouped"], rec["Haralick-196"]["grouped"]),
          ("STK-50 per-image", head["STK-50"]["ungrouped"], rec["VTL-50"]["ungrouped"]),
          ("structural-only", st, rb["concern2_ablation"]["structural_only"]),
          ("union PCA-50", u_pca, rb["concern3_union"]["union_pca50"]),
          ("ambiguity rate", float((margin < 0).mean()), ra["ambiguity_rate_overall"]),
          ("predicts HOG errors", pred["HOG"], rc["vtl_margin_predicts_incumbent_error_auc"]["HOG"]),
          ("predicts Haralick errors", pred["Haralick"], rc["vtl_margin_predicts_incumbent_error_auc"]["Haralick"])]
print("\n== Against the recorded results ==")
for n, a, b in checks:
    print(f"  {n:26s} now {a:.4f}  recorded {b:.4f}  {'same' if abs(a - b) < 5e-4 else 'DIFFERS'}")
print(f"\nwrote {OUT.relative_to(ROOT)}")
