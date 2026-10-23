# Evidence

Sections 1–4 are in `results/recorded/` and reproduce exactly with the scripts in `benchmarks/` (scikit-learn 1.8 on macOS arm64; on other platforms within about 0.003, see [known-issues.md](known-issues.md) §4 and §10). The results under *Also recorded* are kept as records only; their features aren't shipped.

## 1. Microconstituent classification: UHCSDB

961 SEM micrographs of ultrahigh carbon steel, 7 classes (spheroidite, network, pearlite, martensite and three mixtures), from 43 recorded specimens plus 158 micrographs without a specimen record. Standardize, then class-balanced logistic regression, 5-fold `GroupKFold` by specimen, 7-class macro AUC.

| descriptor | dims | grouped by specimen | per image (leaky) |
|---|---|---|---|
| **STK-50** | 50 | **0.832** | 0.926 |
| HOG | 7,200 | 0.728 | 0.790 |
| Haralick GLCM (7 × 7 grid) | 196 | 0.629 | 0.719 |

The per-image split lets micrographs of one specimen fall on both sides and inflates every descriptor (STK by 0.09). Grouping by specimen is the honest number. How the grouping is set up still moves it: see the sensitivity table in [known-issues.md](known-issues.md).

**Stress tests** (`python benchmarks/uhcsdb.py`):

- Drop the 9 acquisition-sensitive channels (tonal, spectral bands, edge sharpness and roughness): **0.829**. The result is structural, not a fingerprint of brightness or focus. Those 9 alone: 0.77.
- Against the union of S2 two-point statistics + Haralick + HOG (7,461 dims): the union scores 0.780, and a 50-component PCA of it (dimension-matched) 0.721. STK-50: 0.832.

**Against a stronger baseline.** A later re-evaluation with stricter grouping and freshly computed features found that an ordinary 83-feature texture set (intensity moments, LBP, GLCM) scores 0.761, against 0.699–0.726 for STK, and that adding STK to it does not clearly help. So STK beats HOG, and Haralick where it was run, but not conventional texture. See [known-issues.md](known-issues.md) §13.

Per class (grouped): network 0.97, spheroidite 0.93, pearlite 0.90, pearlite + widmanstätten 0.89, pearlite + spheroidite 0.82, spheroidite + widmanstätten 0.76, martensite 0.55 (2 specimens; can't be validated).

## 2. Label ambiguity: UHCSDB

No classifier: distances to class centroids in standardized STK space.

- **Mixtures sit between their parents.** Projected onto the axis between two parent classes, pearlite + spheroidite sits at 0.44 of the way from pearlite to spheroidite.
- **Ambiguity concentrates where the taxonomy is fuzzy.** Micrographs closer to another class's centroid than their own: spheroidite + widmanstätten 68 %, pearlite + spheroidite 56 %, pearlite 51 %; spheroidite 30 %, martensite 28 %.
- **It's in the samples, not in STK.** The same ambiguity score predicts where *independent* descriptors fail under grouped CV: HOG errors AUC 0.60, Haralick errors AUC 0.65.

## 3. A second steel: bainite SEM

386 images, 26 specimens, 3 bainite subclasses known for subjective labeling (Figshare 19242903). Grouped by specimen: **macro AUC 0.964**, accuracy 0.927 (`python benchmarks/bainite.py`).

## 4. A property: 2-D permeability

2,000 train / 2,000 test binary porous media with simulated permeability tensors (Zenodo 17711512); target log10 of the mean diagonal permeability (`python benchmarks/permeability.py`).

| features | R² linear | R² boosted |
|---|---|---|
| porosity only | 0.920 | 0.914 |
| porosity + specific surface + Euler + S2 | 0.937 | 0.920 |
| **STK-50 alone** | **0.939** | 0.928 |
| incumbent + STK-50 | 0.938 | 0.929 |

Against a stronger, connectivity-aware incumbent (+ components, skeleton density, distance-transform statistics, throat radius), STK adds ΔR² +0.001 linearly (95 % CI −0.001 to +0.003) and +0.011 with boosting (CI +0.005 to +0.016).

Read: one fixed descriptor, not designed for flow, matches the standard purpose-built ones and adds a little for nonlinear models. It isn't a better permeability predictor. A later re-evaluation agrees (ΔR² +0.009 with boosting; the best conventional linear model is still as good): [known-issues.md](known-issues.md) §7.

## Also recorded

- `arm11_ti64_validation.json`: Ti-6Al-4V EBSD-derived maps, rolling temperature from 72 patches of 18 maps, grouped by map: R² 0.41, Spearman 0.59. Moderate.
- `arm11_alsi_validation.json`: AlSi10Mg as-built vs annealed, 8 maps: AUC 1.0, but one map per state, so indicative only.
- `arm10_phaseA_results.json`, `arm19_256ablation.json`, `arm19_strengthened_gate.json`: dimension-matched and scale ablations behind the above.

The features for these are not shipped; the JSON files are the record.
