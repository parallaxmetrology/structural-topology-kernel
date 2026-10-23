# Structural Topology Kernel (STK)

**50 deterministic, training-free structural coordinates for microstructure images. Drop-in featurizer for matminer.**

One fixed descriptor turns an SEM micrograph, an EBSD-derived map or a binary phase mask into 50 named numbers: where gradient mass sits and how it spreads, void and phase topology, orientation order across scales, texture at two scales, blob statistics, island-and-corridor connectivity, spectral bands, edge sharpness and roughness. No training, no learned parameters, and the same numbers every time.

One representation, several tasks:

- **Classifies steel microstructures.** On ultrahigh carbon steel micrographs (UHCSDB, 7 classes), grouped by specimen so no specimen appears in both training and test, it scores 0.832 macro AUC in the original evaluation, ahead of HOG (0.728) and Haralick (0.629). On a second steel dataset (bainite) it scores 0.964.
- **Predicts 2-D permeability as well as the standard descriptors** (R² 0.939 vs 0.937 for porosity, specific surface, Euler number and two-point statistics).
- **Measures where class labels are ambiguous.** Mixtures sit between their parent classes, and the ambiguity it flags predicts where *other* descriptors fail.

**It is not a better classifier than a well-built texture set.** A stricter re-evaluation (2026-10-23) found that an ordinary 83-feature texture set ranks the UHCSDB classes better, and that adding STK to it does not clearly help. Read [How it compares](#how-it-compares) before citing a number.

The original results reproduce exactly from the feature tables in this repository plus one 22 MB download (scikit-learn 1.8 on macOS arm64; within about 0.003 elsewhere). Details: [docs/evidence.md](docs/evidence.md).

## How it compares

UHCSDB, 7-class macro AUC, grouped by specimen:

| descriptor | dims | original evaluation | stricter re-evaluation |
|---|---|---|---|
| STK-50 | 50 | **0.832** | 0.699 (0.726 without the four `g1`/`g3` coordinates) |
| HOG | 7,200 (50 after PCA in the re-evaluation) | 0.728 | 0.647 |
| Haralick GLCM | 196 | 0.629 | not run separately |
| texture: intensity moments and quantiles, uniform LBP, GLCM statistics | 83 | not run | **0.761** |
| texture + HOG | 133 | not run | 0.688 |
| texture + STK-50 | 133 | not run | 0.721 |

- **The original evaluation** is what this repository ships and reproduces. It uses all 961 micrographs (the 158 without a specimen record each count as their own specimen), standardization and class-balanced logistic regression, and 5-fold `GroupKFold`.
- **The stricter re-evaluation** was a separate local study, and its feature tables are not shipped here. It used the 803 micrographs with a recorded specimen (43 specimens), nested grouped model selection, and freshly computed features from a repaired release of the kernel.

What follows:

- STK beats HOG under both evaluations, and Haralick in the original one. Haralick was not run on its own in the re-evaluation.
- It does not beat a conventional texture set on ranking. Adding STK to that set changes macro AUC by −0.040 (95 % interval −0.098 to +0.020).
- One result favours STK, and it is exploratory: adding STK to the texture set lowers log loss from 2.47 to 1.49, a reduction of 0.98 (interval 0.27 to 1.78). The models were selected for AUC, and no conventional model was tuned or calibrated for probabilities, so this is a lead, not a calibration claim.
- On bainite the texture set also scores higher (0.979 against 0.965).

So treat STK as a compact, interpretable structural representation that is competitive across tasks, not as a stronger classifier than conventional texture features. Details: [known-issues.md](docs/known-issues.md) §13.

## Install

```bash
pip install "structural-topology-kernel @ git+https://github.com/parallaxmetrology/structural-topology-kernel"
```

## Use

```python
import stkernel as stk

c = stk.measure("micrograph.tif")       # 50 coordinates (info bar cropped, 512 px, min-max)
c = stk.measure_field(mask)             # a 2-D array you've already prepared (binary phase map, EBSD map)
df = stk.measure_many("micrographs/")   # one row per image
```

With matminer (or without; the featurizer falls back to a compatible base):

```python
from stkernel import STKFeaturizer
df = STKFeaturizer().featurize_dataframe(df, "image")    # adds stk_delta_x ... stk_tonal_high
```

```bash
stk measure micrograph.tif                    # JSON
stk measure micrographs/ -o features.csv      # a folder
```

## Reproduce

```bash
pip install -e ".[bench]"                   # Python 3.11+; pins scikit-learn 1.8 (fold assignment changed in 1.9)
python tools/fetch_data.py baselines        # HOG + Haralick features for UHCSDB (22 MB)
python benchmarks/uhcsdb.py                 # classification, grouping sensitivity, ablation, union, ambiguity
python benchmarks/bainite.py
python benchmarks/permeability.py           # R², and ΔR² beyond a connectivity-aware incumbent with bootstrap CI
```

Each script prints its results next to the recorded ones in `results/recorded/`.

## What to know before using it

- **The grouped score depends on how folds fall.** 43 specimens, some classes from two to six of them: the headline ranges from about 0.75 to 0.83 depending on how the 158 micrographs without a specimen record are handled, and the stricter re-evaluation above gives 0.70–0.73. STK leads HOG and Haralick in every variant, but not a conventional texture set. Tables in [known-issues.md](docs/known-issues.md).
- **Four coordinates don't compute what their names say** (`g1_*`, `g3_*`), because of an `np.diff(x, 0)` in the research code. The kernel is frozen, so they're documented rather than changed. [coordinates.md](docs/coordinates.md) says what each of the 50 actually measures.
- **On permeability it's competitive, with a small addition for nonlinear models.** It matches purpose-built descriptors; added to them it helps gradient boosting a little, but the best conventional linear model is still as good or better.

## Contribute

- Run it on **your** microstructures (any alloy, ceramic, composite, EBSD or optical) and share how the coordinates separate your classes or track a property, including where they don't.
- Bring a public dataset with **expert disagreement** labels. That's the test this instrument most needs and materials science mostly lacks.
- Spot a coordinate whose description doesn't match how your field would read it.

See [CONTRIBUTING.md](CONTRIBUTING.md). Discussion: [Discussions](https://github.com/parallaxmetrology/structural-topology-kernel/discussions).

## Part of Parallax Metrology

One principle across domains: structure has geometry, and geometry is measurable without knowing what it means. Siblings: [Visual Thinking Lens](https://github.com/parallaxmetrology/visual-thinking-lens) (composition), [Linguistic Kernel](https://github.com/parallaxmetrology/linguistic-kernel) (text), [Inspection Coordinates](https://github.com/parallaxmetrology/inspection-coordinates) (semiconductor inspection), [Spatial Kernel](https://github.com/parallaxmetrology/spatial-kernel) (robot outcomes). Paper: [STK: an interpretable multi-task descriptor for materials](https://www.parallaxmetrology.com/STK_Materials_Paper.html). Lineage: [docs/lineage.md](docs/lineage.md).

## License and data

Code: [MIT](LICENSE). Docs: [CC BY 4.0](LICENSE-CONTENT).

- **UHCS micrographs** (`examples/images/uhcs_*`, and the features in `data/uhcsdb/`): Hecht, DeCost, Francis, Holm, Picard, Webler, *Ultrahigh Carbon Steel Micrographs*, NIST, [hdl.handle.net/11256/940](https://hdl.handle.net/11256/940), CC BY 3.0 US. Data descriptor: DeCost et al., *IMMI* 6, 197–205 (2017), [doi:10.1007/s40192-017-0097-0](https://doi.org/10.1007/s40192-017-0097-0).
- **Bainite SEM** (features in `data/bainite/`): *Classification of bainitic microstructures in SEM image data*, Figshare, [doi:10.6084/m9.figshare.19242903](https://doi.org/10.6084/m9.figshare.19242903), MIT.
- **2-D porous media** (features in `data/permeability/`): Röding et al., Zenodo, [doi:10.5281/zenodo.17711512](https://doi.org/10.5281/zenodo.17711512), CC BY 4.0.
