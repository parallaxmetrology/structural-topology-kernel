# Structural Topology Kernel (STK)

**50 deterministic, training-free structural coordinates for microstructure images. Drop-in featurizer for matminer.**

One fixed descriptor turns an SEM micrograph, an EBSD-derived map or a binary phase mask into 50 named numbers: where gradient mass sits and how it spreads, void and phase topology, orientation order across scales, texture at two scales, blob statistics, island-and-corridor connectivity, spectral bands, edge sharpness and roughness. No training, no learned parameters, and the same numbers every time.

On ultrahigh carbon steel micrographs, grouped by specimen so no specimen appears in both training and test:

| UHCSDB, 7 microconstituent classes, 961 SEM micrographs | dims | macro AUC |
|---|---|---|
| **STK-50** | 50 | **0.832** |
| HOG | 7,200 | 0.728 |
| Haralick GLCM | 196 | 0.629 |

The same representation also gets a second steel dataset (bainite, 0.964 macro AUC by specimen), predicts 2-D permeability as well as the standard porosity/S2 descriptors (R² 0.939 vs 0.937), and measures where class labels are genuinely ambiguous: mixtures sit between their parent classes, and the ambiguity it flags predicts where *other* descriptors fail. Details and caveats: [docs/evidence.md](docs/evidence.md). Every number reproduces exactly from the shipped feature tables.

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

- **The grouped score depends on how folds fall.** 43 specimens, some classes from two to six of them: the headline ranges from about 0.75 to 0.83 depending on how the 158 micrographs without a specimen record are handled. STK leads HOG and Haralick in every variant. Table in [known-issues.md](docs/known-issues.md).
- **Four coordinates don't compute what their names say** (`g1_*`, `g3_*`), because of an `np.diff(x, 0)` in the research code. The kernel is frozen, so they're documented rather than changed. [coordinates.md](docs/coordinates.md) says what each of the 50 actually measures.
- **On permeability it's competitive, not additive.** It matches purpose-built descriptors, and adds little beyond them.

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
