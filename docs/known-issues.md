# Known issues

The kernel is frozen: results across studies are only comparable if it doesn't change. So the issues below are documented, not patched. A corrected kernel would be a new version with its own name and golden values.

## 1. Two coordinate families don't compute what their names say

`compute_g1` and `compute_g3` take differences with `np.diff(x, 0)`. With `n=0` NumPy returns `x` unchanged, so:

- `g1_pathlen` is the sum of the lengths of five centroid *positions* (in normalized image coordinates), not the length of the path the centroid travels across blur scales. `g1_curvature` is the mean angle between successive position vectors, not a turning angle.
- `g3_kvar` and `g3_infl` use contour point positions where tangents were intended, so they are statistics of the angle between successive position vectors along each contour, not of curvature.

They are still deterministic and still carry information (they enter the published results as they are), but don't read them as path length or curvature. With `n=1` they would measure what the names say.

## 2. `sci`'s rhythm term sees only the top of the image

The autocorrelation "rhythm" component of `sci` uses the first 1,000 pixels of the flattened image: about the top two rows at 512 px. It contributes 15 % of `sci`.

## 3. `theta` is a concentration, not an angle

`theta` is 1 − normalized direction entropy (0 isotropic, 1 one direction). The same short name is an angle in Inspection Coordinates.

## 4. Grouped scores move with fold assignment

The headline (0.832) uses `GroupKFold(5)` by specimen. Which specimens land in which fold depends on details that shouldn't matter, and with 43 specimens, some classes held by two to six of them, that moves the score:

| what changed | STK-50 | HOG | Haralick |
|---|---|---|---|
| as published (scikit-learn 1.8, unkeyed micrographs as `singleton_i`) | **0.832** | 0.728 | 0.629 |
| same, scikit-learn 1.9 (changed fold balancing) | 0.830 | 0.726 | 0.627 |
| unkeyed micrographs pooled as one specimen | 0.797 | 0.693 | 0.584 |
| unkeyed micrographs dropped (803 images, 43 specimens) | 0.752 | 0.706 | 0.592 |

STK leads HOG by 0.05–0.10 and Haralick by 0.16–0.21 in every row. The absolute number is not stable to ±0.01, and the strictest reading is about 0.75. Even the type of the specimen labels matters: the dropped row gives 0.768 if the keys are numbers instead of strings, because ties between equal-size specimens are broken by sort order. `pip install ".[bench]"` pins scikit-learn 1.8 so the recorded numbers reproduce exactly on the reference platform (macOS arm64; see §10).

The union comparison is the most fold-sensitive: STK-50 vs a 50-component PCA of S2 + Haralick + HOG is +0.111 with scikit-learn 1.8 and +0.084 with 1.9.

## 5. 158 UHCSDB micrographs have no specimen record

The UHCSDB database gives a specimen key for 803 of the 961 micrographs. The published run treats each of the other 158 as its own specimen, which is the most lenient choice: if several of them share a specimen, some leakage remains. See the table above for the alternatives.

## 6. Small classes can't be validated

Martensite comes from 2 specimens and scores 0.555 under grouping (near chance). Pearlite + spheroidite has 6 specimens. Per-class numbers for these are reported but shouldn't be leaned on.

## 7. Permeability: competitive, with a small nonlinear addition

STK-50 alone predicts 2-D log-permeability about as well as porosity + specific surface + Euler number + two-point statistics (R² 0.939 vs 0.937, linear), but porosity alone already gives 0.920. Added to a connectivity-aware incumbent it contributes nothing linearly (ΔR² +0.001, 95 % CI includes 0) and a small nonlinear amount with gradient boosting (+0.011, CI +0.005 to +0.016).

A separate re-evaluation (2026-10-23, repaired kernel release, 26 conventional features) agrees. Adding STK to gradient boosting raises test R² from 0.921 to 0.931 (ΔR² +0.009, interval +0.004 to +0.015), which is 12 % less squared error in log permeability. The best conventional linear model (0.939) is still as good as or better than that combination. Almost all of the signal is in the 46 coordinates the repair did not touch. The result is also sensitive to representation: enlarging the images to 256 px, which adds no information, changes the boosting gain to +0.026 and the linear one to −0.007.

A strong general descriptor with a real but estimator-dependent addition; not a better permeability predictor.

## 8. Label ambiguity: a structure, not a precise rate

"39.8 % of micrographs sit closer to another class's centroid" depends on the metric and on standardizing over this dataset. The robust part is the pattern: ambiguity concentrates in mixed classes and their parents, and mixed classes sit between their parents. As an independent check, the ambiguity score predicts where HOG and Haralick fail (AUC 0.60–0.65): real, but moderate. UHCSDB has one label per image, so this can't be compared with expert disagreement.

## 9. Bainite: test set only, single consensus labels

The bainite benchmark uses the 386-image test set (26 specimens). The training set is augmented tiles, so it wasn't used. Labels are one consensus label per image; the dataset's documented expert subjectivity isn't available per annotator.

## 10. Platform differences

Golden values were made on macOS arm64 and match there to float precision. Elsewhere OpenCV's Canny/contours and BLAS can shift edge- and contour-based coordinates slightly; the tests use tolerances off the reference platform. Benchmark scores move too: on Linux the bainite AUC comes out 0.966 instead of 0.964 with the same scikit-learn, from differences in the numerical libraries under the logistic regression.

## 11. Failure values

If blob detection or island segmentation throws, `blob_count` or `n_islands` is −1 and the related coordinates are NaN. The benchmarks replace NaN with 0 before standardizing.

## 12. scikit-image < 2

The kernel calls `remove_small_objects(min_size=...)`, deprecated in scikit-image 0.26 and due for removal in 2.0. The dependency is pinned below 2.

## 13. A stronger texture baseline (2026-10-23)

The UHCSDB comparison in this repository is against HOG and Haralick. A stricter re-evaluation added an ordinary texture set and found it ranks the classes better than STK.

The re-evaluation was a separate local study, so its features are not shipped here. Its setup:
- the 803 micrographs with a recorded specimen (43 specimens);
- 5-fold `GroupKFold` outer and 3-fold inner, class-balanced logistic regression with the regularization chosen on the inner folds;
- all imputation, scaling and PCA fitted on training folds only;
- freshly computed features: STK from a repaired release of the kernel (the four `g1`/`g3` coordinates of §1 recomputed as path and turning measures), texture-83, and HOG-7200 reduced to 50 by PCA.

| UHCSDB (803 images, 43 specimens) | macro AUC | 95 % interval | log loss |
|---|---|---|---|
| texture-83 (intensity moments and quantiles; uniform LBP at P8/R1 and P16/R2; GLCM statistics) | **0.761** | 0.694–0.833 | 2.466 |
| STK, 46 untouched coordinates | 0.726 | 0.643–0.824 | 1.827 |
| STK-50 (repaired) | 0.699 | 0.620–0.798 | 1.548 |
| texture-83 + HOG-50 | 0.688 | 0.609–0.768 | 2.141 |
| HOG-50 | 0.647 | 0.568–0.708 | 2.510 |
| texture-83 + STK-50 (follow-up, chosen after seeing the above) | 0.721 | | 1.490 |

| Bainite (386 tiles, 26 source-image groups) | macro AUC |
|---|---|
| texture-83 | 0.979 |
| STK-50 (repaired) | 0.965 |
| STK, 46 untouched coordinates | 0.958 |
| HOG-50 | 0.750 |

What this changes:

- **"Beats conventional descriptors" is retired.** STK beats HOG under every setup tried, and Haralick wherever it was run, but not a well-built texture set. Adding STK to texture does not clearly improve macro AUC: −0.040 (−0.098 to +0.020) on UHCSDB, +0.002 (−0.007 to +0.015) on bainite.
- **One favourable result, exploratory.** Texture + STK lowers UHCSDB log loss from 2.466 to 1.490, a paired reduction of 0.976 (group-bootstrap interval 0.268 to 1.782). Accuracy moves from 61.8 % to 63.0 %, which is within noise. The models were selected for AUC and no conventional model was calibrated, so this is a lead for a probability-focused study, not a calibration claim. It does not recur on bainite.
- **The original 0.832 is not wrong; it answers a more lenient question.** The difference from 0.70–0.73 comes from the cohort (961 against 803 images), the fold and model-selection setup, and a few cached image readouts. It is not caused by the four `g1`/`g3` coordinates: the 46 untouched coordinates alone score 0.726.
- **Limits.** Two specimens carry martensite and one mixed class has only eight eligible images. The bainite groups are source-image filename prefixes, not verified physical specimens. All of this reuses archives that had already been examined, so none of it is fresh independent confirmation.

The kernel in this repository stays frozen, so that the recorded results keep reproducing. The repaired release is a separate artifact.
