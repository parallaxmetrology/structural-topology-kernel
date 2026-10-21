# The 50 coordinates

Input: a 2-D grayscale field, min-max normalized to [0, 1] (call it *L*). Micrograph files go through `prepare_micrograph` first: 8-bit grayscale, dark info bar at the bottom removed, 512 × 512 Lanczos, min-max. Gradients are central differences (`np.gradient`); *G* is the gradient magnitude, normalized to [0, 1]. "Active" pixels are those with normalized *G* ≥ 0.18.

The definitions below describe what the code computes, including where that differs from what a name suggests (flagged ⚠, details in [known-issues.md](known-issues.md)).

## Gradient-field geometry (on *G*)

| coordinate | what it measures |
|---|---|
| `delta_x`, `delta_y` | offset of the gradient-mass centroid from the frame center, as a fraction of width / height |
| `r_v` | share of pixels with almost no gradient (normalized *G* < 0.18): flat area |
| `mu` | share of gradient mass carried by the top quarter of pixels: how concentrated the edges are |
| `rho_r` | active pixels per unit area of their convex hull (× 100, capped at 150): packing |
| `x_p` | share of gradient mass in a border band 15 % wide |
| `theta` | orientation **concentration** of active gradients: 1 − normalized entropy of an 8-bin direction histogram (0 isotropic, 1 one direction). Not an angle. |
| `d_s` | mean distance from active pixels to the nearest inactive one, × 2 / min(W, H): thickness of the active structure |
| `sdi` | mean distance of active pixels from their centroid / image diagonal: spread |

## Luminance structure (on *L*)

| coordinate | what it measures |
|---|---|
| `g1_pathlen`, `g1_curvature` | ⚠ from the luminance centroid after Gaussian blur at σ = 2, 4, 8, 16, 32. As implemented, the sum of the five centroid position lengths and the mean angle between successive position vectors, not a path length and turning angle. |
| `g2_void` | share of pixels below a local (adaptive) threshold after smoothing and morphological cleanup: dark-phase fraction |
| `g2_nvoids` | number of such dark regions (larger than max(64 px, 0.2 % of the image)) |
| `g2_AR` | their mean aspect ratio (major / minor axis) |
| `g2_cut` | area share of the largest one |
| `g3_kvar`, `g3_infl` | ⚠ on Canny contours of *L*: variance of the angle between successive contour-point position vectors, and the rate of sign changes of that angle around its mean. Contour-shape statistics, but not curvature in the usual sense. |
| `g4_oent` | gradient-direction entropy of *L* (36 bins), × 5 (0–5) |
| `g4_slope` | change in that entropy from σ = 1 to σ = 32 blur, per unit log-scale |
| `sci` | composite coherence score, 0–5: regional uniformity of gradient strength (quadrants), direction alignment, consistency of edge strength across blur scales, and an autocorrelation "rhythm" term (⚠ first 1,000 pixels only) |

## Radial organization

| coordinate | what it measures |
|---|---|
| `rc_drc` | how well the radial profile of gradient mass fits an exponential decay around its own centroid, minus the same around the frame center (fit quality = 1 − Jensen–Shannon distance) |
| `rc_dr` | distance of the gradient-mass centroid from the center, normalized to the corner distance |

## Texture (gray-level co-occurrence)

On *L* smoothed at σ = 2 (`_fine`) and σ = 16 (`_coarse`), quantized to 32 levels, distance 1, horizontal and vertical pairs averaged:
`tex_contrast_*`, `tex_entropy_*`, `tex_homogeneity_*`, `tex_energy_*`.

## Blobs (difference of Gaussians, σ 3–20)

| coordinate | what it measures |
|---|---|
| `blob_count` | detected blobs (precipitates, grains, particles); −1 if detection failed |
| `blob_density` | blobs per 10,000 px |
| `blob_size_mean`, `blob_size_cv` | mean blob radius (px) and its coefficient of variation |

## Islands and corridors

The intensity-normalized gradient is smoothed into a density map (smoothing scale set by `rho_r`), peaks are found and grown by watershed into "islands".

| coordinate | what it measures |
|---|---|
| `n_islands` | number of islands (−1 if the step failed) |
| `island_size_cv` | coefficient of variation of island areas |
| `island_spread` | mean distance of island centroids from their mean / diagonal |
| `resistance_mean` | mean of (1 − density) sampled along straight corridors between every pair of island centroids: how "empty" the paths between them are |
| `drain_confidence` | how strongly one island dominates the inbound corridor cost (0 = even, toward 1 = one island stands out) |

## Spectrum, edges, tone

| coordinate | what it measures |
|---|---|
| `ds_low`, `ds_mid`, `ds_high` | share of Fourier power in the lower, middle and upper thirds of radial frequency |
| `edge_density` | share of pixels on Canny edges (thresholds 30, 90 on 8-bit *L*) |
| `nils` | median of \|∇*L*\| / *L* on Canny edge pixels, clipped to 10: NILS-like edge sharpness (not normalized by feature width) |
| `ler` | standard deviation of contour points from a moving-average contour / diagonal: LER-like edge roughness |
| `gap_rv`, `gap_theta` | void ratio and orientation concentration of the "gap field": strong gradients (top quarter) that Canny did not mark as edges |
| `tonal_shadow`, `tonal_mid`, `tonal_high` | share of pixels with *L* < 0.33, 0.33–0.66, ≥ 0.66 |

`ds_*`, `edge_density`, `nils`, `ler` and the three tonal channels are the "acquisition-sensitive" group: they respond to focus, contrast and exposure. The benchmark's ablation drops them to show the classification result doesn't depend on them (0.829 without vs 0.832 with).

## Names shared with other Parallax instruments

Some short names (`delta_x`, `r_v`, `theta`, `d_s`, `nils`, `ler`…) also appear in [Visual Thinking Lens](https://github.com/parallaxmetrology/visual-thinking-lens) and [Inspection Coordinates](https://github.com/parallaxmetrology/inspection-coordinates). Here most of the first group are computed on the gradient field with different thresholds, so values aren't interchangeable across the three.
