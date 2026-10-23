# Changelog

## Documentation, 2026-10-23 (no code change)

- Added the result of a stricter re-evaluation against a conventional texture baseline. An 83-feature texture set ranks UHCSDB classes better than STK (0.761 against 0.70–0.73), and adding STK to it does not clearly improve macro AUC. "Beats conventional descriptors" is retired; STK still beats HOG, and Haralick where it was run. One exploratory probability-score gain is recorded. See README "How it compares" and known-issues §13.
- Permeability wording changed from "competitive, not additive" to "competitive, with a small nonlinear addition", matching the bootstrap and the re-evaluation (known-issues §7).
- The kernel, the feature tables and the recorded results are unchanged and still reproduce.

## 0.1.1 (2026-10-21)

- Fix: `measure()` / `prepare_micrograph()` on NumPy arrays other than 8-bit (float images in [0, 1], 16-bit, non-uint8 color) clipped them to near-black and returned wrong coordinates. Arrays are now converted the way PIL converts the same picture from a file; file paths and PIL images were never affected, nor any published result.
- `stk measure` writes valid JSON (NaN as null).
- The kernel's scikit-image deprecation warning is no longer shown on every call.
- Verified `STKFeaturizer` inside matminer 0.10.1 (serial, `n_jobs=2`, `MultipleFeaturizer`).

## 0.1.0 (2026-10-21)

First public release of the research kernel (June 2026) as `stkernel`: frozen 50-coordinate kernel, micrograph preparation, `measure`/`measure_many`, `STKFeaturizer` (matminer-compatible), `stk` CLI, benchmarks and feature tables that reproduce the recorded UHCSDB, bainite and permeability results exactly on the reference platform.
