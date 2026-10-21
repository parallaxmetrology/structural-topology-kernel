# Changelog

## 0.1.1 (2026-10-21)

- Fix: `measure()` / `prepare_micrograph()` on NumPy arrays other than 8-bit (float images in [0, 1], 16-bit, non-uint8 color) clipped them to near-black and returned wrong coordinates. Arrays are now converted the way PIL converts the same picture from a file; file paths and PIL images were never affected, nor any published result.
- `stk measure` writes valid JSON (NaN as null).
- The kernel's scikit-image deprecation warning is no longer shown on every call.
- Verified `STKFeaturizer` inside matminer 0.10.1 (serial, `n_jobs=2`, `MultipleFeaturizer`).

## 0.1.0 (2026-10-21)

First public release of the research kernel (June 2026) as `stkernel`: frozen 50-coordinate kernel, micrograph preparation, `measure`/`measure_many`, `STKFeaturizer` (matminer-compatible), `stk` CLI, benchmarks and feature tables that reproduce the recorded UHCSDB, bainite and permeability results exactly on the reference platform.
