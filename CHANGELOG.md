# Changelog

## Unreleased

- Added `UtilityGatedRenewal.predict()` for lead-by-lead point forecasts.
- Added `renewal_path_`, `tsb_path_`, and `forecast_path_` fitted attributes.
- Kept `predict_cumulative()` as the cumulative operational forecast used by the frozen Paper-1 benchmark.
- Enforced the fitted horizon for both `predict()` and `predict_cumulative()` so the forecast target remains aligned with the UGRF utility gate.
- Added regression tests verifying that point forecasts sum to the unchanged cumulative forecast.
- Updated public examples and model documentation to distinguish point and cumulative forecasts.
- Updated package documentation to reflect the Apache-2.0 license and PyPI installation.

## 0.1.0 — UGRF v1

- Added the stable `TSB`, `RenewalForecaster`, and `UtilityGatedRenewal` APIs.
- Preserved the exact frozen Paper-1 TSB and Renewal numerical conventions.
- Added the canonical zero-utility gate with TSB fallback when historical evidence is unavailable.
- Added structural renewal diagnostics and occurrence-path simulation helpers.
- Added rolling-origin evaluation utilities.
- Added Paper-1 benchmark reproduction script and frozen ratio guards.
- Kept research-history variants and ML gates outside the user-facing API.
