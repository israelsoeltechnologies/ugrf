# Changelog

## 0.1.0 — UGRF v1

- Added the stable `TSB`, `RenewalForecaster`, and `UtilityGatedRenewal` APIs.
- Preserved the exact frozen Paper-1 TSB and Renewal numerical conventions.
- Added the canonical zero-utility gate with TSB fallback when historical evidence is unavailable.
- Added structural renewal diagnostics and occurrence-path simulation helpers.
- Added rolling-origin evaluation utilities.
- Added Paper-1 benchmark reproduction script and frozen ratio guards.
- Kept research-history variants and ML gates outside the user-facing API.
