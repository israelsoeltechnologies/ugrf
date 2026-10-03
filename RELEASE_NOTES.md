# UGRF v1 / `ugrf-forecasting` 0.1.0 release notes

This is the first packaged release of the frozen Utility-Gated Renewal Forecast architecture.

Validation completed in the build environment:

- 10 deterministic unit/regression tests passed.
- Source modules compiled successfully.
- Wheel built successfully and imported from an isolated target directory.
- Stable top-level API verified as `TSB`, `RenewalForecaster`, `UtilityGatedRenewal`.
- Frozen Paper-1 benchmark reproduction script includes hard guards for Car Parts, AUTO, and RAF at absolute tolerance `5e-6`.

The full three-panel benchmark requires the external public data snapshots and is intentionally kept as a separate reproduction step under `research/paper1/`.

Before public distribution, select and add the desired software license and then add the repository / archival URL to package metadata and `CITATION.cff`.
