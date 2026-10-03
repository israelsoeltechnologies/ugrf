# Frozen-core provenance

This package was migrated from the frozen Paper-1 benchmark core identified as:

`UGRF_FROZEN_CORE_2026-09-27_V3_EXACT`

The stable implementation preserves the canonical forecasting components only:

- TSB with `alpha=0.10`, `beta=0.10`;
- pooled age-shape Renewal expert;
- pooled positive-demand anchor with local shrinkage strength 5;
- SKU-level cumulative absolute-error utility;
- strict zero gate (`utility > 0` selects Renewal; otherwise TSB);
- six-period horizon in the frozen benchmark.

Frozen UGRF/TSB benchmark guards:

- Car Parts: `0.9918971759288252`
- AUTO: `0.9761486034991342`
- RAF: `0.8905672729887494`

Tolerance: absolute `5e-6`.

Broader conventional-model benchmarking, threshold sensitivity, LM-2, ML1, and ML1-B are research analyses and are not part of the stable v0.1.0 public API.
