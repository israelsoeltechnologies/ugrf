# UGRF — Utility-Gated Renewal Forecasting

`ugrf-forecasting` is the reference Python package for **Utility-Gated Renewal Forecasting (UGRF)** for intermittent-demand panels.

This repository packages the frozen Paper-1 forecasting architecture as a small, explicit API:

- `TSB` — Teunter–Syntetos–Babai occurrence/size baseline using the frozen package defaults `alpha=0.10`, `beta=0.10`.
- `RenewalForecaster` — pooled age-dependent renewal hazard plus partially pooled positive-demand magnitude.
- `UtilityGatedRenewal` — per-SKU rolling-origin utility gate that assigns the final forecast to Renewal only when historical cumulative loss improvement over TSB is positive.

The PyPI distribution version is **0.1.0**. In the research project this is the initial public **UGRF v1** package; experimental V1–V4 research-history variants and later ML gates are deliberately not exposed as user-facing models.

## Installation

From the repository root:

```bash
pip install -e .
```

Or install a built wheel:

```bash
pip install dist/ugrf_forecasting-0.1.0-py3-none-any.whl
```

Python 3.10–3.13 is supported.

## Quick start

UGRF is a **panel method** because the Renewal expert pools the age shape and magnitude anchor across SKUs. Rows are SKUs and columns are equally spaced demand periods.

```python
import numpy as np
from ugrf import UtilityGatedRenewal

panel = np.array([
    [0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6],
    [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
    [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
], dtype=float)

model = UtilityGatedRenewal(horizon=3)
model.fit(
    panel,
    utility_origins=[9, 12],   # each historical window is scored on the next 3 periods
    final_origin=15,           # fit final experts using periods [0, 15)
)

forecast = model.predict_cumulative()
print(forecast)
print(model.utility_)
print(model.selected_model_)
```

The gate is fixed at zero:

```text
utility_i = sum_o( |actual_i,o - TSB_i,o| - |actual_i,o - Renewal_i,o| )
use Renewal iff utility_i > 0; otherwise use TSB
```

No dataset-specific utility threshold is estimated by the package.

## Standalone experts

```python
from ugrf import TSB, RenewalForecaster

# One series
baseline = TSB().fit(panel[0, :15])
print(baseline.predict_cumulative(horizon=3))

# Pooled renewal expert
renewal = RenewalForecaster().fit(panel, origin=15)
print(renewal.predict_cumulative(horizon=3))
```

## Diagnostics

Structural helpers live under `ugrf.diagnostics`:

```python
from ugrf.diagnostics import hazard_curve, occurrence_probabilities, simulate_occurrence_paths

ages, hazard = hazard_curve(renewal, panel[0, :15], max_age=12)
q = occurrence_probabilities(renewal, panel[0, :15], horizon=6)
paths = simulate_occurrence_paths(renewal, panel[0, :15], horizon=6, n_paths=1000, seed=7)
```

The simulation helper is diagnostic only. It holds fitted positive magnitude fixed and simulates occurrence paths from the renewal hazard; it is not a new magnitude model.

## Frozen Paper-1 conventions

The package implementation preserves the canonical frozen core:

- six-period cumulative demand was the paper target;
- TSB defaults: `alpha=0.10`, `beta=0.10`;
- Renewal hazard: `logit(h(a)) = alpha_i + beta1*log(a) + beta2*log(a)^2`;
- pooled age coefficients use completed inter-demand intervals and ridge penalty 1 on both age terms;
- local occurrence intercept is fit per SKU with the pooled age shape held fixed;
- positive magnitude uses a pooled log-mean anchor and local log-positive mean with pseudo-count strength 5;
- utility is the sum of rolling-origin **cumulative absolute-error differences**;
- Renewal is selected iff utility is strictly greater than zero;
- no valid utility evidence implies utility zero and therefore TSB fallback.

The frozen benchmark ratios UGRF/TSB are:

| Dataset | UGRF / TSB |
|---|---:|
| Car Parts | 0.9918971759 |
| AUTO | 0.9761486035 |
| RAF | 0.8905672730 |

The script `research/paper1/reproduce_benchmark.py` downloads the same public benchmark snapshots and checks these values to absolute tolerance `5e-6`.

## Repository layout

```text
src/ugrf/                 stable package API
research/paper1/          frozen benchmark reproduction, outside the API
examples/                 small runnable examples
tests/                    deterministic unit tests
docs/                     model specification notes
```

Research-history variants, post-hoc threshold experiments, LM-2, and ML1/ML1-B are intentionally outside the stable UGRF v1 API.

## Citation

See `CITATION.cff`. The associated preprint is **Utility-Gated Renewal Forecasting for Intermittent Demand** by Israel Aloagbaye Igietsemhe, Soel Technologies. A public repository URL / archival identifier can be added once the repository is published.

## License status

A software license has **not yet been selected** for this release candidate. Choose and add the intended license (for example MIT or Apache-2.0) before public distribution.
