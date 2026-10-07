# UGRF

**Utility-Gated Renewal Forecasting for intermittent demand**

UGRF combines a robust TSB baseline with a pooled age-dependent Renewal candidate and assigns forecast responsibility using historical rolling-origin forecast utility.

<div class="grid cards" markdown>

-   :material-download: **Install**

    ---

    ```bash
    pip install ugrf
    ```

-   :material-chart-line: **Forecast**

    ---

    Generate lead-by-lead point forecasts or cumulative horizon forecasts.

-   :material-filter-check: **Gate**

    ---

    Renewal is used only when it has previously reduced forecast loss relative to TSB.

-   :material-flask: **Reproduce**

    ---

    Reproduce the frozen Car Parts, AUTO, and RAF benchmark from the repository.

</div>

## Quick start

```python
import numpy as np
from ugrf import UtilityGatedRenewal

panel = np.array([
    [0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6],
    [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
    [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
], dtype=float)

model = UtilityGatedRenewal(horizon=3).fit(
    panel,
    utility_origins=[9, 12],
    final_origin=15,
)

# Lead-by-lead point forecasts: shape (n_skus, 3)
point_forecasts = model.predict()

# Cumulative forecast over the same 3-period horizon: shape (n_skus,)
cumulative_forecasts = model.predict_cumulative()

print(point_forecasts)
print(cumulative_forecasts)
print(model.selected_model_)
```

## Core idea

For SKU \(i\) and historical origin \(o\), UGRF compares cumulative absolute error from TSB and Renewal:

\[
G_{i,o}=L_{i,o}(\mathrm{TSB})-L_{i,o}(\mathrm{Renewal})
\]

Historical utility is:

\[
U_i=\sum_o G_{i,o}
\]

The final gate is deliberately simple:

\[
\mathrm{Renewal}\quad \text{if } U_i>0,\qquad
\mathrm{TSB}\quad \text{otherwise.}
\]

!!! note "Mechanism evidence is not forecast utility"
    UGRF does not assign forecast responsibility to Renewal merely because an age-dependent renewal structure can be estimated. Renewal must first demonstrate historical out-of-sample forecasting value relative to the available baseline.

## Where to go next

- [Getting Started](getting-started.md) — install UGRF and run a first model.
- [Configuration](configuration.md) — understand every public configuration and the frozen v1 constants.
- [Forecasting](forecasting.md) — point forecasts, cumulative forecasts, shapes, and horizon behavior.
- [Model Specification](model_spec.md) — mathematical details of the frozen architecture.
- [Reproducibility](reproducibility.md) — reproduce the Paper-1 benchmark.
- [API Reference](api.md) — generated directly from the package docstrings.
