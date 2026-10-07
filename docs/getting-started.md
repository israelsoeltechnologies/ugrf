# Getting Started

## Installation

Install the published package from PyPI:

```bash
pip install ugrf
```

For local development from a cloned repository:

```bash
git clone https://github.com/israelsoeltechnologies/ugrf.git
cd ugrf
pip install -e .
```

## Input format

UGRF is a **panel forecasting method**. Supply a two-dimensional array where:

- rows are SKUs or demand series;
- columns are equally spaced chronological periods;
- demand values are finite and non-negative.

```python
import numpy as np

panel = np.array([
    [0, 0, 4, 0, 0, 5, 0, 0, 6],
    [0, 2, 0, 0, 3, 0, 0, 4, 0],
    [1, 0, 0, 1, 0, 0, 2, 0, 0],
], dtype=float)
```

The Renewal expert pools information across rows, so the panel is part of the model definition rather than merely a batch of unrelated calls.

## Fit a model

Choose a forecast horizon and historical utility origins:

```python
from ugrf import UtilityGatedRenewal

model = UtilityGatedRenewal(horizon=3)

model.fit(
    panel,
    utility_origins=[3],
    final_origin=6,
)
```

At each utility origin, TSB and Renewal are fitted using only the history available at that origin and evaluated over the next `horizon` periods.

## Generate forecasts

```python
point_forecasts = model.predict()
cumulative_forecasts = model.predict_cumulative()
```

For `n` SKUs and horizon `h`:

```text
model.predict()             -> shape (n, h)
model.predict_cumulative()  -> shape (n,)
```

For each SKU, the cumulative forecast is the sum of the lead-by-lead point forecasts:

```python
np.allclose(
    model.predict().sum(axis=1),
    model.predict_cumulative(),
)
```

## Inspect the gate

```python
print(model.selected_model_)
print(model.utility_)
```

A strictly positive utility activates Renewal. Zero or negative utility falls back to TSB.

## Inspect the two experts

Cumulative expert forecasts:

```python
print(model.tsb_forecast_)
print(model.renewal_forecast_)
```

Lead-by-lead expert paths:

```python
print(model.tsb_path_)
print(model.renewal_path_)
print(model.forecast_path_)
```

Read [Configuration](configuration.md) before applying UGRF to your own operational dataset.
