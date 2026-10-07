# Diagnostics

UGRF includes structural helpers for understanding the fitted Renewal expert. These diagnostics are explanatory and do **not** override the historical utility gate.

## Hazard curve

```python
from ugrf.diagnostics import hazard_curve

ages, hazard = hazard_curve(
    model.renewal_model_,
    panel[0, :final_origin],
    max_age=18,
)
```

The returned hazard is the conditional occurrence probability at each demand age.

## Marginal occurrence probabilities

```python
from ugrf.diagnostics import occurrence_probabilities

q = occurrence_probabilities(
    model.renewal_model_,
    panel[0, :final_origin],
    horizon=6,
)
```

`q[h]` is the marginal probability of an occurrence at that future lead after integrating over possible event/no-event paths and age resets.

## Series summary

```python
from ugrf.diagnostics import series_summary

summary = series_summary(
    model.renewal_model_,
    panel[0, :final_origin],
    horizon=6,
)
```

The summary includes current age, occurrence intercept, positive magnitude, expected events, cumulative Renewal forecast, and probability of at least one event.

## Simulate occurrence paths

```python
from ugrf.diagnostics import simulate_occurrence_paths

paths = simulate_occurrence_paths(
    model.renewal_model_,
    panel[0, :final_origin],
    horizon=6,
    n_paths=1000,
    seed=7,
)
```

The result has shape `(n_paths, horizon)` and contains binary event indicators.

!!! note
    This helper simulates occurrence paths only. The frozen v1 model holds fitted positive magnitude deterministic in this diagnostic and does not introduce a new probabilistic magnitude model.
