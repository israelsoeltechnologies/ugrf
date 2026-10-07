# Roadmap

## UGRF v1

The stable v1 architecture contains:

- TSB baseline;
- pooled age-dependent Renewal expert;
- partially pooled positive-demand magnitude;
- rolling-origin historical utility;
- strict zero-utility gate;
- lead-by-lead point forecasts;
- cumulative horizon forecasts;
- Renewal structural diagnostics;
- frozen Paper-1 reproduction workflow.

## UGRF v1.1 — planned conformal uncertainty

The next planned extension is a conformal prediction layer around the frozen point forecast:

```text
Frozen UGRF point forecast
        +
temporally valid conformal calibration
        =
prediction interval
```

The conformal layer should not silently change the v1 point forecast or v1 expert-selection rule.

Potential outputs include:

```python
result.point
result.lower
result.upper
result.level
result.selected_model
```

Intermittent demand is skewed and zero-heavy, so the intended direction is temporally valid and potentially asymmetric calibration rather than a generic Gaussian interval.

## Later research directions

Potential later extensions include adaptive conformal calibration, probabilistic magnitude models, learned utility routing, cold-start transfer, environment-aware allocation, and explicit inventory/service-level objectives.
