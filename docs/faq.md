# FAQ

## What does `predict()` return?

`UtilityGatedRenewal.predict()` returns the full lead-by-lead point forecast path.

For 100 SKUs and a six-period horizon:

```python
forecast = model.predict()
forecast.shape
# (100, 6)
```

## What does `predict_cumulative()` return?

One cumulative forecast per SKU over the fitted horizon:

```python
total = model.predict_cumulative()
total.shape
# (100,)
```

## Why can't I fit with horizon 6 and then call `predict(12)`?

Because the gate was learned from six-period historical forecast loss. Refit UGRF with `horizon=12` so the forecasting target and allocation criterion remain aligned.

## Why is UGRF a panel model?

The Renewal expert estimates pooled age structure and a pooled positive-demand anchor across SKUs. Individual SKUs retain local occurrence levels and partially pooled magnitudes.

## Does Renewal always win when there is renewal structure?

No. Renewal receives forecast responsibility only if it previously reduced the historical forecasting loss relative to TSB.

## What happens when utility is exactly zero?

TSB is selected. The gate is strictly `utility > 0`.

## Does UGRF v1 provide prediction intervals?

No. Stable v1 is a point-forecast model. Conformal prediction intervals are planned as a v1.1 uncertainty extension.

## Can I change TSB smoothing parameters?

Yes, but the frozen Paper-1 model uses `alpha=0.10` and `beta=0.10`. Changing them creates a user-configured variant rather than an exact benchmark reproduction.

## Can I change the utility threshold?

Not in the canonical v1 API. The frozen rule is `utility > 0`.
