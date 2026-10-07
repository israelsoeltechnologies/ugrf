# Forecasting

UGRF exposes both the **forecast path** and the **cumulative horizon forecast**.

## Lead-by-lead point forecasts

```python
forecast = model.predict()
```

For `n` SKUs and fitted horizon `h`, the result has shape:

```text
(n, h)
```

Conceptually, one row may look like:

```python
[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

Those are the point forecasts for leads 1 through 6.

## Expert selection applies to the whole path

UGRF makes one gate decision per SKU.

If SKU \(i\) selects Renewal:

\[
\hat{\mathbf y}^{UGRF}_i=\hat{\mathbf y}^{Renewal}_i
\]

If it selects TSB:

\[
\hat{\mathbf y}^{UGRF}_i=\hat{\mathbf y}^{TSB}_i
\]

The gate does not switch experts separately at each future lead.

## Cumulative forecasts

```python
cumulative = model.predict_cumulative()
```

This returns one number per SKU:

\[
\hat D_i^{(H)}=\sum_{h=1}^{H}\hat y_{i,t+h}
\]

```python
point = model.predict()
total = model.predict_cumulative()

assert np.allclose(point.sum(axis=1), total)
```

The cumulative forecast is the target used in the frozen Paper-1 benchmark.

## Why TSB paths are flat

The frozen TSB expert produces a constant future rate. Therefore a TSB-selected SKU may look like:

```python
[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
```

Its six-period cumulative forecast is `4.8`.

The Renewal expert can produce a non-flat path because occurrence probability depends on age and future age-state propagation.

## Fitted attributes

| Attribute | Shape | Meaning |
|---|---|---|
| `forecast_path_` | `(n_series, horizon)` | Final UGRF point-forecast path |
| `tsb_path_` | `(n_series, horizon)` | TSB point-forecast path |
| `renewal_path_` | `(n_series, horizon)` | Renewal point-forecast path |
| `forecast_` | `(n_series,)` | Final cumulative UGRF forecast |
| `tsb_forecast_` | `(n_series,)` | Cumulative TSB forecast |
| `renewal_forecast_` | `(n_series,)` | Cumulative Renewal forecast |
| `selected_model_` | `(n_series,)` | `"TSB"` or `"Renewal"` |
| `utility_` | `(n_series,)` | Historical Renewal utility |
| `utility_evidence_count_` | `(n_series,)` | Number of valid utility windows |

## Retrospective evaluation

If the fitted panel contains the complete future evaluation window:

```python
scores = model.score_final_window()
```

The result includes actual cumulative demand, final UGRF forecast error, TSB error, and Renewal error.
