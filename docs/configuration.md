# Configuration

This page separates **public user configuration** from the **frozen UGRF v1 research constants**.

That distinction matters: UGRF can be used for different operational horizons, but changing the internal research constants produces a variant rather than an exact reproduction of the frozen Paper-1 model.

## `UtilityGatedRenewal`

```python
from ugrf import UtilityGatedRenewal

model = UtilityGatedRenewal(
    horizon=6,
    tsb_alpha=0.10,
    tsb_beta=0.10,
)
```

| Parameter | Default | Meaning | Paper-1 value |
|---|---:|---|---:|
| `horizon` | `6` | Number of future periods in the forecast target and historical utility loss | `6` |
| `tsb_alpha` | `0.10` | TSB smoothing rate for occurrence probability | `0.10` |
| `tsb_beta` | `0.10` | TSB smoothing rate for positive-demand size | `0.10` |

### Horizon is part of the gate

UGRF does not treat the horizon as a cosmetic prediction option. Historical utility is computed for the same horizon that will later be forecast.

A model fitted with:

```python
model = UtilityGatedRenewal(horizon=6)
```

supports:

```python
model.predict()
model.predict(6)
model.predict_cumulative()
model.predict_cumulative(6)
```

but a different horizon requires refitting:

```python
model = UtilityGatedRenewal(horizon=12)
model.fit(panel, utility_origins=..., final_origin=...)
forecast = model.predict()
```

## `fit()` configuration

```python
model.fit(
    panel,
    utility_origins=[30, 33, 36, 39],
    final_origin=45,
)
```

| Argument | Meaning |
|---|---|
| `panel` | 2-D non-negative demand array with shape `(n_series, n_periods)` |
| `utility_origins` | Historical origins used to compare TSB and Renewal |
| `final_origin` | End of the history used to fit the final experts |

### Leakage rule

Every historical utility origin must satisfy:

\[
o + H \leq T_{\text{final}}
\]

For example, with `horizon=6` and `final_origin=45`:

```python
utility_origins=[30, 33, 36, 39]
```

is valid because the final utility window ends at period 45.

## Choosing utility origins

UGRF v1 deliberately does **not** invent utility origins automatically. Useful principles are:

1. each origin must leave a complete evaluation horizon;
2. utility windows must occur before the final forecast origin;
3. origins should represent genuinely historical decisions;
4. avoid selecting origins after examining final-period performance;
5. use the same horizon that matters operationally.

The frozen benchmark uses:

| Dataset | Utility origins | Final origin | Horizon |
|---|---|---:|---:|
| Car Parts | `30, 33, 36, 39` | `45` | `6` |
| AUTO | `12` | `18` | `6` |
| RAF | `48, 54, 60, 66, 72` | `78` | `6` |

## TSB configuration

```python
from ugrf import TSB

model = TSB(alpha=0.10, beta=0.10)
```

| Parameter | Frozen default | Purpose |
|---|---:|---|
| `alpha` | `0.10` | Smooth occurrence probability |
| `beta` | `0.10` | Smooth positive-demand magnitude |

Changing these values is allowed when using `TSB` independently. Changing them inside `UtilityGatedRenewal` creates a user-configured UGRF variant rather than an exact Paper-1 reproduction.

## Renewal configuration

```python
from ugrf import RenewalForecaster

renewal = RenewalForecaster(positive_shrinkage=5.0)
```

| Quantity | Frozen value | Role |
|---|---:|---|
| Positive-magnitude pseudo-count | `5.0` | Shrinks local log-positive magnitude toward the pooled panel anchor |
| Age-shape ridge penalty | `1.0` | Regularizes the two pooled age coefficients |
| Utility threshold | `0.0` | Renewal is selected only if historical utility is strictly positive |

!!! warning "Reproduction boundary"
    If you change the TSB smoothing rates, magnitude shrinkage, utility threshold, recency weighting, or loss definition, describe the result as a **UGRF variant** rather than the frozen UGRF v1 specification.

## Missing and negative values

The stable API expects finite, non-negative demand values.

Prepare missing observations according to the meaning of your dataset before calling UGRF. Do not silently replace missing demand with zero unless zero is the correct domain interpretation.

Negative values are rejected.

## All-zero histories

If an SKU contains no positive demand before the final origin, the frozen Renewal expert has no valid local forecast. UGRF falls back to TSB, which is zero for an all-zero history.

## Operational horizon

For inventory applications, a common interpretation is:

```text
horizon = replenishment lead time expressed in demand periods
```

With weekly demand and a six-week lead time:

```python
model = UtilityGatedRenewal(horizon=6)
```

The historical utility windows should then also be evaluated on six-week cumulative demand.
