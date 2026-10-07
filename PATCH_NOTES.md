# UGRF v1 working update — point-forecast API

This is an **unreleased working-tree update** prepared against the current public repository.

## Main API change

`UtilityGatedRenewal.predict()` now returns the full lead-by-lead point forecast path:

```python
model = UtilityGatedRenewal(horizon=6).fit(...)
point_forecasts = model.predict()
# shape: (n_skus, 6)
```

`predict_cumulative()` remains the cumulative operational target:

```python
cumulative = model.predict_cumulative()
# shape: (n_skus,)
```

For every SKU:

```python
np.allclose(model.predict().sum(axis=1), model.predict_cumulative())
```

The selected UGRF expert is applied to the entire forecast path.

## Frozen-model protection

The existing cumulative TSB, Renewal, gate, and `forecast_` calculations are retained. The new path API is layered on top so the Paper-1 reproduction target is not redefined.

The fitted horizon remains part of the gate definition. A model fitted with horizon 6 accepts `predict()` or `predict(6)` but rejects `predict(12)`.

## Other cleanup

- README now uses `pip install ugrf`.
- Apache-2.0 is reflected in README, `pyproject.toml`, and `CITATION.cff`.
- Changelog records these items under `Unreleased`.
- Existing `RELEASE_NOTES.md` for 0.1.0 should remain historical and unchanged.

## Applying

Copy the files in this ZIP over the same paths in the repository, then run:

```bash
python -m pytest
python -m build
```

During ongoing review, you can commit/push these changes without publishing a new PyPI release. Before making the updated API available through `pip install ugrf`, bump the package version and publish a new immutable release.
