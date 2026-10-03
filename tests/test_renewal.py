import numpy as np

from ugrf import RenewalForecaster
from ugrf.diagnostics import occurrence_probabilities, series_summary


def panel():
    return np.array([
        [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
        [1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0],
        [0, 0, 4, 0, 5, 0, 0, 6, 0, 7, 0, 0],
    ], dtype=float)


def test_renewal_is_deterministic_and_nonnegative():
    p = panel()
    a = RenewalForecaster().fit(p, origin=9)
    b = RenewalForecaster().fit(p, origin=9)
    fa = a.cumulative_for_training_panel(horizon=3)
    fb = b.cumulative_for_training_panel(horizon=3)
    assert np.allclose(fa, fb, equal_nan=True)
    assert np.all(fa[np.isfinite(fa)] >= 0)
    assert np.allclose(a.beta_, b.beta_)


def test_occurrence_probabilities_are_valid():
    p = panel()
    model = RenewalForecaster().fit(p, origin=9)
    q = occurrence_probabilities(model, p[0, :9], horizon=5)
    assert q.shape == (5,)
    assert np.all((q >= 0) & (q <= 1))
    summary = series_summary(model, p[0, :9], horizon=5)
    assert 0 <= summary["probability_at_least_one_event"] <= 1
