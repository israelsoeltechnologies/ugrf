import numpy as np
import pytest

from ugrf import UtilityGatedRenewal


def panel():
    return np.array([
        [0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6],
        [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
        [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)


def test_zero_evidence_means_tsb_fallback():
    model = UtilityGatedRenewal(horizon=3).fit(panel(), utility_origins=[], final_origin=15)
    assert np.all(model.utility_ == 0)
    assert np.all(model.selected_model_ == "TSB")
    assert model.predict_cumulative()[-1] == 0.0


def test_selection_matches_strict_positive_utility():
    model = UtilityGatedRenewal(horizon=3).fit(panel(), utility_origins=[9, 12], final_origin=15)
    expected = (model.utility_ > 0) & np.isfinite(model.renewal_forecast_)
    assert np.array_equal(model.renewal_active_, expected)
    assert np.allclose(
        model.predict_cumulative(),
        np.where(expected, model.renewal_forecast_, model.tsb_forecast_),
        equal_nan=True,
    )


def test_prevents_utility_leakage():
    with pytest.raises(ValueError, match="leaks beyond"):
        UtilityGatedRenewal(horizon=3).fit(panel(), utility_origins=[13], final_origin=15)


def test_horizon_is_part_of_gate_definition():
    model = UtilityGatedRenewal(horizon=3).fit(panel(), utility_origins=[9, 12], final_origin=15)
    with pytest.raises(ValueError, match="different horizon"):
        model.predict_cumulative(horizon=4)
