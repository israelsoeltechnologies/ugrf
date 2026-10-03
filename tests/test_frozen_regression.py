import numpy as np

from ugrf import RenewalForecaster, UtilityGatedRenewal


PANEL = np.array([
    [0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6],
    [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
    [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
], dtype=float)


def test_frozen_synthetic_regression_values():
    renewal = RenewalForecaster().fit(PANEL, origin=15)
    assert np.allclose(
        renewal.beta_,
        [1.4908086763291168, 2.3858486217457173],
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(renewal.global_logmean_, 0.4045134079251824, rtol=0, atol=1e-12)
    assert np.allclose(
        renewal.cumulative_for_training_panel(horizon=3),
        [2.3222926558227273, 2.414842129171921, 1.7942174769938724],
        rtol=0,
        atol=1e-12,
    )

    ugrf = UtilityGatedRenewal(horizon=3).fit(PANEL, utility_origins=[9, 12], final_origin=15)
    assert np.allclose(
        ugrf.utility_,
        [-6.047452020420966, -1.3704724831746482, -0.023117573376824296],
        rtol=0,
        atol=1e-12,
    )
