import numpy as np

from ugrf import TSB


def test_tsb_matches_frozen_update_order():
    y = np.array([0.0, 2.0, 0.0, 4.0, 0.0])
    alpha = beta = 0.10
    positives = y[y > 0]
    p = ((y > 0).sum() + 0.5) / (len(y) + 1.0)
    z = positives.mean()
    for value in y:
        p += alpha * ((1.0 if value > 0 else 0.0) - p)
        if value > 0:
            z += beta * (value - z)
    expected = p * z

    model = TSB(alpha=alpha, beta=beta).fit(y)
    assert np.isclose(model.rate_, expected)
    assert np.isclose(model.predict_cumulative(6), 6 * expected)


def test_all_zero_tsb_is_zero():
    model = TSB().fit([0, 0, 0, 0])
    assert model.predict_cumulative(6) == 0.0
