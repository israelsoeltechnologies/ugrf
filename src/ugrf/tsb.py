from __future__ import annotations

import numpy as np

from ._validation import as_series, validate_horizon


class TSB:
    """Frozen Teunter–Syntetos–Babai baseline used by UGRF.

    Parameters
    ----------
    alpha : float, default=0.10
        Smoothing rate for demand occurrence probability.
    beta : float, default=0.10
        Smoothing rate for positive demand size.

    Notes
    -----
    Initialization and update order intentionally match the frozen Paper-1
    implementation. The fitted forecast is constant across future leads.
    """

    def __init__(self, alpha: float = 0.10, beta: float = 0.10):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 < beta <= 1:
            raise ValueError("beta must be in (0, 1].")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def fit(self, y) -> "TSB":
        y = as_series(y)
        positives = y[y > 0]
        p = ((y > 0).sum() + 0.5) / (len(y) + 1.0)
        z = float(positives.mean()) if len(positives) else 0.0

        for value in y:
            event = 1.0 if value > 0 else 0.0
            p += self.alpha * (event - p)
            if value > 0:
                z += self.beta * (float(value) - z)

        self.p_ = float(p)
        self.z_ = float(z)
        self.rate_ = max(0.0, float(p * z))
        self.n_obs_ = int(len(y))
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "rate_"):
            raise RuntimeError("TSB is not fitted. Call fit() first.")

    def predict(self, horizon: int = 1) -> np.ndarray:
        self._check_fitted()
        h = validate_horizon(horizon)
        return np.full(h, self.rate_, dtype=float)

    def predict_cumulative(self, horizon: int = 6) -> float:
        self._check_fitted()
        h = validate_horizon(horizon)
        return float(h * self.rate_)
