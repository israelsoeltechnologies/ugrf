from __future__ import annotations

import numpy as np

from ._validation import as_panel, validate_horizon, validate_origin, validate_origins
from .evaluation import actual_cumulative, historical_utility, tsb_panel_forecast
from .renewal import RenewalForecaster


class UtilityGatedRenewal:
    """Canonical Utility-Gated Renewal Forecast (UGRF).

    Renewal is selected for SKU i iff the sum of prior rolling-origin
    cumulative absolute-error gains over TSB is strictly positive.
    """

    def __init__(self, horizon: int = 6, tsb_alpha: float = 0.10, tsb_beta: float = 0.10):
        self.horizon = validate_horizon(horizon)
        if not 0 < tsb_alpha <= 1:
            raise ValueError("tsb_alpha must be in (0, 1].")
        if not 0 < tsb_beta <= 1:
            raise ValueError("tsb_beta must be in (0, 1].")
        self.tsb_alpha = float(tsb_alpha)
        self.tsb_beta = float(tsb_beta)

    def fit(self, panel, utility_origins, final_origin: int | None = None) -> "UtilityGatedRenewal":
        arr = as_panel(panel)
        final = validate_origin(final_origin, arr.shape[1])
        origins = validate_origins(tuple(utility_origins), final_origin=final, horizon=self.horizon)

        utility, seen = historical_utility(
            arr,
            origins,
            horizon=self.horizon,
            tsb_alpha=self.tsb_alpha,
            tsb_beta=self.tsb_beta,
        )

        renewal_model = RenewalForecaster().fit(arr, origin=final)
        renewal = renewal_model.cumulative_for_training_panel(horizon=self.horizon)
        tsb = tsb_panel_forecast(
            arr,
            final,
            horizon=self.horizon,
            alpha=self.tsb_alpha,
            beta=self.tsb_beta,
        )

        renewal_valid = np.isfinite(renewal)
        selected = (utility > 0.0) & renewal_valid
        forecast = np.where(selected, renewal, tsb)

        # All-zero histories have no valid frozen Renewal forecast. Their TSB
        # forecast is operationally zero, so expose zero rather than NaN.
        all_zero = ~np.any(arr[:, :final] > 0, axis=1)
        tsb = tsb.copy()
        tsb[all_zero] = 0.0
        forecast = forecast.copy()
        forecast[all_zero] = 0.0

        self.panel_ = arr
        self.final_origin_ = final
        self.utility_origins_ = origins
        self.utility_ = utility
        self.utility_evidence_count_ = seen
        self.renewal_model_ = renewal_model
        self.renewal_forecast_ = renewal
        self.tsb_forecast_ = tsb
        self.renewal_active_ = selected
        self.selected_model_ = np.where(selected, "Renewal", "TSB")
        self.forecast_ = forecast
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "forecast_"):
            raise RuntimeError("UtilityGatedRenewal is not fitted. Call fit() first.")

    def predict_cumulative(self, horizon: int | None = None) -> np.ndarray:
        self._check_fitted()
        if horizon is not None and validate_horizon(horizon) != self.horizon:
            raise ValueError(
                "The UGRF gate was evaluated for a different horizon. "
                "Refit UtilityGatedRenewal with the desired horizon."
            )
        return self.forecast_.copy()

    def score_final_window(self) -> dict[str, np.ndarray]:
        """Evaluate the fitted final-origin forecast when the future window exists."""
        self._check_fitted()
        if self.final_origin_ + self.horizon > self.panel_.shape[1]:
            raise ValueError("The fitted panel does not contain the complete final evaluation window.")
        actual = actual_cumulative(self.panel_, self.final_origin_, self.horizon)
        return {
            "actual": actual,
            "forecast": self.forecast_.copy(),
            "absolute_error": np.abs(actual - self.forecast_),
            "tsb_absolute_error": np.abs(actual - self.tsb_forecast_),
            "renewal_absolute_error": np.abs(actual - self.renewal_forecast_),
        }
