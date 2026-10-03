from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from ._validation import as_panel, validate_horizon
from .renewal import RenewalForecaster
from .tsb import TSB


def actual_cumulative(panel, origin: int, horizon: int = 6) -> np.ndarray:
    arr = as_panel(panel)
    h = validate_horizon(horizon)
    if origin <= 0 or origin + h > arr.shape[1]:
        raise ValueError("origin must leave a complete forecast horizon inside the panel.")
    return arr[:, origin : origin + h].sum(axis=1)


def tsb_panel_forecast(panel, origin: int, horizon: int = 6, *, alpha: float = 0.10, beta: float = 0.10) -> np.ndarray:
    arr = as_panel(panel)
    h = validate_horizon(horizon)
    if origin <= 0 or origin > arr.shape[1]:
        raise ValueError("origin is outside the panel.")
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i, row in enumerate(arr):
        train = row[:origin]
        if np.any(train > 0):
            out[i] = TSB(alpha=alpha, beta=beta).fit(train).predict_cumulative(h)
    return out


def renewal_panel_forecast(panel, origin: int, horizon: int = 6) -> np.ndarray:
    arr = as_panel(panel)
    model = RenewalForecaster().fit(arr, origin=origin)
    return model.cumulative_for_training_panel(horizon=horizon)


def historical_utility(
    panel,
    origins: Sequence[int],
    horizon: int = 6,
    *,
    tsb_alpha: float = 0.10,
    tsb_beta: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute canonical per-SKU UGRF utility and evidence counts."""
    arr = as_panel(panel)
    h = validate_horizon(horizon)
    utility = np.zeros(arr.shape[0], dtype=float)
    seen = np.zeros(arr.shape[0], dtype=int)

    for origin in origins:
        if origin <= 0 or origin + h > arr.shape[1]:
            raise ValueError(f"origin {origin} does not leave a full horizon={h} window.")
        actual = actual_cumulative(arr, origin, h)
        ft = tsb_panel_forecast(arr, origin, h, alpha=tsb_alpha, beta=tsb_beta)
        fr = renewal_panel_forecast(arr, origin, h)
        valid = np.isfinite(ft) & np.isfinite(fr)
        utility[valid] += np.abs(actual[valid] - ft[valid]) - np.abs(actual[valid] - fr[valid])
        seen[valid] += 1

    utility[seen == 0] = 0.0
    return utility, seen
