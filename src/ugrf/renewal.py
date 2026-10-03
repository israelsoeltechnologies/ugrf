from __future__ import annotations

import math
import numpy as np

from ._validation import as_panel, as_series, validate_horizon, validate_origin


RIDGE = 1.0
POSITIVE_SHRINKAGE = 5.0


def _completed_intervals(panel: np.ndarray, origin: int) -> list[int]:
    out: list[int] = []
    for row in panel:
        nz = np.flatnonzero(row[:origin] > 0)
        if len(nz) >= 2:
            out.extend(np.diff(nz).astype(int).tolist())
    return out


def fit_age_shape(panel: np.ndarray, origin: int) -> np.ndarray:
    """Fit the exact frozen pooled age-shape coefficients."""
    intervals = _completed_intervals(panel, origin)
    if not intervals:
        return np.array([0.0, 0.0], dtype=float)

    max_age = max(intervals)
    freq = np.bincount(intervals, minlength=max_age + 1).astype(float)
    risk = np.zeros(max_age, dtype=float)
    events = np.zeros(max_age, dtype=float)
    remaining = float(len(intervals))
    for age in range(1, max_age + 1):
        risk[age - 1] = remaining
        events[age - 1] = freq[age] if age < len(freq) else 0.0
        remaining -= events[age - 1]

    p0 = (len(intervals) + 0.5) / (risk.sum() + 1.0)
    theta = np.array([math.log(p0 / (1.0 - p0)), 0.0, 0.0], dtype=float)

    for _ in range(30):
        grad = np.array([0.0, RIDGE * theta[1], RIDGE * theta[2]], dtype=float)
        hess = np.diag([0.0, RIDGE, RIDGE])

        for j in range(max_age):
            log_age = math.log(j + 1.0)
            x = np.array([1.0, log_age, log_age * log_age], dtype=float)
            eta = float(np.clip(theta @ x, -30, 30))
            p = 1.0 / (1.0 + math.exp(-eta))
            resid = risk[j] * p - events[j]
            weight = risk[j] * p * (1.0 - p)
            grad += resid * x
            hess += weight * np.outer(x, x)

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad

        theta -= step
        if np.linalg.norm(step) < 1e-9:
            break

    return theta[1:3].copy()


def fit_global_positive_logmean(panel: np.ndarray, origin: int) -> float:
    """Fit the frozen panel-level positive-demand log-mean anchor."""
    rows: list[tuple[float, float]] = []
    total = 0.0

    for full in panel:
        age = 1
        for t in range(origin):
            y = float(full[t])
            if y > 0:
                rows.append((y, math.log(age)))
                total += y
            age = 1 if y > 0 else age + 1

    if not rows:
        return 0.0

    theta = np.array([math.log(max(total / len(rows), 1e-6)), 0.0], dtype=float)

    for _ in range(20):
        grad = np.array([0.0, RIDGE * theta[1]], dtype=float)
        hess = np.array([[0.0, 0.0], [0.0, RIDGE]], dtype=float)

        for z, log_age in rows:
            eta = float(np.clip(theta[0] + theta[1] * log_age, -8, 8))
            mu = math.exp(eta)
            resid = mu - z
            grad[0] += resid
            grad[1] += resid * log_age
            hess[0, 0] += mu
            hess[0, 1] += mu * log_age
            hess[1, 0] += mu * log_age
            hess[1, 1] += mu * log_age * log_age

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad

        theta -= step
        if np.linalg.norm(step) < 1e-9:
            break

    return float(theta[0])


def age_effect(age: int, beta: np.ndarray) -> float:
    log_age = math.log(max(1.0, float(age)))
    return float(beta[0] * log_age + beta[1] * log_age * log_age)


def fit_local_occurrence_intercept(y: np.ndarray, beta: np.ndarray) -> float:
    ev = int((y > 0).sum())
    p0 = (ev + 0.5) / (len(y) + 1.0)
    intercept = math.log(p0 / (1.0 - p0))

    age = 1
    offsets = []
    labels = []
    for value in y:
        offsets.append(age_effect(age, beta))
        labels.append(1.0 if value > 0 else 0.0)
        age = 1 if value > 0 else age + 1

    offsets_arr = np.asarray(offsets, dtype=float)
    labels_arr = np.asarray(labels, dtype=float)

    for _ in range(30):
        eta = np.clip(intercept + offsets_arr, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = np.sum(p - labels_arr)
        hess = np.sum(p * (1.0 - p))
        if hess < 1e-12:
            break
        step = grad / hess
        intercept = float(np.clip(intercept - step, -20, 20))
        if abs(step) < 1e-10:
            break

    return float(intercept)


def pooled_positive_logmean(y: np.ndarray, global_logmean: float, k: float = POSITIVE_SHRINKAGE) -> float:
    positives = y[y > 0]
    if len(positives) == 0:
        return float(global_logmean)
    local = float(np.log(np.maximum(positives, 1e-6)).mean())
    weight = len(positives) / (len(positives) + k)
    return float(weight * local + (1.0 - weight) * global_logmean)


def current_age(y: np.ndarray) -> int:
    age = 1
    for value in y:
        age = 1 if value > 0 else age + 1
    return int(age)


def renewal_path(
    y: np.ndarray,
    beta: np.ndarray,
    global_logmean: float,
    horizon: int,
    positive_shrinkage: float = POSITIVE_SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return marginal occurrence probabilities and expected demand by lead."""
    intercept = fit_local_occurrence_intercept(y, beta)
    magnitude = max(
        1.0,
        math.exp(
            float(
                np.clip(
                    pooled_positive_logmean(y, global_logmean, positive_shrinkage),
                    -5,
                    8,
                )
            )
        ),
    )

    state: dict[int, float] = {current_age(y): 1.0}
    occurrence = np.zeros(horizon, dtype=float)

    for lead in range(horizon):
        nxt: dict[int, float] = {}
        event_mass = 0.0
        for age, mass in state.items():
            eta = float(np.clip(intercept + age_effect(age, beta), -30, 30))
            p = 1.0 / (1.0 + math.exp(-eta))
            event = mass * p
            event_mass += event
            nxt[age + 1] = nxt.get(age + 1, 0.0) + mass * (1.0 - p)
        nxt[1] = nxt.get(1, 0.0) + event_mass
        occurrence[lead] = event_mass
        state = nxt

    expected_demand = occurrence * magnitude
    return occurrence, expected_demand, float(intercept), float(magnitude)


class RenewalForecaster:
    """Pooled-age Renewal expert used by UGRF.

    The age-shape coefficients and global positive-demand anchor are pooled
    across a panel. SKU-level occurrence intercepts and magnitudes remain local.
    """

    def __init__(self, positive_shrinkage: float = POSITIVE_SHRINKAGE):
        if positive_shrinkage < 0:
            raise ValueError("positive_shrinkage must be non-negative.")
        self.positive_shrinkage = float(positive_shrinkage)

    def fit(self, panel, origin: int | None = None) -> "RenewalForecaster":
        panel_arr = as_panel(panel)
        origin_int = validate_origin(origin, panel_arr.shape[1])
        self.beta_ = fit_age_shape(panel_arr, origin_int)
        self.global_logmean_ = fit_global_positive_logmean(panel_arr, origin_int)
        self.origin_ = origin_int
        self.n_series_ = int(panel_arr.shape[0])
        self.n_periods_ = int(panel_arr.shape[1])
        self._training_panel = panel_arr
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "beta_"):
            raise RuntimeError("RenewalForecaster is not fitted. Call fit() first.")

    def forecast_series(self, y, horizon: int = 6) -> np.ndarray:
        self._check_fitted()
        y_arr = as_series(y)
        h = validate_horizon(horizon)
        if not np.any(y_arr > 0):
            return np.full(h, np.nan, dtype=float)
        _, expected, _, _ = renewal_path(
            y_arr,
            self.beta_,
            self.global_logmean_,
            h,
            self.positive_shrinkage,
        )
        return expected

    def forecast_series_cumulative(self, y, horizon: int = 6) -> float:
        return float(np.sum(self.forecast_series(y, horizon=horizon)))

    def predict(self, panel=None, horizon: int = 6) -> np.ndarray:
        self._check_fitted()
        h = validate_horizon(horizon)
        panel_arr = self._training_panel if panel is None else as_panel(panel)
        if panel is None:
            histories = panel_arr[:, : self.origin_]
        else:
            histories = panel_arr
        out = np.full((histories.shape[0], h), np.nan, dtype=float)
        for i, y in enumerate(histories):
            if np.any(y > 0):
                out[i] = self.forecast_series(y, horizon=h)
        return out

    def predict_cumulative(self, panel=None, horizon: int = 6) -> np.ndarray:
        forecasts = self.predict(panel=panel, horizon=horizon)
        out = np.nansum(forecasts, axis=1)
        out[np.all(~np.isfinite(forecasts), axis=1)] = np.nan
        return out

    def cumulative_for_training_panel(self, horizon: int = 6) -> np.ndarray:
        """Cumulative forecasts preserving NaN for SKUs with no positive history."""
        self._check_fitted()
        h = validate_horizon(horizon)
        histories = self._training_panel[:, : self.origin_]
        out = np.full(histories.shape[0], np.nan, dtype=float)
        for i, y in enumerate(histories):
            if np.any(y > 0):
                out[i] = self.forecast_series_cumulative(y, horizon=h)
        return out
