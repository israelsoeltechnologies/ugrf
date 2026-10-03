from __future__ import annotations

import math
import numpy as np

from ._validation import as_series, validate_horizon
from .renewal import (
    RenewalForecaster,
    age_effect,
    current_age,
    fit_local_occurrence_intercept,
    renewal_path,
)


def _check(model: RenewalForecaster) -> None:
    if not isinstance(model, RenewalForecaster):
        raise TypeError("model must be a fitted RenewalForecaster.")
    model._check_fitted()


def hazard_curve(model: RenewalForecaster, y, max_age: int = 24) -> tuple[np.ndarray, np.ndarray]:
    _check(model)
    series = as_series(y)
    if max_age <= 0:
        raise ValueError("max_age must be positive.")
    intercept = fit_local_occurrence_intercept(series, model.beta_)
    ages = np.arange(1, int(max_age) + 1)
    values = []
    for age in ages:
        eta = float(np.clip(intercept + age_effect(int(age), model.beta_), -30, 30))
        values.append(1.0 / (1.0 + math.exp(-eta)))
    return ages, np.asarray(values, dtype=float)


def occurrence_probabilities(model: RenewalForecaster, y, horizon: int = 6) -> np.ndarray:
    _check(model)
    series = as_series(y)
    h = validate_horizon(horizon)
    q, _, _, _ = renewal_path(
        series,
        model.beta_,
        model.global_logmean_,
        h,
        model.positive_shrinkage,
    )
    return q


def series_summary(model: RenewalForecaster, y, horizon: int = 6) -> dict[str, float | int]:
    _check(model)
    series = as_series(y)
    h = validate_horizon(horizon)
    q, expected, intercept, magnitude = renewal_path(
        series,
        model.beta_,
        model.global_logmean_,
        h,
        model.positive_shrinkage,
    )
    return {
        "current_age": current_age(series),
        "occurrence_intercept": intercept,
        "positive_magnitude": magnitude,
        "expected_events": float(q.sum()),
        "cumulative_forecast": float(expected.sum()),
        "probability_at_least_one_event": float(1.0 - _prob_no_event(model, series, h)),
    }


def _prob_no_event(model: RenewalForecaster, series: np.ndarray, horizon: int) -> float:
    intercept = fit_local_occurrence_intercept(series, model.beta_)
    age = current_age(series)
    p_none = 1.0
    for _ in range(horizon):
        eta = float(np.clip(intercept + age_effect(age, model.beta_), -30, 30))
        p = 1.0 / (1.0 + math.exp(-eta))
        p_none *= 1.0 - p
        age += 1
    return float(p_none)


def simulate_occurrence_paths(
    model: RenewalForecaster,
    y,
    horizon: int = 6,
    n_paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate binary occurrence paths from the fitted frozen renewal hazard."""
    _check(model)
    series = as_series(y)
    h = validate_horizon(horizon)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    intercept = fit_local_occurrence_intercept(series, model.beta_)
    rng = np.random.default_rng(seed)
    paths = np.zeros((int(n_paths), h), dtype=int)
    ages = np.full(int(n_paths), current_age(series), dtype=int)

    for lead in range(h):
        log_age = np.log(np.maximum(ages.astype(float), 1.0))
        eta = np.clip(intercept + model.beta_[0] * log_age + model.beta_[1] * log_age * log_age, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        events = rng.random(int(n_paths)) < p
        paths[:, lead] = events.astype(int)
        ages = np.where(events, 1, ages + 1)

    return paths
