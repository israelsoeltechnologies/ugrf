from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def as_series(y, *, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return arr


def as_panel(panel, *, name: str = "panel") -> np.ndarray:
    arr = np.asarray(panel, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (n_series, n_periods); got {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one series and one period.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return arr


def validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    return int(horizon)


def validate_origin(origin: int | None, n_periods: int, *, allow_end: bool = True) -> int:
    if origin is None:
        return n_periods
    if isinstance(origin, bool) or int(origin) != origin:
        raise ValueError("origin must be an integer.")
    origin = int(origin)
    upper = n_periods if allow_end else n_periods - 1
    if origin <= 0 or origin > upper:
        raise ValueError(f"origin must be in [1, {upper}].")
    return origin


def validate_origins(origins: Sequence[int], *, final_origin: int, horizon: int) -> tuple[int, ...]:
    out = tuple(int(o) for o in origins)
    for raw, o in zip(origins, out):
        if isinstance(raw, bool) or raw != o:
            raise ValueError("utility_origins must contain integers only.")
        if o <= 0:
            raise ValueError("utility origins must be positive.")
        if o + horizon > final_origin:
            raise ValueError(
                f"utility origin {o} leaks beyond final_origin={final_origin}: "
                f"{o}+horizon({horizon}) > {final_origin}."
            )
    if len(set(out)) != len(out):
        raise ValueError("utility_origins must not contain duplicates.")
    return tuple(sorted(out))
