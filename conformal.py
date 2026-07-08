"""
Split-conformal prediction intervals for regression.

All intervals here are symmetric, fixed-width intervals of the form
    [prediction - width, prediction + width]
built from the (1 - alpha) empirical quantile of absolute calibration
residuals, with the standard finite-sample correction.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def conformal_width(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of absolute residuals."""
    residuals = np.sort(np.abs(residuals))
    n = len(residuals)
    rank = min(ceil((n + 1) * (1 - alpha)), n)
    return float(residuals[rank - 1])


def calibrate(model, X_calib: np.ndarray, y_calib: np.ndarray, alpha: float) -> float:
    """Standard (pooled) split-conformal calibration: one width for everyone."""
    residuals = y_calib - model.predict(X_calib)
    return conformal_width(residuals, alpha)


def calibrate_by_group(
    model, X_calib: np.ndarray, y_calib: np.ndarray, group_calib: np.ndarray, alpha: float
) -> dict[str, float]:
    """Group-specific split-conformal calibration: one width per group."""
    widths = {}
    for g in np.unique(group_calib):
        mask = group_calib == g
        residuals = y_calib[mask] - model.predict(X_calib[mask])
        widths[g] = conformal_width(residuals, alpha)
    return widths


def predict_intervals(model, X: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X)
    return pred - width, pred + width


def evaluate(model, X: np.ndarray, y: np.ndarray, width: float) -> tuple[float, float]:
    """Return (empirical coverage, average interval width) for a pooled width."""
    lower, upper = predict_intervals(model, X, width)
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    avg_width = float(np.mean(upper - lower))
    return coverage, avg_width


def evaluate_by_group(
    model, X: np.ndarray, y: np.ndarray, group: np.ndarray, widths: dict[str, float]
) -> tuple[float, float]:
    """Return (empirical coverage, average interval width) using per-group widths."""
    pred = model.predict(X)
    width_vec = np.array([widths[g] for g in group])
    lower, upper = pred - width_vec, pred + width_vec
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    avg_width = float(np.mean(upper - lower))
    return coverage, avg_width
