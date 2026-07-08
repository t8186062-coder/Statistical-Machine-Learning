"""
All experiments: baseline, distribution-shift sweeps, and attempted fixes.

Every function returns plain data (dicts / pandas DataFrames) so that
plots.py and main.py can consume the results without re-running anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from conformal import calibrate, calibrate_by_group, evaluate, evaluate_by_group
from data_generation import make_data, make_subgroup_data


def make_default_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=seed)


def run_baseline(
    alpha: float = 0.10, n_train: int = 1500, n_calib: int = 500, n_test: int = 1000, seed: int = 0
) -> dict:
    """Fit the base model, calibrate at the nominal (1 - alpha) level, and
    evaluate on an unshifted test set."""
    X_train, y_train = make_data(n_train, seed=seed)
    X_calib, y_calib = make_data(n_calib, seed=seed + 1)
    X_test, y_test = make_data(n_test, seed=seed + 2)

    model = make_default_model(seed).fit(X_train, y_train)
    width = calibrate(model, X_calib, y_calib, alpha)
    coverage, avg_width = evaluate(model, X_test, y_test, width)

    return dict(
        model=model, width=width, alpha=alpha, coverage=coverage, avg_width=avg_width,
        X_train=X_train, y_train=y_train, X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test,
    )


def covariate_shift_sweep(
    model, width: float, shifts: list[float], n_test: int = 1000, noise_std: float = 1.0, seed: int = 100
) -> pd.DataFrame:
    rows = []
    for i, shift in enumerate(shifts):
        X, y = make_data(n_test, seed=seed + i, x_loc=shift, noise_std=noise_std)
        coverage, avg_width = evaluate(model, X, y, width)
        rows.append(dict(shift=shift, coverage=coverage, avg_width=avg_width))
    return pd.DataFrame(rows)


def noise_shift_sweep(
    model, width: float, noise_levels: list[float], n_test: int = 1000, seed: int = 200
) -> pd.DataFrame:
    rows = []
    for i, noise_std in enumerate(noise_levels):
        X, y = make_data(n_test, seed=seed + i, x_loc=0.0, noise_std=noise_std)
        coverage, avg_width = evaluate(model, X, y, width)
        rows.append(dict(noise_std=noise_std, coverage=coverage, avg_width=avg_width))
    return pd.DataFrame(rows)


def subgroup_shift_sweep(
    model, width: float, frac_bs: list[float], n_test: int = 1000,
    noise_std_a: float = 1.0, noise_std_b: float = 2.5, seed: int = 300,
) -> pd.DataFrame:
    rows = []
    for i, frac_b in enumerate(frac_bs):
        X, y, _ = make_subgroup_data(
            n_test, seed=seed + i, frac_b=frac_b, noise_std_a=noise_std_a, noise_std_b=noise_std_b
        )
        coverage, avg_width = evaluate(model, X, y, width)
        rows.append(dict(frac_b=frac_b, coverage=coverage, avg_width=avg_width))
    return pd.DataFrame(rows)


def recalibration_fix(
    model, baseline_width: float, alpha: float, shifts: list[float],
    n_calib: int = 500, n_test: int = 1000, noise_std: float = 1.0, seed: int = 400,
) -> pd.DataFrame:
    """Compare the original (unshifted) calibration width against a width
    recalibrated on a fresh calibration set drawn from the shifted
    distribution itself, at each shift level."""
    rows = []
    for i, shift in enumerate(shifts):
        X_test, y_test = make_data(n_test, seed=seed + 2 * i, x_loc=shift, noise_std=noise_std)
        coverage_std, width_std = evaluate(model, X_test, y_test, baseline_width)

        X_calib_new, y_calib_new = make_data(n_calib, seed=seed + 2 * i + 1, x_loc=shift, noise_std=noise_std)
        new_width = calibrate(model, X_calib_new, y_calib_new, alpha)
        coverage_recal, width_recal = evaluate(model, X_test, y_test, new_width)

        rows.append(dict(
            shift=shift,
            coverage_standard=coverage_std, width_standard=width_std,
            coverage_recalibrated=coverage_recal, width_recalibrated=width_recal,
        ))
    return pd.DataFrame(rows)


def group_specific_fix(
    model, alpha: float, frac_bs: list[float], n_calib: int = 500, n_test: int = 1000,
    noise_std_a: float = 1.0, noise_std_b: float = 2.5, baseline_frac_b: float = 0.2, seed: int = 500,
) -> tuple[pd.DataFrame, float, dict[str, float]]:
    """Calibrate once (pooled width and per-group widths) on the baseline
    group composition, then evaluate both under test sets whose composition
    has shifted away from that baseline."""
    X_calib, y_calib, g_calib = make_subgroup_data(
        n_calib, seed=seed, frac_b=baseline_frac_b, noise_std_a=noise_std_a, noise_std_b=noise_std_b
    )
    pooled_width = calibrate(model, X_calib, y_calib, alpha)
    group_widths = calibrate_by_group(model, X_calib, y_calib, g_calib, alpha)

    rows = []
    for i, frac_b in enumerate(frac_bs):
        X_test, y_test, g_test = make_subgroup_data(
            n_test, seed=seed + 1 + i, frac_b=frac_b, noise_std_a=noise_std_a, noise_std_b=noise_std_b
        )
        coverage_std, width_std = evaluate(model, X_test, y_test, pooled_width)
        coverage_grp, width_grp = evaluate_by_group(model, X_test, y_test, g_test, group_widths)
        rows.append(dict(
            frac_b=frac_b,
            coverage_standard=coverage_std, width_standard=width_std,
            coverage_group_specific=coverage_grp, width_group_specific=width_grp,
        ))
    return pd.DataFrame(rows), pooled_width, group_widths


def calibration_size_experiment(
    model, alpha: float, calib_sizes: list[int], shift: float,
    n_test: int = 1000, noise_std: float = 1.0, repeats: int = 20, seed: int = 600,
) -> pd.DataFrame:
    """At a fixed (shifted) test distribution, recalibrate with freshly
    drawn calibration sets of varying size, repeated several times, to see
    whether a larger calibration set gives more reliable (lower-variance)
    coverage restoration."""
    X_test, y_test = make_data(n_test, seed=seed, x_loc=shift, noise_std=noise_std)

    rows = []
    for size in calib_sizes:
        coverages, widths = [], []
        for r in range(repeats):
            calib_seed = seed + 1000 + size * 100 + r
            X_calib, y_calib = make_data(size, seed=calib_seed, x_loc=shift, noise_std=noise_std)
            width = calibrate(model, X_calib, y_calib, alpha)
            coverage, avg_width = evaluate(model, X_test, y_test, width)
            coverages.append(coverage)
            widths.append(avg_width)
        rows.append(dict(
            calib_size=size,
            mean_coverage=float(np.mean(coverages)), std_coverage=float(np.std(coverages)),
            mean_width=float(np.mean(widths)), std_width=float(np.std(widths)),
        ))
    return pd.DataFrame(rows)


def model_comparison_experiment(
    alpha: float, shifts: list[float], n_train: int = 1500, n_calib: int = 500,
    n_test: int = 1000, noise_std: float = 1.0, seed: int = 700,
) -> pd.DataFrame:
    """Train several regression models on the same data, conformalize each,
    and compare coverage/width as the (noise) shift grows."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }
    X_train, y_train = make_data(n_train, seed=seed)
    X_calib, y_calib = make_data(n_calib, seed=seed + 1)

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        width = calibrate(model, X_calib, y_calib, alpha)
        for i, shift in enumerate(shifts):
            X_test, y_test = make_data(n_test, seed=seed + 2 + i, x_loc=0.0, noise_std=shift)
            coverage, avg_width = evaluate(model, X_test, y_test, width)
            rows.append(dict(model=name, noise_std=shift, coverage=coverage, avg_width=avg_width))
    return pd.DataFrame(rows)
