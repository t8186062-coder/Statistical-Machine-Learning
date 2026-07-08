"""
Synthetic data generation for the conformal-prediction-under-shift experiments.

The ground-truth rule is known because we generate the data ourselves:

    y = 2*x0 - 1.5*x1 + 0.8*x2**2 + sin(2*x3) + noise,  noise ~ Normal(0, noise_std)

Three kinds of distribution shift are supported, matching the experiment plan:
  - covariate shift: shift the mean of the input distribution (x_loc)
  - noise shift: increase the label noise std (noise_std)
  - subgroup shift: change the mixture proportion between a low-noise group A
    and a high-noise group B
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FEATURES = 4


def true_function(X: np.ndarray) -> np.ndarray:
    """The noiseless data-generating rule y = f(x)."""
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 2.0 * x0 - 1.5 * x1 + 0.8 * x2**2 + np.sin(2.0 * x3)


def make_data(
    n: int,
    seed: int,
    x_loc: float = 0.0,
    x_scale: float = 1.0,
    noise_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n i.i.d. samples (X, y) from the synthetic model.

    x_loc / x_scale control covariate shift, noise_std controls noise shift.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=x_loc, scale=x_scale, size=(n, N_FEATURES))
    y = true_function(X) + rng.normal(0.0, noise_std, size=n)
    return X, y


def make_subgroup_data(
    n: int,
    seed: int,
    frac_b: float,
    noise_std_a: float = 1.0,
    noise_std_b: float = 2.5,
    x_loc: float = 0.0,
    x_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a mixture of a low-noise group A and a high-noise group B.

    frac_b is the fraction of the sample coming from group B. The per-group
    conditional distributions never change; only the mixture weight does.
    Returns (X, y, group) where group is an array of 'A'/'B' labels.
    """
    rng = np.random.default_rng(seed)
    n_b = int(round(n * frac_b))
    n_a = n - n_b

    X_a = rng.normal(loc=x_loc, scale=x_scale, size=(n_a, N_FEATURES))
    y_a = true_function(X_a) + rng.normal(0.0, noise_std_a, size=n_a)

    X_b = rng.normal(loc=x_loc, scale=x_scale, size=(n_b, N_FEATURES))
    y_b = true_function(X_b) + rng.normal(0.0, noise_std_b, size=n_b)

    X = np.vstack([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    group = np.array(["A"] * n_a + ["B"] * n_b)

    perm = rng.permutation(len(y))
    return X[perm], y[perm], group[perm]


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def train_calib_test_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.5,
    calib_frac: float = 0.25,
) -> Split:
    """Randomly split (X, y) into train / calibration / test sets."""
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_end = int(train_frac * n)
    calib_end = train_end + int(calib_frac * n)

    train_idx, calib_idx, test_idx = perm[:train_end], perm[train_end:calib_end], perm[calib_end:]
    return Split(
        X_train=X[train_idx], y_train=y[train_idx],
        X_calib=X[calib_idx], y_calib=y[calib_idx],
        X_test=X[test_idx], y_test=y[test_idx],
    )
