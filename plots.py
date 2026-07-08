"""
Figures for the report. All plots are static PNGs (matplotlib, Agg backend).

Palette (validated colorblind-safe, see project notes):
  blue   #2a78d6  - primary series / "fixed" state
  red    #e34948  - "standard / uncorrected" state
  aqua   #1baf7a  - categorical series 2
  yellow #eda100  - categorical series 3
  gray   #898781  - muted reference lines (target coverage)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _target_line(ax, target: float) -> None:
    ax.axhline(target, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(
        f"target coverage ({target:.0%})", xy=(1.0, target), xycoords=("axes fraction", "data"),
        xytext=(-4, 6), textcoords="offset points", ha="right", va="bottom", color=GRAY, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85), zorder=5,
    )


def plot_baseline_calibration(model, X_test, y_test, width, coverage, savepath):
    """Predicted vs actual with the conformal band around the diagonal."""
    pred = model.predict(X_test)
    covered = (y_test >= pred - width) & (y_test <= pred + width)

    lo, hi = min(pred.min(), y_test.min()), max(pred.max(), y_test.max())
    pad = 0.05 * (hi - lo)
    line_x = np.linspace(lo - pad, hi + pad, 100)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.fill_between(line_x, line_x - width, line_x + width, color=BLUE, alpha=0.12, zorder=1, label="90% interval band")
    ax.plot(line_x, line_x, color=GRAY, linewidth=1.5, zorder=2, label="perfect prediction")
    ax.scatter(pred[covered], y_test[covered], s=14, color=BLUE, alpha=0.6, zorder=3, label="covered")
    ax.scatter(pred[~covered], y_test[~covered], s=18, color=RED, alpha=0.8, zorder=4, marker="x", label="not covered")

    ax.set_xlabel("Predicted y")
    ax.set_ylabel("Actual y")
    ax.set_title(f"Baseline calibration (no shift) — empirical coverage {coverage:.1%}")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_shift_curve(df: pd.DataFrame, x_col: str, xlabel: str, title: str, target: float, savepath: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df[x_col], df["coverage"], color=BLUE, marker="o", markersize=7, linewidth=2, zorder=3)
    _target_line(ax, target)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_ylim(min(0.5, df["coverage"].min() - 0.05), 1.02)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_fix_comparison(
    df: pd.DataFrame, x_col: str, xlabel: str, title: str, target: float, savepath: str,
    standard_col: str = "coverage_standard", fixed_col: str = "coverage_recalibrated",
    fixed_label: str = "Recalibrated",
):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df[x_col], df[standard_col], color=RED, marker="x", markersize=8, linewidth=2,
            linestyle="--", label="Standard", zorder=3)
    ax.plot(df[x_col], df[fixed_col], color=BLUE, marker="o", markersize=7, linewidth=2,
            label=fixed_label, zorder=4)
    _target_line(ax, target)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_ylim(min(0.5, df[[standard_col, fixed_col]].min().min() - 0.05), 1.02)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_calibration_size(df: pd.DataFrame, target: float, savepath: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        df["calib_size"], df["mean_coverage"], yerr=df["std_coverage"],
        color=BLUE, marker="o", markersize=7, linewidth=2, capsize=4, zorder=3,
    )
    _target_line(ax, target)
    ax.set_xlabel("Calibration set size (drawn from the shifted distribution)")
    ax.set_ylabel("Mean empirical coverage (± 1 std over repeats)")
    ax.set_title("Recalibration reliability vs. calibration set size")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(df["calib_size"])
    ax.set_xticklabels(df["calib_size"])
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, target: float, savepath: str):
    colors = {"Linear Regression": BLUE, "Random Forest": AQUA, "Gradient Boosting": YELLOW}
    markers = {"Linear Regression": "o", "Random Forest": "s", "Gradient Boosting": "^"}
    styles = {"Linear Regression": "-", "Random Forest": "--", "Gradient Boosting": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for name, group in df.groupby("model", sort=False):
        axes[0].plot(
            group["noise_std"], group["coverage"], color=colors[name], marker=markers[name],
            linestyle=styles[name], markersize=7, linewidth=2, label=name,
        )
        axes[1].plot(
            group["noise_std"], group["avg_width"], color=colors[name], marker=markers[name],
            linestyle=styles[name], markersize=7, linewidth=2, label=name,
        )

    _target_line(axes[0], target)
    axes[0].set_xlabel("Noise std of test distribution")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Coverage under noise shift")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)

    axes[1].set_xlabel("Noise std of test distribution")
    axes[1].set_ylabel("Average interval width")
    axes[1].set_title("Interval width under noise shift")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
