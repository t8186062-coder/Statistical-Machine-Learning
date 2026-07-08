"""
Conformal Prediction Under Distribution Shift
Thinh Truong Nguyen

Runs the full experiment suite end-to-end:
  1. baseline split-conformal calibration (no shift)
  2. covariate / noise / subgroup shift sweeps
  3. attempted fixes: recalibration, group-specific calibration,
     larger calibration sets, alternative regression models
Saves every figure and results table under results/.
"""

from __future__ import annotations

import os

import pandas as pd

from experiments import (
    calibration_size_experiment,
    covariate_shift_sweep,
    group_specific_fix,
    model_comparison_experiment,
    noise_shift_sweep,
    recalibration_fix,
    run_baseline,
    subgroup_shift_sweep,
)
from plots import (
    plot_baseline_calibration,
    plot_calibration_size,
    plot_fix_comparison,
    plot_model_comparison,
    plot_shift_curve,
)

RESULTS_DIR = "results"
ALPHA = 0.10
TARGET_COVERAGE = 1 - ALPHA
SEED = 0

pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Conformal Prediction Under Distribution Shift")
    print("=" * 70)

    # 1. Baseline: fit model, calibrate, check coverage on unshifted test data.
    baseline = run_baseline(alpha=ALPHA, seed=SEED)
    model, width = baseline["model"], baseline["width"]

    section("1. Baseline (no shift)")
    print(f"Target coverage:     {TARGET_COVERAGE:.1%}")
    print(f"Conformal width:     +/- {width:.3f}")
    print(f"Empirical coverage:  {baseline['coverage']:.1%}")
    print(f"Average interval width: {baseline['avg_width']:.3f}")
    plot_baseline_calibration(
        model, baseline["X_test"], baseline["y_test"], width, baseline["coverage"],
        f"{RESULTS_DIR}/baseline.png",
    )

    # 2. Covariate shift sweep.
    cov_shifts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cov_df = covariate_shift_sweep(model, width, cov_shifts)
    cov_df.to_csv(f"{RESULTS_DIR}/covariate_shift.csv", index=False)
    plot_shift_curve(
        cov_df, "shift", "Covariate shift (mean of x)", "Coverage under covariate shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/covariate_shift.png",
    )
    section("2. Covariate shift sweep")
    print(cov_df.to_string(index=False))

    # 3. Noise shift sweep.
    noise_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
    noise_df = noise_shift_sweep(model, width, noise_levels)
    noise_df.to_csv(f"{RESULTS_DIR}/noise_shift.csv", index=False)
    plot_shift_curve(
        noise_df, "noise_std", "Noise std of test distribution", "Coverage under noise shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/noise_shift.png",
    )
    section("3. Noise shift sweep")
    print(noise_df.to_string(index=False))

    # 4. Subgroup shift sweep (population mixture shift; baseline mixture is 20% group B).
    frac_bs = [0.2, 0.35, 0.5, 0.65, 0.8]
    sub_df = subgroup_shift_sweep(model, width, frac_bs)
    sub_df.to_csv(f"{RESULTS_DIR}/subgroup_shift.csv", index=False)
    plot_shift_curve(
        sub_df, "frac_b", "Fraction of test set from high-noise group B", "Coverage under subgroup shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/subgroup_shift.png",
    )
    section("4. Subgroup shift sweep")
    print(sub_df.to_string(index=False))

    # 5. Fix #1: recalibrate on a fresh sample from the shifted (covariate) distribution.
    recal_df = recalibration_fix(model, width, ALPHA, cov_shifts)
    recal_df.to_csv(f"{RESULTS_DIR}/fixes_covariate.csv", index=False)
    plot_fix_comparison(
        recal_df, "shift", "Covariate shift (mean of x)", "Recalibration under covariate shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/fixes_covariate.png",
    )
    section("5. Fix: recalibration under covariate shift")
    print(recal_df.to_string(index=False))

    # 6. Fix #2: group-specific calibration under subgroup (mixture) shift.
    grp_df, pooled_width, group_widths = group_specific_fix(model, ALPHA, frac_bs, baseline_frac_b=0.2)
    grp_df.to_csv(f"{RESULTS_DIR}/fixes_subgroup.csv", index=False)
    plot_fix_comparison(
        grp_df, "frac_b", "Fraction of test set from high-noise group B",
        "Group-specific calibration under subgroup shift", TARGET_COVERAGE,
        f"{RESULTS_DIR}/fixes_subgroup.png",
        standard_col="coverage_standard", fixed_col="coverage_group_specific", fixed_label="Group-specific",
    )
    section("6. Fix: group-specific calibration under subgroup shift")
    print(f"Pooled width: {pooled_width:.3f}   Group widths: "
          f"A={group_widths['A']:.3f}  B={group_widths['B']:.3f}")
    print(grp_df.to_string(index=False))

    # 7. Fix #3: does a larger (shifted-distribution) calibration set help/stabilize recalibration?
    calib_sizes = [200, 500, 1000]
    calib_df = calibration_size_experiment(model, ALPHA, calib_sizes, shift=2.0)
    calib_df.to_csv(f"{RESULTS_DIR}/calibration_size.csv", index=False)
    plot_calibration_size(calib_df, TARGET_COVERAGE, f"{RESULTS_DIR}/calibration_size.png")
    section("7. Fix: calibration set size (recalibrating at covariate shift = 2.0)")
    print(calib_df.to_string(index=False))

    # 8. Fix #4 (comparison): does the choice of base model matter under noise shift?
    model_df = model_comparison_experiment(ALPHA, noise_levels)
    model_df.to_csv(f"{RESULTS_DIR}/method_comparison.csv", index=False)
    plot_model_comparison(model_df, TARGET_COVERAGE, f"{RESULTS_DIR}/method_comparison.png")
    section("8. Model comparison under noise shift")
    print(model_df.to_string(index=False))

    print()
    print("All figures and tables written to results/")


if __name__ == "__main__":
    main()
