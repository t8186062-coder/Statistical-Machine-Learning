"""
Conformal Prediction Under Distribution Shift
Thinh Truong Nguyen
05/04/2026

Version 0 skeleton:
- generate/load data
- split into train, calibration, and test sets
- train a base model
- build conformal prediction intervals
- evaluate coverage under no shift and shifted test data
"""


def make_data():
    """Create or load the dataset."""
    pass


def split_data():
    """Split data into train, calibration, and test sets."""
    pass


def train_model():
    """Train the base prediction model."""
    pass


def build_conformal_intervals():
    """Use calibration errors to build prediction intervals."""
    pass


def apply_distribution_shift():
    """Create a shifted version of the test data."""
    pass


def evaluate_coverage():
    """Measure empirical coverage and average interval width."""
    pass


def main():
    print("Conformal Prediction Under Distribution Shift")
    print("Version 0: project skeleton is ready.")


if __name__ == "__main__":
    main()
