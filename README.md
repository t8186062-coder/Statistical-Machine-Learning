# Conformal Prediction Under Distribution Shift

This project studies how conformal prediction intervals behave when the test distribution differs from the calibration distribution.

Split conformal prediction provides finite-sample coverage guarantees under exchangeability. In practice, however, deployed models often encounter distribution shift. This project investigates how that shift affects empirical coverage and whether simple recalibration methods can restore reliability.

## Research Question

How does distribution shift affect the coverage guarantees of split conformal prediction, and what methods can restore coverage when exchangeability fails?

## Experiments

The project evaluates three forms of distribution shift:

- Covariate shift
- Noise shift
- Subgroup / mixture shift

For each setting, I evaluate:

- Empirical prediction interval coverage
- Prediction interval width
- Coverage degradation as shift strength increases

I also test several possible responses to distribution shift, including:

- Recalibration on shifted data
- Group-specific calibration
- Increasing calibration-set size
- Changing the underlying regression model

## Main Findings

Conformal intervals achieve approximately their target coverage when calibration and test data follow the same distribution.

Under distribution shift, coverage can deteriorate substantially. Recalibration and group-specific calibration can recover much of the lost coverage, although improved coverage can come at the cost of wider prediction intervals.

This highlights an important tradeoff between statistical coverage and the usefulness of the resulting uncertainty intervals.

## Tools

- Python
- scikit-learn
- NumPy
- pandas
- Matplotlib

## Repository Structure

- `Main.py` - main entry point
- `conformal.py` - conformal prediction implementation
- `data_generation.py` - synthetic data and distribution-shift generation
- `experiments.py` - experimental pipeline
- `plots.py` - visualization utilities
- `results/` - experimental outputs
- `report.pdf` - full project report

## AI Tool Usage

AI tools, including Claude, were used during development for coding assistance, debugging, and discussing possible implementation improvements.

The research question, experimental design, methodological decisions, validation, analysis, and interpretation of the results were performed by the author. AI-generated suggestions and code were reviewed and verified before being incorporated into the project.

## Author

Thinh Truong Nguyen  
University of California, Berkeley  
Mathematics and Statistics
