# Student Performance Prediction

> A reproducible, leakage-aware machine-learning evaluation for predicting
> whether a student passes based on study behavior and academic indicators.

This project trains and evaluates a logistic-regression classifier on a
balanced dataset of 200 student records. It demonstrates a complete evaluation
workflow: stratified data splitting, training-only imputation, feature
standardization, holdout metrics, five-fold cross-validation, and diagnostic
visualizations.

## Results at a glance

The fixed test split contains 40 records and was not used to fit the model.

| Metric | Test score |
| --- | ---: |
| Accuracy | **0.825** |
| Precision | **0.842** |
| Recall | **0.800** |
| F1 score | **0.821** |
| ROC AUC | **0.915** |

Five-fold cross-validation on the development set produced a mean ROC AUC of
`0.842 ± 0.076` and mean accuracy of `0.738 ± 0.038`. These results are useful
for demonstrating the workflow, not for making real-world decisions about
students.

## What the model uses

The model predicts the binary `Passed` target from four numeric features:

- `Study_Hours_Per_Week`
- `Attendance_Percent`
- `Previous_Score`
- `Assignments_Completed_Percent`

`Student_ID` is treated as an identifier and excluded from training. The source
dataset is balanced: 100 passing and 100 failing records. Eight feature values
are missing across three columns; missing values are replaced with training-set
medians inside the modeling pipeline.

## Methodology

1. Load `student_model_evaluation_dataset.csv`.
2. Separate the identifier, predictors, and target.
3. Create stratified 60/20/20 train, validation, and test splits.
4. Fit a pipeline containing:
   - median imputation,
   - standard scaling,
   - logistic regression with `max_iter=1000`.
5. Evaluate the fitted model on train, validation, and test data.
6. Run shuffled, stratified five-fold cross-validation on the combined
   train-and-validation development set.
7. Export metrics, model coefficients, and figures.

The imputer and scaler are fitted as part of each training operation. This
prevents information from validation or test records leaking into preprocessing.
The workflow is deterministic with `random_state=42`.

## Quick start

### Prerequisites

- Python 3.9 or newer
- Internet access during the first dependency installation

### 1. Clone and enter the project

```bash
git clone https://github.com/Wolfof420Street/student_performance_prediction.git
cd student_performance_prediction
```

### 2. Create the virtual environment

On systems with the standard `venv` module:

```bash
python3 -m venv .venv
```

If `venv` is unavailable, install the user-level `virtualenv` tool instead:

```bash
python3 -m pip install --user --break-system-packages virtualenv
~/.local/bin/virtualenv .venv --python=python3
```

### 3. Install dependencies

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Run the analysis

Run from the project directory so the script can locate the CSV file:

```bash
python analysis.py
```

You can also run it without activating the environment:

```bash
.venv/bin/python analysis.py
```

## Outputs

The script creates the following files in the project directory:

| File | Description |
| --- | --- |
| `results.json` | Dataset summary, medians, coefficients, holdout metrics, and cross-validation results |
| `fig_confusion.png` | Test-set confusion matrix |
| `fig_roc.png` | Validation and test ROC curves |
| `fig_proba.png` | Test-set predicted-probability distributions |

Generated outputs and the local `.venv` directory are excluded from version
control by [`.gitignore`](.gitignore).

## Project structure

```text
.
├── analysis.py
├── requirements.txt
├── student_model_evaluation_dataset.csv
├── README.md
└── .gitignore
```

## Reproducibility

The dependency versions are pinned in
[`requirements.txt`](requirements.txt), and the analysis uses a fixed seed:

```python
SEED = 42
```

To regenerate every result and figure from a clean environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

## Responsible use and limitations

This is an educational modeling example using a small, static dataset. The
reported scores may not generalize to other schools, cohorts, or grading
systems. Predictions should not be used as the sole basis for interventions,
admissions, discipline, or other high-impact decisions. A production system
would require representative data, stronger validation, fairness and privacy
reviews, monitoring for drift, calibration checks, and review by qualified
educators and domain experts.
