# AINT602 ML 30-day readmission pipeline
# Mathew Chamley 27 November 2026

## What this is

A working MLOps pipeline that takes patient data from ingestion through to a ranked prediction queue. 
It predicts the risk of a 30-day hospital readmission, using synthetic clinical data (10K patients, 14 columns). 
It supports triage prioritisation. 
It is decision support only and does not make clinical decisions.

Data source: Dataset B

## Results

| Model | Validation accuracy | Validation ROC-AUC | Validation recall |
|---|---|---|---|
| Baseline (always predicts the average) | 0.860 | 0.500 | 0.000 |
| Logistic regression (selected) | 0.661 | 0.620 | 0.527 |
| Random forest | 0.659 | 0.614 | 0.546 |

The selected model was scored once on the test set: accuracy 0.644, ROC-AUC 0.620, recall 0.558, precision 0.209. 
It made 436 false positives and 91 false negatives. 
The baseline scores 0.861 accuracy on the test set but finds no readmissions, so models are judged on ROC-AUC and recall, not accuracy.

## How to run it

Python 3.14 is used locally and in CI.

```
git clone https://github.com/mattchamley/aint602.git
cd aint602
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
python -m src.train
python -m src.predict
```

Run every command from the project folder. To rank a different file of new patients:

```
python -m src.predict --input path/to/patients.csv --output reports/predictions.csv
```

To see the MLflow experiment runs:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## How it works

| Stage | File | What it does |
|---|---|---|
| Settings | `config.yaml` | Paths, random seed, split sizes and model settings |
| Ingest | `src/ingest.py` | Reads the CSV and fingerprints it with an MD5 hash |
| Validate | `src/validate.py` | Checks columns, ranges and categories. Training drops bad rows and reports them. Prediction stops with an error |
| Features | `src/features.py` | Adds pulse pressure, comorbidity count and age band, then builds one scikit-learn pipeline |
| Evaluate | `src/evaluate.py` | Accuracy, ROC-AUC, PR-AUC, recall, precision, Brier, confusion counts and results by group |
| Train | `src/train.py` | 70/15/15 stratified split, baseline and two candidates, MLflow tracking, saves the best |
| Predict | `src/predict.py` | Validates new patients, ranks them by risk and appends to a prediction log |

## Folders

```
config.yaml           settings
data/                 original dataset (never edited) and a 20-patient sample for predict
src/                  the pipeline code
tests/                26 pytest tests
models/               saved pipeline and model_info.json
reports/              metrics, validation report, group results, ROC plot, predictions, log
evidence/             screenshots for the report appendix
.github/workflows/    GitHub Actions: tests, training and prediction on every push
```

## Reproducibility and tracking

- The random seed (42) and split sizes are in `config.yaml`.
- `models/model_info.json` records the git commit, the data hash, the MLflow run ID, the seed and the test scores for the saved model.
- Each MLflow run records the model settings, the data hash and the scores.
- Library versions are pinned in `requirements.txt`.
- GitHub Actions re-runs the tests, training and prediction on a clean machine.
- The test set is scored once, after the model has been chosen on the validation set.

## Assumptions and limits

- Validation ranges are wide plausibility limits, not clinical normal ranges. Units are assumed (age in years, blood pressure in mmHg, glucose, cholesterol and creatinine in mg/dL) because the dataset does not state them.
- 136 rows (1.4%) were dropped because diastolic blood pressure was higher than systolic.
- `mortality` and `patient_id` are never used as inputs. Mortality is an outcome that is not known at triage, so using it would leak the answer.
- The models are balanced for the rare readmitted group. This makes the predicted probabilities higher than real risk (a mean of 0.452 against a training rate of 0.139), so use the ranking and not the percentage.
- The signal in this data is weak (ROC-AUC about 0.62).
- Recall differs between age bands (0.47 to 0.72 on the test set). The groups for sex "Other" (18 patients) and ages 0-17 (31 patients) are too small to judge.
- The data is synthetic, so results say nothing about real patients.

