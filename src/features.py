# AINT602 ML Pipeline Features Script
# Matt Chamley 26 November 2026

# Feature choices

# pulse_pressure: systolic minus diastolic blood pressure. A widened pulse pressure
# is linked to higher cardiovascular risk (Homan, Bordes and Cichowski, 2023).
# comorbidity_count: diabetes plus hypertension, 0 to 2. The odds of 30-day readmission
# rise with the number of chronic conditions (Aubert et al., 2019). This is a
# simplified version, using only the two conditions in the data.

# age_band: 0-17, 18-39, 40-59, 60-74, 75+. Used as a model input and to report
# results by age group. The 75+ cut-off is selected as the older-adult threshold.

# Mortality and patient_id are left out of the inputs. Mortality is an outcome that
# is not known at triage, so using it would leak the answer. Patient ID is only a label.
# These features are assumptions built from the columns available.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.validate import INPUTS

NUMBER_FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol",
    "creatinine", "diabetes", "hypertension", "pulse_pressure", "comorbidity_count",
]
CATEGORY_FEATURES = ["sex", "diagnosis", "age_band"]

AGE_BINS = [-1, 17, 39, 59, 74, 200]
AGE_LABELS = ["0-17", "18-39", "40-59", "60-74", "75+"]


def add_features(df):
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["comorbidity_count"] = df["diabetes"] + df["hypertension"]
    df["age_band"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    return df


def build_pipeline(model):
    number_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    category_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numbers", number_steps, NUMBER_FEATURES),
        ("categories", category_steps, CATEGORY_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", model),
    ])


def get_inputs(df):
    return df[INPUTS]
