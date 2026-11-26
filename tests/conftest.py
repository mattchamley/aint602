# AINT602 ML Pipeline Fake Dataset Test Shared by Every Test
# Matt Chamley 27 November 2026

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    n = 300
    diastolic = rng.normal(80, 8, n).round(0)
    df = pd.DataFrame({
        "age": rng.integers(1, 90, n),
        "sex": rng.choice(["Female", "Male", "Other"], n),
        "bmi": rng.normal(27, 4, n).round(1),
        "systolic_bp": diastolic + rng.normal(45, 8, n).round(0),
        "diastolic_bp": diastolic,
        "glucose": rng.normal(105, 15, n).round(1),
        "cholesterol": rng.normal(190, 25, n).round(1),
        "creatinine": rng.normal(1.0, 0.2, n).round(2),
        "diabetes": rng.integers(0, 2, n),
        "hypertension": rng.integers(0, 2, n),
        "diagnosis": rng.choice(["Normal", "Pneumonia", "Heart Failure", "Sepsis"], n),
    })
    df["readmission_30d"] = (rng.random(n) < 0.15 + 0.1 * df["hypertension"]).astype(int)
    return df
