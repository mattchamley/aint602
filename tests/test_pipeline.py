# AINT602 ML Pipeline Pipeline Test Script
# Matt Chamley 27 November 2026

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline, get_inputs
from src.predict import predict


def fit_small_model(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    return build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)


def test_saved_model_gives_same_predictions(small_data, tmp_path):
    model = fit_small_model(small_data)
    X = get_inputs(small_data)
    joblib.dump(model, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_prediction_output_shape_and_range(small_data):
    model = fit_small_model(small_data)
    new_patients = small_data.drop(columns=["readmission_30d"]).head(10)
    result, report = predict(new_patients, model)
    assert len(result) == 10
    assert result["readmission_probability"].between(0, 1).all()
    assert set(result["high_risk_flag"]) <= {0, 1}


def test_queue_rank_one_is_the_highest_risk(small_data):
    model = fit_small_model(small_data)
    new_patients = small_data.drop(columns=["readmission_30d"]).head(10)
    result, report = predict(new_patients, model)
    assert result.iloc[0]["queue_rank"] == 1
    assert result.iloc[0]["readmission_probability"] == result["readmission_probability"].max()


def test_prediction_works_with_missing_values(small_data):
    model = fit_small_model(small_data)
    new_patients = small_data.drop(columns=["readmission_30d"]).head(5)
    new_patients.loc[0, "glucose"] = None
    result, report = predict(new_patients, model)
    assert result["readmission_probability"].notna().all()


def test_extra_columns_are_carried_through_but_not_used(small_data):
    model = fit_small_model(small_data)
    new_patients = small_data.drop(columns=["readmission_30d"]).head(5).copy()
    new_patients["patient_id"] = range(5)
    result, report = predict(new_patients, model)
    assert len(result) == 5
    assert "patient_id" in result.columns
