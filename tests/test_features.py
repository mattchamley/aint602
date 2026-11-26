# AINT602 ML Pipeline Features Test
# Matt Chamley 27 November 2026

import pandas as pd

from src.features import add_features, build_pipeline, get_inputs
from src.validate import INPUTS


def make_row(**changes):
    row = {"age": 50, "systolic_bp": 130, "diastolic_bp": 80, "diabetes": 1, "hypertension": 1}
    row.update(changes)
    return pd.DataFrame([row])


def test_pulse_pressure():
    assert add_features(make_row()).loc[0, "pulse_pressure"] == 50


def test_comorbidity_count():
    assert add_features(make_row()).loc[0, "comorbidity_count"] == 2
    assert add_features(make_row(diabetes=0)).loc[0, "comorbidity_count"] == 1


def test_age_bands():
    ages = [5, 25, 45, 65, 85]
    bands = [add_features(make_row(age=a)).loc[0, "age_band"] for a in ages]
    assert bands == ["0-17", "18-39", "40-59", "60-74", "75+"]


def test_age_band_edges():
    ages = [17, 18, 39, 40, 74, 75]
    bands = [add_features(make_row(age=a)).loc[0, "age_band"] for a in ages]
    assert bands == ["0-17", "18-39", "18-39", "40-59", "60-74", "75+"]


def test_original_data_is_not_changed():
    df = make_row()
    add_features(df)
    assert "pulse_pressure" not in df.columns


def test_model_inputs_leave_out_outcome_and_id_columns(small_data):
    small_data["mortality"] = 0
    small_data["patient_id"] = range(len(small_data))
    inputs = get_inputs(small_data)
    assert list(inputs.columns) == INPUTS
    assert not {"mortality", "patient_id", "readmission_30d"} & set(inputs.columns)


def test_missing_values_get_an_indicator_column(small_data):
    from sklearn.linear_model import LogisticRegression

    small_data.loc[0, "glucose"] = None
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(get_inputs(small_data), small_data["readmission_30d"])
    names = model.named_steps["prep"].get_feature_names_out()
    assert any("missingindicator" in n for n in names)
