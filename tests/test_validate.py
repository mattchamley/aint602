# AINT602 ML Pipeline Rule Validation Tests
# Matt Chamley 27 November 2026

import pandas as pd
import pytest

from src.validate import validate


def test_clean_data_keeps_all_rows(small_data):
    clean, report = validate(small_data, target="readmission_30d")
    assert len(clean) == len(small_data)
    assert report["rows_out"] == len(small_data)


def test_missing_column_stops_the_run(small_data):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(small_data.drop(columns=["glucose"]), target="readmission_30d")


def test_out_of_range_age_is_dropped(small_data):
    small_data.loc[0, "age"] = 250
    clean, report = validate(small_data, target="readmission_30d")
    assert len(clean) == len(small_data) - 1
    assert report["problems"]["age out of range"] == 1


def test_unknown_category_is_dropped(small_data):
    small_data.loc[1, "diagnosis"] = "Flu"
    clean, report = validate(small_data, target="readmission_30d")
    assert len(clean) == len(small_data) - 1
    assert report["problems"]["diagnosis not recognised"] == 1


def test_target_that_is_not_0_or_1_is_dropped(small_data):
    small_data.loc[2, "readmission_30d"] = 2
    clean, report = validate(small_data, target="readmission_30d")
    assert len(clean) == len(small_data) - 1


def test_diastolic_higher_than_systolic_is_dropped(small_data):
    small_data.loc[3, "diastolic_bp"] = small_data.loc[3, "systolic_bp"] + 10
    clean, report = validate(small_data, target="readmission_30d")
    assert len(clean) == len(small_data) - 1
    assert report["problems"]["diastolic higher than systolic"] == 1


def test_duplicate_rows_are_removed(small_data):
    doubled = pd.concat([small_data, small_data.iloc[[0]]])
    clean, report = validate(doubled, target="readmission_30d")
    assert report["duplicate_rows"] == 1
    assert len(clean) == len(small_data)


def test_missing_values_are_counted_not_dropped(small_data):
    small_data.loc[4, "glucose"] = None
    clean, report = validate(small_data, target="readmission_30d")
    assert report["missing_values"]["glucose"] == 1
    assert len(clean) == len(small_data)


def test_bad_rows_stop_a_prediction_run(small_data):
    inputs = small_data.drop(columns=["readmission_30d"])
    inputs.loc[5, "sex"] = "Unknown"
    with pytest.raises(ValueError, match="Invalid rows"):
        validate(inputs)
