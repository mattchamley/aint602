# AINT602 ML Pipeline Evaluation Test Script
# Matt Chamley 27 November 2026

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import confusion, group_report, score
from src.features import build_pipeline, get_inputs


def test_score_includes_accuracy_and_it_is_a_share(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert 0 <= score(model, X, y)["accuracy"] <= 1


def test_baseline_accuracy_equals_the_majority_class_share(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    baseline = build_pipeline(DummyClassifier(strategy="prior")).fit(X, y)
    assert abs(score(baseline, X, y)["accuracy"] - (1 - y.mean())) < 1e-9


def test_baseline_finds_no_readmissions(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    baseline = build_pipeline(DummyClassifier(strategy="prior")).fit(X, y)
    assert score(baseline, X, y)["recall"] == 0


def test_confusion_counts_add_up_to_all_patients(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert sum(confusion(model, X, y).values()) == len(y)


def test_group_report_covers_every_patient(small_data):
    X = get_inputs(small_data)
    y = small_data["readmission_30d"]
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    table = group_report(model, X, y, "age_band")
    assert table["patients"].sum() == len(y)
