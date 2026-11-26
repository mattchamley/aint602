
# AINT620 ML Pipeline Training Script
# Matt Chamley 26 November 2026

import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluate import confusion, score
from src.features import build_pipeline, get_inputs
from src.ingest import file_hash, load_config, load_data
from src.validate import validate


def rounded(scores):
    return {k: round(v, 3) for k, v in scores.items()}


def split_data(X, y, config):
    holdout = config["val_size"] + config["test_size"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=config["random_state"])
    test_share = config["test_size"] / holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=test_share, stratify=y_rest,
        random_state=config["random_state"])
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    config = load_config()
    raw = load_data(config["data_path"])
    clean, report = validate(raw, target=config["target"])
    print("Validation report:", report)

    X = get_inputs(clean)
    y = clean[config["target"]]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    print(f"Train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")
    print(f"Readmission rate: train {y_train.mean():.3f}, validation {y_val.mean():.3f}, test {y_test.mean():.3f}")

    seed = config["random_state"]
    models = {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(**config["logistic_regression"]),
        "random_forest": RandomForestClassifier(**config["random_forest"], random_state=seed),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config["mlflow_experiment"])
    data_hash = file_hash(config["data_path"])

    fitted, val_scores, run_ids = {}, {}, {}
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        val_scores[name] = score(pipeline, X_val, y_val)
        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_params({"data_hash": data_hash, "split_random_state": seed})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_scores[name].items()})
        print(name, "validation:", rounded(val_scores[name]))
        print(name, "confusion:", confusion(pipeline, X_val, y_val))

    candidates = [n for n in models if n != "baseline_dummy"]
    best_name = max(candidates, key=lambda n: val_scores[n]["roc_auc"])
    best = fitted[best_name]
    print("Selected model:", best_name)

    test_scores = score(best, X_test, y_test)
    baseline_test = score(fitted["baseline_dummy"], X_test, y_test)
    print("Test scores (selected):", rounded(test_scores))
    print("Test scores (baseline):", rounded(baseline_test))
    print("Test confusion (selected):", confusion(best, X_test, y_test))

    with mlflow.start_run(run_id=run_ids[best_name]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})


if __name__ == "__main__":
    main()
