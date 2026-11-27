# AINT602 ML Pipeline Training Script
# Matt Chamley 26 November 2026

import json
import subprocess
from datetime import datetime

import joblib
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluate import confusion, group_report, save_roc_plot, score
from src.features import build_pipeline, get_inputs
from src.ingest import file_hash, load_config, load_data
from src.validate import validate


def rounded(scores):
    return {k: round(v, 3) for k, v in scores.items()}


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def git_commit():
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


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
    reports = config["reports_dir"]
    raw = load_data(config["data_path"])
    clean, report = validate(raw, target=config["target"])
    save_json(report, f"{reports}/validation_report.json")
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
    commit = git_commit()

    fitted, val_scores, run_ids = {}, {}, {}
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        val_scores[name] = score(pipeline, X_val, y_val)
        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_params({"data_hash": data_hash, "git_commit": commit,
                               "split_random_state": seed})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_scores[name].items()})
        print(name, "validation:", rounded(val_scores[name]))
        print(name, "confusion:", confusion(pipeline, X_val, y_val))

    candidates = [n for n in models if n != "baseline_dummy"]
    best_name = max(candidates, key=lambda n: val_scores[n]["roc_auc"])
    best = fitted[best_name]
    print("Selected model:", best_name)

    test_scores = score(best, X_test, y_test)
    baseline_test = score(fitted["baseline_dummy"], X_test, y_test)
    matrix = confusion(best, X_test, y_test)
    print("Test scores (selected):", rounded(test_scores))
    print("Test scores (baseline):", rounded(baseline_test))
    print("Test confusion (selected):", matrix)

    save_roc_plot(fitted, X_val, y_val, f"{reports}/roc_validation.png")
    group_report(best, X_test, y_test, "sex").to_csv(f"{reports}/results_by_sex.csv", index=False)
    group_report(best, X_test, y_test, "age_band").to_csv(f"{reports}/results_by_age_band.csv", index=False)
    save_json({
        "selected_model": best_name,
        "validation": val_scores,
        "test_selected": test_scores,
        "test_baseline": baseline_test,
        "confusion_matrix_test": matrix,
    }, f"{reports}/metrics.json")

    joblib.dump(best, config["model_path"])
    save_json({
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "data_file": config["data_path"],
        "data_hash": data_hash,
        "rows_used": len(clean),
        "training_readmission_rate": float(y_train.mean()),
        "random_state": seed,
        "mlflow_run_id": run_ids[best_name],
        "test_scores": test_scores,
    }, config["model_info_path"])

    X_test.head(20).to_csv(config["sample_patients_path"], index=False)

    with mlflow.start_run(run_id=run_ids[best_name]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})
        mlflow.log_artifact(config["model_path"])
        mlflow.log_artifact(config["model_info_path"])
        mlflow.log_artifact(f"{reports}/metrics.json")


if __name__ == "__main__":
    main()
