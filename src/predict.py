# AINT602 ML Pipeline Predict script
# Matt Chamley 26 November 2026

import argparse
import json
import os
from datetime import datetime

import joblib

from src.features import get_inputs
from src.ingest import load_config, load_data
from src.validate import validate


def predict(df, model):
    clean, report = validate(df)
    inputs = get_inputs(clean)
    result = clean.copy()
    result["readmission_probability"] = model.predict_proba(inputs)[:, 1]
    result["high_risk_flag"] = model.predict(inputs)
    result["queue_rank"] = result["readmission_probability"].rank(ascending=False, method="first").astype(int)
    return result.sort_values("queue_rank"), report


def write_log(result, info, path):
    log = result.copy()
    log["model_name"] = info["model_name"]
    log["model_commit"] = info["git_commit"]
    log["predicted_at"] = datetime.now().isoformat(timespec="seconds")
    log.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_new_patients.csv")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()

    config = load_config()
    model = joblib.load(config["model_path"])
    with open(config["model_info_path"]) as f:
        info = json.load(f)

    new_data = load_data(args.input)
    result, report = predict(new_data, model)
    result.to_csv(args.output, index=False)
    write_log(result, info, config["prediction_log_path"])

    print(f"Model: {info['model_name']} (git {info['git_commit']}, data {info['data_hash'][:8]})")
    print(f"Rows predicted: {len(result)}")
    print(f"Missing values in input: {report['missing_values'] or 'none'}")
    print(f"Flagged high risk: {result['high_risk_flag'].mean():.1%}")
    print(f"Mean predicted probability: {result['readmission_probability'].mean():.3f}")
    print(f"Training readmission rate: {info['training_readmission_rate']:.3f}")
    print(f"Saved ranked predictions to {args.output} and appended to {config['prediction_log_path']}")


if __name__ == "__main__":
    main()
