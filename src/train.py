# AINT620 ML Pipeline Training Script
# Matt Chamley 26 November 2026

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.evaluate import confusion, score
from src.features import build_pipeline, get_inputs
from src.ingest import load_config, load_data
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

    baseline = build_pipeline(DummyClassifier(strategy="prior"))
    baseline.fit(X_train, y_train)
    print("Baseline validation scores:", rounded(score(baseline, X_val, y_val)))
    print("Baseline confusion matrix:", confusion(baseline, X_val, y_val))


if __name__ == "__main__":
    main()
