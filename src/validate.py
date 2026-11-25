# AINT602 ML Pipeline Data Validation
# Matt Chamley 26 November 2026

import pandas as pd

INPUTS = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp", "glucose",
    "cholesterol", "creatinine", "diabetes", "hypertension", "diagnosis",
]

RANGES = {
    "age": (0, 100),
    "bmi": (10, 70),
    "systolic_bp": (60, 250),
    "diastolic_bp": (30, 150),
    "glucose": (30, 600),
    "cholesterol": (50, 500),
    "creatinine": (0.1, 15),
}

CATEGORIES = {
    "sex": ["Female", "Male", "Other"],
    "diagnosis": ["Normal", "Pneumonia", "Heart Failure", "Sepsis"],
}


def check_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def find_problems(df, target=None):
    problems = {}
    for col, (low, high) in RANGES.items():
        problems[f"{col} out of range"] = (df[col] < low) | (df[col] > high)
    for col in ["diabetes", "hypertension"]:
        problems[f"{col} not 0 or 1"] = df[col].notna() & ~df[col].isin([0, 1])
    for col, allowed in CATEGORIES.items():
        problems[f"{col} not recognised"] = df[col].notna() & ~df[col].isin(allowed)
    problems["diastolic higher than systolic"] = df["diastolic_bp"] > df["systolic_bp"]
    if target:
        problems["target not 0 or 1"] = ~df[target].isin([0, 1])
    return problems


def validate(df, target=None):
    check_columns(df, INPUTS + ([target] if target else []))
    problems = find_problems(df, target)
    bad_rows = pd.concat(problems.values(), axis=1).any(axis=1)

    report = {
        "rows_in": len(df),
        "missing_values": {c: int(n) for c, n in df[INPUTS].isna().sum().items() if n > 0},
        "problems": {k: int(v.sum()) for k, v in problems.items() if v.sum() > 0},
    }

    if target is None:
        if bad_rows.any():
            raise ValueError(f"Invalid rows {list(df.index[bad_rows])}: {report['problems']}")
        report["rows_out"] = len(df)
        return df, report

    report["duplicate_rows"] = int(df.duplicated().sum())
    clean = df[~bad_rows].drop_duplicates()
    report["rows_out"] = len(clean)
    return clean, report
