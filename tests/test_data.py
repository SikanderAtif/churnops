import pandas as pd
import pytest

from src.data.preprocess import add_features, build_spec, clean, split
from src.data.validate import validate
from src.features import prepare_features


def make_raw(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "customerID": f"C{i:04d}", "gender": "Male" if i % 2 else "Female",
            "SeniorCitizen": str(i % 2), "Partner": "Yes", "Dependents": "No",
            "tenure": str(i % 12), "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": ["DSL", "Fiber optic", "No"][i % 3],
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": "70.5",
            "TotalCharges": " " if i % 12 == 0 else str(70.5 * (i % 12)),
            "Churn": "Yes" if i % 4 == 0 else "No",
        })
    return pd.DataFrame(rows)


def test_validation_passes_on_clean_data():
    assert validate(make_raw(), max_missing_fraction=0.2)["passed"]


def test_validation_catches_missing_column():
    report = validate(make_raw().drop(columns=["Contract"]), max_missing_fraction=0.2)
    assert not report["passed"] and "missing columns" in report["errors"][0]


def test_validation_catches_unknown_category():
    raw = make_raw()
    raw.loc[0, "Contract"] = "Five year"
    assert not validate(raw, max_missing_fraction=0.2)["passed"]


def test_clean_fills_first_month_total_charges_and_encodes_target():
    df = clean(make_raw(), "Churn", "customerID")
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    assert "customerID" not in df.columns


def test_split_is_stratified_and_disjoint():
    df = add_features(clean(make_raw(200), "Churn", "customerID"))
    train, val, test = split(df, "Churn", 0.15, 0.15, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
    for part in (train, val, test):
        assert abs(part["Churn"].mean() - df["Churn"].mean()) < 0.05


def test_prepare_features_handles_unseen_and_missing_values():
    df = add_features(clean(make_raw(), "Churn", "customerID"))
    spec = build_spec(df, "Churn")
    new = df.head(3).copy()
    new.loc[0, "MonthlyCharges"] = None
    X, y = prepare_features(new, spec)
    assert X["MonthlyCharges"].isna().sum() == 0
    assert list(X.columns) == spec["numeric_features"] + spec["categorical_features"]
