"""Clean the raw data, engineer features, split, impute, and write processed splits.

Tree models (LightGBM, CatBoost) don't need feature scaling, and both handle
categorical columns natively, so encoding here means fixing each column's
category set rather than one-hot expanding it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import ROOT, get_logger, load_params, save_json
from src.data.schema import CATEGORICAL, NUMERIC, SERVICE_COLUMNS

log = get_logger("preprocess")


def clean(df: pd.DataFrame, target: str, id_column: str) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    for col in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Customers in their first month (tenure 0) have a blank TotalCharges: nothing billed yet.
    first_month = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[first_month, "TotalCharges"] = 0.0

    # SeniorCitizen is a 0/1 flag; treat it as categorical like the other Yes/No fields.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df[target] = df[target].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    return df.drop(columns=[id_column])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only, so computing them before the split leaks nothing."""
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, np.nan), df["MonthlyCharges"]
    )
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    return df


def split(df: pd.DataFrame, target: str, test_size: float, val_size: float, seed: int):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[target], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_spec(train: pd.DataFrame, target: str) -> dict:
    """Feature spec fitted on the training split only (imputation values, category sets)."""
    categorical = [c for c in CATEGORICAL if c != target] + ["SeniorCitizen"]
    numeric = [c for c in NUMERIC if c != "SeniorCitizen"] + ["num_services", "avg_monthly_spend", "charge_increase"]
    impute = {c: float(train[c].median()) for c in numeric}
    impute.update({c: str(train[c].mode().iloc[0]) for c in categorical})
    categories = {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in categorical}
    return {
        "target": target,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "impute": impute,
        "categories": categories,
    }


def main() -> None:
    params = load_params()
    d = params["data"]
    out_dir = ROOT / d["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / d["raw_path"], dtype=str)
    df = add_features(clean(raw, d["target"], d["id_column"]))
    train, val, test = split(df, d["target"], d["test_size"], d["val_size"], params["seed"])

    spec = build_spec(train, d["target"])
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        log.info("%s: %d rows, churn rate %.2f%%", name, len(part), 100 * part[d["target"]].mean())
    save_json(spec, out_dir / "feature_spec.json")
    save_json(
        {
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
            "churn_rate": {n: round(float(p[d["target"]].mean()), 4) for n, p in [("train", train), ("val", val), ("test", test)]},
            "n_features": len(spec["numeric_features"]) + len(spec["categorical_features"]),
        },
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
