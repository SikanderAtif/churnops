"""Validate the raw dataset against the expected schema before anything trains on it."""
from __future__ import annotations

import sys

import pandas as pd

from src.common import ROOT, get_logger, load_params, save_json
from src.data.schema import CATEGORICAL, ID_COLUMN, NUMERIC, REQUIRED_COLUMNS

log = get_logger("validate")


def validate(df: pd.DataFrame, max_missing_fraction: float) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return {"passed": False, "errors": errors, "warnings": warnings, "n_rows": len(df)}

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"unexpected extra columns (ignored): {extra}")

    if df[ID_COLUMN].duplicated().any():
        errors.append(f"{int(df[ID_COLUMN].duplicated().sum())} duplicate {ID_COLUMN} values")

    missing_report = {}
    for col, allowed in CATEGORICAL.items():
        values = df[col].astype("string").str.strip()
        n_missing = int(values.isna().sum() + (values == "").sum())
        missing_report[col] = n_missing
        bad = sorted(set(values.dropna()) - set(allowed) - {""})
        if bad:
            errors.append(f"{col}: unexpected categories {bad}")

    for col, (lo, hi) in NUMERIC.items():
        raw = df[col].astype("string").str.strip().replace("", pd.NA)
        numeric = pd.to_numeric(raw, errors="coerce")
        non_numeric = int((numeric.isna() & raw.notna()).sum())
        if non_numeric:
            errors.append(f"{col}: {non_numeric} non-numeric values")
        missing_report[col] = int(numeric.isna().sum())
        if lo is not None and (numeric < lo).any():
            errors.append(f"{col}: values below {lo}")
        if hi is not None and (numeric > hi).any():
            errors.append(f"{col}: values above {hi}")

    for col, n in missing_report.items():
        frac = n / len(df)
        if frac > max_missing_fraction:
            errors.append(f"{col}: {frac:.2%} missing exceeds {max_missing_fraction:.2%}")
        elif n:
            warnings.append(f"{col}: {n} missing values (will be handled in preprocessing)")

    churn_rate = float((df["Churn"].str.strip() == "Yes").mean())
    if not 0.05 <= churn_rate <= 0.60:
        warnings.append(f"churn rate {churn_rate:.2%} is outside the expected range")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "churn_rate": round(churn_rate, 4),
        "missing_values": {k: v for k, v in missing_report.items() if v},
    }


def main() -> None:
    params = load_params()
    df = pd.read_csv(ROOT / params["data"]["raw_path"], dtype=str)
    report = validate(df, params["validation"]["max_missing_fraction"])
    save_json(report, ROOT / "reports/validation.json")
    for w in report["warnings"]:
        log.warning(w)
    if not report["passed"]:
        for e in report["errors"]:
            log.error(e)
        sys.exit(1)
    log.info("Validation passed: %d rows, churn rate %.2f%%", report["n_rows"], 100 * report["churn_rate"])


if __name__ == "__main__":
    main()
