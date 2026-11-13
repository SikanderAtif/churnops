"""Turn processed splits into model-ready matrices using the saved feature spec.

The same functions will be reused by the FastAPI service in Phase 3, so training
and serving apply identical transformations.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def prepare_features(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y). Categorical columns get a fixed category set so encodings are stable."""
    target = spec["target"]
    y = df[target].astype(int) if target in df.columns else None
    X = df[spec["numeric_features"] + spec["categorical_features"]].copy()
    for col in spec["numeric_features"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(spec["impute"][col]).astype(float)
    for col in spec["categorical_features"]:
        values = X[col].astype("string").fillna(spec["impute"][col]).astype(str)
        X[col] = pd.Categorical(values, categories=spec["categories"][col])
    return X, y


def for_catboost(X: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """CatBoost wants categorical columns as plain strings."""
    X = X.copy()
    for col in spec["categorical_features"]:
        X[col] = X[col].astype(str)
    return X
