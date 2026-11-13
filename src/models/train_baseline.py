"""Baseline: LightGBM with near-default settings, logged to MLflow."""
from __future__ import annotations

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature

from src.common import ROOT, get_logger, load_json, load_params, run_tags, save_json, setup_mlflow
from src.features import prepare_features
from src.metrics import best_threshold, classification_metrics

log = get_logger("train_baseline")
MODEL_DIR = ROOT / "models/baseline"


def main() -> None:
    params = load_params()
    cfg = params["baseline"]
    proc = ROOT / params["data"]["processed_dir"]
    spec = load_json(proc / "feature_spec.json")
    X_tr, y_tr = prepare_features(pd.read_parquet(proc / "train.parquet"), spec)
    X_va, y_va = prepare_features(pd.read_parquet(proc / "val.parquet"), spec)

    model = LGBMClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        class_weight="balanced",   # ~26% churners, so weight the minority class up
        random_state=params["seed"],
        verbose=-1,
    )

    mlflow = setup_mlflow(params)
    with mlflow.start_run(run_name="baseline-lightgbm") as run:
        mlflow.set_tags(run_tags(params, role="baseline", framework="lightgbm"))
        mlflow.log_params({**cfg, "class_weight": "balanced", "seed": params["seed"], "n_train": len(X_tr)})

        model.fit(X_tr, y_tr, categorical_feature=spec["categorical_features"])
        p_va = model.predict_proba(X_va)[:, 1]
        threshold = best_threshold(y_va, p_va, params["selection"]["min_recall"])
        metrics = classification_metrics(y_va, p_va, threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("decision_threshold", threshold)

        importance = pd.DataFrame(
            {"feature": X_tr.columns, "importance": model.booster_.feature_importance("gain")}
        ).sort_values("importance", ascending=False)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(MODEL_DIR / "feature_importance.csv"))
        mlflow.log_artifact(str(proc / "feature_spec.json"))

        info = mlflow.lightgbm.log_model(
            model, name="model", signature=infer_signature(X_va, p_va)
        )

    joblib.dump(model, MODEL_DIR / "model.joblib")
    save_json(
        {"run_id": run.info.run_id, "model_uri": info.model_uri, "framework": "lightgbm",
         "threshold": threshold},
        MODEL_DIR / "run_info.json",
    )
    save_json({"threshold": threshold, **{f"val_{k}": v for k, v in metrics.items()}},
              ROOT / "metrics/baseline.json")
    log.info("Baseline val F1=%.4f recall=%.4f ROC-AUC=%.4f (threshold %.2f)",
             metrics["f1"], metrics["recall"], metrics["roc_auc"], threshold)


if __name__ == "__main__":
    main()
