"""Champion: CatBoost tuned with Optuna. Every trial is a nested MLflow run."""
from __future__ import annotations

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score

from src.common import ROOT, get_logger, load_json, load_params, run_tags, save_json, setup_mlflow
from src.features import for_catboost, prepare_features
from src.metrics import best_threshold, classification_metrics

log = get_logger("tune_champion")
MODEL_DIR = ROOT / "models/champion"


def build_model(hp: dict, cfg: dict, cat_features: list[str], seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        **hp,
        iterations=cfg["max_iterations"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        auto_class_weights="Balanced",
        cat_features=cat_features,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )


def sample_params(trial: optuna.Trial, space: dict) -> dict:
    return {
        "depth": trial.suggest_int("depth", *space["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *space["l2_leaf_reg"], log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", *space["bagging_temperature"]),
        "random_strength": trial.suggest_float("random_strength", *space["random_strength"]),
        "border_count": trial.suggest_int("border_count", *space["border_count"]),
    }


def main() -> None:
    params = load_params()
    cfg = params["champion"]
    seed = params["seed"]
    proc = ROOT / params["data"]["processed_dir"]
    spec = load_json(proc / "feature_spec.json")
    cats = spec["categorical_features"]
    X_tr, y_tr = prepare_features(pd.read_parquet(proc / "train.parquet"), spec)
    X_va, y_va = prepare_features(pd.read_parquet(proc / "val.parquet"), spec)
    X_tr, X_va = for_catboost(X_tr, spec), for_catboost(X_va, spec)

    mlflow = setup_mlflow(params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="champion-catboost-optuna") as parent:
        mlflow.set_tags(run_tags(params, role="champion", framework="catboost"))
        mlflow.log_params({"n_trials": cfg["n_trials"], "max_iterations": cfg["max_iterations"],
                           "early_stopping_rounds": cfg["early_stopping_rounds"],
                           "optuna_objective": "val_pr_auc", "seed": seed, "n_train": len(X_tr)})

        def objective(trial: optuna.Trial) -> float:
            hp = sample_params(trial, cfg["search_space"])
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                model = build_model(hp, cfg, cats, seed)
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                p_va = model.predict_proba(X_va)[:, 1]
                pr_auc = average_precision_score(y_va, p_va)
                mlflow.log_params(hp)
                mlflow.log_metrics({"val_pr_auc": pr_auc, "best_iteration": model.get_best_iteration()})
            log.info("trial %d: PR-AUC=%.4f", trial.number, pr_auc)
            return pr_auc

        # PR-AUC is threshold-free and suits the imbalanced target; F1 is optimised afterwards via the threshold.
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    study_name="catboost-churn")
        study.optimize(objective, n_trials=cfg["n_trials"], timeout=cfg["timeout_seconds"])

        best = study.best_params
        model = build_model(best, cfg, cats, seed)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        p_va = model.predict_proba(X_va)[:, 1]
        threshold = best_threshold(y_va, p_va, params["selection"]["min_recall"])
        metrics = classification_metrics(y_va, p_va, threshold)

        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({"decision_threshold": threshold, "best_iteration": model.get_best_iteration(),
                            "optuna_best_pr_auc": study.best_value})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(MODEL_DIR / "optuna_trials.csv", index=False)
        pd.DataFrame({"feature": X_tr.columns, "importance": model.get_feature_importance()}) \
            .sort_values("importance", ascending=False).to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        for f in ["optuna_trials.csv", "feature_importance.csv"]:
            mlflow.log_artifact(str(MODEL_DIR / f))
        mlflow.log_artifact(str(proc / "feature_spec.json"))

        info = mlflow.catboost.log_model(model, name="model", signature=infer_signature(X_va, p_va))

    model.save_model(str(MODEL_DIR / "model.cbm"))
    save_json({"run_id": parent.info.run_id, "model_uri": info.model_uri, "framework": "catboost",
               "threshold": threshold, "best_params": best}, MODEL_DIR / "run_info.json")
    save_json({"threshold": threshold, "n_trials_completed": len(study.trials),
               **{f"val_{k}": v for k, v in metrics.items()}}, ROOT / "metrics/champion.json")
    log.info("Champion val F1=%.4f recall=%.4f ROC-AUC=%.4f (threshold %.2f) best=%s",
             metrics["f1"], metrics["recall"], metrics["roc_auc"], threshold, best)


if __name__ == "__main__":
    main()
