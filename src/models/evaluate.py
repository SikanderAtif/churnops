"""Score both models on the held-out test set and pick the champion.

Selection uses validation metrics (F1, gated on a minimum recall) so the test set
stays an unbiased final estimate.
"""
from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from src.common import ROOT, get_logger, load_json, load_params, save_json, setup_mlflow
from src.features import for_catboost, prepare_features
from src.metrics import classification_metrics

log = get_logger("evaluate")
CANDIDATES = {"baseline": ROOT / "models/baseline", "champion": ROOT / "models/champion"}


def predict(name: str, model_dir, X: pd.DataFrame, spec: dict):
    if name == "baseline":
        return joblib.load(model_dir / "model.joblib").predict_proba(X)[:, 1]
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))
    return model.predict_proba(for_catboost(X, spec))[:, 1]


def select(results: dict, metric: str, min_recall: float) -> str:
    eligible = {n: r for n, r in results.items() if r["val"]["recall"] >= min_recall}
    pool = eligible or results
    if not eligible:
        log.warning("No model reached val recall %.2f; choosing on %s alone", min_recall, metric)
    return max(pool, key=lambda n: pool[n]["val"][metric])


def main() -> None:
    params = load_params()
    sel = params["selection"]
    proc = ROOT / params["data"]["processed_dir"]
    spec = load_json(proc / "feature_spec.json")
    X_te, y_te = prepare_features(pd.read_parquet(proc / "test.parquet"), spec)

    mlflow = setup_mlflow(params)
    results = {}
    for name, model_dir in CANDIDATES.items():
        info = load_json(model_dir / "run_info.json")
        val = {k.removeprefix("val_"): v for k, v in load_json(ROOT / f"metrics/{name}.json").items() if k.startswith("val_")}
        test = classification_metrics(y_te, predict(name, model_dir, X_te, spec), info["threshold"])
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        results[name] = {"framework": info["framework"], "run_id": info["run_id"], "model_uri": info["model_uri"],
                         "threshold": info["threshold"], "val": val, "test": test}
        log.info("%s test: F1=%.4f recall=%.4f ROC-AUC=%.4f", name, test["f1"], test["recall"], test["roc_auc"])

    winner = select(results, sel["primary_metric"], sel["min_recall"])
    save_json({"selected": winner, "selection_rule": sel, "models": results}, ROOT / "metrics/evaluation.json")
    save_json({f"{n}_test_{k}": results[n]["test"][k] for n in results for k in ("f1", "recall", "precision", "roc_auc", "pr_auc")},
              ROOT / "metrics/test.json")

    rows = ["| Model | Framework | Split | F1 | Recall | Precision | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        for split in ("val", "test"):
            m = r[split]
            rows.append(f"| {name} | {r['framework']} | {split} | {m['f1']:.4f} | {m['recall']:.4f} | "
                        f"{m['precision']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")
    report = (f"# Model comparison\n\nSelected for registry alias `champion`: **{winner}** "
              f"({results[winner]['framework']})\n\nRule: best validation {sel['primary_metric']} among models "
              f"with validation recall >= {sel['min_recall']}.\n\n" + "\n".join(rows) + "\n")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/model_comparison.md").write_text(report)
    log.info("Selected: %s", winner)


if __name__ == "__main__":
    main()
