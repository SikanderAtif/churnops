"""Register both models in the MLflow Model Registry with version tags and aliases.

The selected model gets the alias `champion`, the other gets `challenger`.
Serving (Phase 3) loads `models:/<name>@champion`, so promoting a model is an alias move.
"""
from __future__ import annotations

from mlflow import MlflowClient

from src.common import ROOT, data_version, get_logger, git_commit, load_json, load_params, save_json, setup_mlflow

log = get_logger("register")


def main() -> None:
    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    evaluation = load_json(ROOT / "metrics/evaluation.json")
    mlflow = setup_mlflow(params)
    client = MlflowClient()

    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(
            name, description="ChurnOps telecom churn classifier. Alias 'champion' is the model to serve.",
            tags={"project": "churnops", "task": "binary-classification"},
        )

    registered = {}
    for role, r in evaluation["models"].items():
        alias = "champion" if role == evaluation["selected"] else "challenger"
        mv = mlflow.register_model(r["model_uri"], name)
        tags = {
            "framework": r["framework"],
            "trained_as": role,
            "alias_at_registration": alias,
            "decision_threshold": f"{r['threshold']:.2f}",
            "data_version": data_version(params["data"]["raw_path"]),
            "git_commit": git_commit(),
            "val_f1": f"{r['val']['f1']:.4f}",
            "val_recall": f"{r['val']['recall']:.4f}",
            "test_f1": f"{r['test']['f1']:.4f}",
            "test_recall": f"{r['test']['recall']:.4f}",
            "test_roc_auc": f"{r['test']['roc_auc']:.4f}",
        }
        for k, v in tags.items():
            client.set_model_version_tag(name, mv.version, k, v)
        client.set_registered_model_alias(name, alias, mv.version)
        registered[role] = {"version": mv.version, "alias": alias, **tags}
        log.info("Registered %s v%s as @%s", r["framework"], mv.version, alias)

    save_json({"registered_model": name, "versions": registered}, ROOT / "reports/registry.json")


if __name__ == "__main__":
    main()
