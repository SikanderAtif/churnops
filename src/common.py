"""Shared helpers: config loading, logging, hashing, MLflow setup."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path | str = ROOT / "params.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def file_md5(path: Path | str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def data_version(raw_path: Path | str) -> str:
    """Return the DVC md5 of the raw dataset (falls back to hashing the file)."""
    raw_path = ROOT / raw_path
    dvc_file = raw_path.with_name(raw_path.name + ".dvc")
    if dvc_file.exists():
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta["outs"][0]["md5"]
    return file_md5(raw_path)


def setup_mlflow(params: dict[str, Any]):
    """Point MLflow at the configured tracking server and experiment."""
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    return mlflow


def run_tags(params: dict[str, Any], role: str, framework: str) -> dict[str, str]:
    """Tags attached to every training run so it can be traced back to code + data."""
    return {
        "project": "churnops",
        "phase": "1",
        "model_role": role,
        "framework": framework,
        "git_commit": git_commit(),
        "data_version": data_version(params["data"]["raw_path"]),
    }
