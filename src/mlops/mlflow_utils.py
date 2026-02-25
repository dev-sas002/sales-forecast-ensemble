"""MLflow helpers for starting runs and logging params/metrics."""

from __future__ import annotations

import mlflow


def start_run(params: dict[str, str] | None = None):
    """Start an MLflow run and optionally log params."""
    run = mlflow.start_run()
    if params:
        for k, v in params.items():
            mlflow.log_param(k, v)
    return run


def log_metrics(metrics: dict[str, float]) -> None:
    """Log each key-value in metrics as an MLflow metric."""
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))
