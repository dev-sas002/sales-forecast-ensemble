"""
Airflow DAG over the same pipeline steps `scripts/run_pipeline.py` runs.

Optional: Airflow is not a core dependency and nothing else imports this. It
is here because a weekly forecast is a scheduled job in every real deployment,
and the DAG makes the dependency order explicit.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_DATA_DIR = "./data"


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ingest(**kwargs):
    data_dir = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
    _run(
        [
            "python",
            "-m",
            "src.ingest.ingest",
            "--credit_csv",
            f"{data_dir}/credit.csv",
            "--panel_csv",
            f"{data_dir}/panel.csv",
        ]
    )


def run_link(**kwargs):
    _run(["python", "-m", "src.linking.link_panel_credit"])


def run_rake(**kwargs):
    _run(["python", "-m", "src.weighting.raking"])


def build_feat(**kwargs):
    _run(["python", "-m", "src.features.build_features"])


def train_lgb(**kwargs):
    # Quantiles come from src.config.QUANTILES; the trainer takes no
    # --quantiles flag and argparse exits non-zero if one is passed.
    _run(["python", "-m", "src.models.train_lgb_quantile", "--horizon", "12"])


def ensemble(**kwargs):
    _run(["python", "-m", "src.ensemble.ensemble_and_reconcile"])


def backtest(**kwargs):
    # Non-zero exit when the model stops beating the baseline or the interval
    # drifts off nominal, so the schedule fails loudly rather than publishing
    # a forecast nobody scored.
    _run(["python", "-m", "src.evaluation.backtest", "--folds", "4", "--horizon", "8"])


default_args = {
    "owner": "forecast_mvp",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "forecast_mvp_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@weekly",
    default_args=default_args,
    catchup=False,
    tags=["mvp"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=run_ingest)
    link = PythonOperator(task_id="link", python_callable=run_link)
    rake = PythonOperator(task_id="rake", python_callable=run_rake)
    features = PythonOperator(task_id="build_features", python_callable=build_feat)
    lgb = PythonOperator(task_id="train_lgb", python_callable=train_lgb)
    ens = PythonOperator(task_id="ensemble", python_callable=ensemble)
    score = PythonOperator(task_id="backtest", python_callable=backtest)

    # No train_deepar task: src/models/train_deepar_gluonts.py is a stub that
    # raises NotImplementedError, so scheduling it made every DAG run fail. It
    # is also not in src.models.registry, which is what stops its stale output
    # being blended — see src/models/registry.py. Register it here and there
    # together when the trainer exists.
    ingest >> link >> rake >> features >> lgb >> ens >> score
