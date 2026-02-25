"""
What drove one series' median forecast, read from the trained booster.

`ForecastFacts.drivers` is the part of an explanation that ties it to the
model rather than to a plausible story, so it is computed here from the saved
median booster's `pred_contrib` on the same rows the forecast was made from.
With no trained model on disk, or a feature table without the model's inputs,
there are no drivers and the explanation leaves that sentence out.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.config import QUANTILES
from src.models.train_lgb_quantile import feature_contributions, future_panel

MEDIAN = QUANTILES[len(QUANTILES) // 2]


@lru_cache(maxsize=4)
def _load_booster(path: str, mtime_ns: int, size: int) -> lgb.Booster:  # noqa: ARG001
    # The stat fields are part of the cache key, so a retrained model is
    # picked up on the next call without a restart.
    return lgb.Booster(model_file=path)


def median_booster(model_dir: Path) -> lgb.Booster | None:
    """The saved median-quantile booster, or None if the pipeline has not trained one."""
    path = model_dir / f"lgb_q{MEDIAN}.txt"
    if not path.exists():
        return None
    st = path.stat()
    return _load_booster(str(path), st.st_mtime_ns, st.st_size)


def _model_inputs(model_dir: Path) -> list[str] | None:
    path = model_dir / "feature_columns.json"
    return json.loads(path.read_text()) if path.exists() else None


def series_drivers(
    features: pd.DataFrame,
    sku: str,
    region: str,
    horizon: int,
    model_dir: Path,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    (feature, mean contribution in units) for one series' next `horizon` weeks.

    Empty when there is nothing to attribute: no model, no rows for the
    series, or a feature table that lacks columns the model was trained on.
    """
    booster = median_booster(model_dir)
    if booster is None or horizon < 1:
        return []

    series = features[(features["sku"] == sku) & (features["region"] == region)]
    if series.empty:
        return []

    future = future_panel(series, horizon)
    inputs = _model_inputs(model_dir) or booster.feature_name()
    if any(col not in future.columns for col in inputs):
        return []
    return feature_contributions(booster, future, top_n=top_n)
