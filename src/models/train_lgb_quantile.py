"""
Quantile demand forecasting with LightGBM.

Design decisions worth stating:

**Direct multi-horizon, one model per quantile.** Rather than a separate model
per (quantile, horizon) pair — 36 models for a 12-week horizon — the horizon
is a *feature*, and each quantile gets one model trained across all horizons.
Three models instead of thirty-six, one place for a bug to hide instead of
thirty-six, and forecasting a horizon the training loop never enumerated is a
matter of passing a different number.

**Direct, not recursive.** A recursive forecaster feeds its own week-1
prediction back in to produce week 2, so errors compound and the uncertainty
bands are optimistic in a way nothing in the output reveals. Predicting week
t+h directly from what is known at t keeps each horizon's error independent
and honest.

**Quantiles are sorted after prediction.** Three independently fitted models
have no constraint tying them together, so q0.9 can land below q0.5 — a
"prediction interval" with a negative width. Sorting each row's quantiles is
the standard, distribution-free repair, and it can only reduce pinball loss.

**Intervals are conformalised.** Quantile regression is not calibrated: the
measured 80% interval covered about 72% of held-out actuals, which is a model
overstating its own precision. `src.evaluation.conformal` widens the interval
by an offset learned on a held-out split; the offset is saved beside the
boosters and applied at forecast time. See that module for the method and the
measured effect.

The module-level functions are the pipeline's working interface and are what
the tests exercise. :class:`LightGBMQuantileForecaster` wraps them to satisfy
the registry protocol, so the ensemble discovers this model rather than
naming its output file from a downstream constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import FEATURE_DIR, MODEL_DIR, QUANTILES
from src.features.build_features import load_feature_table
from src.features.registry import HISTORY_PREFIXES
from src.models import registry

TARGET = "units_sold"
KEYS = ["sku", "region"]
CATEGORICAL = ["sku", "region"]


def history_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs present in `df`, selected by the registry's prefixes."""
    return sorted(c for c in df.columns if c.startswith(HISTORY_PREFIXES))


def make_supervised(features: pd.DataFrame, horizons: range) -> pd.DataFrame:
    """
    Reshape one row per week into one row per (origin week, horizon).

    For origin t and horizon h the target is units at t+h, while every feature
    comes from t — except the calendar of the target week, which is known in
    advance and is what lets the model place the forecast in the year.
    """
    hist_cols = history_columns(features)
    features = features.sort_values(KEYS + ["week"]).reset_index(drop=True)
    grouped = features.groupby(KEYS, sort=False)

    frames = []
    for h in horizons:
        block = features[KEYS + ["week"] + hist_cols].copy()
        block["horizon"] = h
        block["target_week"] = grouped["week"].shift(-h)
        block[TARGET] = grouped[TARGET].shift(-h)
        frames.append(block)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["target_week", TARGET])

    target_week = pd.to_datetime(panel["target_week"])
    panel["target_weekofyear"] = target_week.dt.isocalendar().week.astype(int)
    panel["target_month"] = target_week.dt.month.astype(int)

    return panel


def feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = history_columns(panel) + ["horizon", "target_weekofyear", "target_month"] + CATEGORICAL
    X = panel[cols].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X, cols


def train_quantile_models(
    panel: pd.DataFrame,
    quantiles: tuple[float, ...] = QUANTILES,
    num_boost_round: int = 300,
    seed: int = 7,
) -> tuple[dict[float, lgb.Booster], list[str]]:
    """Fit one LightGBM booster per quantile."""
    X, cols = feature_matrix(panel)
    y = panel[TARGET].astype(float)

    models: dict[float, lgb.Booster] = {}
    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": seed,
            # Determinism: a portfolio metric that moves between runs on
            # identical input is not a measurement.
            "deterministic": True,
            "num_threads": 4,
        }
        models[q] = lgb.train(
            params,
            lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL),
            num_boost_round=num_boost_round,
        )
    return models, cols


def predict_quantiles(
    models: dict[float, lgb.Booster],
    panel: pd.DataFrame,
    suffix: str = "lgb",
    widen: float = 0.0,
) -> pd.DataFrame:
    """
    Predict every quantile and enforce monotonicity across them.

    `widen` is the conformal offset: the outer quantiles are pushed out by it
    before clipping, which is what turns a nominal 80% interval into one that
    actually covers 80%. Zero means "report the raw quantile regression",
    which is what the backtest measures before calibration.
    """
    X, _ = feature_matrix(panel)
    quantiles = sorted(models)

    preds = np.column_stack([models[q].predict(X) for q in quantiles])
    if widen > 0 and preds.shape[1] >= 2:
        preds[:, 0] -= widen
        preds[:, -1] += widen
    # Demand is non-negative; then sort so the interval cannot be inverted.
    preds = np.clip(preds, 0.0, None)
    preds = np.sort(preds, axis=1)

    out = panel[KEYS + ["target_week", "horizon"]].copy()
    out = out.rename(columns={"target_week": "date"})
    for i, q in enumerate(quantiles):
        out[f"q{q}_{suffix}"] = preds[:, i]
    return out.reset_index(drop=True)


def forecast_from_last_origin(
    models: dict[float, lgb.Booster],
    features: pd.DataFrame,
    horizon: int,
    widen: float = 0.0,
) -> pd.DataFrame:
    """
    Forecast forward from each series' most recent week.

    The training panel drops rows whose target is unknown, which is exactly the
    future we want to predict — so the forecast rows are rebuilt here from the
    last observed origin rather than reused from training.
    """
    hist_cols = history_columns(features)
    last = (
        features.sort_values(KEYS + ["week"])
        .groupby(KEYS, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )

    rows = []
    for h in range(1, horizon + 1):
        block = last[KEYS + ["week"] + hist_cols].copy()
        block["horizon"] = h
        block["date"] = pd.to_datetime(block["week"]) + pd.Timedelta(weeks=h)
        rows.append(block)

    future = pd.concat(rows, ignore_index=True)
    dates = pd.to_datetime(future["date"])
    future["target_weekofyear"] = dates.dt.isocalendar().week.astype(int)
    future["target_month"] = dates.dt.month.astype(int)
    future["target_week"] = future["date"]

    return predict_quantiles(models, future, widen=widen)


def feature_contributions(
    model: lgb.Booster, panel: pd.DataFrame, top_n: int = 5
) -> list[list[tuple[str, float]]]:
    """
    Per-row feature attributions from LightGBM's own `pred_contrib`.

    Exact tree SHAP values, not a surrogate: each row's contributions plus the
    base value sum to that row's prediction. This is what the explanation
    layer narrates, which is what keeps the narrative tied to the model rather
    than to a plausible story about it.
    """
    X, cols = feature_matrix(panel)
    contrib = model.predict(X, pred_contrib=True)
    # The final column is the base value; the rest align with `cols`.
    values = np.asarray(contrib)[:, :-1]

    out = []
    for row in values:
        order = np.argsort(-np.abs(row))[:top_n]
        out.append([(cols[i], float(row[i])) for i in order])
    return out


@registry.register
class LightGBMQuantileForecaster:
    """The registry's view of this module. See :mod:`src.models.registry`."""

    name = "lgb"

    def __init__(self) -> None:
        self.models: dict[float, lgb.Booster] = {}
        self.columns: list[str] = []
        self.widen: float = 0.0

    def fit(
        self,
        panel: pd.DataFrame,
        quantiles: tuple[float, ...] = QUANTILES,
        num_boost_round: int = 300,
        seed: int = 7,
    ) -> LightGBMQuantileForecaster:
        self.models, self.columns = train_quantile_models(
            panel, quantiles=quantiles, num_boost_round=num_boost_round, seed=seed
        )
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        return predict_quantiles(self.models, panel, suffix=self.name, widen=self.widen)

    def forecast(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        return forecast_from_last_origin(self.models, features, horizon, widen=self.widen)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for q, booster in self.models.items():
            booster.save_model(str(model_dir / f"lgb_q{q}.txt"))
        (model_dir / "feature_columns.json").write_text(json.dumps(self.columns, indent=2))
        (model_dir / "calibration.json").write_text(json.dumps({"widen": self.widen}, indent=2))

    def load(self, model_dir: Path) -> LightGBMQuantileForecaster:
        self.models = {
            q: lgb.Booster(model_file=str(model_dir / f"lgb_q{q}.txt"))
            for q in QUANTILES
            if (model_dir / f"lgb_q{q}.txt").exists()
        }
        cols_path = model_dir / "feature_columns.json"
        if cols_path.exists():
            self.columns = json.loads(cols_path.read_text())
        calib = model_dir / "calibration.json"
        if calib.exists():
            self.widen = float(json.loads(calib.read_text()).get("widen", 0.0))
        return self


def main(
    horizon: int = 12,
    num_boost_round: int = 300,
    seed: int = 7,
    calibrate: bool = True,
) -> Path:
    from src.evaluation.conformal import fit_conformal_offset

    features = load_feature_table()
    panel = make_supervised(features, range(1, horizon + 1))
    print(
        f"  training rows: {len(panel):,}  "
        f"({panel.groupby(KEYS).ngroups} series x {horizon} horizons)"
    )

    model = LightGBMQuantileForecaster().fit(panel, num_boost_round=num_boost_round, seed=seed)

    if calibrate:
        result = fit_conformal_offset(panel, num_boost_round=num_boost_round, seed=seed)
        model.widen = result.offset
        print(
            f"  conformal offset: {result.offset:.2f} units  "
            f"(calibration coverage {result.raw_coverage:.1%} → "
            f"{result.calibrated_coverage:.1%}, nominal {result.nominal:.0%})"
        )

    model.save(MODEL_DIR)

    forecast = model.forecast(features, horizon)
    out = FEATURE_DIR / registry.output_filename(model.name)
    forecast.to_parquet(out, index=False)
    print(f"  saved {len(forecast):,} forecast rows → {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quantile LightGBM forecasters.")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip conformal interval calibration (report raw quantile regression)",
    )
    args = parser.parse_args()
    main(
        horizon=args.horizon,
        num_boost_round=args.rounds,
        seed=args.seed,
        calibrate=not args.no_calibrate,
    )
