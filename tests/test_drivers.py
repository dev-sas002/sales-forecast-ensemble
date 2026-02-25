"""
Driver attribution for one series, from a booster saved the way the pipeline
saves it, and its route into `/explain`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.explain.drivers import median_booster, series_drivers
from src.features.build_features import add_history_features
from src.models.train_lgb_quantile import (
    LightGBMQuantileForecaster,
    feature_matrix,
    future_panel,
    make_supervised,
)
from src.serving import store
from src.serving.app import app


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    weeks = pd.date_range("2023-01-02", periods=80, freq="W-MON")
    for sku, base in (("SKU-A", 90.0), ("SKU-B", 30.0)):
        for i, week in enumerate(weeks):
            units = base * (1 + 0.3 * np.sin(2 * np.pi * i / 52)) + rng.normal(0, 3)
            rows.append(
                {
                    "sku": sku,
                    "region": "NE",
                    "week": week,
                    "units_sold": float(max(0.0, round(units))),
                    "sales_dollars": units * 3,
                    "n_transactions": units / 2,
                    "avg_price": 3.0,
                }
            )
    return add_history_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def model_dir(features, tmp_path_factory):
    panel = make_supervised(features, range(1, 5))
    out = tmp_path_factory.mktemp("models")
    LightGBMQuantileForecaster().fit(panel, num_boost_round=30, seed=1).save(out)
    return out


class TestSeriesDrivers:
    def test_no_model_means_no_drivers(self, features, tmp_path):
        assert series_drivers(features, "SKU-A", "NE", 4, tmp_path) == []

    def test_an_unknown_series_has_no_drivers(self, features, model_dir):
        assert series_drivers(features, "SKU-Z", "NE", 4, model_dir) == []

    def test_a_table_without_the_model_inputs_has_no_drivers(self, features, model_dir):
        thin = features[["sku", "region", "week", "units_sold", "avg_price"]]
        assert series_drivers(thin, "SKU-A", "NE", 4, model_dir) == []

    def test_drivers_are_ranked_by_size_and_named_after_model_inputs(self, features, model_dir):
        drivers = series_drivers(features, "SKU-A", "NE", 4, model_dir, top_n=4)
        assert len(drivers) == 4
        sizes = [abs(v) for _, v in drivers]
        assert sizes == sorted(sizes, reverse=True)
        _, inputs = feature_matrix(future_panel(features, 1))
        assert {name for name, _ in drivers} <= set(inputs)

    def test_contributions_add_up_to_the_mean_prediction(self, features, model_dir):
        # SHAP additivity, computed on one series alone: the categorical
        # columns must be mapped onto the training categories, or the sum
        # would not match a prediction made on the full panel.
        booster = median_booster(model_dir)
        everything = series_drivers(features, "SKU-B", "NE", 4, model_dir, top_n=100)

        future = future_panel(features, 4)
        X, _ = feature_matrix(future)
        mask = (future["sku"] == "SKU-B").to_numpy()
        contrib = booster.predict(X, pred_contrib=True)[mask]
        predicted = booster.predict(X)[mask].mean()

        base = contrib[:, -1].mean()
        assert base + sum(v for _, v in everything) == pytest.approx(predicted, rel=1e-6)


class TestExplainEndpoint:
    def test_explain_reports_the_drivers_of_the_median(
        self, features, model_dir, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        features.to_parquet(tmp_path / "features.parquet", index=False)
        forecast = LightGBMQuantileForecaster().load(model_dir).forecast(features, 4)
        forecast.to_parquet(tmp_path / "forecast_lgb.parquet", index=False)
        monkeypatch.setattr("src.serving.app.FEATURE_DIR", tmp_path)
        monkeypatch.setattr("src.serving.app.MODEL_DIR", model_dir)
        store.clear_cache()

        response = TestClient(app).post(
            "/explain", json={"sku": "SKU-A", "region": "NE", "horizon_weeks": 4}
        )
        store.clear_cache()

        assert response.status_code == 200
        body = response.json()
        drivers = body["facts"]["drivers"]
        assert drivers, "the explanation should name what drove the forecast"
        assert all(d["description"] for d in drivers)
        assert "largest influences" in body["explanation"]
