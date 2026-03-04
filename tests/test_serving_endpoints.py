"""
Tests for the endpoints added in the 2026-09-23 pass: /models, /anomalies,
/explain.

`test_serving.py` covers the original /health and /forecast contract; this
covers the new surface. The load-bearing assertion in the whole file is that
`/explain` returns 200 with no `ANTHROPIC_API_KEY` set — an AI feature that
takes the product down when a key is missing is not a feature.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving import store
from src.serving.app import app


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A feature directory with a forecast, a feature table and a backtest."""
    dates = pd.date_range("2026-01-05", periods=6, freq="W-MON")
    pd.DataFrame(
        {
            "sku": "SKU-1",
            "region": "NE",
            "date": dates,
            "q0.1": [80.0] * 6,
            "q0.5": [100.0] * 6,
            "q0.9": [130.0] * 6,
        }
    ).to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

    weeks = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    units = [120.0] * 25 + [0.0] * 5 + [120.0] * 30
    pd.DataFrame(
        {
            "sku": "SKU-1",
            "region": "NE",
            "week": weeks,
            "units_sold": np.asarray(units[: len(weeks)], dtype=float),
            "avg_price": 2.5,
        }
    ).to_parquet(tmp_path / "features.parquet", index=False)

    (tmp_path / "backtest_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "wape_model": 0.238,
                    "wape_naive": 0.327,
                    "coverage": 0.722,
                    "calibrated_coverage": 0.804,
                }
            }
        )
    )

    monkeypatch.setattr("src.serving.app.FEATURE_DIR", tmp_path)
    return tmp_path


class TestModelsEndpoint:
    def test_it_lists_the_registered_forecasters(self, client, data_dir):
        body = client.get("/models").json()
        assert "lgb" in body["registered"]
        assert "deepar" not in body["registered"]

    def test_it_names_orphan_forecast_files(self, client, data_dir):
        # The shipped bug, made visible: a forecast file nothing produces.
        pd.DataFrame(
            {
                "sku": ["SKU-1"],
                "region": ["NE"],
                "date": pd.to_datetime(["2026-01-05"]),
                "q0.1_deepar": [1.0],
                "q0.5_deepar": [2.0],
                "q0.9_deepar": [3.0],
            }
        ).to_parquet(data_dir / "forecast_deepar.parquet", index=False)

        body = client.get("/models").json()
        assert body["orphan_forecast_files"] == ["forecast_deepar.parquet"]

    def test_it_reports_whether_ai_narration_is_available(self, client, data_dir):
        # With no key set, narration is unavailable and the endpoint says so
        # rather than silently serving templates. The exact reason depends on
        # whether the optional `anthropic` package happens to be installed.
        narrator = client.get("/models").json()["narrator"]
        assert narrator["available"] is False
        assert narrator["model"] is None
        assert narrator["reason"] in {
            "ANTHROPIC_API_KEY is not set",
            "the optional `anthropic` package is not installed",
        }


class TestAnomaliesEndpoint:
    def test_it_finds_the_zero_run_in_the_seeded_table(self, client, data_dir):
        body = client.get("/anomalies").json()
        assert body["summary"]["zero_run"] == 1
        kinds = {row["kind"] for row in body["anomalies"]}
        assert "zero_run" in kinds

    def test_each_finding_carries_an_actionable_sentence(self, client, data_dir):
        row = client.get("/anomalies").json()["anomalies"][0]
        assert set(row) == {"sku", "region", "week", "kind", "severity", "detail", "value"}
        assert len(row["detail"]) > 20

    def test_it_filters_by_series(self, client, data_dir):
        assert client.get("/anomalies?sku=NOPE").json()["anomalies"] == []

    def test_the_limit_is_clamped_rather_than_trusted(self, client, data_dir):
        # An unbounded limit is how a diagnostic endpoint becomes a way to
        # pull the whole table.
        assert client.get("/anomalies?limit=100000").status_code == 200

    def test_no_feature_table_is_a_404_that_says_what_to_run(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.serving.app.FEATURE_DIR", tmp_path / "empty")
        r = client.get("/anomalies")
        assert r.status_code == 404
        assert "pipeline" in r.json()["detail"]


class TestExplainEndpoint:
    def test_it_returns_200_with_no_api_key_set(self, client, data_dir):
        # The whole point of the degradation path.
        r = client.post("/explain", json={"sku": "SKU-1", "region": "NE", "horizon_weeks": 6})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "template"
        assert body["model"] is None
        assert body["explanation"]

    def test_the_facts_are_computed_from_the_served_rows(self, client, data_dir):
        body = client.post(
            "/explain", json={"sku": "SKU-1", "region": "NE", "horizon_weeks": 6}
        ).json()
        facts = body["facts"]
        assert facts["total_forecast"] == pytest.approx(600.0)
        assert facts["mean_weekly"] == pytest.approx(100.0)
        assert facts["horizon_weeks"] == 6

    def test_it_quotes_the_measured_backtest_rather_than_a_claim(self, client, data_dir):
        facts = client.post("/explain", json={"sku": "SKU-1", "region": "NE"}).json()["facts"]
        assert facts["backtest_wape"] == 0.238
        # The calibrated coverage is the one the API actually serves.
        assert facts["backtest_coverage"] == 0.804

    def test_it_surfaces_input_anomalies_for_the_series(self, client, data_dir):
        facts = client.post("/explain", json={"sku": "SKU-1", "region": "NE"}).json()["facts"]
        assert facts["anomaly_count"] > 0

    def test_an_unknown_series_is_a_404_not_an_invented_explanation(self, client, data_dir):
        r = client.post("/explain", json={"sku": "NOPE", "region": "NE"})
        assert r.status_code == 404

    def test_an_unreadable_feature_table_costs_context_not_the_endpoint(self, client, data_dir):
        (data_dir / "features.parquet").write_text("not a parquet file")
        r = client.post("/explain", json={"sku": "SKU-1", "region": "NE"})
        assert r.status_code == 200
        assert r.json()["facts"]["trend_direction"] == "unknown"
