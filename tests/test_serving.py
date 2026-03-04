"""Tests for serving API: /health, /forecast with TestClient and mocked parquet."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import _normalize_forecast_columns, app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_forecast_404_when_no_file(client: TestClient) -> None:
    with patch("src.serving.app.FEATURE_DIR", Path("/nonexistent_empty_dir")):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4})
    assert r.status_code == 404
    assert "No forecast" in r.json()["detail"]


def test_forecast_200_with_mock_parquet(client: TestClient, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sku": ["S1", "S1"],
            "region": ["NE", "NE"],
            "date": ["2025-02-01", "2025-02-08"],
            "q0.1": [1.0, 2.0],
            "q0.5": [2.0, 3.0],
            "q0.9": [3.0, 4.0],
        }
    )
    df.to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert data[0]["sku"] == "S1" and data[0]["region"] == "NE"
    assert "q0.1" in data[0] and "q0.5" in data[0] and "q0.9" in data[0]


def test_forecast_404_sku_region_not_in_file(client: TestClient, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sku": ["S2"],
            "region": ["SE"],
            "date": ["2025-02-01"],
            "q0.1": [1.0],
            "q0.5": [2.0],
            "q0.9": [3.0],
        }
    )
    df.to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4})
    assert r.status_code == 404
    assert "No forecast found" in r.json()["detail"]


def test_normalize_forecast_columns_lgb_suffix() -> None:
    df = pd.DataFrame({"q0.1_lgb": [1], "q0.5_lgb": [2], "q0.9_lgb": [3]})
    out = _normalize_forecast_columns(df)
    assert "q0.1" in out.columns and out["q0.1"].iloc[0] == 1


def test_a_zero_or_negative_horizon_is_rejected(client: TestClient) -> None:
    # Unvalidated, horizon_weeks=-2 reached `df.head(-2)`, which silently drops
    # the last two rows and returns a 200 with a forecast.
    for bad in (0, -2):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": bad})
        assert r.status_code == 422, f"horizon_weeks={bad} was accepted"


def test_an_absurd_horizon_is_rejected(client: TestClient) -> None:
    r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 10_000})
    assert r.status_code == 422


def test_horizon_truncates_to_the_earliest_weeks(client: TestClient, tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "sku": ["S1"] * 4,
            "region": ["NE"] * 4,
            # Deliberately out of order on disk.
            "date": ["2025-02-22", "2025-02-01", "2025-02-15", "2025-02-08"],
            "q0.1": [4.0, 1.0, 3.0, 2.0],
            "q0.5": [8.0, 2.0, 6.0, 4.0],
            "q0.9": [12.0, 3.0, 9.0, 6.0],
        }
    ).to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 2})

    assert r.status_code == 200
    data = r.json()
    assert [row["date"] for row in data] == ["2025-02-01", "2025-02-08"]
    assert [row["q0.5"] for row in data] == [2.0, 4.0]


def test_the_lgb_forecast_is_served_when_no_ensemble_exists(
    client: TestClient, tmp_path: Path
) -> None:
    # The suffixed column names the trainer writes must be normalised, or the
    # API 500s on a pipeline that ran the trainer but not the ensemble.
    pd.DataFrame(
        {
            "sku": ["S1"],
            "region": ["NE"],
            "date": ["2025-03-03"],
            "q0.1_lgb": [1.0],
            "q0.5_lgb": [2.0],
            "q0.9_lgb": [3.0],
        }
    ).to_parquet(tmp_path / "forecast_lgb.parquet", index=False)

    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4})

    assert r.status_code == 200
    assert r.json() == [
        {"sku": "S1", "region": "NE", "date": "2025-03-03", "q0.1": 1.0, "q0.5": 2.0, "q0.9": 3.0}
    ]


def test_a_forecast_file_missing_a_quantile_is_a_clear_error(
    client: TestClient, tmp_path: Path
) -> None:
    pd.DataFrame(
        {"sku": ["S1"], "region": ["NE"], "date": ["2025-03-03"], "q0.5": [2.0]}
    ).to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4})

    # Previously a bare KeyError escaped as an unhandled 500 with a pandas
    # traceback and no hint about which file was wrong.
    assert r.status_code == 500
    assert "missing quantile columns" in r.json()["detail"]


def test_forecast_columns_follow_the_configured_quantiles() -> None:
    from src.config import quantile_cols
    from src.serving.app import FORECAST_COLS

    assert quantile_cols() == FORECAST_COLS
