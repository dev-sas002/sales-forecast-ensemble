"""
Tests for the cached forecast store.

A cache is only an improvement if it is never stale. The interesting tests
here are the invalidation ones: a pipeline run must be visible to a running
API on the next request, with no restart and no TTL, or the API will serve
last week's forecast forever and nothing will say so.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from src.serving import store


@pytest.fixture(autouse=True)
def clean_cache():
    store.clear_cache()
    yield
    store.clear_cache()


def write_forecast(path, medians, sku="SKU-1", region="NE", suffix=""):
    dates = pd.date_range("2026-01-05", periods=len(medians), freq="W-MON")
    pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "date": dates,
            f"q0.1{suffix}": [m * 0.8 for m in medians],
            f"q0.5{suffix}": list(medians),
            f"q0.9{suffix}": [m * 1.2 for m in medians],
        }
    ).to_parquet(path, index=False)


class TestLoading:
    def test_the_ensemble_is_preferred_over_a_single_model(self, tmp_path):
        write_forecast(tmp_path / "forecast_ensemble.parquet", [10.0])
        write_forecast(tmp_path / "forecast_lgb.parquet", [99.0], suffix="_lgb")

        table = store.load_forecasts(tmp_path)
        assert table.source == "forecast_ensemble.parquet"
        assert table.slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 10.0

    def test_a_single_model_file_is_served_when_there_is_no_ensemble(self, tmp_path):
        write_forecast(tmp_path / "forecast_lgb.parquet", [42.0], suffix="_lgb")
        table = store.load_forecasts(tmp_path)
        # The suffixed columns the trainer writes are normalised to canonical.
        assert table.slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 42.0

    def test_no_file_at_all_names_the_command_that_makes_one(self, tmp_path):
        with pytest.raises(store.ForecastNotAvailable, match="make pipeline"):
            store.load_forecasts(tmp_path)

    def test_a_file_missing_a_quantile_is_a_typed_error(self, tmp_path):
        pd.DataFrame(
            {"sku": ["S"], "region": ["NE"], "date": ["2026-01-05"], "q0.5": [1.0]}
        ).to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

        with pytest.raises(store.ForecastSchemaError, match=r"q0\.1"):
            store.load_forecasts(tmp_path)


class TestIndexing:
    def test_rows_come_back_in_date_order_however_they_were_written(self, tmp_path):
        pd.DataFrame(
            {
                "sku": ["S1"] * 4,
                "region": ["NE"] * 4,
                "date": pd.to_datetime(["2026-02-23", "2026-02-02", "2026-02-16", "2026-02-09"]),
                "q0.1": [4.0, 1.0, 3.0, 2.0],
                "q0.5": [8.0, 2.0, 6.0, 4.0],
                "q0.9": [12.0, 3.0, 9.0, 6.0],
            }
        ).to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

        rows = store.load_forecasts(tmp_path).slice("S1", "NE", 2)
        assert rows["date"].tolist() == ["2026-02-02", "2026-02-09"]

    def test_an_unknown_series_returns_none_rather_than_an_empty_guess(self, tmp_path):
        write_forecast(tmp_path / "forecast_ensemble.parquet", [10.0])
        assert store.load_forecasts(tmp_path).slice("NOPE", "NE", 4) is None

    def test_series_do_not_bleed_into_each_other(self, tmp_path):
        a = pd.DataFrame(
            {
                "sku": ["A", "B"],
                "region": ["NE", "NE"],
                "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                "q0.1": [1.0, 10.0],
                "q0.5": [2.0, 20.0],
                "q0.9": [3.0, 30.0],
            }
        )
        a.to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)

        table = store.load_forecasts(tmp_path)
        assert table.slice("A", "NE", 10)["q0.5"].tolist() == [2.0]
        assert table.slice("B", "NE", 10)["q0.5"].tolist() == [20.0]
        assert table.n_rows == 2
        assert table.series == [("A", "NE"), ("B", "NE")]

    def test_the_horizon_truncates_from_the_front(self, tmp_path):
        write_forecast(tmp_path / "forecast_ensemble.parquet", [1.0, 2.0, 3.0, 4.0])
        rows = store.load_forecasts(tmp_path).slice("SKU-1", "NE", 2)
        assert rows["q0.5"].tolist() == [1.0, 2.0]


class TestCaching:
    def test_a_second_call_returns_the_same_object(self, tmp_path):
        write_forecast(tmp_path / "forecast_ensemble.parquet", [10.0])
        first = store.load_forecasts(tmp_path)
        second = store.load_forecasts(tmp_path)
        assert first is second, "the table was re-read when nothing had changed"

    def test_rewriting_the_file_invalidates_the_cache(self, tmp_path):
        path = tmp_path / "forecast_ensemble.parquet"
        write_forecast(path, [10.0])
        assert store.load_forecasts(tmp_path).slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 10.0

        # A pipeline run must be picked up without restarting the API. The
        # stat-based key means no TTL and no stale window.
        time.sleep(0.01)
        write_forecast(path, [999.0])
        os.utime(path, None)

        assert store.load_forecasts(tmp_path).slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 999.0

    def test_pointing_at_a_different_directory_does_not_serve_the_old_one(self, tmp_path):
        # The tests move FEATURE_DIR under the API; a path-blind cache would
        # serve another directory's forecast.
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        write_forecast(a / "forecast_ensemble.parquet", [1.0])
        write_forecast(b / "forecast_ensemble.parquet", [2.0])

        assert store.load_forecasts(a).slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 1.0
        assert store.load_forecasts(b).slice("SKU-1", "NE", 1)["q0.5"].iloc[0] == 2.0

    def test_clearing_the_cache_forces_a_reload(self, tmp_path):
        write_forecast(tmp_path / "forecast_ensemble.parquet", [10.0])
        first = store.load_forecasts(tmp_path)
        store.clear_cache()
        assert store.load_forecasts(tmp_path) is not first


class TestColumnNormalisation:
    def test_suffixed_columns_are_renamed_to_canonical(self):
        df = pd.DataFrame({"q0.1_lgb": [1], "q0.5_lgb": [2], "q0.9_lgb": [3]})
        out = store.normalize_forecast_columns(df, ["q0.1", "q0.5", "q0.9"])
        assert out["q0.1"].iloc[0] == 1

    def test_canonical_columns_are_left_alone(self):
        df = pd.DataFrame({"q0.1": [1], "q0.5": [2], "q0.9": [3]})
        out = store.normalize_forecast_columns(df, ["q0.1", "q0.5", "q0.9"])
        assert out is df
