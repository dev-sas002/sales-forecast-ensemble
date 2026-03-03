"""
Tests for the dashboard's data layer.

These exist because none of this was testable before: loading, derivation and
layout lived in one function, so a linkage rate could only be checked by
opening a browser. The first test is the one that keeps it that way — if
`src.dashboard.data` ever imports Streamlit, the separation has collapsed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.dashboard.data import (
    backtest_frames,
    build_forecast_view,
    feature_glossary,
    linkage_rate,
    load_dashboard_data,
    pipeline_stages,
    quality_summary,
    series_options,
)


@pytest.fixture
def dirs(tmp_path):
    landing = tmp_path / "landing"
    features = tmp_path / "features"
    landing.mkdir()
    features.mkdir()
    return landing, features


def write_features(path, sku="SKU-1", region="NE", n=60, units=None):
    weeks = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    values = units if units is not None else [100.0] * n
    pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "week": weeks,
            "units_sold": np.asarray(values, dtype=float),
            "avg_price": 2.5,
            "lag_1_units": np.asarray(values, dtype=float),
            "roll_4w_mean": np.asarray(values, dtype=float),
            "weekofyear": weeks.isocalendar().week.astype(int),
        }
    ).to_parquet(path, index=False)


def write_forecast(path, sku="SKU-1", region="NE", n=8, suffix=""):
    dates = pd.date_range("2025-03-03", periods=n, freq="W-MON")
    pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "date": dates,
            f"q0.1{suffix}": [80.0] * n,
            f"q0.5{suffix}": [110.0] * n,
            f"q0.9{suffix}": [140.0] * n,
        }
    ).to_parquet(path, index=False)


class TestTheSeparationHolds:
    def test_the_data_layer_does_not_import_streamlit(self):
        # The property that makes everything else here testable. A view import
        # creeping into the data module is how the two get welded back
        # together, and it happens one convenience call at a time.
        #
        # Parsed rather than grepped: the module's own docstring talks about
        # Streamlit, and a substring check would either fail on the prose or
        # have to be loosened until it stopped checking anything.
        import ast
        import pathlib

        import src.dashboard.data as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert "streamlit" not in imported, (
            "the dashboard's data layer imports Streamlit; the point of the "
            "split is that this module is testable without a browser"
        )


class TestLoading:
    def test_a_half_run_pipeline_loads_what_exists(self, dirs):
        landing, features = dirs
        pd.DataFrame({"txn_id": [1], "customer_id": ["C1"], "quantity": [2]}).to_parquet(
            landing / "credit_txn.parquet", index=False
        )

        data = load_dashboard_data(landing, features)
        assert data.credit is not None
        assert data.features is None
        assert data.forecast is None
        assert data.has_any

    def test_an_empty_directory_is_not_an_error(self, dirs):
        data = load_dashboard_data(*dirs)
        assert not data.has_any
        assert data.errors == []

    def test_a_corrupt_file_is_recorded_rather_than_raised(self, dirs):
        landing, features = dirs
        (landing / "credit_txn.parquet").write_text("not parquet")
        data = load_dashboard_data(landing, features)
        assert data.credit is None
        assert any("credit_txn" in e for e in data.errors)

    def test_the_ensemble_is_preferred_over_a_single_model(self, dirs):
        _, features = dirs
        write_forecast(features / "forecast_ensemble.parquet")
        write_forecast(features / "forecast_lgb.parquet", suffix="_lgb")
        data = load_dashboard_data(*dirs)
        assert "q0.5" in data.forecast.columns

    def test_a_single_model_file_has_its_suffix_stripped(self, dirs):
        _, features = dirs
        write_forecast(features / "forecast_lgb.parquet", suffix="_lgb")
        data = load_dashboard_data(*dirs)
        assert "q0.5" in data.forecast.columns

    def test_an_orphan_file_is_reported_and_never_shown_as_a_forecast(self, dirs):
        # The shipped bug: a forecast file no code writes, presented as if it
        # were a forecast the system had made.
        _, features = dirs
        write_forecast(features / "forecast_deepar.parquet", suffix="_deepar")
        data = load_dashboard_data(*dirs)
        assert data.orphan_forecasts == ["forecast_deepar.parquet"]
        assert data.forecast is None

    def test_the_backtest_report_is_parsed_when_present(self, dirs):
        _, features = dirs
        (features / "backtest_report.json").write_text(
            json.dumps({"summary": {"n_folds": 4}, "folds": [], "per_horizon": []})
        )
        data = load_dashboard_data(*dirs)
        assert data.backtest["summary"]["n_folds"] == 4


class TestPipelineStages:
    def test_every_stage_is_listed_even_when_nothing_has_run(self, dirs):
        stages = pipeline_stages(load_dashboard_data(*dirs))
        assert [s.name for s in stages] == ["Ingest", "Link", "Features", "Forecast", "Backtest"]
        assert not any(s.done for s in stages)

    def test_a_done_stage_carries_a_count(self, dirs):
        _, features = dirs
        write_features(features / "features.parquet")
        stages = {s.name: s for s in pipeline_stages(load_dashboard_data(*dirs))}
        assert stages["Features"].done
        assert "60" in stages["Features"].detail


class TestDerivations:
    def test_linkage_rate_is_linked_over_credit(self, dirs):
        landing, _ = dirs
        pd.DataFrame({"txn_id": range(10), "customer_id": ["C"] * 10}).to_parquet(
            landing / "credit_txn.parquet", index=False
        )
        pd.DataFrame({"txn_id": range(4), "cust_hash": ["h"] * 4}).to_parquet(
            landing / "linked_panel_credit.parquet", index=False
        )
        assert linkage_rate(load_dashboard_data(*dirs)) == pytest.approx(0.4)

    def test_linkage_rate_is_none_rather_than_a_divide_by_zero(self, dirs):
        assert linkage_rate(load_dashboard_data(*dirs)) is None

    def test_the_glossary_comes_from_the_feature_registry(self, dirs):
        _, feature_dir = dirs
        write_features(feature_dir / "features.parquet")
        data = load_dashboard_data(*dirs)
        glossary = feature_glossary(data.features)
        assert set(glossary["feature"]) >= {"lag_1_units", "roll_4w_mean"}
        # Descriptions are prose, not the column name repeated back.
        assert all(len(d) > 10 for d in glossary["description"])

    def test_series_options_never_offer_a_pair_with_no_forecast(self, dirs):
        _, features = dirs
        write_forecast(features / "forecast_ensemble.parquet", sku="A", region="NE")
        options = series_options(load_dashboard_data(*dirs))
        assert options == {"A": ["NE"]}


class TestForecastView:
    def test_it_joins_history_to_forecast_for_one_series(self, dirs):
        _, features = dirs
        write_features(features / "features.parquet")
        write_forecast(features / "forecast_ensemble.parquet")

        view = build_forecast_view(load_dashboard_data(*dirs), "SKU-1", "NE")
        assert view is not None
        assert len(view.forecast) == 8
        assert not view.history.empty
        assert view.total == pytest.approx(880.0)
        assert view.mean_weekly == pytest.approx(110.0)
        assert view.recent_mean == pytest.approx(100.0)
        assert view.change_vs_recent == pytest.approx(0.1)

    def test_an_unknown_series_returns_none(self, dirs):
        _, features = dirs
        write_forecast(features / "forecast_ensemble.parquet")
        assert build_forecast_view(load_dashboard_data(*dirs), "NOPE", "NE") is None

    def test_a_forecast_without_history_still_renders(self, dirs):
        _, features = dirs
        write_forecast(features / "forecast_ensemble.parquet")
        view = build_forecast_view(load_dashboard_data(*dirs), "SKU-1", "NE")
        assert view.history.empty
        assert view.recent_mean is None
        assert view.change_vs_recent is None

    def test_anomalies_are_attached_to_the_series(self, dirs):
        _, features = dirs
        units = [120.0] * 25 + [0.0] * 5 + [120.0] * 30
        write_features(features / "features.parquet", n=60, units=units)
        write_forecast(features / "forecast_ensemble.parquet")

        view = build_forecast_view(load_dashboard_data(*dirs), "SKU-1", "NE")
        assert not view.anomalies.empty


class TestQualityAndBacktestFrames:
    def test_quality_summary_handles_a_missing_table(self):
        counts, found = quality_summary(None)
        assert counts["total"] == 0
        assert found.empty

    def test_backtest_frames_add_the_improvement_column(self):
        report = {
            "folds": [{"cutoff": "2025-01-06", "wape_model": 0.2, "wape_naive": 0.4}],
            "per_horizon": [{"horizon": 1, "wape": 0.2, "coverage": 0.8}],
        }
        horizons, folds = backtest_frames(report)
        assert folds["improvement"].iloc[0] == pytest.approx(0.5)
        assert len(horizons) == 1

    def test_backtest_frames_survive_an_empty_report(self):
        horizons, folds = backtest_frames({})
        assert horizons.empty and folds.empty
