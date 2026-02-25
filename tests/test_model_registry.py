"""
Tests for the model registry and the orphan-artifact guard.

The regression being pinned is specific and was shipped: a committed
`forecast_deepar.parquet` from a dead toy dataset was blended into every local
ensemble run, even though the DeepAR trainer has only ever been a stub that
raises. Deleting the file fixes today; refusing to blend a forecast no
registered model produces fixes the class.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ensemble.ensemble_and_reconcile import (
    OrphanForecastError,
    collect_sources,
)
from src.models import registry


def forecast_frame(model: str | None = None, sku="SKU-1", region="NE"):
    suffix = f"_{model}" if model else ""
    return pd.DataFrame(
        {
            "sku": [sku, sku],
            "region": [region, region],
            "date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
            f"q0.1{suffix}": [8.0, 9.0],
            f"q0.5{suffix}": [10.0, 11.0],
            f"q0.9{suffix}": [12.0, 13.0],
        }
    )


class TestTheRegistry:
    def test_lightgbm_is_registered(self):
        # Importing the trainer registers it; if that stops happening the
        # ensemble silently has nothing to blend.
        import src.models.train_lgb_quantile  # noqa: F401

        assert "lgb" in registry.registered()

    def test_the_stub_models_are_not_registered(self):
        # train_deepar_gluonts and train_pymc_hierarchical raise
        # NotImplementedError. Registration is a claim that a model works.
        assert "deepar" not in registry.registered()
        assert "pymc" not in registry.registered()

    def test_output_filename_follows_the_convention(self):
        assert registry.output_filename("lgb") == "forecast_lgb.parquet"

    def test_registering_a_duplicate_name_is_an_error(self):
        class Clash:
            name = "lgb"

        with pytest.raises(ValueError, match="already registered"):
            registry.register(Clash)

    def test_get_names_the_alternatives_when_a_model_is_unknown(self):
        with pytest.raises(KeyError, match="registered:"):
            registry.get("nonexistent")


class TestScanningForecastFiles:
    def test_a_registered_model_file_is_recognised(self, tmp_path):
        forecast_frame("lgb").to_parquet(tmp_path / "forecast_lgb.parquet")
        produced, orphans = registry.scan_forecasts(tmp_path)
        assert set(produced) == {"lgb"}
        assert orphans == []

    def test_a_file_with_no_producer_is_an_orphan(self, tmp_path):
        forecast_frame("deepar").to_parquet(tmp_path / "forecast_deepar.parquet")
        produced, orphans = registry.scan_forecasts(tmp_path)
        assert produced == {}
        assert [p.name for p in orphans] == ["forecast_deepar.parquet"]

    def test_the_ensembles_own_output_is_not_an_orphan(self, tmp_path):
        forecast_frame().to_parquet(tmp_path / "forecast_ensemble.parquet")
        produced, orphans = registry.scan_forecasts(tmp_path)
        assert produced == {} and orphans == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        produced, orphans = registry.scan_forecasts(tmp_path / "nope")
        assert produced == {} and orphans == []


class TestTheEnsembleIgnoresOrphans:
    def test_an_orphan_is_not_blended(self, tmp_path):
        # This is the shipped regression, reproduced: a real model's forecast
        # beside an artifact nothing wrote. The blend must be the real one.
        forecast_frame("lgb").to_parquet(tmp_path / "forecast_lgb.parquet")
        stale = forecast_frame("deepar")
        stale[["q0.1_deepar", "q0.5_deepar", "q0.9_deepar"]] *= 100
        stale.to_parquet(tmp_path / "forecast_deepar.parquet")

        frames, orphans = collect_sources(tmp_path)

        assert set(frames) == {"lgb"}
        assert [p.name for p in orphans] == ["forecast_deepar.parquet"]

    def test_an_orphan_can_be_blended_on_request(self, tmp_path):
        # There is a legitimate case — a forecast produced outside this repo —
        # but it has to be asked for.
        forecast_frame("lgb").to_parquet(tmp_path / "forecast_lgb.parquet")
        forecast_frame("deepar").to_parquet(tmp_path / "forecast_deepar.parquet")

        frames, _ = collect_sources(tmp_path, include_unregistered=True)
        assert set(frames) == {"lgb", "deepar"}

    def test_orphans_alone_are_refused_rather_than_silently_blended(self, tmp_path):
        # The exact shipped state: the only forecast on disk has no producer.
        # Building an "ensemble" out of it is what used to happen.
        forecast_frame("deepar").to_parquet(tmp_path / "forecast_deepar.parquet")

        with pytest.raises(OrphanForecastError, match="no registered forecaster"):
            collect_sources(tmp_path)

    def test_no_files_at_all_is_a_different_error(self, tmp_path):
        frames, orphans = collect_sources(tmp_path)
        assert frames == {} and orphans == []
