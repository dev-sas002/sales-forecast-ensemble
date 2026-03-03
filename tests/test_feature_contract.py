"""
Tests for the feature-table schema contract.

The second of the two shipped data problems: `features.parquet` was committed
carrying `rolling_4w_mean` — the *leaking* trailing window from before the
2026-09-22 fix — while the trainer selects its inputs by the `roll_` prefix.
Training against that file would have fed the model the leaking column, and
nothing would have raised.

Deleting the file fixes today. The sidecar fixes the class: a feature table
now records the registry that produced it, and `load_feature_table` refuses
one it does not recognise rather than quietly training on it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf


@pytest.fixture
def feature_dir(tmp_path, monkeypatch):
    """A DATA_DIR-shaped tree with a linked parquet ready to build from."""
    landing = tmp_path / "landing"
    features = tmp_path / "features"
    landing.mkdir()
    features.mkdir()

    rng = np.random.default_rng(5)
    dates = pd.date_range("2025-01-06", periods=60, freq="W-MON")
    rows = []
    for i, day in enumerate(dates):
        for _ in range(3):
            rows.append(
                {
                    "txn_id": len(rows) + 1,
                    "txn_date": day,
                    "sku": "SKU-1",
                    "region": "NE",
                    "quantity": int(rng.integers(1, 9)),
                    "amount": 10.0,
                    "price": 2.5 + 0.01 * i,
                }
            )
    pd.DataFrame(rows).to_parquet(landing / "linked_panel_credit.parquet", index=False)

    monkeypatch.setattr(bf, "LANDING_DIR", landing)
    monkeypatch.setattr(bf, "FEATURE_DIR", features)
    return features


class TestTheSidecarIsWritten:
    def test_build_writes_a_schema_sidecar(self, feature_dir):
        out = bf.build()
        sidecar = feature_dir / "features.parquet.schema.json"
        assert sidecar.exists()

        meta = json.loads(sidecar.read_text())
        assert meta["schema_version"] == bf.FEATURE_SCHEMA_VERSION
        assert meta["features"] == bf.history_feature_names()
        assert meta["rows"] == len(pd.read_parquet(out))

    def test_a_freshly_built_table_loads(self, feature_dir):
        bf.build()
        loaded = bf.load_feature_table(feature_dir / "features.parquet")
        assert not loaded.empty
        for name in bf.history_feature_names():
            assert name in loaded.columns


class TestStaleTablesAreRefused:
    def test_a_table_with_no_sidecar_is_refused(self, feature_dir):
        # Exactly the shipped file: a parquet from an older pipeline, with no
        # record of which registry wrote it.
        path = feature_dir / "features.parquet"
        pd.DataFrame(
            {
                "sku": ["SKU123"],
                "region": ["NE"],
                "week": pd.to_datetime(["2025-01-06"]),
                "units_sold": [4.0],
                "lag_1_units": [8.0],
                "rolling_4w_mean": [6.0],  # the leaking column
            }
        ).to_parquet(path, index=False)

        with pytest.raises(ValueError, match="predates feature-schema checking"):
            bf.load_feature_table(path)

    def test_a_table_from_a_different_registry_is_refused_and_says_why(self, feature_dir):
        bf.build()
        path = feature_dir / "features.parquet"
        sidecar = feature_dir / "features.parquet.schema.json"
        meta = json.loads(sidecar.read_text())
        meta["features"] = [*meta["features"], "rolling_4w_mean"]
        sidecar.write_text(json.dumps(meta))

        with pytest.raises(ValueError) as excinfo:
            bf.load_feature_table(path)
        # The message must name the offending column, or the reader has to
        # diff two parquet files to find out what is wrong.
        assert "rolling_4w_mean" in str(excinfo.value)
        assert "Regenerate" in str(excinfo.value)

    def test_an_older_schema_version_is_refused(self, feature_dir):
        bf.build()
        sidecar = feature_dir / "features.parquet.schema.json"
        meta = json.loads(sidecar.read_text())
        meta["schema_version"] = bf.FEATURE_SCHEMA_VERSION - 1
        sidecar.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="different feature registry"):
            bf.load_feature_table(feature_dir / "features.parquet")

    def test_a_missing_table_says_how_to_make_one(self, feature_dir):
        with pytest.raises(FileNotFoundError, match="run_pipeline"):
            bf.load_feature_table(feature_dir / "features.parquet")
