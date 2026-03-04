"""Tests for src.ingest: read_csv, write_parquet, main validation and error handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest.ingest import IngestError, read_csv


def test_read_csv_not_found() -> None:
    with pytest.raises(IngestError, match="File not found"):
        read_csv("/nonexistent/path.csv")


def test_write_parquet_and_read_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ingest_mod.write_parquet(df, "test_out")
    assert (tmp_path / "test_out.parquet").exists()
    back = pd.read_parquet(tmp_path / "test_out.parquet")
    pd.testing.assert_frame_equal(back, df)


def test_main_missing_columns_credit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    # Credit missing 'quantity'
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
        }
    ).to_csv(credit_csv, index=False)
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
            "household_size": [3],
        }
    ).to_csv(panel_csv, index=False)
    with pytest.raises(ValueError, match="credit_csv missing required columns"):
        ingest_mod.main(str(credit_csv), str(panel_csv))


def test_main_missing_columns_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
            "quantity": [5],
        }
    ).to_csv(credit_csv, index=False)
    # Panel missing household_size
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
        }
    ).to_csv(panel_csv, index=False)
    with pytest.raises(ValueError, match="panel_csv missing required columns"):
        ingest_mod.main(str(credit_csv), str(panel_csv))


def test_main_success_writes_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
            "quantity": [5],
        }
    ).to_csv(credit_csv, index=False)
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
            "household_size": [3],
        }
    ).to_csv(panel_csv, index=False)
    ingest_mod.main(str(credit_csv), str(panel_csv))
    assert (tmp_path / "credit_txn.parquet").exists()
    assert (tmp_path / "panel.parquet").exists()
