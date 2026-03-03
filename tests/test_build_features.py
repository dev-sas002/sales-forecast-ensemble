"""Tests for build_features: build() with minimal linked DataFrame, required columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_build_requires_linked_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.features.build_features as bf

    monkeypatch.setattr(bf, "LANDING_DIR", tmp_path)
    monkeypatch.setattr(bf, "FEATURE_DIR", tmp_path / "feat")
    (tmp_path / "feat").mkdir(parents=True, exist_ok=True)
    # Linked missing 'price'
    linked = pd.DataFrame(
        {
            "txn_date": ["2025-01-06"],
            "sku": ["S1"],
            "region": ["NE"],
            "quantity": [5],
            "amount": [12.5],
            "txn_id": [1],
        }
    )
    linked.to_parquet(tmp_path / "linked_panel_credit.parquet", index=False)
    with pytest.raises(ValueError, match="linked_panel_credit missing required columns"):
        bf.build()


def test_build_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.features.build_features as bf

    monkeypatch.setattr(bf, "LANDING_DIR", tmp_path)
    monkeypatch.setattr(bf, "FEATURE_DIR", tmp_path / "feat")
    (tmp_path / "feat").mkdir(parents=True, exist_ok=True)
    linked = pd.DataFrame(
        {
            "txn_date": ["2025-01-06", "2025-01-13"],
            "sku": ["S1", "S1"],
            "region": ["NE", "NE"],
            "quantity": [5, 3],
            "amount": [12.5, 7.5],
            "txn_id": [1, 2],
            "price": [2.5, 2.5],
        }
    )
    linked.to_parquet(tmp_path / "linked_panel_credit.parquet", index=False)
    out_path = bf.build()
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert "units_sold" in df.columns and "lag_1_units" in df.columns


# ---------------------------------------------------------------------------
# Weekly aggregation. These run on a handful of hand-written transactions, so
# the expected numbers can be checked by eye rather than by rerunning the code.
# ---------------------------------------------------------------------------


def _txns(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_aggregate_weekly_collapses_a_week_to_one_row() -> None:
    from src.features.build_features import aggregate_weekly

    # Tuesday and Thursday of the same week, plus one the following Monday.
    weekly = aggregate_weekly(
        _txns(
            [
                {
                    "txn_date": "2025-01-07",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 5,
                    "amount": 10.0,
                    "txn_id": 1,
                    "price": 2.0,
                },
                {
                    "txn_date": "2025-01-09",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 3,
                    "amount": 9.0,
                    "txn_id": 2,
                    "price": 3.0,
                },
                {
                    "txn_date": "2025-01-13",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 7,
                    "amount": 28.0,
                    "txn_id": 3,
                    "price": 4.0,
                },
            ]
        )
    )

    assert len(weekly) == 2
    assert weekly["units_sold"].tolist() == [8, 7]
    assert weekly["sales_dollars"].tolist() == [19.0, 28.0]
    assert weekly["n_transactions"].tolist() == [2, 1]
    assert weekly["avg_price"].tolist() == [2.5, 4.0]


def test_aggregate_weekly_labels_weeks_with_their_monday() -> None:
    from src.features.build_features import aggregate_weekly

    weekly = aggregate_weekly(
        _txns(
            [
                # A Sunday: still part of the week that began the previous Monday.
                {
                    "txn_date": "2025-01-12",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 1,
                    "amount": 2.0,
                    "txn_id": 1,
                    "price": 2.0,
                },
            ]
        )
    )
    week = pd.Timestamp(weekly["week"].iloc[0])
    assert week == pd.Timestamp("2025-01-06")
    assert week.dayofweek == 0, "week labels must be Mondays"


def test_aggregate_weekly_keeps_series_separate() -> None:
    from src.features.build_features import aggregate_weekly

    weekly = aggregate_weekly(
        _txns(
            [
                {
                    "txn_date": "2025-01-07",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 5,
                    "amount": 10.0,
                    "txn_id": 1,
                    "price": 2.0,
                },
                {
                    "txn_date": "2025-01-07",
                    "sku": "S1",
                    "region": "SE",
                    "quantity": 2,
                    "amount": 4.0,
                    "txn_id": 2,
                    "price": 2.0,
                },
                {
                    "txn_date": "2025-01-07",
                    "sku": "S2",
                    "region": "NE",
                    "quantity": 9,
                    "amount": 18.0,
                    "txn_id": 3,
                    "price": 2.0,
                },
            ]
        )
    )
    assert len(weekly) == 3
    assert set(zip(weekly["sku"], weekly["region"], strict=True)) == {
        ("S1", "NE"),
        ("S1", "SE"),
        ("S2", "NE"),
    }


def test_repeated_txn_ids_in_a_week_count_once() -> None:
    # n_transactions is nunique, not size: a duplicated row from a re-delivered
    # feed must not inflate the transaction count.
    from src.features.build_features import aggregate_weekly

    weekly = aggregate_weekly(
        _txns(
            [
                {
                    "txn_date": "2025-01-07",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 5,
                    "amount": 10.0,
                    "txn_id": 1,
                    "price": 2.0,
                },
                {
                    "txn_date": "2025-01-08",
                    "sku": "S1",
                    "region": "NE",
                    "quantity": 5,
                    "amount": 10.0,
                    "txn_id": 1,
                    "price": 2.0,
                },
            ]
        )
    )
    assert weekly["n_transactions"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Price features. Price is the one input that is partly known in advance, so
# it is the easiest place to reintroduce a leak by accident.
# ---------------------------------------------------------------------------


def _series(units, prices, start="2024-01-01"):
    weeks = pd.date_range(start, periods=len(units), freq="W-MON")
    return pd.DataFrame(
        {
            "sku": "S1",
            "region": "NE",
            "week": weeks,
            "units_sold": [float(u) for u in units],
            "sales_dollars": [float(u) * p for u, p in zip(units, prices, strict=True)],
            "n_transactions": [float(u) for u in units],
            "avg_price": [float(p) for p in prices],
        }
    )


def test_lag_1_price_is_last_weeks_price_not_this_weeks() -> None:
    from src.features.build_features import add_history_features

    prices = [2.0, 2.0, 1.0, 2.0, 2.0]
    df = add_history_features(_series([10] * 5, prices))
    assert pd.isna(df["lag_1_price"].iloc[0])
    assert df["lag_1_price"].tolist()[1:] == prices[:-1]
    # The promo week (price 1.0) must not see its own discount.
    assert df["lag_1_price"].iloc[2] == 2.0


def test_price_vs_13w_compares_lagged_price_to_lagged_history() -> None:
    from src.features.build_features import add_history_features

    prices = [4.0, 4.0, 4.0, 2.0]
    df = add_history_features(_series([10] * 4, prices))
    # Row 3: lagged price is 4.0, trailing mean of lagged prices is 4.0.
    assert df["price_vs_13w"].iloc[3] == pytest.approx(1.0)
    # Nothing on any row may reflect week 3's own 2.0.
    assert not (df["price_vs_13w"].dropna() < 1.0).any()


def test_derived_feature_columns_stay_numeric() -> None:
    # trend_1_over_4 and price_vs_13w divide by a series with zeros replaced.
    # If the replacement value promotes the column to object dtype, LightGBM
    # refuses the matrix at fit time with an error about the wrong column.
    from src.features.build_features import add_history_features

    df = add_history_features(_series([0, 0, 5, 10, 20, 30], [2.0] * 6))
    for col in ("trend_1_over_4", "price_vs_13w"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is {df[col].dtype}"


def test_calendar_features_describe_the_row_s_own_week() -> None:
    from src.features.build_features import add_history_features

    df = add_history_features(_series([1, 2, 3], [2.0] * 3, start="2024-12-30"))
    weeks = pd.to_datetime(df["week"])
    assert df["month"].tolist() == weeks.dt.month.tolist()
    assert df["year"].tolist() == weeks.dt.year.tolist()
    assert df["weekofyear"].tolist() == weeks.dt.isocalendar().week.tolist()
