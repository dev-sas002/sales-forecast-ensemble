"""
Leakage tests.

These are the most important tests in the repository. A leak does not raise or
crash — it inflates backtest scores and is only discovered when the model meets
real data and performs nothing like the number that justified shipping it. So
the property is checked by construction rather than by reading the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_history_features, fill_missing_weeks

FEATURE_PREFIXES = ("lag_", "roll_", "trend_", "price_")


def series(units, sku="SKU-1", region="NE", start="2024-01-01"):
    weeks = pd.date_range(start, periods=len(units), freq="W-MON")
    return pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "week": weeks,
            "units_sold": np.asarray(units, dtype=float),
            "sales_dollars": np.asarray(units, dtype=float) * 2.0,
            "n_transactions": np.asarray(units, dtype=float),
            "avg_price": 2.0,
        }
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


class TestNoTargetLeakage:
    def test_changing_only_the_last_week_changes_no_feature_before_it(self):
        """
        The decisive test.

        Take a series, change the final week's units, and rebuild. Every
        feature on every *earlier* row must be untouched. If any trailing
        window reaches forward, this fails.
        """
        base = series([10, 12, 14, 16, 18, 20, 22, 24])
        bumped = base.copy()
        bumped.loc[bumped.index[-1], "units_sold"] = 9999.0

        a = add_history_features(base)
        b = add_history_features(bumped)

        cols = feature_columns(a)
        assert cols, "no feature columns found — the test is not checking anything"
        pd.testing.assert_frame_equal(a.iloc[:-1][cols], b.iloc[:-1][cols], check_dtype=False)

    def test_no_feature_equals_the_target_it_predicts(self):
        # A rolling window that includes the current row makes roll_1w == target.
        df = add_history_features(series([5, 9, 13, 2, 40, 7, 7, 31]))
        for col in feature_columns(df):
            same = (df[col] == df["units_sold"]).fillna(False)
            # Allow an incidental coincidence, not a wholesale match.
            assert same.sum() < len(df) - 2, f"{col} tracks the target too closely"

    def test_the_first_row_of_a_series_has_no_history(self):
        df = add_history_features(series([10, 20, 30]))
        first = df.iloc[0]
        assert pd.isna(first["lag_1_units"])
        # A trailing mean with nothing behind it must be null, not zero — zero
        # is a claim about demand, null is an admission of no history.
        assert pd.isna(first["roll_4w_mean"])

    def test_lag_1_is_literally_the_previous_week(self):
        units = [3, 1, 4, 1, 5, 9, 2, 6]
        df = add_history_features(series(units))
        assert df["lag_1_units"].tolist()[1:] == [float(u) for u in units[:-1]]

    def test_history_does_not_cross_between_series(self):
        # Two series concatenated: the second must not inherit the first's tail.
        a = series([100, 100, 100], sku="SKU-A")
        b = series([1, 2, 3], sku="SKU-B")
        df = add_history_features(pd.concat([a, b], ignore_index=True))
        first_of_b = df[df["sku"] == "SKU-B"].iloc[0]
        assert pd.isna(first_of_b["lag_1_units"]), "lag bled across series boundary"


class TestGapFilling:
    def test_a_missing_week_becomes_an_explicit_zero(self):
        df = series([10, 20, 30, 40])
        df = df.drop(index=2).reset_index(drop=True)  # remove the third week

        filled = fill_missing_weeks(df)

        assert len(filled) == 4
        assert filled["units_sold"].tolist() == [10.0, 20.0, 0.0, 40.0]

    def test_price_is_carried_across_a_gap_rather_than_zeroed(self):
        # Zero units is true. Zero price is not — nothing was sold at $0.00.
        df = series([10, 20, 30, 40])
        df.loc[1, "avg_price"] = 3.0
        df = df.drop(index=2).reset_index(drop=True)

        filled = fill_missing_weeks(df)
        assert (filled["avg_price"] > 0).all()

    def test_lags_are_correct_across_a_filled_gap(self):
        # The reason gap-filling exists: without it shift(1) reaches over the
        # hole and reports a value from two weeks ago as "last week".
        df = series([10, 20, 30, 40])
        df = df.drop(index=2).reset_index(drop=True)

        featured = add_history_features(fill_missing_weeks(df))
        lags = featured["lag_1_units"]

        assert pd.isna(lags.iloc[0])
        # Week 3 was absent and is now an explicit zero, so week 4's "last
        # week" is 0 — not the 20 it would report if shift() had skipped it.
        assert lags.iloc[1:].tolist() == [10.0, 20.0, 0.0]


@pytest.mark.parametrize("window", [4, 13, 52])
def test_trailing_means_exclude_the_current_week(window):
    units = list(range(1, 61))
    df = add_history_features(series(units))
    row = df.iloc[40]
    expected = np.mean(units[max(0, 40 - window) : 40])
    assert row[f"roll_{window}w_mean"] == pytest.approx(expected)
