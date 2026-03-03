"""
Tests for the feature registry — the seam, and the guarantee it encodes.

`test_no_leakage.py` proves the *current* features do not reach forward.
These prove the stronger, structural property: that a feature written through
this interface tomorrow cannot reach forward either, because the interface
does not expose the current week at all. That is the difference between a
codebase that happens to be correct and one that stays correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import registry
from src.features.build_features import add_history_features
from src.features.registry import (
    History,
    HistoryFeature,
    history_feature_names,
    validate_registry,
)


def series(units, sku="SKU-1", region="NE"):
    weeks = pd.date_range("2024-01-01", periods=len(units), freq="W-MON")
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


def a_history(units, prices=None) -> History:
    """A History built the same way build_features builds one."""
    df = series(units)
    if prices is not None:
        df["avg_price"] = prices
    prior_units = df.groupby(["sku", "region"], sort=False)["units_sold"].shift(1)
    prior_price = df.groupby(["sku", "region"], sort=False)["avg_price"].shift(1)
    idx = [df["sku"], df["region"]]
    return History(
        prior_units=prior_units.groupby(idx, sort=False),
        prior_price=prior_price.groupby(idx, sort=False),
    )


class TestTheCurrentWeekIsUnreachable:
    """The whole point of the History type."""

    def test_lag_zero_is_rejected(self):
        # lag(0) would be "this week's units", which is the target. It must
        # not be expressible, not merely discouraged.
        with pytest.raises(ValueError, match="at least 1 week back"):
            a_history([1, 2, 3, 4]).lag(0)

    def test_a_negative_lag_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1 week back"):
            a_history([1, 2, 3, 4]).lag(-3)

    def test_price_lag_zero_is_rejected(self):
        # The realised average price reflects the mix that sold that week, so
        # it is as much of a leak as the target itself.
        with pytest.raises(ValueError, match="at least 1 week back"):
            a_history([1, 2, 3, 4]).price_lag(0)

    def test_lag_one_is_last_week_not_this_week(self):
        units = [5.0, 8.0, 13.0, 21.0]
        got = a_history(units).lag(1).tolist()
        assert pd.isna(got[0])
        assert got[1:] == units[:-1]

    def test_lag_n_is_the_raw_series_shifted_n(self):
        units = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        h = a_history(units)
        for n in (1, 2, 3):
            got = h.lag(n)
            assert got.iloc[n:].tolist() == units[: len(units) - n]
            assert got.iloc[:n].isna().all()

    def test_a_trailing_mean_never_includes_the_row_it_describes(self):
        units = [10.0, 20.0, 30.0, 40.0, 50.0]
        got = a_history(units).mean(3, min_periods=1)
        # Row 3's window is weeks 0..2, not 0..3.
        assert got.iloc[3] == pytest.approx(np.mean(units[:3]))


class TestEveryRegisteredFeatureIsBackwardLooking:
    """
    Enumerated from the registry, so a feature added later is covered here
    without anyone remembering to add a test for it.
    """

    @pytest.mark.parametrize("name", history_feature_names())
    def test_changing_the_last_week_leaves_this_feature_untouched_before_it(self, name):
        base = series([12, 14, 11, 19, 23, 17, 25, 31, 22, 28])
        bumped = base.copy()
        bumped.loc[bumped.index[-1], "units_sold"] = 99_999.0

        a = add_history_features(base)[name].iloc[:-1]
        b = add_history_features(bumped)[name].iloc[:-1]
        pd.testing.assert_series_equal(a, b, check_dtype=False)

    @pytest.mark.parametrize("name", history_feature_names())
    def test_this_feature_is_numeric(self, name):
        # The regression that broke LightGBM: a pd.NA divide-guard promoted a
        # derived column to object dtype, but only for series containing a
        # zero trailing window — so it failed on sparse data and not dense.
        df = add_history_features(series([4, 0, 0, 0, 0, 7, 9, 0, 0, 11]))
        assert pd.api.types.is_numeric_dtype(df[name]), f"{name} is not numeric"


class TestRegistryInvariants:
    def test_the_shipped_registry_is_valid(self):
        validate_registry()

    def test_a_duplicate_name_is_rejected(self):
        dupe = HistoryFeature("lag_1_units", "x", lambda h: h.lag(1))
        with pytest.raises(ValueError, match="duplicate history feature names"):
            validate_registry([dupe, dupe])

    def test_a_name_the_trainer_would_ignore_is_rejected(self):
        # The trainer selects model inputs by prefix. A feature named outside
        # that set is computed, written to parquet and silently never used.
        stray = HistoryFeature("units_last_week", "x", lambda h: h.lag(1))
        with pytest.raises(ValueError, match="do not start with"):
            validate_registry([stray])

    def test_every_registered_name_reaches_the_built_frame(self):
        built = add_history_features(series(list(range(1, 15))))
        missing = [n for n in history_feature_names() if n not in built.columns]
        assert not missing, f"registered but not built: {missing}"

    def test_descriptions_cover_every_feature(self):
        # The dashboard renders these; a missing one shows a raw column name.
        described = registry.describe()
        for feature in (*registry.history_features(), *registry.calendar_features()):
            assert described.get(feature.name), f"{feature.name} has no description"
