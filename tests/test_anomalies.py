"""
Tests for input anomaly detection.

Each test builds a series whose defect is obvious by eye, so a failure means
the detector is wrong rather than the expectation being arbitrary. The
negative cases matter as much as the positive ones: a detector that flags
ordinary seasonality is noise, and a planner learns to ignore it within a
week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.quality.anomalies import (
    AnomalyRules,
    detect,
    modified_zscore,
    summarise,
)


def weekly(units, prices=None, sku="SKU-1", region="NE"):
    weeks = pd.date_range("2025-01-06", periods=len(units), freq="W-MON")
    return pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "week": weeks,
            "units_sold": np.asarray(units, dtype=float),
            "avg_price": (
                np.full(len(units), 2.5) if prices is None else np.asarray(prices, dtype=float)
            ),
        }
    )


class TestModifiedZScore:
    def test_a_constant_series_has_no_outliers(self):
        # MAD is zero; dividing by it would report every week as infinitely
        # anomalous, which is the exact opposite of the truth.
        z = modified_zscore(pd.Series([7.0] * 10))
        assert (z == 0).all()
        assert np.isfinite(z).all()

    def test_it_finds_the_outlier_the_mean_and_sd_would_hide(self):
        # A single 100x week inflates the SD enough to mask itself; the median
        # and MAD are not moved by it.
        values = pd.Series([10.0] * 20 + [1000.0])
        z = modified_zscore(values)
        assert abs(z.iloc[-1]) > 3.5


class TestLevelAnomalies:
    def test_a_spike_is_flagged_with_its_week(self):
        units = [100.0] * 12
        units[6] = 900.0
        found = detect(weekly(units))
        spikes = found[found["kind"] == "level_spike"]
        assert len(spikes) == 1
        assert pd.Timestamp(spikes.iloc[0]["week"]) == pd.Timestamp("2025-02-17")
        assert spikes.iloc[0]["value"] == 900.0

    def test_a_collapse_is_flagged_separately_from_a_spike(self):
        units = [200.0] * 12
        units[3] = 4.0
        found = detect(weekly(units))
        assert set(found["kind"]) >= {"level_collapse"}

    def test_ordinary_seasonal_variation_is_not_flagged(self):
        # A clean annual cycle is signal, not an anomaly. Flagging it would
        # make the whole feature noise.
        i = np.arange(104)
        units = 200 + 60 * np.cos(2 * np.pi * i / 52)
        found = detect(weekly(units))
        assert found[found["kind"].str.startswith("level_")].empty

    def test_noise_alone_does_not_trip_the_threshold(self):
        rng = np.random.default_rng(4)
        units = np.clip(rng.normal(150, 15, 120), 0, None)
        found = detect(weekly(units))
        assert found[found["kind"].str.startswith("level_")].empty


class TestZeroRuns:
    def test_a_run_of_zero_weeks_in_a_busy_series_is_flagged(self):
        units = [150.0] * 10 + [0.0, 0.0, 0.0, 0.0] + [150.0] * 10
        found = detect(weekly(units))
        runs = found[found["kind"] == "zero_run"]
        assert len(runs) == 1
        assert runs.iloc[0]["severity"] == 4.0
        assert "stockout" in runs.iloc[0]["detail"]

    def test_a_single_quiet_week_is_not_a_run(self):
        units = [150.0] * 10 + [0.0] + [150.0] * 10
        assert detect(weekly(units))["kind"].tolist().count("zero_run") == 0

    def test_a_slow_mover_that_is_mostly_zero_is_not_flagged(self):
        # A SKU that genuinely sells a few units a month is not a broken feed.
        units = [0.0, 0.0, 0.0, 3.0] * 8
        assert detect(weekly(units))["kind"].tolist().count("zero_run") == 0

    def test_a_trailing_zero_run_is_closed_by_the_sentinel(self):
        # The run at the end of the series has no non-zero week after it, so
        # without the sentinel the loop never emits it.
        units = [120.0] * 10 + [0.0, 0.0, 0.0, 0.0]
        runs = detect(weekly(units))
        assert (runs["kind"] == "zero_run").sum() == 1


class TestPriceAnomalies:
    def test_a_large_price_move_is_flagged_with_its_direction(self):
        prices = [2.50] * 6 + [5.00] * 6
        found = detect(weekly([100.0] * 12, prices=prices))
        jumps = found[found["kind"] == "price_jump"]
        assert len(jumps) == 1
        assert "+100%" in jumps.iloc[0]["detail"]

    def test_a_small_price_move_is_ignored(self):
        prices = [2.50, 2.55, 2.60, 2.58, 2.62, 2.59]
        found = detect(weekly([100.0] * 6, prices=prices))
        assert found[found["kind"] == "price_jump"].empty

    def test_a_table_without_prices_still_works(self):
        df = weekly([100.0] * 8).drop(columns="avg_price")
        detect(df)  # must not raise


class TestTheContract:
    def test_the_thresholds_are_tunable_without_editing_the_module(self):
        # A noisy series with a mild bump: below the shipped threshold, above
        # a loosened one. The point is that a caller can retune without
        # forking the module.
        rng = np.random.default_rng(11)
        units = list(np.clip(rng.normal(150, 20, 60), 0, None))
        units[30] = 215.0

        assert detect(weekly(units)).empty
        loose = detect(weekly(units), AnomalyRules(z_threshold=1.5))
        assert not loose.empty

    def test_series_are_scored_independently(self):
        # A spike in one SKU must not shift another SKU's median.
        calm = weekly([100.0] * 12, sku="SKU-A")
        spiky = weekly([10.0] * 6 + [900.0] + [10.0] * 5, sku="SKU-B")
        found = detect(pd.concat([calm, spiky], ignore_index=True))
        assert set(found["sku"]) == {"SKU-B"}

    def test_findings_come_back_most_severe_first(self):
        units = [100.0] * 20
        units[4] = 600.0
        units[9] = 1200.0
        found = detect(weekly(units))
        assert found["severity"].is_monotonic_decreasing

    def test_an_empty_table_returns_an_empty_frame_with_the_right_columns(self):
        out = detect(weekly([]).head(0))
        assert out.empty
        assert list(out.columns) == ["sku", "region", "week", "kind", "severity", "detail", "value"]

    def test_a_missing_required_column_is_a_clear_error(self):
        with pytest.raises(ValueError, match="units_sold"):
            detect(weekly([1.0, 2.0]).drop(columns="units_sold"))

    def test_summarise_reports_every_kind_even_at_zero(self):
        counts = summarise(detect(weekly([100.0] * 10)))
        assert counts == {
            "level_spike": 0,
            "level_collapse": 0,
            "zero_run": 0,
            "price_jump": 0,
            "total": 0,
        }

    def test_summarise_counts_what_was_found(self):
        units = [150.0] * 10 + [0.0] * 4 + [150.0] * 10
        counts = summarise(detect(weekly(units)))
        assert counts["zero_run"] == 1
        assert counts["total"] == sum(
            counts[k] for k in ("level_spike", "level_collapse", "zero_run", "price_jump")
        )
