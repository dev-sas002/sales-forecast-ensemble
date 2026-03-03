"""
Tests for combining model forecasts.

The bug these are built around: the previous implementation outer-joined the
two models' forecasts and averaged with `.fillna(0)`, so a week only one model
covered came out at half that model's forecast. It never raised, and the output
looked entirely reasonable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import QUANTILES
from src.ensemble.ensemble_and_reconcile import combine


def forecast(model: str, dates, values, sku="SKU-1", region="NE"):
    """One model's forecast; `values` is the median, with a fixed band around it."""
    return pd.DataFrame(
        {
            "sku": sku,
            "region": region,
            "date": pd.to_datetime(dates),
            f"q0.1_{model}": [v * 0.8 for v in values],
            f"q0.5_{model}": list(values),
            f"q0.9_{model}": [v * 1.2 for v in values],
        }
    )


class TestAveraging:
    def test_two_models_are_averaged(self):
        a = forecast("lgb", ["2025-01-06"], [100.0])
        b = forecast("deepar", ["2025-01-06"], [200.0])

        out = combine({"lgb": a, "deepar": b})

        assert out["q0.5"].iloc[0] == pytest.approx(150.0)

    def test_a_week_only_one_model_covers_keeps_that_model_value(self):
        """The regression. This must be 100, not 50."""
        a = forecast("lgb", ["2025-01-06", "2025-01-13"], [100.0, 100.0])
        b = forecast("deepar", ["2025-01-06"], [100.0])

        out = combine({"lgb": a, "deepar": b}).sort_values("date")

        assert out["q0.5"].tolist() == pytest.approx([100.0, 100.0])

    def test_a_single_model_passes_through_unchanged(self):
        a = forecast("lgb", ["2025-01-06"], [123.0])
        out = combine({"lgb": a})
        assert out["q0.5"].iloc[0] == pytest.approx(123.0)


class TestOutputContract:
    def test_quantiles_are_ordered(self):
        a = forecast("lgb", ["2025-01-06"], [50.0])
        out = combine({"lgb": a})
        assert out["q0.1"].iloc[0] <= out["q0.5"].iloc[0] <= out["q0.9"].iloc[0]

    def test_crossed_input_quantiles_are_repaired(self):
        a = forecast("lgb", ["2025-01-06"], [50.0])
        # Invert the band: q0.9 below q0.1.
        a.loc[0, "q0.1_lgb"] = 90.0
        a.loc[0, "q0.9_lgb"] = 10.0

        out = combine({"lgb": a})

        assert out["q0.1"].iloc[0] <= out["q0.5"].iloc[0] <= out["q0.9"].iloc[0]

    def test_negative_forecasts_are_clipped_to_zero(self):
        a = forecast("lgb", ["2025-01-06"], [-30.0])
        out = combine({"lgb": a})
        assert (out[[f"q{q}" for q in QUANTILES]] >= 0).all().all()

    def test_no_forecasts_at_all_is_an_error_not_an_empty_file(self):
        with pytest.raises(FileNotFoundError, match="No forecast files"):
            combine({})

    def test_a_model_missing_a_quantile_column_fails_loudly(self):
        # Silently substituting 0 for a missing column is how the original
        # produced all-zero ensembles without complaint.
        a = forecast("lgb", ["2025-01-06"], [10.0]).drop(columns=["q0.9_lgb"])
        with pytest.raises(KeyError, match=r"quantile 0\.9"):
            combine({"lgb": a})

    def test_series_are_kept_separate(self):
        a = forecast("lgb", ["2025-01-06"], [100.0], sku="SKU-A")
        b = forecast("lgb", ["2025-01-06"], [200.0], sku="SKU-B")
        out = combine({"lgb": pd.concat([a, b], ignore_index=True)})
        assert out.sort_values("sku")["q0.5"].tolist() == pytest.approx([100.0, 200.0])


class TestMergeMechanics:
    def test_the_average_is_over_the_models_that_covered_the_row(self):
        # Three "models" (the SOURCES map is extensible), two of which cover a
        # given week: the mean must be over two, not three.
        a = forecast("lgb", ["2025-01-06"], [90.0])
        b = forecast("deepar", ["2025-01-06"], [110.0])
        c = forecast("other", ["2025-01-13"], [1000.0])

        out = combine({"lgb": a, "deepar": b, "other": c}).sort_values("date")

        assert out["q0.5"].tolist() == pytest.approx([100.0, 1000.0])

    def test_a_bare_quantile_column_is_accepted_as_well_as_a_suffixed_one(self):
        # forecast_ensemble.parquet has bare q0.5; forecast_lgb.parquet has
        # q0.5_lgb. _tidy must take either, or re-ensembling a previous output
        # fails with a KeyError.
        bare = pd.DataFrame(
            {
                "sku": ["SKU-1"],
                "region": ["NE"],
                "date": pd.to_datetime(["2025-01-06"]),
                "q0.1": [8.0],
                "q0.5": [10.0],
                "q0.9": [12.0],
            }
        )
        out = combine({"lgb": bare})
        assert out["q0.5"].iloc[0] == pytest.approx(10.0)

    def test_string_dates_and_timestamps_line_up(self):
        # One model writing ISO strings and another writing timestamps must
        # still join; otherwise every row looks single-model covered.
        a = forecast("lgb", ["2025-01-06"], [100.0])
        b = forecast("deepar", ["2025-01-06"], [200.0])
        b["date"] = b["date"].dt.strftime("%Y-%m-%d")

        out = combine({"lgb": a, "deepar": b})

        assert len(out) == 1
        assert out["q0.5"].iloc[0] == pytest.approx(150.0)

    def test_output_is_sorted_and_reindexed(self):
        a = forecast("lgb", ["2025-02-03", "2025-01-06"], [2.0, 1.0])
        out = combine({"lgb": a})
        assert out["date"].is_monotonic_increasing
        assert out.index.tolist() == list(range(len(out)))

    def test_result_has_one_row_per_series_week(self):
        a = forecast("lgb", ["2025-01-06", "2025-01-13"], [10.0, 20.0], sku="SKU-A")
        b = forecast("deepar", ["2025-01-06", "2025-01-13"], [30.0, 40.0], sku="SKU-A")
        out = combine({"lgb": a, "deepar": b})
        assert len(out) == 2
        assert not out.duplicated(subset=["sku", "region", "date"]).any()

    def test_crossed_quantiles_are_repaired_without_changing_the_set(self):
        a = forecast("lgb", ["2025-01-06"], [50.0])
        a.loc[0, "q0.1_lgb"] = 90.0
        a.loc[0, "q0.9_lgb"] = 10.0

        out = combine({"lgb": a})

        # Sorting reorders the three numbers; it must not invent new ones.
        assert sorted(out[["q0.1", "q0.5", "q0.9"]].iloc[0].tolist()) == pytest.approx(
            [10.0, 50.0, 90.0]
        )
