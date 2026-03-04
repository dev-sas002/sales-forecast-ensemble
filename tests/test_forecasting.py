"""
Tests for the quantile forecaster and the backtest protocol.

Two different things are checked here. The model tests pin properties a
forecast must satisfy to be usable at all — ordered quantiles, non-negative
demand, the right shape. The backtest tests pin the *protocol*, which matters
more: a harness that accidentally trains on the weeks it scores will report an
excellent number forever.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import QUANTILES
from src.evaluation.backtest import evaluate_fold, seasonal_naive
from src.features.build_features import add_history_features
from src.models.train_lgb_quantile import (
    forecast_from_last_origin,
    make_supervised,
    predict_quantiles,
    train_quantile_models,
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Two years of two seasonal series — enough to fit, fast to run."""
    rng = np.random.default_rng(3)
    rows = []
    weeks = pd.date_range("2023-01-02", periods=110, freq="W-MON")
    for sku, base in (("SKU-A", 100.0), ("SKU-B", 40.0)):
        for i, week in enumerate(weeks):
            seasonal = 1 + 0.3 * np.cos(2 * np.pi * (i % 52) / 52)
            units = max(0.0, base * seasonal + rng.normal(0, base * 0.06))
            rows.append(
                {
                    "sku": sku,
                    "region": "NE",
                    "week": week,
                    "units_sold": float(round(units)),
                    "sales_dollars": units * 3,
                    "n_transactions": units / 2,
                    "avg_price": 3.0,
                }
            )
    return add_history_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def trained(features):
    panel = make_supervised(features, range(1, 5))
    models, _ = train_quantile_models(panel, num_boost_round=40, seed=1)
    return models, panel


class TestSupervisedReshape:
    def test_target_is_the_value_h_weeks_after_the_origin(self, features):
        panel = make_supervised(features, range(1, 4))
        row = panel.iloc[0]
        actual = features[
            (features["sku"] == row["sku"])
            & (features["region"] == row["region"])
            & (features["week"] == row["target_week"])
        ]["units_sold"].iloc[0]
        assert row["units_sold"] == actual

    def test_rows_without_a_known_target_are_dropped(self, features):
        panel = make_supervised(features, range(1, 5))
        assert panel["units_sold"].notna().all()
        assert panel["target_week"].notna().all()

    def test_every_horizon_is_represented(self, features):
        panel = make_supervised(features, range(1, 5))
        assert sorted(panel["horizon"].unique()) == [1, 2, 3, 4]

    def test_targets_never_precede_their_origin(self, features):
        panel = make_supervised(features, range(1, 5))
        assert (pd.to_datetime(panel["target_week"]) > pd.to_datetime(panel["week"])).all()


class TestPredictions:
    def test_quantiles_are_ordered_for_every_row(self, trained):
        models, panel = trained
        preds = predict_quantiles(models, panel.head(200))
        cols = [f"q{q}_lgb" for q in QUANTILES]
        values = preds[cols].to_numpy()
        assert (np.diff(values, axis=1) >= 0).all(), "quantile crossing in output"

    def test_forecasts_are_never_negative(self, trained):
        models, panel = trained
        preds = predict_quantiles(models, panel.head(200))
        assert (preds[[f"q{q}_lgb" for q in QUANTILES]] >= 0).all().all()

    def test_the_median_is_in_the_right_ballpark(self, trained, features):
        # Not an accuracy assertion — a smoke test that the model learned the
        # level at all rather than predicting near zero or wildly high.
        models, panel = trained
        preds = predict_quantiles(models, panel)
        for sku, expected in (("SKU-A", 100.0), ("SKU-B", 40.0)):
            median = preds[preds["sku"] == sku]["q0.5_lgb"].mean()
            assert 0.5 * expected < median < 1.6 * expected

    def test_forecast_horizon_is_produced_for_every_series(self, trained, features):
        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=6)
        assert len(out) == 2 * 6
        assert sorted(out["horizon"].unique()) == [1, 2, 3, 4, 5, 6]

    def test_forecast_dates_start_after_the_last_observed_week(self, trained, features):
        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=3)
        assert pd.to_datetime(out["date"]).min() > pd.to_datetime(features["week"]).max()


class TestBacktestProtocol:
    def test_a_fold_scores_only_weeks_after_its_cutoff(self, features):
        cutoff = pd.Timestamp("2024-08-05")
        fold = evaluate_fold(features, cutoff, horizon=4, rounds=30, seed=1)
        assert fold is not None
        assert fold.n_rows > 0
        assert fold.cutoff == cutoff

    def test_coverage_is_a_proportion(self, features):
        fold = evaluate_fold(features, pd.Timestamp("2024-08-05"), 4, 30, 1)
        assert 0.0 <= fold.coverage <= 1.0

    def test_a_cutoff_with_too_little_history_is_skipped_not_scored(self, features):
        # Returning a score from three weeks of history would be worse than
        # returning nothing, because it would go in the table.
        assert evaluate_fold(features, pd.Timestamp("2023-01-16"), 4, 30, 1) is None

    def test_metrics_are_finite(self, features):
        fold = evaluate_fold(features, pd.Timestamp("2024-08-05"), 4, 30, 1)
        assert np.isfinite(fold.wape_model)
        assert np.isfinite(fold.wape_naive)
        assert all(np.isfinite(v) for v in fold.pinball.values())

    def test_backtest_is_deterministic(self, features):
        a = evaluate_fold(features, pd.Timestamp("2024-08-05"), 4, 30, 1)
        b = evaluate_fold(features, pd.Timestamp("2024-08-05"), 4, 30, 1)
        # A score that moves between identical runs is not a measurement.
        assert a.wape_model == pytest.approx(b.wape_model)


class TestSeasonalNaiveBaseline:
    def test_it_uses_the_same_week_last_year(self, features):
        targets = pd.DataFrame(
            {
                "sku": ["SKU-A"],
                "region": ["NE"],
                "date": [pd.Timestamp("2024-07-01")],
            }
        )
        expected = features[
            (features["sku"] == "SKU-A") & (features["week"] == pd.Timestamp("2023-07-03"))
        ]
        # Not conditional: if that week is not in the fixture the test is
        # asserting nothing, which is worse than failing.
        assert len(expected) == 1, "fixture no longer contains the lookback week"
        baseline = seasonal_naive(features, targets)
        assert baseline.iloc[0] == pytest.approx(expected["units_sold"].iloc[0])

    def test_it_falls_back_when_a_year_of_history_is_missing(self, features):
        short = features[features["week"] < "2023-06-01"]
        targets = pd.DataFrame(
            {"sku": ["SKU-A"], "region": ["NE"], "date": [pd.Timestamp("2023-06-05")]}
        )
        value = seasonal_naive(short, targets).iloc[0]
        assert np.isfinite(value)


class TestForecastPath:
    """The path `make pipeline` actually runs to produce forecast_lgb.parquet."""

    def test_forecast_dates_are_exactly_h_weeks_after_the_last_origin(self, trained, features):
        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=5)

        last_week = pd.to_datetime(features["week"]).max()
        for h in range(1, 6):
            dates = pd.to_datetime(out[out["horizon"] == h]["date"]).unique()
            assert list(dates) == [last_week + pd.Timedelta(weeks=h)]

    def test_forecast_covers_every_series_exactly_once_per_horizon(self, trained, features):
        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=4)
        counts = out.groupby(["sku", "region", "horizon"]).size()
        assert (counts == 1).all()

    def test_forecast_quantiles_are_ordered_and_non_negative(self, trained, features):
        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=4)
        cols = [f"q{q}_lgb" for q in QUANTILES]
        values = out[cols].to_numpy()
        assert (values >= 0).all()
        assert (np.diff(values, axis=1) >= 0).all()

    def test_forecast_columns_match_what_the_ensemble_expects(self, trained, features):
        from src.ensemble.ensemble_and_reconcile import combine

        models, _ = trained
        out = forecast_from_last_origin(models, features, horizon=3)
        # combine() raises KeyError if a quantile column is missing, so this
        # pins the contract between the trainer and the ensemble.
        merged = combine({"lgb": out})
        assert len(merged) == len(out)
        assert list(merged.columns) == ["sku", "region", "date"] + [f"q{q}" for q in QUANTILES]

    def test_forecasting_twice_gives_the_same_numbers(self, trained, features):
        models, _ = trained
        a = forecast_from_last_origin(models, features, horizon=3)
        b = forecast_from_last_origin(models, features, horizon=3)
        pd.testing.assert_frame_equal(a, b)

    def test_training_twice_with_the_same_seed_gives_the_same_model(self, features):
        panel = make_supervised(features, range(1, 3))
        m1, _ = train_quantile_models(panel, num_boost_round=15, seed=4)
        m2, _ = train_quantile_models(panel, num_boost_round=15, seed=4)
        p1 = predict_quantiles(m1, panel.head(50))
        p2 = predict_quantiles(m2, panel.head(50))
        pd.testing.assert_frame_equal(p1, p2)


class TestFeatureMatrixContract:
    def test_the_target_column_is_never_a_model_input(self, features):
        from src.models.train_lgb_quantile import TARGET, feature_matrix

        panel = make_supervised(features, range(1, 3))
        _, cols = feature_matrix(panel)
        assert TARGET not in cols
        assert "target_week" not in cols
        assert "week" not in cols

    def test_horizon_and_target_calendar_are_inputs(self, features):
        from src.models.train_lgb_quantile import feature_matrix

        panel = make_supervised(features, range(1, 3))
        _, cols = feature_matrix(panel)
        for expected in ("horizon", "target_weekofyear", "target_month"):
            assert expected in cols

    def test_the_feature_matrix_is_all_numeric_or_categorical(self, features):
        from src.models.train_lgb_quantile import CATEGORICAL, feature_matrix

        panel = make_supervised(features, range(1, 3))
        X, _ = feature_matrix(panel)
        for col in X.columns:
            if col in CATEGORICAL:
                assert isinstance(X[col].dtype, pd.CategoricalDtype)
            else:
                assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is {X[col].dtype}"


class TestSeasonalNaiveFallback:
    def test_the_fallback_is_the_latest_week_even_if_rows_are_unordered(self):
        # The fallback used to be `series[TARGET].iloc[-1]` on the frame as
        # given, so a feature table that arrived in any order other than
        # sorted-by-week produced a baseline from an arbitrary week.
        weeks = pd.date_range("2024-01-01", periods=5, freq="W-MON")
        df = pd.DataFrame(
            {
                "sku": "S1",
                "region": "NE",
                "week": weeks,
                "units_sold": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        shuffled = df.iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)
        targets = pd.DataFrame(
            {"sku": ["S1"], "region": ["NE"], "date": [pd.Timestamp("2024-03-04")]}
        )
        # No week 52 weeks before the target, so the fallback is used.
        assert seasonal_naive(shuffled, targets).iloc[0] == pytest.approx(5.0)

    def test_an_unknown_series_falls_back_to_zero_rather_than_raising(self):
        weeks = pd.date_range("2024-01-01", periods=3, freq="W-MON")
        df = pd.DataFrame(
            {"sku": "S1", "region": "NE", "week": weeks, "units_sold": [1.0, 2.0, 3.0]}
        )
        targets = pd.DataFrame(
            {"sku": ["S9"], "region": ["XX"], "date": [pd.Timestamp("2024-02-05")]}
        )
        assert seasonal_naive(df, targets).iloc[0] == 0.0

    def test_it_prefers_last_year_over_the_fallback(self):
        weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
        units = [float(i) for i in range(60)]
        df = pd.DataFrame({"sku": "S1", "region": "NE", "week": weeks, "units_sold": units})
        target_date = weeks[0] + pd.Timedelta(weeks=52)
        targets = pd.DataFrame({"sku": ["S1"], "region": ["NE"], "date": [target_date]})
        # 52 weeks before the target is week 0, whose value is 0.0 — not the
        # last observed value of 59.0.
        assert seasonal_naive(df, targets).iloc[0] == pytest.approx(0.0)
